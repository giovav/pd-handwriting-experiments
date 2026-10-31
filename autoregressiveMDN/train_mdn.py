"""
train_mdn.py: fine-tuning dell'MDN (Graves 2013) su un fold, con teacher forcing.

Parte dai pesi IAM-OnDB. Non si congela nulla: i parametri nuovi (embedding, ingressi
dei canali extra, teste pen e aux) usano lr_new senza weight decay, quelli pretrained
lr_pretrained con weight decay. Early stopping sulla loss di validazione.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# I moduli condivisi stanno in common/ (se c'è una copia accanto allo script, vale quella).
sys.path.insert(1, str(Path(__file__).resolve().parents[1] / "common"))

import train_common as tc
from train_common import dp
from mdn_head import mdn_loss
from mdn_model import PaHaWMDN, generate, teacher_forcing_inputs
from pretrained import load_pretrained_mdn, new_parameter_names


def build_mdn(config, pretrained_dir: str | None = None, verbose: bool = True) -> PaHaWMDN:
    """MDN con i pesi ufficiali (IAM-OnDB)."""
    model = PaHaWMDN(config)
    report = load_pretrained_mdn(model, tc.pretrained_file(config.pretrained_path, pretrained_dir))
    if verbose:
        report.print(model)
    return model


def load_finetuned_mdn(config, checkpoint_path, device="cpu") -> PaHaWMDN:
    """MDN fine-tuned, letto da un checkpoint del training."""
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = PaHaWMDN(config)
    model.load_state_dict(saved["model_state_dict"])
    return model.to(device).eval()


def parameter_groups(model: PaHaWMDN, config) -> list[dict]:
    """
    Gruppi per AdamW: parametri nuovi a lr_new senza weight decay, pesi pretrained a
    lr_pretrained con weight decay.
    """
    new = set(new_parameter_names(model))
    named = list(model.named_parameters())
    return [
        {"params": [p for n, p in named if n in new], "lr": config.lr_new, "weight_decay": 0.0},
        {"params": [p for n, p in named if n not in new], "lr": config.lr_pretrained,
         "weight_decay": config.weight_decay},
    ]


def loss_terms(model: PaHaWMDN, batch: dict) -> dict:
    """I tre termini della loss in teacher forcing."""
    x, mask = batch["x"], batch["mask"]
    raw, _ = model(teacher_forcing_inputs(x), batch["y"], batch["task"])
    return mdn_loss(raw, x, mask)


def weighted_total(terms: dict, config) -> torch.Tensor:
    return config.gmm_weight * terms["gmm"] + config.aux_weight * terms["aux"] + config.pen_weight * terms["pen"]


@torch.no_grad()
def final_report(model, val_set, stats, train_set, device, config, seed: int) -> dict:
    """
    Report finale sul miglior modello: termini della loss di validazione (in totale e per
    classe) e una preview con una registrazione generata per ogni (classe, task), per
    controllare che la generazione libera resti finita.
    """
    model.eval()
    report = {}
    for label, name in ((None, "all"), (0, "healthy"), (1, "pd")):
        items = [i for i, item in enumerate(val_set.items) if label is None or item["y"] == label]
        if not items:
            continue
        sums, steps = {"gmm": 0.0, "aux": 0.0, "pen": 0.0}, 0
        for i in items:
            batch = tc.to_device(dp.collate_fn([val_set[i]]), device)
            n = int(batch["mask"].sum())
            for key, value in loss_terms(model, batch).items():
                sums[key] += value.item() * n
            steps += n
        report[f"val_{name}"] = {key: round(value / steps, 5) for key, value in sums.items()}

    ys, tasks, lengths = [], [], []
    for y in (0, 1):
        for task in range(config.num_tasks):
            same = [len(item["x"]) for item in train_set.items if item["y"] == y and item["task"] == task]
            if same:
                ys.append(y), tasks.append(task), lengths.append(int(np.median(same)))
                
    in_air_pressure = float((0.0 - stats.mean[dp.PRESSURE]) / stats.std[dp.PRESSURE])
    generator = torch.Generator(device=device).manual_seed(seed)
    
    x, mask = generate(model, torch.tensor(ys, device=device), torch.tensor(tasks, device=device),
                       torch.tensor(lengths), bias=config.sampling_bias,
                       in_air_pressure=in_air_pressure, generator=generator)
                       
    x, mask = x.cpu().numpy(), mask.cpu().numpy()
    finite = bool(np.isfinite(x).all())
    offsets = np.abs(x[..., :2][mask])
    
    report["preview"] = {"recordings": len(ys), "all_finite": finite,
                         "max_abs_offset_normalized": round(float(offsets.max()), 2) if finite else None,
                         "pen_down_share": round(float(x[..., dp.PEN][mask].mean()), 3)}
                         
    raw = [dp.denormalize(x[i, :lengths[i]], stats) for i in range(len(ys))]
    report["_preview_arrays"] = (raw, ys, tasks)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tuning dell'MDN su un fold.")
    parser.add_argument("--fold", type=int, required=True, help="fold, da 0 a 3")
    parser.add_argument("--epochs", type=int, default=200, help="numero massimo di epoche")
    parser.add_argument("--patience", type=int, default=20, help="epoche senza miglioramento della loss di validazione prima di fermarsi")
    parser.add_argument("--batch_size", type=int, default=16, help="registrazioni per batch, al massimo")
    parser.add_argument("--step_budget", type=float, default=40_000,
                        help="limite di registrazioni x lunghezza con padding per batch; abbassarlo se la GPU esaurisce la memoria")
    parser.add_argument("--accum", type=int, default=1, help="batch accumulati per ogni aggiornamento dei pesi")
    parser.add_argument("--clip", type=float, default=1.0, help="clipping della norma del gradiente")
    parser.add_argument("--max_seq_len", type=int, default=None, help="esclude le registrazioni più lunghe di così (in campioni)")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out_root", type=str, default="runs/mdn")
    tc.add_data_arguments(parser)
    parser.add_argument("--resume", action="store_true", help="continua dall'ultimo checkpoint del run")
    parser.add_argument("--overwrite", action="store_true", help="cancella il run esistente e ricomincia")
    parser.add_argument("--smoke", action="store_true",
                        help="prova rapida: 2 epoche da 3 batch, risultati in <out_root>_smoke")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    limit = 3 if args.smoke else None
    if args.smoke:
        args.epochs, args.step_budget = 2, min(args.step_budget, 6_000)
        args.out_root, args.overwrite = args.out_root + "_smoke", True

    tc.seed_everything(args.seed)
    device = tc.get_device()
    config = tc.load_config(args.config)
    out_dir = tc.run_dir(args.fold, args.out_root)
    print(f"Device: {device}    risultati: {out_dir}")

    train_set, val_set, stats = tc.build_datasets(args.fold, out_dir, args.max_seq_len,
                                                  args.data_file, args.split_file)
                                                  
    train_sampler = tc.StepBudgetSampler(train_set.lengths, args.batch_size, args.step_budget,
                                         shuffle=True, seed=args.seed)
    val_sampler = tc.StepBudgetSampler(val_set.lengths, args.batch_size, args.step_budget, shuffle=False)
    train_loader = DataLoader(train_set, batch_sampler=train_sampler, collate_fn=dp.collate_fn)
    val_loader = DataLoader(val_set, batch_sampler=val_sampler, collate_fn=dp.collate_fn)
    print(f"Batch: {train_sampler.describe()}")

    model = build_mdn(config, args.pretrained_dir).to(device)
    n_new = sum(p.numel() for n, p in model.named_parameters() if n in set(new_parameter_names(model)))
    n_all = sum(p.numel() for p in model.parameters())
    print(f"MDN: {n_all / 1e6:.2f} M parametri addestrabili, "
          f"di cui {n_new / 1e6:.3f} M nuovi (lr {config.lr_new:g}); gli altri dal checkpoint IAM-OnDB (lr {config.lr_pretrained:g})")

    terms = {"sum": {}, "n": 0, "last": {}}

    def loss_fn(batch):
        parts = loss_terms(model, batch)
        terms["last"] = {k: round(v.item(), 4) for k, v in parts.items()}
        terms["n"] += 1
        for name, value in parts.items():
            terms["sum"][name] = terms["sum"].get(name, 0.0) + value.item()
        return weighted_total(parts, config)

    def term_means() -> dict:
        means = {name: total / max(terms["n"], 1) for name, total in terms["sum"].items()}
        terms["sum"], terms["n"] = {}, 0
        return means

    explain = lambda batch: f"termini della loss: {terms['last']}"

    optimizer = tc.make_optimizer(parameter_groups(model, config))
    trainable = list(model.parameters())

    stopper = tc.EarlyStopper(args.patience)
    state = {"step": 0}
    start_epoch = 0
    last_path, best_path = out_dir / "mdn_last.pt", out_dir / "mdn_best.pt"
    tc.check_not_overwriting(last_path, args.resume, args.overwrite)

    if args.resume and last_path.exists():
        saved = torch.load(last_path, map_location="cpu", weights_only=True)
        tc.check_checkpoint_is_healthy(saved["model_state_dict"], last_path)
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        stopper.load_state_dict(saved["stopper"])
        state = {"step": saved["step"]}
        start_epoch = saved["epoch"] + 1
        print(f"Ripreso dall'epoca {saved['epoch']} (migliore loss di validazione: {stopper.best:.4f})")

    log = tc.CsvLog(out_dir / "mdn_log.csv",
                    ["epoch", "train_loss", "val_loss", "best", "seconds", "train_gmm", "train_aux", "train_pen"],
                    keep_existing=args.resume)
    timer = tc.Timer()

    for epoch in range(start_epoch, args.epochs):
        if stopper.should_stop:
            print("Questo run si era già fermato per early stopping: non c'è niente da fare.")
            break
            
        model.train()
        tc.seed_everything(args.seed + epoch + 1)
        term_means()
        
        train_loss = tc.train_one_epoch(loss_fn, train_loader, optimizer, trainable, device, state,
                                        accum=args.accum, clip=args.clip, limit_batches=limit, explain=explain)
        train_terms = term_means()

        model.eval()
        val_loss = tc.evaluate(loss_fn, val_loader, device, limit_batches=limit)
        tc.require_finite(val_loss, f"loss di validazione all'epoca {epoch}")
        tc.check_checkpoint_is_healthy(model.state_dict(), "pesi del modello")

        is_best = stopper.update(val_loss, epoch)
        if is_best:
            tc.save_checkpoint({"model_state_dict": model.state_dict(), "epoch": epoch, "val_loss": val_loss,
                                "fold": args.fold}, best_path)
                                
        tc.save_checkpoint({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                            "epoch": epoch, "step": state["step"], "stopper": stopper.state_dict(),
                            "fold": args.fold}, last_path)

        seconds = timer.lap()
        log.write(epoch=epoch, train_loss=round(train_loss, 5), val_loss=round(val_loss, 5),
                  best=int(is_best), seconds=round(seconds, 1),
                  **{f"train_{name}": round(value, 5) for name, value in train_terms.items()})
                  
        print(f"Epoca {epoch:3d}  TrainLoss {train_loss:8.4f}   ValLoss {val_loss:8.4f}"
              f"{'  *Best Model*' if is_best else ''}   ({seconds:.0f} s)   "
              + "  ".join(f"{name} {value:.3f}" for name, value in train_terms.items()))

        if stopper.should_stop:
            print(f"Early stopping: {args.patience} epoche senza miglioramento.")
            break

    best = torch.load(best_path, map_location="cpu", weights_only=True)
    model.load_state_dict(best["model_state_dict"])
    report = final_report(model, val_set, stats, train_set, device, config, args.seed)
    raw, ys, tasks = report.pop("_preview_arrays")
    
    np.savez_compressed(out_dir / "mdn_preview.npz", lengths=np.array([len(r) for r in raw]),
                        features=np.concatenate(raw), labels=np.array(ys), tasks=np.array(tasks))
    report.update({"fold": args.fold, "best_epoch": best["epoch"], "best_val_loss": round(best["val_loss"], 5)})
    
    with open(out_dir / "mdn_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nTraining finito. Epoca migliore: {best['epoch']}  (ValLoss {best['val_loss']:.4f})")
    for key in ("val_all", "val_healthy", "val_pd"):
        if key in report:
            print(f"   {key:12s} " + "  ".join(f"{k} {v:.3f}" for k, v in report[key].items()))
            
    p = report["preview"]
    print(f"Preview ({p['recordings']} registrazioni generate): tutti i valori finiti: {p['all_finite']}, "
          f"|offset| massimo {p['max_abs_offset_normalized']} dev. std, penna giù {100 * p['pen_down_share']:.0f}%")
    print(f"Salvato: {best_path}")


if __name__ == "__main__":
    OutOfMemory = getattr(torch, "OutOfMemoryError", torch.cuda.OutOfMemoryError)
    try:
        main()
    except OutOfMemory as error:
        raise SystemExit(
            f"\nMemoria GPU esaurita ({str(error).splitlines()[0]})\n"
            f"Abbassare il limite, per esempio --step_budget 20000 (ora 40000), e rilanciare con --resume: "
            f"le epoche già salvate non vanno perse."
        )