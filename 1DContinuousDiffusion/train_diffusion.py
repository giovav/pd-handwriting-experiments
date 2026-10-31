"""
train_diffusion.py: fine-tuning del modello di Luhman e Luhman (2020) su un fold.

Parte da pretrained/luhman_pahaw_init.pt, scritto da convert_official_weights.py. Loss di
diffusion.py, tre gruppi di parametri con AdamW, early stopping sulla loss di validazione
calcolata con rumore fisso e media su più passate.
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
from diffusion import diffusion_loss, sample
from diffusion_model import PaHaWDiffusionWriter
from pretrained import load_pahaw_init, new_parameter_names


def build_model(config, pretrained_dir: str | None = None) -> PaHaWDiffusionWriter:
    """Rete PaHaW con i pesi di luhman_pahaw_init.pt."""
    model = PaHaWDiffusionWriter(config)
    path = tc.pretrained_file(config.pretrained_path, pretrained_dir)
    record = load_pahaw_init(model, path)
    
    print(f"\n=== Diffusion1D: pesi di partenza da {path} ===")
    print(f"   SHA-256 del file .h5 ufficiale: {record['source_h5_sha256'][:16]}...;  "
          f"{len(record['loaded'])} tensori caricati ({len(record['flipped'])} col segno invertito), nuovi: {record['new']}")
    return model


def load_finetuned_diffusion(config, checkpoint_path, device="cpu") -> PaHaWDiffusionWriter:
    """Rete fine-tuned, letta da un checkpoint del training."""
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = PaHaWDiffusionWriter(config)
    model.load_state_dict(saved["model_state_dict"])
    return model.to(device).eval()


REQUIRED_KEYS = ("c1", "c2", "c3", "num_att_layers", "num_classes", "num_tasks", "pretrained_path",
                 "noise_xy_weight", "noise_aux_weight", "pen_weight", "lr_new", "lr_pretrained", "weight_decay")


def check_config(config, path) -> None:
    """Si ferma se nella configurazione manca una delle chiavi necessarie."""
    missing = [key for key in REQUIRED_KEYS if not hasattr(config, key)]
    if missing:
        raise SystemExit(f"In {path} mancano delle impostazioni: "
                         f"{', '.join(missing)}.\nIl file di configurazione è probabilmente una versione vecchia.")


def parameter_groups(model: PaHaWDiffusionWriter, config) -> list[dict]:
    """
    Tre gruppi per AdamW:
        1. parametri nuovi (embedding, input_new, output_new): lr_new, senza weight decay;
        2. matrici e kernel pretrained: lr_pretrained, con weight decay;
        3. bias pretrained: lr_pretrained, senza weight decay.
    """
    new = set(new_parameter_names(model))
    named = list(model.named_parameters())
    return [
        {"params": [p for n, p in named if n in new], "lr": config.lr_new, "weight_decay": 0.0},
        {"params": [p for n, p in named if n not in new and p.dim() > 1], "lr": config.lr_pretrained,
         "weight_decay": config.weight_decay},
        {"params": [p for n, p in named if n not in new and p.dim() <= 1], "lr": config.lr_pretrained,
         "weight_decay": 0.0},
    ]


def weighted_total(terms: dict, config) -> torch.Tensor:
    return (config.noise_xy_weight * terms["noise_xy"] + config.noise_aux_weight * terms["noise_aux"]
            + config.pen_weight * terms["pen"])


class HalfResolutionBudgetSampler(tc.AttentionBudgetSampler):
    """
    AttentionBudgetSampler con le lunghezze moltiplicate per 4.

    L'attention più costosa di questa rete è a T/2 (enc3), non a T/8 come in DiffInk, per cui
    il sampler comune ha le lunghezze in frame T/8.
    """
    def __init__(self, lengths, max_batch_size, budget, shuffle, seed=None):
        super().__init__([4 * n for n in lengths], max_batch_size, budget, shuffle, seed=seed)


@torch.no_grad()
def final_report(model, val_set, train_set, stats, device, config, seed: int, repeats: int) -> dict:
    """
    Report finale sul miglior modello: termini della loss di validazione (in totale e per
    classe, rumore fisso, media su più passate) e una preview con una registrazione generata
    per ogni (classe, task) con il sampler completo.
    """
    model.eval()
    report = {}
    for label, name in ((None, "all"), (0, "healthy"), (1, "pd")):
        items = [i for i, item in enumerate(val_set.items) if label is None or item["y"] == label]
        if not items:
            continue
            
        sums, steps = {"noise_xy": 0.0, "noise_aux": 0.0, "pen": 0.0}, 0
        tc.seed_everything(seed)
        
        for _ in range(repeats):
            for i in items:
                batch = tc.to_device(dp.collate_fn([val_set[i]]), device)
                n = int(batch["mask"].sum())
                terms = diffusion_loss(model, batch["x"], batch["mask"], batch["y"], batch["task"])
                for key, value in terms.items():
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
    
    x, mask = sample(model, torch.tensor(ys, device=device), torch.tensor(tasks, device=device),
                     torch.tensor(lengths), in_air_pressure=in_air_pressure, generator=generator)
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
    parser = argparse.ArgumentParser(description="Fine-tuning del modello di diffusione 1-D su un fold.")
    parser.add_argument("--fold", type=int, required=True, help="fold, da 0 a 3")
    parser.add_argument("--epochs", type=int, default=300, help="numero massimo di epoche")
    parser.add_argument("--patience", type=int, default=30, help="epoche senza miglioramento della loss di validazione prima di fermarsi")
    parser.add_argument("--batch_size", type=int, default=16, help="registrazioni per batch, al massimo")
    parser.add_argument("--attention_budget", type=float, default=4e7,
                        help="limite di batch x (T/2)^2 per batch; abbassarlo se la GPU esaurisce la memoria")
    parser.add_argument("--accum", type=int, default=1, help="batch accumulati per ogni aggiornamento dei pesi")
    parser.add_argument("--clip", type=float, default=1.0, help="clipping della norma del gradiente")
    parser.add_argument("--val_repeats", type=int, default=4, help="passate sulla validazione per epoca (la loss dipende dal rumore)")
    parser.add_argument("--max_seq_len", type=int, default=None, help="esclude le registrazioni più lunghe di così (in campioni)")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out_root", type=str, default="runs/diffusion")
    tc.add_data_arguments(parser)
    parser.add_argument("--resume", action="store_true", help="continua dall'ultimo checkpoint del run")
    parser.add_argument("--overwrite", action="store_true", help="cancella il run esistente e ricomincia")
    parser.add_argument("--smoke", action="store_true",
                        help="prova rapida: 2 epoche da 3 batch, risultati in <out_root>_smoke")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    limit = 3 if args.smoke else None
    if args.smoke:
        args.epochs, args.val_repeats = 2, 1
        args.out_root, args.overwrite = args.out_root + "_smoke", True

    tc.seed_everything(args.seed)
    device = tc.get_device()
    config = tc.load_config(args.config)
    check_config(config, args.config or tc.MODEL_DIR / "configs" / "pahaw_config.yaml")
    out_dir = tc.run_dir(args.fold, args.out_root)
    print(f"Device: {device}    risultati: {out_dir}")

    train_set, val_set, stats = tc.build_datasets(args.fold, out_dir, args.max_seq_len,
                                                  args.data_file, args.split_file)
                                                  
    train_sampler = HalfResolutionBudgetSampler(train_set.lengths, args.batch_size, args.attention_budget,
                                                shuffle=True, seed=args.seed)
    val_sampler = HalfResolutionBudgetSampler(val_set.lengths, args.batch_size, args.attention_budget, shuffle=False)
    
    train_loader = DataLoader(train_set, batch_sampler=train_sampler, collate_fn=dp.collate_fn)
    val_loader = DataLoader(val_set, batch_sampler=val_sampler, collate_fn=dp.collate_fn)
    print(f"Batch: {train_sampler.describe()}")

    model = build_model(config, args.pretrained_dir).to(device)
    groups = parameter_groups(model, config)
    sizes = [sum(p.numel() for p in g["params"]) for g in groups]
    print(f"Rete: {sum(sizes) / 1e6:.2f} M parametri addestrabili: {sizes[0]:,} nuovi "
          f"(lr {config.lr_new:g}), {sizes[1] + sizes[2]:,} pretrained (lr {config.lr_pretrained:g}; "
          f"weight decay {config.weight_decay:g} su {sizes[1]:,} valori di matrici e kernel)")

    terms = {"sum": {}, "n": 0, "last": {}}

    def loss_fn(batch):
        parts = diffusion_loss(model, batch["x"], batch["mask"], batch["y"], batch["task"])
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
    optimizer = tc.make_optimizer(groups)
    trainable = list(model.parameters())

    stopper = tc.EarlyStopper(args.patience)
    state = {"step": 0}
    start_epoch = 0
    last_path, best_path = out_dir / "diffusion_last.pt", out_dir / "diffusion_best.pt"
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

    log = tc.CsvLog(out_dir / "diffusion_log.csv",
                    ["epoch", "train_loss", "val_loss", "best", "seconds", "train_noise_xy", "train_noise_aux",
                     "train_pen"], keep_existing=args.resume)
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
        val_loss = tc.evaluate(loss_fn, val_loader, device, repeats=args.val_repeats, limit_batches=limit)
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
    report = final_report(model, val_set, train_set, stats, device, config, args.seed,
                          repeats=1 if args.smoke else args.val_repeats)
                          
    raw, ys, tasks = report.pop("_preview_arrays")
    np.savez_compressed(out_dir / "diffusion_preview.npz", lengths=np.array([len(r) for r in raw]),
                        features=np.concatenate(raw), labels=np.array(ys), tasks=np.array(tasks))
                        
    report.update({"fold": args.fold, "best_epoch": best["epoch"], "best_val_loss": round(best["val_loss"], 5)})
    with open(out_dir / "diffusion_report.json", "w") as f:
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
            f"Abbassare il limite, per esempio --attention_budget 2e7 (ora 4e7), e rilanciare con --resume: "
            f"le epoche già salvate non vanno perse."
        )