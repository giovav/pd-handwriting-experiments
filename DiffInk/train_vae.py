"""
train_vae.py: stadio 1 di DiffInk, fine-tuning del VAE su un fold.

Si caricano i pesi ufficiali. Per --warmup_epochs si addestrano solo i due layer nuovi
(primo conv dell'encoder e layer d'uscita), poi tutto il VAE con due learning rate,
fino all'early stopping sulla loss di validazione. Le registrazioni più lunghe di
--crop_len vengono tagliate a una finestra casuale, perché l'attention del decoder a
risoluzione piena cresce col quadrato della lunghezza.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# I moduli condivisi stanno in common/ (se c'è una copia accanto allo script, vale quella).
sys.path.insert(1, str(Path(__file__).resolve().parents[1] / "common"))

import train_common as tc
from train_common import dp
from finetune import configure_vae_finetuning, param_groups, print_trainable
from model.output_head import decode_output
from model.vae import VAE
from pahaw_steps import vae_training_loss
from pretrained import load_pretrained_vae


class CropCollate:
    """collate_fn che taglia a una finestra casuale di crop_len le registrazioni più lunghe."""

    def __init__(self, crop_len: int):
        self.crop_len = crop_len

    def __call__(self, items: list[dict]) -> dict:
        cropped = []
        for item in items:
            length = item["x"].shape[0]
            if length > self.crop_len:
                start = random.randint(0, length - self.crop_len)
                item = {**item, "x": item["x"][start:start + self.crop_len]}
            cropped.append(item)
        return dp.collate_fn(cropped)


def make_loader(dataset, batch_size: int, crop_len: int, shuffle: bool) -> DataLoader:
    """DataLoader a bucket con le lunghezze dopo il taglio."""
    lengths = [min(n, crop_len) for n in dataset.lengths]
    sampler = dp.BucketBatchSampler(lengths, batch_size, shuffle=shuffle)
    return DataLoader(dataset, batch_sampler=sampler, collate_fn=CropCollate(crop_len))


@torch.no_grad()
def final_report(vae, loader, device, num_mixture: int) -> dict:
    """
    Report finale sul miglior VAE, sui dati di validazione: RMSE di ricostruzione per canale
    (unità normalizzate), accuratezza dello stato della penna, media e deviazione standard dei latenti.
    """
    vae.eval()
    torch.manual_seed(0)
    squared_error = torch.zeros(dp.N_CONTINUOUS, device=device)
    n_steps = torch.zeros(dp.N_CONTINUOUS, device=device)
    pen_correct, pen_total = 0.0, 0.0
    latent_sum, latent_sq, latent_n = 0.0, 0.0, 0.0

    for batch in loader:
        batch = tc.to_device(batch, device)
        x, mask = batch["x"], batch["mask"]

        z, mu, _, latent_mask = vae.encode(x, mask, sample=False)
        rebuilt = decode_output(vae.decode(z, mask), mask, num_mixture)

        weight = mask.unsqueeze(-1).float().repeat(1, 1, dp.N_CONTINUOUS)
        weight[..., dp.PRESSURE] *= x[..., dp.PEN]
        difference = (rebuilt[..., :dp.N_CONTINUOUS] - x[..., :dp.N_CONTINUOUS]) ** 2
        squared_error += (difference * weight).sum(dim=(0, 1))
        n_steps += weight.sum(dim=(0, 1))

        pen_correct += ((rebuilt[..., dp.PEN] == x[..., dp.PEN]) & mask).sum().item()
        pen_total += mask.sum().item()

        real_latents = mu[latent_mask]
        latent_sum += real_latents.sum().item()
        latent_sq += (real_latents ** 2).sum().item()
        latent_n += real_latents.numel()

    rmse = torch.sqrt(squared_error / n_steps.clamp(min=1)).tolist()
    latent_mean = latent_sum / latent_n
    latent_std = (latent_sq / latent_n - latent_mean ** 2) ** 0.5
    
    return {
        "reconstruction_rmse_normalized": dict(zip(dp.CHANNEL_NAMES[:dp.N_CONTINUOUS], [round(v, 4) for v in rmse])),
        "pen_state_accuracy": round(pen_correct / pen_total, 4),
        "latent_mean": round(latent_mean, 4),
        "latent_std": round(latent_std, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stadio 1: fine-tuning del VAE su un fold.")
    parser.add_argument("--fold", type=int, required=True, help="fold, da 0 a 3")
    parser.add_argument("--epochs", type=int, default=200, help="numero massimo di epoche")
    parser.add_argument("--warmup_epochs", type=int, default=10, help="epoche iniziali in cui si addestrano solo i due layer nuovi")
    parser.add_argument("--warmup_lr", type=float, default=1e-3, help="learning rate del warm-up")
    parser.add_argument("--patience", type=int, default=20, help="epoche senza miglioramento della loss di validazione prima di fermarsi")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accum", type=int, default=2, help="batch accumulati per ogni aggiornamento dei pesi")
    parser.add_argument("--crop_len", type=int, default=2048, help="lunghezza massima delle finestre di training (multiplo di 8)")
    parser.add_argument("--max_seq_len", type=int, default=None, help="esclude le registrazioni più lunghe di così (in campioni)")
    parser.add_argument("--clip_sigma", type=float, default=None,
                        help="limita dx, dy normalizzati a +-clip_sigma (non serve con le regole sui gap attuali)")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out_root", type=str, default="runs/diffink")
    tc.add_data_arguments(parser)
    parser.add_argument("--resume", action="store_true", help="continua da vae_last.pt")
    parser.add_argument("--overwrite", action="store_true", help="cancella il run esistente e ricomincia")
    parser.add_argument("--smoke", action="store_true", help="prova rapida: 2 epoche da 3 batch, risultati in <out_root>_smoke")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    assert args.crop_len % 8 == 0, "--crop_len deve essere un multiplo di 8"

    limit = 3 if args.smoke else None
    if args.smoke:
        args.epochs, args.warmup_epochs = 2, 1
        args.out_root, args.overwrite = args.out_root + "_smoke", True

    tc.seed_everything(args.seed)
    device = tc.get_device()
    config = tc.load_config(args.config)
    out_dir = tc.run_dir(args.fold, args.out_root)
    print(f"Device: {device}    risultati: {out_dir}")

    train_set, val_set, _ = tc.build_datasets(args.fold, out_dir, args.max_seq_len,
                                              args.data_file, args.split_file, args.clip_sigma)
    train_loader = make_loader(train_set, args.batch_size, args.crop_len, shuffle=True)
    val_loader = make_loader(val_set, args.batch_size, args.crop_len, shuffle=False)

    vae = VAE(config)
    report = load_pretrained_vae(vae, tc.pretrained_file(config.pretrained_vae_path, args.pretrained_dir))
    new_names = report.new
    vae.to(device)

    terms = {"sum": {}, "n": 0, "last": {}}

    def loss_fn(batch):
        loss, parts = vae_training_loss(vae, batch, config)
        terms["last"] = parts
        terms["n"] += 1
        for name, value in parts.items():
            terms["sum"][name] = terms["sum"].get(name, 0.0) + value
        return loss

    def term_means() -> dict:
        means = {name: total / max(terms["n"], 1) for name, total in terms["sum"].items()}
        terms["sum"], terms["n"] = {}, 0
        return means

    @torch.no_grad()
    def explain(batch) -> str:
        _, mu, logvar, latent_mask = vae.encode(batch["x"], batch["mask"], sample=False)
        return (f"termini della loss {terms['last']}; latente |media| max {mu.abs().max().item():.1f}, "
                f"log-varianza da {logvar.min().item():.1f} a {logvar.max().item():.1f}")

    def start_phase(phase: str) -> torch.optim.Optimizer:
        configure_vae_finetuning(vae, new_names, phase)
        print_trainable(vae, f"VAE, fase '{phase}'")
        lr_new = args.warmup_lr if phase == "warmup" else config.lr_new
        return tc.make_optimizer(param_groups(vae, new_names, lr_new, config.lr_pretrained, config.weight_decay))

    phase_of = lambda epoch: "warmup" if epoch < args.warmup_epochs else "full"

    stopper = tc.EarlyStopper(args.patience)
    state = {"step": 0}
    start_epoch = 0
    last_path, best_path = out_dir / "vae_last.pt", out_dir / "vae_best.pt"
    tc.check_not_overwriting(last_path, args.resume, args.overwrite)

    if args.resume and last_path.exists():
        saved = torch.load(last_path, map_location="cpu", weights_only=True)
        tc.check_checkpoint_is_healthy(saved["model_state_dict"], last_path)
        assert saved.get("clip_sigma") == args.clip_sigma, "--clip_sigma diverso da quello del run da riprendere"
        
        vae.load_state_dict(saved["model_state_dict"])
        start_epoch = saved["epoch"] + 1
        optimizer = start_phase(phase_of(start_epoch))
        if saved["phase"] == phase_of(start_epoch):
            optimizer.load_state_dict(saved["optimizer_state_dict"])
            state = {"step": saved["step"]}
        stopper.load_state_dict(saved["stopper"])
        print(f"Ripreso dall'epoca {saved['epoch']} (migliore loss di validazione: {stopper.best:.4f})")
    else:
        optimizer = start_phase(phase_of(0))

    log = tc.CsvLog(out_dir / "vae_log.csv",
                    ["epoch", "phase", "train_loss", "val_loss", "best", "seconds",
                     "train_gmm", "train_aux", "train_pen", "train_kl"],
                    keep_existing=args.resume)
    timer = tc.Timer()

    for epoch in range(start_epoch, args.epochs):
        phase = phase_of(epoch)
        if phase == "full" and stopper.should_stop:
            print("Questo run si era già fermato per early stopping: non c'è niente da fare.")
            break
            
        if epoch == args.warmup_epochs and epoch != start_epoch:
            optimizer = start_phase("full")
            state = {"step": 0}

        vae.train()
        tc.seed_everything(args.seed + epoch + 1)
        trainable = [p for p in vae.parameters() if p.requires_grad]
        term_means()
        
        train_loss = tc.train_one_epoch(loss_fn, train_loader, optimizer, trainable, device, state,
                                        accum=args.accum, limit_batches=limit, explain=explain)
        train_terms = term_means()

        vae.eval()
        val_loss = tc.evaluate(loss_fn, val_loader, device, limit_batches=limit)
        tc.require_finite(val_loss, f"loss di validazione all'epoca {epoch}")
        tc.check_checkpoint_is_healthy(vae.state_dict(), "pesi del modello")

        is_best = stopper.update(val_loss, epoch)
        if phase == "warmup":
            stopper.bad_epochs = 0
            
        if is_best:
            tc.save_checkpoint({"model_state_dict": vae.state_dict(), "epoch": epoch, "val_loss": val_loss,
                                "fold": args.fold, "clip_sigma": args.clip_sigma}, best_path)
                                
        tc.save_checkpoint({"model_state_dict": vae.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                            "epoch": epoch, "phase": phase, "step": state["step"],
                            "stopper": stopper.state_dict(), "fold": args.fold,
                            "clip_sigma": args.clip_sigma}, last_path)

        seconds = timer.lap()
        log.write(epoch=epoch, phase=phase, train_loss=round(train_loss, 5), val_loss=round(val_loss, 5),
                  best=int(is_best), seconds=round(seconds, 1),
                  **{f"train_{name}": round(value, 5) for name, value in train_terms.items()})
                  
        print(f"Epoca {epoch:3d} [{phase:6s}] TrainLoss {train_loss:8.4f}   ValLoss {val_loss:8.4f}"
              f"{'  *Best Model*' if is_best else ''}   ({seconds:.0f} s)   "
              + "  ".join(f"{name} {value:.3f}" for name, value in train_terms.items()))

        if phase == "full" and stopper.should_stop:
            print(f"Early stopping: {args.patience} epoche senza miglioramento.")
            break

    best = torch.load(best_path, map_location="cpu", weights_only=True)
    vae.load_state_dict(best["model_state_dict"])
    summary = final_report(vae, val_loader, device, config.num_mixture)
    summary.update({"fold": args.fold, "best_epoch": best["epoch"], "best_val_loss": round(best["val_loss"], 5)})
    
    with open(out_dir / "vae_report.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nTraining finito. Epoca migliore: {best['epoch']}  (ValLoss {best['val_loss']:.4f})")
    print("RMSE di ricostruzione sulla validazione (unità normalizzate):")
    for name, value in summary["reconstruction_rmse_normalized"].items():
        print(f"   {name:9s} {value:.3f}")
    print(f"   accuratezza dello stato della penna: {100 * summary['pen_state_accuracy']:.1f}%")
    print(f"Latenti: media {summary['latent_mean']:.3f}, deviazione standard {summary['latent_std']:.3f}")
    print(f"Salvato: {best_path}")


if __name__ == "__main__":
    main()