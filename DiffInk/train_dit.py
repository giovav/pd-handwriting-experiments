"""
train_dit.py: stadio 2 di DiffInk, fine-tuning del DiT su un fold.

Carica il VAE fine-tuned dello stesso fold e lo congela, carica i pesi ufficiali del
DiT e applica la strategia di config.dit_strategy (finetune.py). Il training usa le
registrazioni intere, la classe nascosta con probabilità p_uncond ed early stopping
sulla loss di validazione. Salva solo i pesi addestrabili (dit_best.pt).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# I moduli condivisi stanno in common/ (se c'è una copia accanto allo script, vale quella).
sys.path.insert(1, str(Path(__file__).resolve().parents[1] / "common"))

import train_common as tc
from train_common import dp
from finetune import configure_dit_finetuning, param_groups, print_trainable, trainable_state_dict
from model.diffusion import Diffusion
from model.dit import DiT
from model.vae import VAE
from pahaw_steps import dit_training_loss
from pretrained import load_pretrained_dit


def build_dit(config, strategy: str, rank: int, pretrained_dir: str | None = None):
    """DiT con i pesi ufficiali e la strategia di fine-tuning applicata."""
    dit = DiT(config)
    report = load_pretrained_dit(dit, tc.pretrained_file(config.pretrained_dit_path, pretrained_dir))
    configure_dit_finetuning(dit, strategy, rank)
    return dit, report.new


def load_finetuned_dit(config, checkpoint_path, pretrained_dir: str | None = None) -> DiT:
    """DiT fine-tuned = pesi ufficiali + strategia + pesi salvati in dit_best.pt."""
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    dit, _ = build_dit(config, saved["strategy"], saved["rank"], pretrained_dir)
    result = dit.load_state_dict(saved["trainable_state_dict"], strict=False)
    assert not result.unexpected_keys, f"Il checkpoint contiene tensori che il modello non ha: {result.unexpected_keys[:3]}"
    return dit


def main() -> None:
    parser = argparse.ArgumentParser(description="Stadio 2: fine-tuning del DiT su un fold.")
    parser.add_argument("--fold", type=int, required=True, help="fold, da 0 a 3")
    parser.add_argument("--epochs", type=int, default=300, help="numero massimo di epoche")
    parser.add_argument("--patience", type=int, default=30, help="epoche senza miglioramento della loss di validazione prima di fermarsi")
    parser.add_argument("--batch_size", type=int, default=8, help="registrazioni per batch, al massimo")
    parser.add_argument("--attention_budget", type=float, default=3e6,
                        help="limite di batch x N^2 (N = frame latenti); abbassarlo se la GPU esaurisce la memoria")
    parser.add_argument("--accum", type=int, default=1, help="batch accumulati per ogni aggiornamento dei pesi")
    parser.add_argument("--val_repeats", type=int, default=4, help="passate sulla validazione per epoca (la loss dipende dal rumore)")
    parser.add_argument("--max_seq_len", type=int, default=None, help="esclude le registrazioni più lunghe di così (in campioni)")
    parser.add_argument("--vae_ckpt", type=str, default=None, help="checkpoint del VAE (default: <out_root>/fold<k>/vae_best.pt)")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out_root", type=str, default="runs/diffink")
    tc.add_data_arguments(parser)
    parser.add_argument("--resume", action="store_true", help="continua da dit_last.pt")
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
    out_dir = tc.run_dir(args.fold, args.out_root)
    print(f"Device: {device}    risultati: {out_dir}")

    vae_path = args.vae_ckpt or out_dir / "vae_best.pt"
    try:
        saved_vae = torch.load(vae_path, map_location="cpu", weights_only=True)
    except FileNotFoundError:
        raise SystemExit(f"Checkpoint del VAE non trovato: {vae_path}. Prima va eseguito train_vae.py.")
    
    assert saved_vae["fold"] == args.fold, "Il VAE è stato addestrato su un altro fold: non si può usare (vedrebbe i pazienti di test)."
    tc.check_checkpoint_is_healthy(saved_vae["model_state_dict"], vae_path)

    clip_sigma = saved_vae.get("clip_sigma")
    train_set, val_set, _ = tc.build_datasets(args.fold, out_dir, args.max_seq_len,
                                              args.data_file, args.split_file, clip_sigma)

    train_sampler = tc.AttentionBudgetSampler(train_set.lengths, args.batch_size, args.attention_budget, shuffle=True)
    val_sampler = tc.AttentionBudgetSampler(val_set.lengths, args.batch_size, args.attention_budget, shuffle=False)
    train_loader = DataLoader(train_set, batch_sampler=train_sampler, collate_fn=dp.collate_fn)
    val_loader = DataLoader(val_set, batch_sampler=val_sampler, collate_fn=dp.collate_fn)
    
    print(f"Batch: {train_sampler.describe()}")
    
    vae = VAE(config)
    vae.load_state_dict(saved_vae["model_state_dict"])
    vae.to(device).eval()
    for p in vae.parameters():
        p.requires_grad = False
    print(f"VAE da {vae_path} (epoca {saved_vae['epoch']}, loss {saved_vae['val_loss']:.4f}), congelato.")

    dit, new_names = build_dit(config, config.dit_strategy, config.lora_rank, args.pretrained_dir)
    dit.to(device)
    print_trainable(dit, f"DiT, strategia '{config.dit_strategy}'")

    diffusion = Diffusion(noise_steps=config.noise_steps, schedule_type=config.schedule_type, device=device)
    loss_fn = lambda batch: dit_training_loss(dit, vae, diffusion, batch, p_uncond=config.p_uncond)

    optimizer = tc.make_optimizer(param_groups(dit, new_names, config.lr_new, config.lr_pretrained, config.weight_decay))
    trainable = [p for p in dit.parameters() if p.requires_grad]

    stopper = tc.EarlyStopper(args.patience)
    state = {"step": 0}
    start_epoch = 0
    last_path, best_path = out_dir / "dit_last.pt", out_dir / "dit_best.pt"
    tc.check_not_overwriting(last_path, args.resume, args.overwrite)

    if args.resume and last_path.exists():
        saved = torch.load(last_path, map_location="cpu", weights_only=True)
        assert saved["strategy"] == config.dit_strategy and saved["rank"] == config.lora_rank, \
            "dit_strategy o lora_rank diversi da quelli del run da riprendere."
        tc.check_checkpoint_is_healthy(saved["trainable_state_dict"], last_path)
        dit.load_state_dict(saved["trainable_state_dict"], strict=False)
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        stopper.load_state_dict(saved["stopper"])
        state = {"step": saved["step"]}
        start_epoch = saved["epoch"] + 1
        print(f"Ripreso dall'epoca {saved['epoch']} (migliore loss di validazione: {stopper.best:.4f})")

    log = tc.CsvLog(out_dir / "dit_log.csv", ["epoch", "train_loss", "val_loss", "best", "seconds"],
                    keep_existing=args.resume)
    timer = tc.Timer()
    what = {"strategy": config.dit_strategy, "rank": config.lora_rank, "fold": args.fold}

    for epoch in range(start_epoch, args.epochs):
        if stopper.should_stop:
            print("Questo run si era già fermato per early stopping: non c'è niente da fare.")
            break
            
        dit.train()
        tc.seed_everything(args.seed + epoch + 1)
        train_loss = tc.train_one_epoch(loss_fn, train_loader, optimizer, trainable, device, state,
                                        accum=args.accum, limit_batches=limit)

        dit.eval()
        val_loss = tc.evaluate(loss_fn, val_loader, device, repeats=args.val_repeats, limit_batches=limit)
        tc.require_finite(val_loss, f"loss di validazione all'epoca {epoch}")

        is_best = stopper.update(val_loss, epoch)
        weights = trainable_state_dict(dit)
        tc.check_checkpoint_is_healthy(weights, "pesi del modello")
        
        if is_best:
            tc.save_checkpoint({"trainable_state_dict": weights, "epoch": epoch, "val_loss": val_loss, **what}, best_path)
            
        tc.save_checkpoint({"trainable_state_dict": weights, "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch, "step": state["step"], "stopper": stopper.state_dict(), **what}, last_path)

        seconds = timer.lap()
        log.write(epoch=epoch, train_loss=round(train_loss, 5), val_loss=round(val_loss, 5),
                  best=int(is_best), seconds=round(seconds, 1))
                  
        print(f"Epoch {epoch:3d}  TrainLoss {train_loss:8.4f}   ValLoss {val_loss:8.4f}"
              f"{'  *Best Model*' if is_best else ''}   (lap {seconds:.0f}s)")

        if stopper.should_stop:
            print(f"Early stopping: {args.patience} epoche senza miglioramento.")
            break

    print(f"\nTraining finito. Epoca migliore: {stopper.best_epoch}  (ValLoss {stopper.best:.4f})")
    print(f"Salvato: {best_path}")


if __name__ == "__main__":
    OutOfMemory = getattr(torch, "OutOfMemoryError", torch.cuda.OutOfMemoryError)
    try:
        main()
    except OutOfMemory as error:
        raise SystemExit(
            f"\nMemoria GPU esaurita ({str(error).splitlines()[0]})\n"
            f"Abbassare --attention_budget (ora 3e6) e rilanciare con --resume: "
            f"le epoche già salvate non vanno perse."
        )