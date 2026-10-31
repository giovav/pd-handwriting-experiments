"""
generate_synthetic.py (DiffInk): dataset sintetici dal DiffInk fine-tuned.

Dalla cartella principale:

    python DiffInk/generate_synthetic.py --folds 0 1 2 3 --runs_dir runs/diffink \\
        --data_file packed_export/pahaw_packed.npz --split_file packed_export/splits_4fold.json \\
        --pretrained_dir <cartella con i checkpoint ufficiali> --cfg_scale 1.0

Per ogni fold:
    1. il DiT di runs/diffink/fold<k>/dit_best.pt (pesi ufficiali + adattatori, per
       questo serve --pretrained_dir) genera i latenti da rumore puro con DDIM, per la
       classe e il task di ogni elemento del piano, con classifier-free guidance rispetto
       alla classe nulla (indice 2) di forza --cfg_scale;
    2. il VAE congelato dello stesso fold (vae_best.pt) decodifica ogni latente;
    3. ogni uscita del decoder diventa una registrazione come nell'originale
       (trainer/dit_trainer.py, sample_from_params(..., greedy=True)): a ogni step si
       estrae una componente della mixture secondo i pesi e se ne prende il centro per
       (dx, dy); penna = stato più probabile; pressure, azimuth, altitude = medie;
    4. si torna alle unità originali con le costanti del fold e si salva
       synthetic/diffink/fold<k>_cfg<s>.npz (+ .json).

Le lunghezze sono quelle del piano di synthetic_common.py. Il VAE lavora su lunghezze
multiple di 8: ogni registrazione è generata alla lunghezza arrotondata al multiplo di 8
successivo e poi tagliata.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

# I moduli condivisi stanno in common/ (se c'è una copia accanto allo script, vale quella).
sys.path.insert(1, str(Path(__file__).resolve().parents[1] / "common"))

import synthetic_common as sc
import train_common as tc
from train_common import dp
from model.diffusion import Diffusion
from model.output_head import PEN, PRESSURE, split_output
from model.vae import VAE
from pahaw_steps import lengths_to_mask
from train_dit import load_finetuned_dit

MODEL = "diffink"


@torch.no_grad()
def decode_like_original(output: torch.Tensor, mask: torch.Tensor, num_mixture: int, in_air_pressure: float,
                         generator: torch.Generator) -> torch.Tensor:
    """
    Uscita del decoder [B, T, D_out] -> registrazione normalizzata [B, T, 6], come
    sample_from_params(greedy=True) dell'originale: componente estratta secondo i pesi della
    mixture, centro usato per (dx, dy); penna: logit > 0; altri canali: le medie.
    """
    p = split_output(output, num_mixture)
    B, T, K = p["log_pi"].shape
    probs = p["log_pi"].exp().reshape(B * T, K)
    component = torch.multinomial(probs, 1, generator=generator).reshape(B, T, 1)
    dx = torch.gather(p["mu1"], -1, component).squeeze(-1)
    dy = torch.gather(p["mu2"], -1, component).squeeze(-1)
    pen = (p["pen_logit"] > 0).float()
    pressure = torch.where(pen.bool(), p["aux_mu"][..., 0], torch.full_like(pen, in_air_pressure))
    x = torch.stack([dx, dy, pressure, p["aux_mu"][..., 1], p["aux_mu"][..., 2], pen], dim=-1)
    assert x.shape[-1] == 6 and PEN == 5 and PRESSURE == 2
    return x * mask.unsqueeze(-1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera i dataset sintetici con il DiffInk fine-tuned.")
    sc.add_common_arguments(parser)
    parser.add_argument("--pretrained_dir", type=str, default=None,
                        help="cartella con i checkpoint ufficiali di DiffInk (il DiT si ricostruisce da questi + adattatori)")
    parser.add_argument("--cfg_scale", type=float, default=None,
                        help="forza della classifier-free guidance; 1 = modello condizionato come addestrato, >1 = spinta "
                             "più forte verso la classe. Default: cfg_scale della configurazione")
    parser.add_argument("--sampling_timesteps", type=int, default=None, help="passi DDIM; default: quelli della configurazione")
    parser.add_argument("--batch_size", type=int, default=16, help="numero massimo di registrazioni generate insieme")
    parser.add_argument("--attention_budget", type=float, default=3e6,
                        help="limite per batch: dimensione x (T/8)^2, come nel training del DiT")
    args = parser.parse_args()
    if args.smoke:
        args.n_per_cell, args.out_dir = args.n_per_cell or 2, args.out_dir + "_smoke"

    device = tc.get_device()
    config = tc.load_config(args.config)
    cfg_scale = config.cfg_scale if args.cfg_scale is None else args.cfg_scale
    steps = config.sampling_timesteps if args.sampling_timesteps is None else args.sampling_timesteps
    tag = f"_cfg{cfg_scale:g}"
    print(f"Device: {device}   DiffInk   classifier-free guidance {cfg_scale:g} (classe nulla 2), {steps} passi DDIM")
    diffusion = Diffusion(noise_steps=config.noise_steps, schedule_type=config.schedule_type, device=device)

    for fold in args.folds:
        npz_path, json_path = sc.output_paths(args.out_dir, MODEL, fold, tag)
        if npz_path.exists() and not args.overwrite:
            print(f"\nFold {fold}: {npz_path} esiste già, saltato (--overwrite per rifarlo)")
            continue
        run_dir = Path(args.runs_dir) / f"fold{fold}"
        vae_path, dit_path = run_dir / "vae_best.pt", run_dir / "dit_best.pt"
        recordings, split = sc.load_fold_data(fold, args.data_file, args.split_file)
        stats = sc.fold_stats(run_dir, recordings, split)
        plan = sc.make_plan(recordings, split, fold, args.n_per_cell, args.seed)

        saved_vae = torch.load(vae_path, map_location="cpu", weights_only=True)
        saved_dit = torch.load(dit_path, map_location="cpu", weights_only=True)
        for path, saved in ((vae_path, saved_vae), (dit_path, saved_dit)):
            assert saved.get("fold", fold) == fold, f"{path} è stato addestrato sul fold {saved['fold']}"
        vae = VAE(config)
        vae.load_state_dict(saved_vae["model_state_dict"])
        vae.to(device).eval()
        dit = load_finetuned_dit(config, dit_path, args.pretrained_dir).to(device).eval()
        assert dit.null_class == 2, "la classe nulla del DiT deve essere 2"

        sc.seed_all(args.seed + fold)                       # il rumore iniziale di DDIM usa il generatore globale
        generator = torch.Generator(device=device).manual_seed(args.seed + fold)
        groups = sc.batches(plan, args.batch_size, cost=lambda b, t: b * (t / 8) ** 2, budget=args.attention_budget)
        print(f"\nFold {fold}: {len(plan)} registrazioni in {len(groups)} batch, da {dit_path} (epoca "
              f"{saved_dit['epoch']}) e {vae_path} (epoca {saved_vae['epoch']})")
        features, start, splits = [None] * len(plan), time.time(), []

        def run(group):
            y = torch.tensor([plan[i]["label"] for i in group], device=device)
            task = torch.tensor([plan[i]["task"] for i in group], device=device)
            lengths = torch.tensor([plan[i]["length"] for i in group], device=device)
            mask = lengths_to_mask(lengths)                                        # [B, T], T multiplo di 8
            latent_mask = mask.reshape(mask.shape[0], -1, 8).any(dim=2)            # [B, T/8]
            with torch.no_grad():
                z = diffusion.ddim_sample(dit, latent_mask, y, task, sampling_timesteps=steps, cfg_scale=cfg_scale)
                output = vae.decode(z, mask)
                x = decode_like_original(output, mask, config.num_mixture, sc.in_air_pressure(stats), generator)
            x = x.cpu().numpy()
            return [dp.denormalize(x[row, :plan[i]["length"]], stats) for row, i in enumerate(group)]

        for n, group in enumerate(groups):
            for i, f in zip(group, sc.generate_in_pieces(run, group, splits)):
                features[i] = f
            print(f"   batch {n + 1}/{len(groups)}: {len(group)} registrazioni fino a {plan[group[-1]]['length']} step "
                  f"({time.time() - start:.0f} s finora)")

        meta = {"model": MODEL, "fold": fold, "seed": args.seed, "batches_split_for_memory": splits,
                "checkpoints": {"dit": str(dit_path), "dit_sha256": sc.file_sha256(dit_path), "dit_epoch": saved_dit["epoch"],
                                "vae": str(vae_path), "vae_sha256": sc.file_sha256(vae_path), "vae_epoch": saved_vae["epoch"]},
                "settings": {"cfg_scale": cfg_scale, "null_class": 2, "ddim_steps": steps, "eta": 0.0,
                             "decoding": "component drawn from the mixture weights, its centre taken (original greedy=True)",
                             "length_rounding": "generated at the next multiple of 8, cut to the drawn length"}}
        summary = sc.check_and_save(features, plan, npz_path, json_path, meta, recordings, split)
        sc.print_summary(summary)
        print(f"   salvato {npz_path}")


if __name__ == "__main__":
    main()
