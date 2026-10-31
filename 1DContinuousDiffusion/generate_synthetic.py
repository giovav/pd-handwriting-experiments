"""
generate_synthetic.py (diffusione 1-D): dataset sintetici dal modello fine-tuned.

Per ogni fold carica diffusion_best.pt e genera le registrazioni del piano di
synthetic_common.py partendo da rumore puro, con il sampler originale (60 passi). Lo stato
della penna è preso dall'ultima previsione; l'uscita è riportata alle unità originali con
le costanti del fold.
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
from diffusion import N_LEVELS, sample
from train_diffusion import load_finetuned_diffusion

MODEL = "diffusion"


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera i dataset sintetici con il modello di diffusione 1-D fine-tuned.")
    sc.add_common_arguments(parser)
    parser.add_argument("--batch_size", type=int, default=32, help="numero massimo di registrazioni generate insieme")
    parser.add_argument("--attention_budget", type=float, default=4e7,
                        help="limite per batch: dimensione x (T/2)^2, come nel training")
    args = parser.parse_args()
    if args.smoke:
        args.n_per_cell, args.out_dir = args.n_per_cell or 2, args.out_dir + "_smoke"

    device = tc.get_device()
    config = tc.load_config(args.config)
    print(f"Device: {device}   diffusione 1-D   {N_LEVELS} passi (sampler originale)")

    for fold in args.folds:
        npz_path, json_path = sc.output_paths(args.out_dir, MODEL, fold)
        if npz_path.exists() and not args.overwrite:
            print(f"\nFold {fold}: {npz_path} esiste già, saltato (--overwrite per rifarlo).")
            continue
            
        run_dir = Path(args.runs_dir) / f"fold{fold}"
        checkpoint = run_dir / "diffusion_best.pt"
        recordings, split = sc.load_fold_data(fold, args.data_file, args.split_file)
        stats = sc.fold_stats(run_dir, recordings, split)
        plan = sc.make_plan(recordings, split, fold, args.n_per_cell, args.seed)
        
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        assert saved.get("fold", fold) == fold, f"{checkpoint} è stato addestrato sul fold {saved['fold']}"
        model = load_finetuned_diffusion(config, checkpoint, device)

        sc.seed_all(args.seed + fold)
        generator = torch.Generator(device=device).manual_seed(args.seed + fold)
        
        groups = sc.batches(plan, args.batch_size, cost=lambda b, t: b * (t / 2) ** 2, budget=args.attention_budget)
        print(f"\nFold {fold}: {len(plan)} registrazioni in {len(groups)} batch, "
              f"da {checkpoint} (epoca {saved['epoch']})")
              
        features, start, splits = [None] * len(plan), time.time(), []

        def run(group):
            """Genera le registrazioni di un gruppo del piano."""
            y = torch.tensor([plan[i]["label"] for i in group], device=device)
            task = torch.tensor([plan[i]["task"] for i in group], device=device)
            lengths = torch.tensor([plan[i]["length"] for i in group])
            
            x, _ = sample(model, y, task, lengths, in_air_pressure=sc.in_air_pressure(stats), generator=generator)
            x = x.cpu().numpy()
            
            return [dp.denormalize(x[row, :plan[i]["length"]], stats) for row, i in enumerate(group)]

        for n, group in enumerate(groups):
            for i, f in zip(group, sc.generate_in_pieces(run, group, splits)):
                features[i] = f
            print(f"   batch {n + 1}/{len(groups)}: {len(group)} registrazioni fino a {plan[group[-1]]['length']} step "
                  f"({time.time() - start:.0f} s finora)")

        meta = {"model": MODEL, "fold": fold, "checkpoint": str(checkpoint), "checkpoint_sha256": sc.file_sha256(checkpoint),
                "checkpoint_epoch": saved["epoch"], "seed": args.seed, "batches_split_for_memory": splits,
                "settings": {"reverse_steps": N_LEVELS, "sampler": "original, diffusion_mode='new'",
                             "pen": "from the last step's prediction"}}
                             
        summary = sc.check_and_save(features, plan, npz_path, json_path, meta, recordings, split)
        sc.print_summary(summary)
        print(f"   salvato {npz_path}")


if __name__ == "__main__":
    main()