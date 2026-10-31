"""
generate_synthetic.py (MDN): dataset sintetici dall'MDN fine-tuned.

Generazione senza teacher forcing: si parte dal vettore di zeri, a ogni step si
campiona dalla distribuzione prevista (con il sampling bias di Graves, 0 di default) e
il campione diventa l'ingresso dello step successivo. Le lunghezze sono quelle del piano
di synthetic_common.py; l'uscita è riportata alle unità originali con le costanti del fold.
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
from mdn_model import generate
from train_mdn import load_finetuned_mdn

MODEL = "mdn"


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera i dataset sintetici con l'MDN fine-tuned.")
    sc.add_common_arguments(parser)
    parser.add_argument("--bias", type=float, default=None, help="sampling bias di Graves (2013); 0 = distribuzione appresa. Default: sampling_bias della configurazione")
    parser.add_argument("--batch_size", type=int, default=64, help="numero massimo di registrazioni generate insieme")
    args = parser.parse_args()
    if args.smoke:
        args.n_per_cell, args.out_dir = args.n_per_cell or 2, args.out_dir + "_smoke"

    device = tc.get_device()
    config = tc.load_config(args.config)
    bias = config.sampling_bias if args.bias is None else args.bias
    tag = "" if bias == 0 else f"_bias{bias:g}"
    print(f"Device: {device}   MDN   sampling bias: {bias:g}")

    for fold in args.folds:
        npz_path, json_path = sc.output_paths(args.out_dir, MODEL, fold, tag)
        if npz_path.exists() and not args.overwrite:
            print(f"\nFold {fold}: {npz_path} esiste già, saltato (--overwrite per rifarlo).")
            continue
            
        run_dir = Path(args.runs_dir) / f"fold{fold}"
        checkpoint = run_dir / "mdn_best.pt"
        recordings, split = sc.load_fold_data(fold, args.data_file, args.split_file)
        stats = sc.fold_stats(run_dir, recordings, split)
        plan = sc.make_plan(recordings, split, fold, args.n_per_cell, args.seed)
        
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        assert saved.get("fold", fold) == fold, f"{checkpoint} è stato addestrato sul fold {saved['fold']}"
        model = load_finetuned_mdn(config, checkpoint, device)

        sc.seed_all(args.seed + fold)
        generator = torch.Generator(device=device).manual_seed(args.seed + fold)
        
        groups = sc.batches(plan, args.batch_size, cost=lambda b, t: 0, budget=1)
        print(f"\nFold {fold}: {len(plan)} registrazioni in {len(groups)} batch, "
              f"da {checkpoint} (epoca {saved['epoch']})")
              
        features, start, splits = [None] * len(plan), time.time(), []

        def run(group):
            """Genera le registrazioni di un gruppo del piano."""
            y = torch.tensor([plan[i]["label"] for i in group], device=device)
            task = torch.tensor([plan[i]["task"] for i in group], device=device)
            lengths = torch.tensor([plan[i]["length"] for i in group])
            
            x, _ = generate(model, y, task, lengths, bias=bias, in_air_pressure=sc.in_air_pressure(stats),
                            generator=generator)
            x = x.cpu().numpy()
            
            return [dp.denormalize(x[row, :plan[i]["length"]], stats) for row, i in enumerate(group)]

        for n, group in enumerate(groups):
            for i, f in zip(group, sc.generate_in_pieces(run, group, splits)):
                features[i] = f
            print(f"   batch {n + 1}/{len(groups)}: {len(group)} registrazioni fino a {plan[group[-1]]['length']} step "
                  f"({time.time() - start:.0f} s finora)")

        meta = {"model": MODEL, "fold": fold, "checkpoint": str(checkpoint), "checkpoint_sha256": sc.file_sha256(checkpoint),
                "checkpoint_epoch": saved["epoch"], "seed": args.seed, "batches_split_for_memory": splits,
                "settings": {"graves_bias": bias, "start": "zero vector", "teacher_forcing": False}}
                
        summary = sc.check_and_save(features, plan, npz_path, json_path, meta, recordings, split)
        sc.print_summary(summary)
        print(f"   salvato {npz_path}")


if __name__ == "__main__":
    main()