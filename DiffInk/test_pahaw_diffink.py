"""
test_pahaw_diffink.py: controlli del DiffInk modificato per PaHaW.

Dalla cartella DiffInk:

    python test_pahaw_diffink.py

Le parti A e B usano dati casuali e copie piccole delle reti, quindi girano in circa un
minuto anche senza GPU e senza PaHaW. Controllano che il codice sia collegato bene, non
la qualità di un modello addestrato. La parte C gira solo se trova i checkpoint ufficiali
ai percorsi di configs/pahaw_config.yaml: li carica nelle reti a grandezza piena e stampa
il report di caricamento.
"""

import copy
import os
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

# I moduli condivisi stanno in common/ (se c'è una copia accanto allo script, vale quella).
sys.path.insert(1, str(Path(__file__).resolve().parents[1] / "common"))

from dataset_prep import N_CHANNELS, PEN, PRESSURE, collate_fn
from finetune import (LoRALinear, configure_dit_finetuning, configure_vae_finetuning,
                      param_groups, print_trainable, trainable_state_dict)
from model.blocks import MaskedGroupNorm
from model.diffusion import Diffusion
from model.dit import DiT
from model.output_head import output_dim
from model.vae import VAE
from pahaw_steps import dit_training_loss, generate, vae_training_loss
from pretrained import load_pretrained_dit, load_pretrained_vae
from utils.utils import ModelConfig, load_config_from_yaml

torch.manual_seed(0)
full_config = load_config_from_yaml("configs/pahaw_config.yaml")

# Dimensioni piccole per le parti A e B (quelle vere servono solo nella parte C).
TEXT_DIM = 24
small = ModelConfig({**full_config,
                     "latent_dim": 32, "hidden_dims": [32, 48, 32], "decoder_dims": [32, 48, 32],
                     "trans_hidden_dim": 32, "dim": 64, "depth": 2, "heads": 2, "dim_head": 32})


def fake_item(length: int, label: int, task: int) -> dict:
    """Una 'registrazione' casuale nel formato di PaHaWSequenceDataset."""
    x = torch.randn(length, N_CHANNELS)
    x[:, PEN] = (torch.rand(length) > 0.2).float()        # pen è 0/1, quasi sempre sulla superficie
    x[:, PRESSURE] = x[:, PRESSURE] * x[:, PEN]
    return {"x": x, "y": label, "task": task, "patient_id": "fake"}


def pretend_trained(dit: DiT) -> None:
    """
    Un DiT appena creato restituisce esattamente 0 (alcuni layer partono da zero apposta) e
    i controlli passerebbero per forza. Un checkpoint addestrato lì ha valori veri, quindi per
    il test li riempiamo con piccoli numeri casuali.
    """
    for m in [dit.proj_out, dit.norm_out.linear] + [b.attn_norm.linear for b in dit.transformer_blocks]:
        nn.init.normal_(m.weight, std=0.05)


batch = collate_fn([fake_item(501, 1, 0), fake_item(333, 0, 3), fake_item(160, 1, 7), fake_item(64, 0, 5)])
B, T, _ = batch["x"].shape
diffusion = Diffusion(noise_steps=small.noise_steps, schedule_type=small.schedule_type)
print(f"batch: x {tuple(batch['x'].shape)}, lunghezze reali {batch['lengths'].tolist()}")

# =============================================================================
print("\n===== PARTE A: collegamenti (reti piccole, dati casuali) =====")
# =============================================================================
vae, dit = VAE(small), DiT(small)

print("[A1] shape attraverso il VAE")
z, mu, logvar, latent_mask = vae.encode(batch["x"], batch["mask"])
output = vae.decode(z, batch["mask"])
print(f"     x {tuple(batch['x'].shape)} -> z {tuple(z.shape)} -> output {tuple(output.shape)}")
assert z.shape == (B, T // 8, small.latent_dim)
assert output.shape == (B, T, output_dim(small.num_mixture))
assert (z[~latent_mask] == 0).all(), "i frame latenti di padding devono essere zero"

print("[A2] MaskedGroupNorm coincide con nn.GroupNorm(1, C) senza padding")
h = torch.randn(3, 16, 40)
diff = (MaskedGroupNorm(16)(h, torch.ones(3, 1, 40)) - nn.GroupNorm(1, 16)(h)).abs().max().item()
print(f"     differenza massima {diff:.1e}")
assert diff < 1e-5

print("[A3] il padding non cambia il risultato (VAE)")
vae.eval()
alone = collate_fn([fake_item(160, 1, 7)])
alone["x"] = batch["x"][2:3, :160].clone()
with torch.no_grad():
    _, mu_alone, _, _ = vae.encode(alone["x"], alone["mask"], sample=False)
    _, mu_batch, _, _ = vae.encode(batch["x"], batch["mask"], sample=False)
    d_enc = (mu_alone[0] - mu_batch[2, :20]).abs().max().item()
    d_dec = (vae.decode(mu_alone, alone["mask"])[0] - vae.decode(mu_batch, batch["mask"])[2, :160]).abs().max().item()
print(f"     differenza encoder {d_enc:.1e}, decoder {d_dec:.1e}")
assert d_enc < 1e-4 and d_dec < 1e-3
vae.train()

print("[A4] loss e gradienti del VAE")
loss, parts = vae_training_loss(vae, batch, small)
loss.backward()
print("     " + ", ".join(f"{k} {v:.3f}" for k, v in parts.items()))
assert torch.isfinite(loss)
vae.zero_grad()

print("[A5] DiT: invarianza al padding; classe, task e classe nulla cambiano l'uscita")
pretend_trained(dit)
nn.init.normal_(dit.class_embed.weight, std=0.5)       # come se il fine-tuning li avesse spostati
nn.init.normal_(dit.task_embed.weight, std=0.5)
dit.eval()
t = torch.tensor([100, 500, 900, 300])
with torch.no_grad():
    base = dit(z, t, batch["y"], batch["task"], latent_mask)
    alone_out = dit(z[2:3, :20], t[2:3], batch["y"][2:3], batch["task"][2:3], latent_mask[2:3, :20])
    d_pad = (alone_out[0] - base[2, :20]).abs().max().item()
    change = lambda other: (other - base)[latent_mask].abs().mean().item()
    c_class = change(dit(z, t, 1 - batch["y"], batch["task"], latent_mask))
    c_null = change(dit(z, t, torch.full_like(batch["y"], dit.null_class), batch["task"], latent_mask))
    c_task = change(dit(z, t, batch["y"], (batch["task"] + 1) % 8, latent_mask))
print(f"     differenza col padding {d_pad:.1e}; effetto della classe {c_class:.3f}, classe nulla {c_null:.3f}, task {c_task:.3f}")
assert base.shape == z.shape and d_pad < 1e-4 and min(c_class, c_null, c_task) > 1e-3

print("[A6] generazione, senza e con guidance")
y, task, lengths = torch.tensor([0, 1, 1]), torch.tensor([2, 2, 6]), torch.tensor([300, 451, 97])
for scale in (1.0, 2.0):
    feats, mask = generate(dit, vae, diffusion, y, task, lengths, small.num_mixture,
                           cfg_scale=scale, sampling_timesteps=5)
    assert feats.shape == (3, 456, N_CHANNELS) and mask.sum(1).tolist() == lengths.tolist()
    assert torch.isfinite(feats).all() and (feats[~mask] == 0).all()
    assert set(feats[..., PEN].unique().tolist()) <= {0.0, 1.0}
print(f"     feature {tuple(feats.shape)}, lunghezze {mask.sum(1).tolist()}")

# =============================================================================
print("\n===== PARTE B: caricamento di pesi in formato ufficiale e freezing (reti piccole) =====")
# =============================================================================
# Checkpoint finto con la struttura di quello ufficiale: nomi dei layer originali,
# 5 canali di ingresso, 123 uscite, ramo del testo, proiezione di ingresso con le colonne
# del testo, moduli OCR e writer, prefisso 'module.' del multi-GPU.
torch.manual_seed(1)
src_vae = VAE(ModelConfig({**vars(small), "in_channels": 5}))
src_dit = DiT(small)
pretend_trained(src_dit)

RENAME_BACK = {"norm1": "block.0", "conv1": "block.2", "norm2": "block.3", "conv2": "block.5"}
vae_ckpt = {}
for name, tensor in src_vae.state_dict().items():
    for ours, theirs in RENAME_BACK.items():
        name = name.replace(f".{ours}.", f".{theirs}.")
    vae_ckpt[name] = tensor
vae_ckpt["transformer_decoder.fc.weight"] = torch.randn(123, small.trans_hidden_dim)
vae_ckpt["transformer_decoder.fc.bias"] = torch.randn(123)
vae_ckpt["ocr_model.fc.weight"] = torch.randn(10, 4)
vae_ckpt["style_classifier.fc.weight"] = torch.randn(10, 4)

dit_ckpt = dict(src_dit.state_dict())
del dit_ckpt["class_embed.weight"], dit_ckpt["task_embed.weight"]
text_columns = torch.randn(small.dim, TEXT_DIM)
dit_ckpt["input_embed.proj.weight"] = torch.cat([dit_ckpt["input_embed.proj.weight"], text_columns], dim=1)
dit_ckpt["text_embed.text_embed.weight"] = torch.randn(1000, TEXT_DIM)

folder = tempfile.mkdtemp()
vae_path, dit_path = os.path.join(folder, "vae.pt"), os.path.join(folder, "dit.pt")
torch.save({"epoch": 99, "model_state_dict": vae_ckpt}, vae_path)
torch.save({"epoch": 0, "model_state_dict": {"module." + k: v for k, v in dit_ckpt.items()}}, dit_path)

print("[B1] VAE: si carica tutto tranne il primo conv e il layer d'uscita")
vae = VAE(small)
vae_report = load_pretrained_vae(vae, vae_path)
assert vae_report.new == ["encoder.conv_1.bias", "encoder.conv_1.weight",
                          "transformer_decoder.fc.bias", "transformer_decoder.fc.weight"]
same = torch.equal(vae.encoder.res_stack_2.layers[1].conv1.weight, src_vae.encoder.res_stack_2.layers[1].conv1.weight)
assert same, "un layer residuo rinominato non è stato caricato"

print("\n[B2] DiT: si carica tutto tranne i due embedding clinici; la proiezione di ingresso viene tagliata")
dit = DiT(small)
dit_report = load_pretrained_dit(dit, dit_path)
assert dit_report.new == ["class_embed.weight", "task_embed.weight"] and len(dit_report.sliced) == 1
assert torch.equal(dit.input_embed.proj.weight, src_dit.input_embed.proj.weight)

print("\n[B3] il DiT caricato calcola esattamente quello che calcola la rete del checkpoint")
src_dit.eval(); dit.eval()
nn.init.zeros_(src_dit.class_embed.weight); nn.init.zeros_(src_dit.task_embed.weight)
with torch.no_grad():
    reference = src_dit(z, t, batch["y"], batch["task"], latent_mask)
    loaded = dit(z, t, batch["y"], batch["task"], latent_mask)
print(f"     differenza massima {(reference - loaded).abs().max().item():.1e}")
assert torch.allclose(reference, loaded, atol=1e-6)

print("[B4] strategie di freezing")
for strategy in ("minimal", "lora", "full"):
    d = copy.deepcopy(dit)
    configure_dit_finetuning(d, strategy, rank=4)
    trainable, total = print_trainable(d, f"     DiT '{strategy}'")
    with torch.no_grad():
        unchanged = torch.allclose(d.eval()(z, t, batch["y"], batch["task"], latent_mask), loaded, atol=1e-6)
    assert unchanged, "aggiungere gli adattatori non deve cambiare l'uscita prima del training"
    if strategy != "full":
        assert isinstance(d.transformer_blocks[0].attn_norm.linear, LoRALinear)
        assert not d.transformer_blocks[0].ff.ff[2].weight.requires_grad, "il feed-forward deve essere congelato"
        assert d.class_embed.weight.requires_grad

print("[B5] un passo di training muove solo ciò che può muoversi")
d = copy.deepcopy(dit); configure_dit_finetuning(d, "lora", rank=4); d.train(); vae.eval()
before = {n: p.detach().clone() for n, p in d.named_parameters()}
groups = param_groups(d, dit_report.new, small.lr_new, small.lr_pretrained, small.weight_decay)
optimizer = torch.optim.AdamW(groups)
for _ in range(3):
    loss = dit_training_loss(d, vae, diffusion, batch, p_uncond=small.p_uncond)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
moved = {n for n, p in d.named_parameters() if not torch.equal(p, before[n])}
allowed = {n for n, p in d.named_parameters() if p.requires_grad}
print(f"     {len(moved)} tensori cambiati, {len(before) - len(moved)} rimasti esattamente ai valori caricati")
assert moved and moved <= allowed
assert "class_embed.weight" in moved and "task_embed.weight" in moved
assert all(p.grad is None for p in vae.parameters()), "nello stadio 2 il VAE non deve ricevere gradiente"
print(f"     learning rate: {[(len(g['params']), g['lr']) for g in groups]}  (tensori, lr)")

saved = trainable_state_dict(d)
restored = copy.deepcopy(dit); configure_dit_finetuning(restored, "lora", rank=4)
restored.load_state_dict(saved, strict=False)
assert all(torch.equal(p, dict(d.named_parameters())[n]) for n, p in restored.named_parameters())
print(f"     checkpoint degli adattatori: {len(saved)} tensori ricostruiscono esattamente il modello fine-tuned")

print("[B6] fine-tuning del VAE in due fasi")
for phase in ("warmup", "full"):
    configure_vae_finetuning(vae, vae_report.new, phase)
    trainable, total = print_trainable(vae, f"     VAE '{phase}'")
configure_vae_finetuning(vae, vae_report.new, "warmup"); vae.train()
optimizer = torch.optim.AdamW(param_groups(vae, vae_report.new, 1e-3, 1e-4, 0.0))
history = []
for _ in range(40):
    loss, _ = vae_training_loss(vae, batch, small)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    history.append(loss.item())
print(f"     warm-up (solo layer nuovi): loss {history[0]:.3f} -> {history[-1]:.3f}")
assert history[-1] < history[0]
assert torch.equal(vae.encoder.res_stack_2.layers[1].conv1.weight, src_vae.encoder.res_stack_2.layers[1].conv1.weight)

# =============================================================================
print("\n===== PARTE C: i checkpoint ufficiali veri (reti a grandezza piena) =====")
# =============================================================================
config = ModelConfig(full_config)
if os.path.exists(config.pretrained_vae_path) and os.path.exists(config.pretrained_dit_path):
    vae, dit = VAE(config), DiT(config)
    vae_report = load_pretrained_vae(vae, config.pretrained_vae_path)
    dit_report = load_pretrained_dit(dit, config.pretrained_dit_path)
    configure_dit_finetuning(dit, config.dit_strategy, rank=config.lora_rank)
    print()
    print_trainable(dit, f"DiT '{config.dit_strategy}'")
    loss = dit_training_loss(dit, vae.eval(), diffusion, batch, p_uncond=config.p_uncond)
    print(f"una loss del DiT su dati casuali: {loss.item():.3f}")
else:
    print(f"saltata: nessun file in {config.pretrained_vae_path} e {config.pretrained_dit_path}")

print("\nTutti i controlli superati.")
