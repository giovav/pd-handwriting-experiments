"""
test_pahaw_diffusion.py: controlli del modello di diffusione 1-D. Dalla cartella del modello,
dopo convert_official_weights.py:

    python test_pahaw_diffusion.py

    A  i pesi convertiti riproducono le uscite TensorFlow salvate;
    B  la rete PaHaW calcola la rete originale senza cross-attention sul testo;
    C  il padding non cambia il risultato, e funziona anche con T = 4096;
    D  loss: valori finiti, padding ignorato, pressione esclusa in aria, gradiente nei layer nuovi;
    E  generazione: lunghezze, valori finiti, pressione in aria;
    F  report di caricamento, e luhman_pahaw_init.pt uguale all'adattamento ricalcolato.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import torch
import yaml

from convert_official_weights import compare_with_tensorflow
from diffusion import diffusion_loss, sample
from diffusion_model import PaHaWDiffusionWriter
from luhman_layers import OriginalDiffusionWriter
from pretrained import load_pahaw_init, load_pretrained_diffusion, new_parameter_names

torch.manual_seed(0)
HERE = Path(__file__).resolve().parent
config = types.SimpleNamespace(**yaml.safe_load(open(HERE / "configs" / "pahaw_config.yaml")))
CONVERTED = HERE / config.official_weights_path
INIT = HERE / config.pretrained_path
failures = []


def check(condition, what: str) -> None:
    """Registra l'esito di un controllo."""
    print(f"   [{'ok' if condition else 'FAILED'}] {what}")
    if not condition:
        failures.append(what)


if not (CONVERTED.exists() and INIT.exists()):
    raise SystemExit(f"Mancano {CONVERTED} o {INIT}. "
                     f"Vanno creati con convert_official_weights.py.")

official = torch.load(CONVERTED, map_location="cpu", weights_only=True)


def pahaw_model() -> PaHaWDiffusionWriter:
    """Rete PaHaW con i pesi adattati, in modalità eval."""
    model = PaHaWDiffusionWriter(config)
    load_pretrained_diffusion(model, CONVERTED)
    return model.eval()


# =============================================================================
print("\nA  Pesi convertiti (PyTorch) e uscite di riferimento (TensorFlow)")
# =============================================================================
differences = compare_with_tensorflow(official)
print("   differenza massima: " + ", ".join(f"{k} {v:.2e}" for k, v in differences.items()))
check(differences["noise prediction"] < 1e-3 and differences["pen-lift probability"] < 1e-3,
      "la rete PyTorch riproduce le uscite del codice TensorFlow")


# =============================================================================
print("\nB  Rete PaHaW e rete originale senza cross-attention sul testo")
# =============================================================================
original = OriginalDiffusionWriter()
original.load_state_dict(official)
original.eval()
for block in (original.enc3, original.enc5, *original.att_layers):
    block.with_text = False
model = pahaw_model()

B, T = 3, 96
x = torch.randn(B, T, 5)
sigma = torch.tensor([0.05, 0.5, 0.97])
y, task = torch.tensor([0, 1, 1]), torch.tensor([0, 4, 7])
with torch.no_grad():
    dummy_text, dummy_style = torch.ones(B, 4, dtype=torch.long), torch.zeros(B, 14, 1280)
    ref_eps, ref_lift = original(x[..., :2], dummy_text, sigma.reshape(B, 1, 1), dummy_style)
    eps, pen_logit = model(x, sigma, y, task)

d_eps = (eps[..., :2] - ref_eps).abs().max().item()
d_pen = (torch.sigmoid(pen_logit) - (1 - ref_lift[..., 0])).abs().max().item()

print(f"   differenza massima: rumore (dx, dy) {d_eps:.1e}, probabilità della penna {d_pen:.1e}")
check(d_eps < 1e-5 and d_pen < 1e-5, "stesso rumore per dx, dy e P(penna giù) = 1 - P(penna su)")
check(bool((eps[..., 2:] == 0).all()), "prima del fine-tuning il rumore previsto per i canali nuovi è zero")

with torch.no_grad():
    eps_other, _ = model(x, sigma, torch.tensor([1, 0, 0]), torch.tensor([3, 3, 3]))
check(torch.equal(eps, eps_other), "prima del fine-tuning classe e task non hanno effetto (embedding a zero)")


# =============================================================================
print("\nC  Padding")
# =============================================================================
with torch.no_grad():
    n = 61
    alone_x = torch.zeros(1, 64, 5)
    alone_x[0, :n] = x[0, :n]
    alone_mask = torch.arange(64)[None] < n
    alone, alone_pen = model(alone_x, sigma[:1], y[:1], task[:1], alone_mask)
    
    batch_x = torch.randn(2, 160, 5) * 50
    batch_x[0, :n] = x[0, :n]
    batch_mask = torch.zeros(2, 160, dtype=torch.bool)
    batch_mask[0, :n] = True
    batch_mask[1, :150] = True
    batched, batched_pen = model(batch_x, torch.cat([sigma[:1], sigma[1:2]]), torch.tensor([0, 1]),
                                 torch.tensor([0, 2]), batch_mask)
                                 
diff = max((alone[0, :n] - batched[0, :n]).abs().max().item(), (alone_pen[0, :n] - batched_pen[0, :n]).abs().max().item())
check(diff < 1e-4, f"da sola e dentro un batch con padding la registrazione dà lo stesso risultato (differenza {diff:.1e})")
check(bool((batched[0, n:] == 0).all()), "le uscite nelle posizioni di padding sono zero")

with torch.no_grad():
    long_eps, _ = model(torch.randn(1, 4096, 5), sigma[:1], y[:1], task[:1], torch.ones(1, 4096, dtype=torch.bool))
check(bool(torch.isfinite(long_eps).all()), "funziona con T = 4096 (la tabella di 2000 posizioni dell'originale si fermava a 4000 step)")


# =============================================================================
print("\nD  Loss")
# =============================================================================
model.train()
data = torch.randn(2, 80, 6)
data[..., 5] = (torch.rand(2, 80) > 0.3).float()
mask = torch.ones(2, 80, dtype=torch.bool)
mask[1, 50:] = False
data[1, 50:] = 0
g = torch.Generator().manual_seed(1)
terms = diffusion_loss(model, data, mask, y[:2], task[:2], generator=g)

check(all(torch.isfinite(v) for v in terms.values()), "termini della loss finiti: " + ", ".join(f"{k} {v.item():.3f}" for k, v in terms.items()))

garbage = data.clone()
garbage[1, 50:, :5] = 1e3
terms_garbage = diffusion_loss(model, garbage, mask, y[:2], task[:2], generator=torch.Generator().manual_seed(1))
check(all(abs(terms[k].item() - terms_garbage[k].item()) < 1e-5 for k in terms), "il padding non entra nella loss")
model.zero_grad()
sum(terms.values()).backward()

zero_net = lambda noisy, s, yy, tt, m: (torch.zeros_like(noisy), torch.zeros(noisy.shape[:2]))
big = torch.zeros(4, 4000, 6)
down = diffusion_loss(zero_net, big.index_fill(2, torch.tensor([5]), 1.0), torch.ones(4, 4000, dtype=torch.bool),
                      y[:1].repeat(4), task[:1].repeat(4))["noise_aux"].item()
up = diffusion_loss(zero_net, big, torch.ones(4, 4000, dtype=torch.bool), y[:1].repeat(4), task[:1].repeat(4))["noise_aux"].item()
check(abs(down - 3) < 0.1 and abs(up - 2) < 0.1,
      f"la pressione conta solo con la penna appoggiata (giù {down:.2f}, su {up:.2f})")

named = dict(model.named_parameters())
no_grad = [n for n in new_parameter_names(model) if named[n].grad is None or named[n].grad.abs().sum() == 0]
check(not no_grad, f"il gradiente arriva ai layer nuovi {no_grad or ''}")


# =============================================================================
print("\nE  Generazione")
# =============================================================================
model.eval()
out, out_mask = sample(model, torch.tensor([0, 1]), torch.tensor([1, 6]), torch.tensor([45, 30]),
                       in_air_pressure=-3.0, generator=torch.Generator().manual_seed(0))
check(out.shape == (2, 48, 6) and out_mask.sum().item() == 75, "le lunghezze generate sono quelle richieste (T arrotondato a un multiplo di 8)")
check(bool(torch.isfinite(out).all()), "tutti i valori generati sono finiti")

air = (out[..., 5] == 0) & out_mask
check(bool((out[..., 2][air] == -3.0).all()), "la pressione è al valore in aria dove la penna è sollevata")


# =============================================================================
print("\nF  Report di caricamento")
# =============================================================================
fresh = PaHaWDiffusionWriter(config)
report = load_pretrained_diffusion(fresh, CONVERTED)
report.print(fresh)
check(report.new == ["class_embed.weight", "input_new.weight", "output_new.bias", "output_new.weight",
                     "task_embed.weight"], "gli unici tensori nuovi sono gli embedding e i layer input_new, output_new")
check(len(report.flipped) == 2, "pen_dense ha il segno invertito rispetto a pen_lifts_dense")

init_model = PaHaWDiffusionWriter(config)
record = load_pahaw_init(init_model, INIT)
same = all(torch.equal(a, b) for a, b in zip(init_model.state_dict().values(), fresh.state_dict().values()))
check(same, "luhman_pahaw_init.pt coincide con l'adattamento ricalcolato dai pesi ufficiali")

print("\nTutti i controlli superati." if not failures else f"\n{len(failures)} CONTROLLI NON SUPERATI: {failures}")
sys.exit(1 if failures else 0)