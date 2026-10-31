"""
test_pahaw_mdn.py: controlli dell'MDN per PaHaW. Dalla cartella autoregressiveMDN:

    python test_pahaw_mdn.py

Deve finire con "Tutti i controlli superati."

    A  la rete PaHaW, caricata con i pesi ufficiali, calcola esattamente quello che
       calcola il codice originale (stessi dx, dy, end-of-stroke a 0, finestra vuota).
       Serve il pacchetto originale handwriting_synthesis/; se manca, la parte è saltata.
    B  meccanica: lo shift del teacher forcing, il padding non cambia né le uscite
       reali né la loss, la loss coincide con la formula originale, il gradiente
       arriva a tutte le parti nuove, la condizione cambia l'uscita, la generazione
       funziona e resta finita.
    C  il report di caricamento del checkpoint ufficiale.
"""

from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import torch
import yaml

from mdn_head import mdn_loss, mixture_parameters
from mdn_model import PaHaWMDN, generate, teacher_forcing_inputs
from pretrained import load_pretrained_mdn, new_parameter_names

torch.manual_seed(0)
HERE = Path(__file__).resolve().parent
config = types.SimpleNamespace(**yaml.safe_load(open(HERE / "configs" / "pahaw_config.yaml")))
CHECKPOINT = HERE / config.pretrained_path
failures = []


def check(condition: bool, what: str) -> None:
    print(f"   [{'ok' if condition else 'FAILED'}] {what}")
    if not condition:
        failures.append(what)


def fresh_model() -> PaHaWMDN:
    model = PaHaWMDN(config)
    if CHECKPOINT.exists():
        load_pretrained_mdn(model, CHECKPOINT)
    return model.eval()


def random_batch(B=3, T=40):
    x = torch.randn(B, T, 6)
    x[..., 5] = (torch.rand(B, T) > 0.3).float()
    return x, torch.tensor([0, 1, 1][:B]), torch.tensor([0, 3, 7][:B])


# =============================================================================
print("\nA  Stessi risultati del codice originale")
# =============================================================================
try:
    sys.path.insert(1, str(HERE))
    from handwriting_synthesis.models import SynthesisNetwork      # la rete ORIGINALE
    from handwriting_synthesis.losses import Mixture               # la likelihood ORIGINALE
    have_original = True
except ImportError:
    have_original = False
    print("   saltata: handwriting_synthesis/ (il pacchetto originale) non è in questa cartella")

if have_original:
    original = SynthesisNetwork(3, config.hidden_size, config.cond_dim, torch.device("cpu"))
    if CHECKPOINT.exists():
        original.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=True))
        source = "pesi ufficiali"
    else:
        source = "pesi casuali (file ufficiale non trovato)"
    original.eval()
    ours = PaHaWMDN(config)
    probe = Path(HERE / "_stand_in.pt")
    torch.save(original.state_dict(), probe)
    load_pretrained_mdn(ours, probe)
    probe.unlink()
    ours.eval()

    x, y, task = random_batch()
    x_original = torch.cat([x[..., :2], torch.zeros_like(x[..., :1])], dim=-1)   # [dx, dy, end-of-stroke = 0]
    empty_text = torch.zeros(x.shape[0], 7, config.cond_dim)                     # righe one-hot tutte a zero -> finestra 0
    with torch.no_grad():
        (pi, mu, sd, ro), _ = original(x_original, empty_text)
        raw, _ = ours(x, y, task)
    differences = {
        "pi": (torch.softmax(raw["pi"], -1) - pi).abs().max().item(),
        "mu": (raw["mu"] - mu).abs().max().item(),
        "sd": ((raw["sd"].exp() - sd).abs() / sd).max().item(),          # differenza relativa: exp() amplifica
        "ro": (raw["ro"].tanh() - ro).abs().max().item(),
    }
    print(f"   ({source}) differenze massime: " + ", ".join(f"{k} {v:.1e}" for k, v in differences.items()))
    check(max(differences.values()) < 1e-4, "le uscite coincidono con la rete originale (mixture per dx, dy)")

    # Il termine (dx, dy) della nostra loss coincide con la likelihood originale, step per step.
    # Le due protezioni (larghezza minima SIGMA_MIN, correlazione massima RHO_MAX) sono
    # spente per questo confronto; il loro effetto è mostrato a parte.
    import mdn_head
    with torch.no_grad():
        target = torch.randn_like(x)
        target[..., 5] = 1.0
        mask = torch.ones(x.shape[:2], dtype=torch.bool)
        with_safeguards = mdn_loss(raw, target, mask)["gmm"].item()
        saved_limits = mdn_head.SIGMA_MIN, mdn_head.RHO_MAX
        mdn_head.SIGMA_MIN, mdn_head.RHO_MAX = 0.0, 1.0
        ours_gmm = mdn_loss(raw, target, mask)["gmm"].item()
        mdn_head.SIGMA_MIN, mdn_head.RHO_MAX = saved_limits
        K = config.num_mixture
        flat = lambda t: t.reshape(-1, t.shape[-1])
        original_mixture = Mixture(flat(pi), flat(mu), flat(sd), flat(ro))
        original_gmm = -original_mixture.log_density(target[..., 0].reshape(-1), target[..., 1].reshape(-1)).item()
        original_gmm /= mask.sum().item()                                 # originale: somma; nostra: media
    print(f"   costo (dx, dy) per step: nostro {ours_gmm:.5f}, formula originale {original_gmm:.5f} "
          f"(nostro con le protezioni: {with_safeguards:.5f})")
    check(abs(ours_gmm - original_gmm) < 1e-3 * max(1.0, abs(original_gmm)),
          "la likelihood della mixture coincide con la formula originale (Graves 2013, sez. 4.1)")

# =============================================================================
print("\nB  Meccanica")
# =============================================================================
model = fresh_model()
x, y, task = random_batch()

inputs = teacher_forcing_inputs(x)
check(bool((inputs[:, 0] == 0).all()) and torch.equal(inputs[:, 1:], x[:, :-1]),
      "teacher forcing: l'ingresso a t è lo step reale t-1, il primo ingresso è il vettore di zeri")

# Padding: registrazione 0 da sola (lunghezza 25) e dentro un batch con padding a 40.
with torch.no_grad():
    alone, _ = model(teacher_forcing_inputs(x[:1, :25]), y[:1], task[:1])
    padded_x = x.clone()
    padded_x[0, 25:] = 0.0
    batch_out, _ = model(teacher_forcing_inputs(padded_x), y, task)
same = max((alone[k][0] - batch_out[k][0, :25]).abs().max().item() for k in alone)
check(same < 1e-5, f"il padding non cambia nessuna uscita reale (differenza massima {same:.1e})")

mask = torch.ones(x.shape[:2], dtype=torch.bool)
mask[0, 25:] = False
with torch.no_grad():
    loss_padded = mdn_loss(batch_out, padded_x, mask)
    garbage_x = padded_x.clone()
    garbage_x[0, 25:] = 1e6                                          # qualsiasi valore nel padding
    garbage_out, _ = model(teacher_forcing_inputs(garbage_x), y, task)
    loss_garbage = mdn_loss(garbage_out, garbage_x, mask)
check(all(abs(loss_padded[k] - loss_garbage[k]) < 1e-5 for k in loss_padded),
      "la loss ignora quello che c'è nel padding")

# Il gradiente arriva a tutte le parti nuove (partono tutte da zero).
model.train()
model.zero_grad()
terms = mdn_loss(model(teacher_forcing_inputs(x), y, task)[0], x, mask)
sum(terms.values()).backward()
named = dict(model.named_parameters())
no_gradient = [n for n in new_parameter_names(model) if named[n].grad is None or named[n].grad.abs().sum() == 0]
check(not no_gradient, f"ogni parametro nuovo riceve un gradiente {no_gradient or ''}")
check(named["lstm1.W_xh"].grad.abs().sum() > 0, "le LSTM pretrained ricevono un gradiente")
check(all(torch.isfinite(t) for t in terms.values()), "termini della loss finiti: " +
      ", ".join(f"{k} {v.item():.3f}" for k, v in terms.items()))

# La condizione arriva all'uscita quando il suo embedding non è zero.
with torch.no_grad():
    model.eval()
    before, _ = model(x[:1], torch.tensor([0]), task[:1])
    model.class_embed.weight[0].normal_(0, 0.5)
    after, _ = model(x[:1], torch.tensor([0]), task[:1])
    other, _ = model(x[:1], torch.tensor([1]), task[:1])
check((before["mu"] - after["mu"]).abs().max() > 1e-3 and (after["mu"] - other["mu"]).abs().max() > 1e-3,
      "l'embedding di classe cambia la distribuzione prevista")

# Generazione.
model = fresh_model()
g = torch.Generator().manual_seed(0)
out, out_mask = generate(model, torch.tensor([0, 1]), torch.tensor([2, 5]), torch.tensor([30, 18]),
                         bias=0.5, in_air_pressure=-3.0, generator=g)
check(out.shape == (2, 30, 6) and out_mask.sum().item() == 48, "generate: shape e lunghezze")
check(bool(torch.isfinite(out).all()), "generate: tutti i valori finiti")
air = (out[..., 5] == 0) & out_mask
check(bool((out[..., 2][air] == -3.0).all()), "generate: pressione al valore in aria dove la penna è sollevata")
check(bool((out[1, 18:] == 0).all()), "generate: gli step oltre la lunghezza della registrazione sono zero (padding)")

# Sampling bias (Graves 2013, sez. 5.4): pesi più netti, larghezze più strette.
p0 = mixture_parameters(batch_out, bias=0.0)
p1 = mixture_parameters(batch_out, bias=1.0)
check(bool((p1["sigma1"] <= p0["sigma1"] + 1e-6).all()) and
      bool((p1["log_pi"].max(-1).values >= p0["log_pi"].max(-1).values - 1e-6).all()),
      "il sampling bias restringe le gaussiane e rende più netti i pesi della mixture")

# =============================================================================
print("\nC  Il checkpoint ufficiale")
# =============================================================================
if not CHECKPOINT.exists():
    print(f"   saltata: {CHECKPOINT} non trovato")
else:
    model = PaHaWMDN(config)
    report = load_pretrained_mdn(model, CHECKPOINT)
    report.print(model)
    check(len(report.loaded) == 32 and len(report.sliced) == 3 and len(report.discarded) == 8,
          "32 tensori caricati (3 senza la riga dell'end-of-stroke), 8 scartati")
    check(all(model.state_dict()[n].abs().sum() == 0 for n in report.new), "tutti i tensori nuovi partono da zero")

print("\nTutti i controlli superati." if not failures else f"\n{len(failures)} CONTROLLI NON SUPERATI: {failures}")
sys.exit(1 if failures else 0)
