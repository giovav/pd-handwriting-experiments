"""
inspect_checkpoint.py: stampa il contenuto di un checkpoint ufficiale di DiffInk.

Moduli, numero di parametri e le shape che fissano la configurazione, da confrontare
con i valori [PRETRAINED] di configs/pahaw_config.yaml prima del fine-tuning.
Per il DiT stampa anche quanta parte del peso adaLN del blocco 0 sta su una sola
direzione (vedi finetune.py).

    python inspect_checkpoint.py pretrained/dit_epoch_1.pt
"""

import sys
from collections import defaultdict

import torch

from pretrained import read_state_dict

# tensore -> significato della sua shape
LANDMARKS = [
    ("encoder.conv_1.weight",                "[hidden_dims[0], in_channels, 3]"),
    ("conv_mu.weight",                       "[latent_dim, hidden_dims[2], 1]"),
    ("transformer_decoder.input_proj.weight", "[trans_hidden_dim, decoder_dims[2]]"),
    ("transformer_decoder.fc.weight",        "[decoder outputs, trans_hidden_dim]"),
    ("input_embed.proj.weight",              "[dim, latent_dim + text_dim]"),
    ("proj_out.weight",                      "[latent_dim, dim]"),
    ("text_embed.text_embed.weight",         "[number of characters + 1, text_dim]"),
    ("transformer_blocks.0.attn.to_q.weight", "[heads * dim_head, dim]"),
    ("transformer_blocks.0.attn_norm.linear.weight", "[6 * dim, dim]"),
]


def main(path: str) -> None:
    raw = torch.load(path, map_location="cpu", weights_only=True)
    print(f"\nFile: {path}")
    print(f"Chiavi principali: {list(raw.keys())[:8] if isinstance(raw, dict) else type(raw)}")
    if isinstance(raw, dict) and "epoch" in raw:
        print(f"Epoca salvata: {raw['epoch']}")

    state = read_state_dict(path)
    total = sum(t.numel() for t in state.values())
    print(f"Tensori: {len(state)}   parametri: {total / 1e6:.1f} M")

    groups = defaultdict(lambda: [0, 0])
    for name, tensor in state.items():
        top = name.split(".")[0]
        groups[top][0] += 1
        groups[top][1] += tensor.numel()
        
    print("\nModuli:")
    for top, (count, values) in sorted(groups.items(), key=lambda kv: -kv[1][1]):
        print(f"   {top:22s} {count:4d} tensori  {values / 1e6:8.2f} M parametri")

    print("\nShape che fissano la configurazione:")
    for name, meaning in LANDMARKS:
        if name in state:
            print(f"   {name:46s} {str(tuple(state[name].shape)):18s} = {meaning}")
    blocks = {int(n.split(".")[1]) for n in state if n.startswith("transformer_blocks.")}
    if blocks:
        print(f"   blocchi del DiT: {max(blocks) + 1}")

    # Quota dell'energia del peso adaLN (blocco 0) sul primo valore singolare.
    # Se è vicina al 100%, il layer produce di fatto un'uscita costante: per rispondere
    # alla condizione clinica gli serve un adattatore (LoRA).
    key = "transformer_blocks.0.attn_norm.linear.weight"
    if key in state:
        singular = torch.linalg.svdvals(state[key].float())
        share = (singular[0] ** 2 / (singular ** 2).sum()).item()
        print(f"\nPeso adaLN del blocco 0: {100 * share:.1f}% dell'energia su una sola direzione")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Uso: python inspect_checkpoint.py percorso/del/checkpoint.pt")
    main(sys.argv[1])