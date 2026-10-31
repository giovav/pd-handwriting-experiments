# Benchmark generativo su PaHaW

Codice della tesi. Tre modelli generativi di scrittura a mano online, partiti da pesi
pubblici, vengono adattati con fine-tuning al dataset PaHaW per generare registrazioni
sintetiche di soggetti sani e con malattia di Parkinson, condizionate su classe e task.
I dataset sintetici vengono poi valutati con un classificatore CNN-BiLSTM addestrato sui
dati reali e, viceversa, usati per addestrarlo.

| Cartella | Modello | Pesi di partenza |
|---|---|---|
| `DiffInk/` | DiffInk (Pan et al., ICLR 2026): VAE + Diffusion Transformer | ufficiali degli autori |
| `autoregressiveMDN/` | LSTM + mixture density network (Graves, 2013) | reimplementazione PyTorch di E. Dolotov |
| `1DContinuousDiffusion/` | diffusione 1-D (Luhman e Luhman, 2020) | ufficiali degli autori, convertiti da Keras a PyTorch |

Ogni cartella ha un `PAHAW_README.md` con la citazione del modello originale. Per l'MDN e
per la diffusione 1-D, che non usano il codice degli autori così com'è, lo stesso file
riporta anche la corrispondenza tra l'implementazione e l'articolo.

## Struttura

```
.
├── common/                     moduli condivisi da tutti i modelli
│   ├── dataset_prep.py         lettura dei .svc, 6 canali, normalizzazione, DataLoader
│   ├── create_split.py         split per paziente, 4 fold
│   ├── packed_data.py          salvataggio e lettura dei dati elaborati in un unico .npz
│   ├── train_common.py         ciclo di training, early stopping, checkpoint
│   └── synthetic_common.py     piano di generazione e salvataggio dei dataset sintetici
├── export_packed_data.py       esporta dati pseudonimizzati e split in un unico file
├── check_jumps.py              controllo delle regole sui gap del tablet (dati grezzi)
├── DiffInk/                    train_vae.py, train_dit.py, generate_synthetic.py, test
├── autoregressiveMDN/          train_mdn.py, generate_synthetic.py, test
├── 1DContinuousDiffusion/      convert_official_weights.py, train_diffusion.py,
│                               generate_synthetic.py, test
├── classifier/                 classificatore e valutazione dei dataset sintetici
└── output/                     punteggi ottenuti dai modelli
```

