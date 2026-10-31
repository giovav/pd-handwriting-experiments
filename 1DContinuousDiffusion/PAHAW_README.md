# Modello di diffusione 1-D: corrispondenza con Luhman e Luhman (2020)

**Articolo.** T. Luhman, E. Luhman, *Diffusion Models for Handwriting Generation*,
arXiv:2011.06704v1. I numeri di equazioni, algoritmi, sezioni e figure qui sotto si riferiscono a
questa versione.

**Codice e pesi ufficiali.** `tcl9876/Diffusion-Handwriting-Generation` (licenza MIT), citato
nell'abstract dell'articolo. Architettura in `nn.py`, pesi in `weights/model_weights.h5`
(TensorFlow/Keras).

**File del progetto** (`1DContinuousDiffusion/`): `luhman_layers.py`, `tools/make_weight_map.py`,
`weight_map.json`, `reference/tf_reference.npz`, `convert_official_weights.py`, `diffusion_model.py`,
`pretrained.py`, `diffusion.py`, `configs/pahaw_config.yaml`, `test_pahaw_diffusion.py`.

---

## 1. Perché serve una conversione in PyTorch

Il benchmark è scritto in PyTorch, e ogni modello deve partire dai pesi ufficiali per il
fine-tuning. I pesi degli autori esistono solo come file Keras `.h5`, legato al loro codice
TensorFlow.

Esiste un port PyTorch di terzi, `sleep3r/Diffusion-Handwriting-Generation.pytorch`, con la stessa
architettura e dei pesi. Quei pesi però sono stati riaddestrati dall'autore del port
(1 gennaio 2026) e non sono quelli degli autori, quindi il port non è stato usato.

Procedura seguita:

1. Riscrivere ogni layer di `nn.py` in PyTorch con gli stessi nomi degli attributi (`luhman_layers.py`).
2. Associare ogni tensore Keras a un solo parametro PyTorch seguendo il percorso degli attributi, e
   verificare l'associazione sui valori (`weight_map.json`).
3. Convertire il file `.h5` ufficiale e verificare la rete convertita sulle uscite del codice
   TensorFlow.
4. Togliere dalla rete verificata il condizionamento su testo e stile, e verificare che la parte
   rimasta calcoli la stessa funzione.

## 2. Architettura originale e articolo

La rete di `nn.py` (`DiffusionWriter`, $c_1 = 128$, $c_2 = 192$, $c_3 = 256$, due layer di
attention) è stata confrontata con la Sezione 3 e l'Appendice A dell'articolo.

| Articolo | `nn.py` / `luhman_layers.py` | Note |
|---|---|---|
| Livello di rumore $\sqrt{\bar\alpha}$ passato in una rete feed-forward di due layer fully connected (Sez. 3) | `sigma_ffn = FeedForward(1, 32, 2048)` | Uscita: un vettore di 32 valori per il livello di rumore. |
| Condizionamento con trasformazioni affini lungo l'asse dei canali, scale e bias da un layer fully connected (Sez. 3, Fig. 6) | `AffineTransformLayer`: $x \odot \gamma(s) + \beta(s)$, con `gamma_dense`, `beta_dense` che leggono il vettore di 32 valori | Layer FiLM dopo ogni convoluzione, attention e feed-forward. |
| Blocco convoluzionale: 3 layer convoluzionali e una skip connection convoluzionale, trasformazione affine dopo ogni layer (Sez. 3, Fig. 7) | `ConvSubLayer`: `conv1` (kernel 3), `conv2` (kernel 3), `fc` (dense per posizione, equivalente a kernel 1), `conv_skip`; `affine1–3` | L'originale passa `dils[0]` (sempre 1) a entrambe le convoluzioni, quindi nessuna convoluzione è dilatata. Riprodotto così. |
| Blocco di attention: cross-attention sull'encoder di testo e stile, self-attention, feed-forward; layer normalization seguita da una trasformazione affine dopo ognuno; positional encoding sinusoidali sommati a query e key; posizioni del testo scalate alle risoluzioni più alte (Sez. 3, Fig. 8) | `DecoderLayer`: `text_dense`, `mha`, `affine0`, `affine1` (cross-attention); `mha2`, `affine2` (self-attention); `ffn`, `affine3`; `pos_factor` 4, 2 e 1 | Teste: 3 a $T/2$ (`enc3`), 4 a $T/4$ (`enc5`), 6 a $T/8$ (`att_layers`). |
| Encoder di testo e stile: embedding dei caratteri, feature MobileNetV2 di un'immagine dello scrittore, attention tra i due, feed-forward (Sez. 3, Fig. 10) | `TextStyleEncoder` | Tolto nel modello PaHaW (Sezione 6). |
| Blocchi di downsampling e upsampling con lunghe skip connection convoluzionali (Sez. 3, Fig. 9) | Tre average pooling ($T \to T/8$), upsampling nearest-neighbour, `skip_conv1–3` | — |
| Uscite: rumore previsto per gli offset della penna e stima $\hat d_0$ dello stato della penna, con tutti i parametri condivisi (Sez. 2.2) | `output_dense` (128 → 2), `pen_lifts_dense` (128 → 1, sigmoid) | `pen_lifts_dense` predice il bit della penna dell'articolo: 1 quando la penna è sollevata. |
| 10,0 milioni di parametri (Sez. 5.1) | File `.h5`: 333 tensori, 10.031.651 valori | 10 tensori appartengono a layer normalization congelate nell'originale (`trainable=False`). Hanno tutti scala 1 e shift 0, quindi sono l'identità e non vengono convertiti. |
| Preprocessing: ogni esempio diviso per la sua deviazione standard; tratti con direzioni simili uniti (Sez. 5.1) | Scala per riga; tre passate che uniscono ognuna il 20% dei punti; righe di al massimo 480 punti | Non riprodotto (Sezione 8). |

## 3. I layer Keras riscritti in PyTorch: `luhman_layers.py`

Ogni classe di `nn.py` è stata riscritta con gli stessi nomi degli attributi:
`FeedForward`, `AffineTransformLayer`, `ConvSubLayer`, `MultiHeadAttention`, `DecoderLayer`,
`TextStyleEncoder`, `OriginalDiffusionWriter`, e le funzioni `positional_encoding`, `avg_pool`,
`upsample`, `layer_norm`. Si ricostruisce la rete originale completa, comprese le parti che PaHaW non
usa, per poter verificare la conversione sul codice degli autori.

Corrispondenze tra le primitive dei due framework:

| Keras | PyTorch |
|---|---|
| `LayerNormalization` congelata (scala e shift identità) | Normalizzazione senza parametri affini (`layer_norm`) |
| `Dense` sull'ultimo asse | `nn.Linear` |
| `Conv1D(kernel_size=3, padding='same')` su tensori channels-last | `nn.Conv1d` con padding 1, applicato a tensori channels-last (`Conv1dLast`) |
| Logit dell'attention $+\,(-10^{12})$ sulle key mascherate | `F.scaled_dot_product_attention` con maschera booleana sulle key (peso esattamente 0) |
| Tabella di positional encoding di 2000 posizioni | Le stesse sinusoidi calcolate per qualsiasi lunghezza |

Il kernel di attention fuso evita di tenere in memoria la matrice dei punteggi $T \times T$. Le
registrazioni PaHaW arrivano a 16.071 step, contro i 480 punti per riga dei dati originali. La
tabella di 2000 posizioni dell'originale farebbe fallire l'attention a $T/2$ oltre i 4.000 step; il
calcolo al momento dà valori identici sotto quella lunghezza.

Ogni blocco accetta una maschera di padding opzionale, e `DecoderLayer` ha l'opzione `with_text`.
Nessuna delle due cambia il calcolo originale quando non è usata.

## 4. Mappa dei pesi per percorso di attributi: `weight_map.json`

Keras dà ai layer nomi con contatori globali (`dense_17`, `dense_108`). Associare i pesi per nome
dipenderebbe dall'ordine in cui lo script degli autori ha creato i layer, e uno sfasamento
sposterebbe dei pesi senza che nessuno se ne accorga. La mappa è quindi costruita dal modello stesso
degli autori:

1. `tools/make_weight_map.py` costruisce il modello TensorFlow originale con il `nn.py` originale e
   carica il file ufficiale con la chiamata degli autori, `model.load_weights`.
2. Per ogni nome di parametro PyTorch segue lo stesso percorso di attributi nel modello Keras (per
   esempio `enc3.mha2.wq.weight` → `model.enc3.mha2.wq.kernel`).
3. Cerca l'unico dataset del `.h5` con valori identici a quella variabile Keras, e si ferma se ne
   trova zero o più di uno.
4. Scrive la tabella: 323 parametri, ognuno con il suo percorso nel `.h5` e il layout. Esempi:

```json
"input_dense.weight":     {"h5": "dense/score_network/dense/kernel:0", "layout": "dense"},
"sigma_ffn.dense1.weight": {"h5": "sequential/dense_1/kernel:0",       "layout": "dense"}
```

Lo strumento salva anche le uscite del modello originale su ingressi casuali fissi
(`reference/tf_reference.npz`), così la conversione si può verificare in seguito senza TensorFlow.

`convert_official_weights.py` legge il `.h5` con h5py e copia ogni tensore nel suo nome PyTorch. I
kernel Dense vengono trasposti ($[\text{in}, \text{out}] \to [\text{out}, \text{in}]$) e quelli Conv1D
permutati ($[k, \text{in}, \text{out}] \to [\text{out}, \text{in}, k]$). Il risultato viene caricato
in `OriginalDiffusionWriter` con `strict=True`.

## 5. Verifica della conversione

Il convertitore fa due controlli prima di scrivere `pretrained/luhman_official.pt`:

| Controllo | Ingressi | Differenza assoluta massima |
|---|---|---|
| 1a: uscite TensorFlow salvate | Testo, stile e tre livelli di rumore casuali fissi | $2{,}2 \times 10^{-5}$ sul rumore previsto (valori fino a 3,8); $1{,}5 \times 10^{-5}$ sulla probabilità di penna sollevata (`pretrained/conversion_report.txt`) |
| 1b (`--verify_tf`): codice TensorFlow originale eseguito al momento della conversione | Ingressi casuali nuovi | circa $4 \times 10^{-5}$ |

Differenze dell'ordine di $10^{-5}$ su uscite dell'ordine di 1 sono arrotondamenti float32 tra due
implementazioni di una rete profonda. Il file viene scritto solo se ogni differenza è sotto
$10^{-3}$. La rete convertita ha 323 tensori (10.028.451 valori). Lo SHA-256 del file di partenza
viene stampato e salvato insieme al risultato.

## 6. Rimozione del condizionamento sul testo

### 6.1 Encoder e cross-attention

Le registrazioni PaHaW non hanno trascrizione (il task fissa il contenuto), e imitare un certo
scrittore non è un obiettivo. `PaHaWDiffusionWriter` (`diffusion_model.py`) è la rete
verificata senza:

- il `TextStyleEncoder`;
- in ogni blocco di attention (`enc3`, `enc5`, `att_layers`), il ramo di cross-attention
  `text_dense`, `affine0`, `mha`, `affine1`.

In `DecoderLayer` il sotto-layer di cross-attention calcola
$x_2 = \mathrm{affine}_1(\mathrm{LN}(\mathrm{MHA}(x + PE, \text{testo})), s) + x$. Con
`with_text=False` il layer pone $x_2 = x$, che è ciò che calcola l'originale quando la
cross-attention non aggiunge nulla. Self-attention, feed-forward, tutte le convoluzioni, le skip
connection, `sigma_ffn` e gli altri layer FiLM mantengono nomi, dimensioni e pesi: 6,00 dei 10,03
milioni di parametri.

Togliendo la cross-attention sul testo e nient'altro, la rete ufficiale genera gruppi di tratti
corsivi simili a parole, separati da sollevamenti della penna a distanze plausibili tra parole.

### 6.2 Condizione clinica

Il vettore del livello di rumore $s = \mathrm{sigma\_ffn}(\sqrt{\bar\alpha})$ è l'unico segnale
globale che ogni blocco legge, attraverso i suoi layer FiLM. Classe e task vengono sommati a questo
vettore:

$$s' = \mathrm{sigma\_ffn}(\sqrt{\bar\alpha}) + E_{\text{class}}[y] + E_{\text{task}}[\text{task}], \qquad E_{\text{class}} \in \mathbb{R}^{2 \times 32},\; E_{\text{task}} \in \mathbb{R}^{8 \times 32}$$

La dimensione è 32 perché è quella del vettore letto dai layer FiLM. Entrambi gli embedding partono
da zero, quindi la rete caricata calcola la funzione pretrained, e il gradiente arriva agli embedding
attraverso i layer FiLM.

### 6.3 Canali di ingresso e uscita

`input_dense` (2 → 128) e `output_dense` (128 → 2) mantengono le shape ufficiali e servono $dx, dy$. Pressure, azimuth e altitude entrano attraverso `input_new` (3 → 128, senza bias),
sommato all'uscita di `input_dense`. Il loro rumore è previsto da `output_new` (128 → 3), concatenato
dopo `output_dense`. Entrambi i layer nuovi partono da zero. Prima del fine-tuning la rete quindi
ignora i canali nuovi e prevede per loro rumore zero, e il calcolo su $dx, dy$ resta invariato.

### 6.4 Testa della penna

La Sezione 2.2 definisce il bit della penna come 0 quando la penna è giù e 1 altrimenti;
`pen_lifts_dense` predice questa probabilità. Il canale pen di PaHaW vale 1 quando la penna tocca la
superficie. Siccome $1 - \sigma(z) = \sigma(-z)$, peso e bias di `pen_lifts_dense` vengono
moltiplicati per $-1$ e salvati come `pen_dense`. La testa restituisce così esattamente
la probabilità complementare, e la conoscenza pretrained di dove si interrompono i tratti resta.

### 6.5 Riepilogo dell'adattamento e verifica

`load_pretrained_diffusion` (`pretrained.py`) si ferma se un parametro diverso da quelli dichiarati
nuovi resterebbe senza pesi.

| Gruppo | Tensori | Trattamento |
|---|---|---|
| Convoluzioni, self-attention, feed-forward, FiLM, skip, `sigma_ffn`, `input_dense`, `output_dense` | 218 | Copiati invariati |
| `pen_lifts_dense` → `pen_dense` | 2 (dei 218) | Segno invertito |
| `class_embed`, `task_embed`, `input_new`, `output_new` (peso, bias) | 5 (1.091 valori) | Nuovi, inizializzati a zero |
| Encoder di testo e stile e rami di cross-attention | 105 | Scartati |

Il check 2 del convertitore confronta la rete adattata con `OriginalDiffusionWriter` eseguito senza
cross-attention, entrambi con i pesi convertiti:

- rumore previsto per $dx, dy$: identico (differenza 0);
- probabilità della penna: complementare entro $6 \times 10^{-8}$;
- canali nuovi: uscita zero;
- classe e task: nessun effetto prima del training.

Il risultato viene scritto in `pretrained/luhman_pahaw_init.pt` (formato `pahaw-diffusion-init-v2`)
con lo SHA-256 del file di partenza e gli elenchi dei tensori caricati, con il segno invertito, nuovi
e scartati. `test_pahaw_diffusion.py` ripete questi controlli e verifica che il file salvato
coincida con l'adattamento ricalcolato dai pesi ufficiali.

## 7. Dalle equazioni al codice per training e campionamento

`diffusion.py`:

| Articolo | Codice | Note |
|---|---|---|
| $T = 60$; $\beta_t = 0{,}02 + \mathrm{Exponential}(10^{-5}, 0{,}4)$ (Sez. 5.1) | `beta_set`: `0.02 + exp(linspace(log 1e-5, log 0.4, 60))`; `alpha_set`: $\bar\alpha_t = \prod_s (1 - \beta_s)$, eq. (6) | — |
| $\sqrt{\bar\alpha} \sim \mathrm{Uniform}(l_{t-1}, l_t)$, condizionamento sul livello di rumore continuo (Sez. 2.2; Alg. 1, righe 3–4) | `sample_alpha_bar`: un indice di livello estratto uniformemente, $\bar\alpha$ estratto uniformemente tra due livelli vicini; alla rete si passa $\sqrt{\bar\alpha}$ | Riproduce il codice ufficiale, che interpola $\bar\alpha$ invece di $\sqrt{\bar\alpha}$ dentro ogni intervallo. |
| $y_t = \sqrt{\bar\alpha}\, y_0 + \sqrt{1 - \bar\alpha}\, \epsilon$, eq. (6); Alg. 1, riga 6 | `noisy = sqrt(a) * clean + sqrt(1 - a) * eps` | Applicato ai 5 canali continui; lo stato della penna non riceve rumore. |
| $L_\text{stroke} = \lVert \epsilon - \epsilon_\theta(y_t, c, s, \sqrt{\bar\alpha}) \rVert_2^2$, eq. (13) | `noise_xy`: errore quadratico del rumore previsto, sommato su $dx, dy$ e mediato sugli step reali | Eq. (8) senza $C_t$, come detto nella Sez. 2.1. |
| — | `noise_aux`: lo stesso per pressure, azimuth, altitude; pressure solo con la penna sulla superficie | Canali nuovi. |
| $L_\text{drawn}$, eq. (14), pesata con $\bar\alpha$ (Alg. 1, riga 8) | `pen`: binary cross-entropy dello stato della penna moltiplicata per $\bar\alpha$ | — |
| Campionamento, eq. (12); Alg. 2: $y_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(y_t - \sqrt{1-\bar\alpha_t}\, \epsilon_\theta\right) + \sqrt{1 - \bar\alpha_{t-1}}\, z$ | `sample`: `x = (x - sqrt(1 - a) * eps_pred) / sqrt(1 - b)`; `x = x + sqrt(1 - a_next) * z` | $1 - \beta_t = \alpha_t$. Il codice ufficiale (`diffusion_mode='new'`, il default) pone $\bar\alpha_{\text{next}} = 1$ negli ultimi due passi; riprodotto. |
| $d_0 = d_\theta(y_t, \ldots)$ letto all'ultimo passo (Alg. 2, riga 6) | `pen = (pen_logit > 0)` dall'ultima chiamata della rete | Dove la penna è in aria, la pressione è messa al suo valore in aria. |

## 8. Differenze rispetto all'articolo

| Aspetto | Articolo / codice ufficiale | Implementazione | Motivo |
|---|---|---|---|
| Condizione | Testo e immagine dello stile dello scrittore | Embedding di classe e task sommati al vettore del livello di rumore | PaHaW non ha trascrizioni e non si imita uno scrittore |
| Canali continui | $dx, dy$ | $dx, dy$, pressure, azimuth, altitude | Canali di PaHaW; i canali nuovi passano per layer inizializzati a zero |
| Preprocessing | Scala per riga con la deviazione standard; unione dei punti (Sez. 5.1) | Normalizzazione globale comune al benchmark; nessuna unione dei punti | La scala per riga cancellerebbe la micrografia; unire i punti cambierebbe la frequenza di campionamento su cui si misurano velocità e tremore |
| Padding | Padding a zero, non mascherato | Maschera di padding a ogni risoluzione; key mascherate nell'attention; posizioni di padding azzerate prima di ogni convoluzione | Le uscite di una registrazione non dipendono dal batch (verificato entro $2 \times 10^{-6}$) |
| Aggregazione della loss | Media che include le posizioni di padding | Media solo sugli step reali | Stesso motivo |
| Ottimizzatore | Adam ($\beta_1 = 0{,}9$, $\beta_2 = 0{,}98$), clipping della norma del gradiente a 100, schedule inverse square-root con 10.000 step di warm-up (Sez. 5.1) | AdamW; matrici pretrained a lr $10^{-5}$ con weight decay 0,01, parametri nuovi a lr $10^{-4}$; clipping della norma del gradiente a 1; warm-up; early stopping sui pazienti di validazione | Fine-tuning da pesi pretrained; stesse protezioni per tutti i modelli del benchmark |
| Lunghezza della sequenza | Decisa dal testo | Data in generazione, estratta dalle registrazioni reali della stessa classe e task | Non c'è testo |

La differenza di preprocessing è un cambio di dominio tra i dati di pre-addestramento (punti più
radi, scala per riga) e PaHaW, che il fine-tuning deve assorbire.
