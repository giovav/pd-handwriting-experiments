# MDN autoregressivo: corrispondenza con Graves (2013)

**Articolo.** A. Graves, *Generating Sequences With Recurrent Neural Networks*, arXiv:1308.0850v5.
I numeri di equazioni e sezioni qui sotto si riferiscono a questa versione.

**Pesi.** `X-rayLaser/pytorch-handwriting-synthesis-toolkit` (E. Dolotov, licenza MIT, vedi `LICENSE`),
checkpoint della synthesis network `checkpoints/Epoch_56/model.pt`, addestrato su IAM-OnDB.
Graves non ha pubblicato pesi; questo repository reimplementa in PyTorch la synthesis network della
Sezione 5 e ne pubblica i checkpoint addestrati.

**File del progetto** (`autoregressiveMDN/`): `mdn_model.py`, `mdn_head.py`, `pretrained.py`,
`train_mdn.py`, `configs/pahaw_config.yaml`, `test_pahaw_mdn.py`.

---

## 1. Verifica del repository rispetto all'articolo

Prima di qualsiasi modifica, il codice del repository e il checkpoint Epoch_56 sono stati confrontati
con l'articolo. Il checkpoint contiene 3.764.151 valori, in accordo con i "circa 3,7M" pesi della
Sezione 5.2.

| Articolo | Repository | Esito |
|---|---|---|
| LSTM con peephole, eq. (7)–(11): i gate di input e forget leggono $c_{t-1}$, quello di output legge $c_t$; pesi peephole diagonali (Sez. 2.1) | `PeepholeLSTM`: `W_ci`, `W_cf`, `W_co` sono vettori; pesi di input e ricorrenti salvati in un'unica matrice `W_xh` applicata a $[x_t, h_{t-1}]$ | Identico. Un'unica matrice concatenata equivale alle matrici separate $W_{x\cdot}$, $W_{h\cdot}$. |
| Synthesis network, eq. (52)–(53): 3 layer nascosti da 400 celle; l'ingresso arriva a ogni layer; ogni layer arriva all'uscita; finestra $w_{t-1}$ nel layer 1, $w_t$ nei layer 2 e 3 (Sez. 5.1–5.2) | `SynthesisNetwork(3, 400, …)`: `lstm1` legge $[x, w_{t-1}]$, `lstm2` legge $[x, h^1, w_t]$, `lstm3` legge $[x, h^2, w_t]$; l'uscita legge $[h^1, h^2, h^3]$ | Identico. `W_xh` ha 483 righe nel layer 1 (3 ingressi + 80 finestra + 400 ricorrenti) e 883 nei layer 2–3 (3 + 400 + 80 + 400). |
| Soft window, eq. (46)–(51): $K = 10$ gaussiane, $\alpha = \exp(\hat\alpha)$, $\beta = \exp(\hat\beta)$, $\kappa_t = \kappa_{t-1} + \exp(\hat\kappa)$ | `SoftWindow`, 10 componenti, stesse formule; posizioni dei caratteri contate da 0 | Identico a meno di uno sfasamento di indice assorbito da $\kappa$. Finestra di dimensione 80 (il set di caratteri del repository) invece di 57 (Sez. 5.2). |
| Uscita a mixture, eq. (17)–(22): $M = 20$ gaussiane bivariate; $\pi$ = softmax, $\mu$ = identità, $\sigma = \exp$, $\rho = \tanh$, $e = 1/(1+\exp(\hat e))$ | `MixtureDensityLayer(1200, 20)`: softmax, identità, exp, tanh; $e = \mathrm{sigmoid}(\hat e)$ | Identico tranne la convenzione di segno di $\hat e$, una riparametrizzazione equivalente. |
| Loss della sequenza, eq. (23)–(26) | `losses.nll_loss`: stessi termini, costanti 1e-8 e 1e-16 aggiunte dentro i logaritmi, somma sugli step, media sul batch | Identico a meno delle costanti di stabilità. |
| Biased sampling, eq. (61)–(62) | `MixtureDensityLayer` con `bias` | Identico. |
| RMSProp, eq. (38)–(45): $\aleph = 0{,}95$, momentum 0,9, learning rate $10^{-4}$, stabilizzatore $10^{-4}$; derivate dell'uscita limitate a $[-100, 100]$, derivate delle LSTM a $[-10, 10]$ (Sez. 4.2) | `CustomRMSprop(lr=1e-4, alpha=0.95, eps=1e-4, centered=True)`, momentum 0,9; clipping opzionale dei gradienti dei parametri per valore | Ottimizzatore identico. Il clipping è applicato ai gradienti dei parametri, non alle derivate interne alla rete. |
| Preprocessing: offset normalizzati a media 0 e deviazione standard 1 sul training set (Sez. 4) | Offset standardizzati con media e deviazione standard del dataset (`meta.json`: dx media 8,2, std 42,3; dy media 0,12, std 37,1) | Coerente. |

Il repository è stato quindi accettato come implementazione fedele delle Sezioni 2, 4.1 e 5 dell'articolo.

## 2. Dalle equazioni al codice nell'implementazione PaHaW

`mdn_model.py` e `mdn_head.py` sostituiscono `SynthesisNetwork`, `losses.py` e le funzioni di
campionamento del repository. Nomi e shape di tutti i parametri mantenuti sono quelli del repository,
quindi il checkpoint si carica direttamente.

### 2.1 Layer nascosti: eq. (52)–(53)

| Articolo | Codice (`PaHaWMDN.forward`) |
|---|---|
| $h^1_t = \mathcal{H}(W_{ih^1} x_t + W_{h^1h^1} h^1_{t-1} + W_{wh^1} w_{t-1} + b^1_h)$ | `z1 = lstm1.project(cat[dx, dy, cond]) + extra_input[0](new)`; `h1 = lstm1(z1)` |
| $h^n_t = \mathcal{H}(W_{ih^n} x_t + W_{h^{n-1}h^n} h^{n-1}_t + W_{h^nh^n} h^n_{t-1} + W_{wh^n} w_t + b^n_h)$, $n = 2, 3$ | `z2 = lstm2.project(cat[dx, dy, h1, cond]) + extra_input[1](new)`, e allo stesso modo per il layer 3 |
| $\hat y_t = b_y + \sum_n W_{h^n y} h^n_t$, eq. (17) | `mixture(cat[h1, h2, h3])` |

Il termine della finestra $W_{wh^n} w_t$ è sostituito da $W_{wh^n}\,c$, dove $c$ è il vettore della
condizione clinica (Sezione 3.2). Le righe $W_{wh^n}$ sono quelle del checkpoint.

### 2.2 LSTM con peephole: eq. (7)–(11)

`_recurrence` in `mdn_model.py`:

| Articolo | Codice |
|---|---|
| (7) $i_t = \sigma(W_{xi}x_t + W_{hi}h_{t-1} + W_{ci}c_{t-1} + b_i)$ | `i = sigmoid(z_i + c * W_ci + b_i)` |
| (8) $f_t = \sigma(W_{xf}x_t + W_{hf}h_{t-1} + W_{cf}c_{t-1} + b_f)$ | `f = sigmoid(z_f + c * W_cf + b_f)` |
| (9) $c_t = f_t c_{t-1} + i_t \tanh(W_{xc}x_t + W_{hc}h_{t-1} + b_c)$ | `c = f * c + i * tanh(z_c + b_c)` |
| (10) $o_t = \sigma(W_{xo}x_t + W_{ho}h_{t-1} + W_{co}c_t + b_o)$ | `o = sigmoid(z_o + c * W_co + b_o)`, con la `c` aggiornata |
| (11) $h_t = o_t \tanh(c_t)$ | `h = o * tanh(c)` |

Qui $z = x_t W_x + h_{t-1} W_h$, diviso nei quattro blocchi dei gate `z_i, z_f, z_c, z_o`. $W_x$ e
$W_h$ sono le righe di ingresso e quelle ricorrenti della matrice salvata `W_xh`. I pesi peephole
`W_ci`, `W_cf`, `W_co` sono vettori (prodotti elemento per elemento), come prevede l'articolo per
matrici peephole diagonali.

Il contributo dell'ingresso $x_t W_x$ è calcolato per tutti gli step con un solo prodotto matriciale
(`PeepholeLSTM.project`); dentro il ciclo nel tempo si calcola solo $h_{t-1} W_h$. È un
riordino algebrico delle eq. (7)–(10), possibile perché la finestra, la cui posizione dipendeva da
$h^1_t$, è stata tolta dal ciclo. Il ciclo è in Python semplice: il TorchScript usato dal repository
è risultato più lento nel backward.

### 2.3 Uscita a mixture: eq. (18)–(22)

`mixture_parameters` in `mdn_head.py`, con $M = 20$ (`num_mixture`):

| Articolo | Codice | Differenza |
|---|---|---|
| (19) $\pi^j_t = \mathrm{softmax}(\hat\pi^j_t)$ | `log_softmax(pi * (1 + bias))` | Calcolato in scala logaritmica. |
| (20) $\mu^j_t = \hat\mu^j_t$ | `mu1, mu2` | Nessuna. |
| (21) $\sigma^j_t = \exp(\hat\sigma^j_t)$ | `exp(clamp(sd, max=5)) + SIGMA_MIN` | Minimo `SIGMA_MIN = 0.01`, massimo $e^5$. |
| (22) $\rho^j_t = \tanh(\hat\rho^j_t)$ | `tanh(ro) * RHO_MAX` | $\lvert\rho\rvert \le$ `RHO_MAX` $= 0{,}99$. |
| (18) end-of-stroke $e_t$ | `pen_logit` (Bernoulli, logit di "penna sulla superficie") | Variabile diversa; vedi Sezione 3.4. |

Le prime $K$ uscite di `mu` (e di `sd`) sono i centri (le larghezze) dx delle $K$ componenti, le
successive $K$ i centri (le larghezze) dy, come nel repository.

### 2.4 Loss della sequenza: eq. (23)–(26)

`mdn_loss` in `mdn_head.py` calcola i termini dell'eq. (26):

- Il termine della mixture $-\log \sum_j \pi^j_t \mathcal{N}(x_{t+1} \mid \mu^j_t, \sigma^j_t, \rho^j_t)$
  usa la normale bivariata delle eq. (24)–(25) in forma logaritmica:
  `log_gauss` $= -Z/(2(1-\rho^2)) - \log(2\pi\sigma_1\sigma_2) - \tfrac12\log(1-\rho^2)$.
  La somma sulle componenti è fatta con `logsumexp`.
- Il termine di Bernoulli $-\log e_t$ o $-\log(1-e_t)$ diventa la binary cross-entropy dello stato
  della penna di PaHaW.
- Si aggiunge una negative log-likelihood gaussiana per pressure, azimuth e altitude. La pressione
  conta solo dove la penna tocca la superficie.

Ogni termine è mediato sugli step reali (senza padding) invece che sommato su $t$. Cambia la scala
del gradiente, non il minimo. Un singolo step contribuisce al massimo `NLL_CAP = 100` nat.

### 2.5 Primo ingresso nullo (Sez. 2)

La Sezione 2 dice che $x_1$ è un vettore nullo, così il primo ingresso reale viene previsto senza
informazione precedente. `teacher_forcing_inputs` costruisce la sequenza di ingresso spostando la
registrazione di uno step a destra, con un vettore di zeri al primo posto. `generate` parte dallo
stesso vettore di zeri.

### 2.6 Biased sampling: eq. (61)–(62)

`mixture_parameters(raw, bias)` applica $\pi = \mathrm{softmax}(\hat\pi(1+b))$ e
$\sigma = \exp(\hat\sigma - b)$. La stessa riduzione delle larghezze vale per le gaussiane dei tre
canali aggiunti. `sample_step` estrae una componente, poi una coppia $(dx, dy)$ correlata dalla
normale bivariata di quella componente, poi lo stato della penna e i canali aggiunti. Il bias
configurato è $b = 0$ (unbiased sampling, Sez. 5.3).

## 3. Modifiche strutturali e trasferimento dei pesi

### 3.1 Rimozione della finestra sul testo: eq. (46)–(51)

`SoftWindow`, la sequenza one-hot dei caratteri e il test di fine testo sono stati tolti.
Le registrazioni di PaHaW non hanno trascrizione: il contenuto di ogni registrazione è fissato dal
task. I tensori `window.*` del checkpoint, che calcolano $(\hat\alpha, \hat\beta, \hat\kappa)$ da
$h^1_t$ (eq. 48), vengono scartati.

### 3.2 Condizione clinica al posto del vettore della finestra

Nelle eq. (52)–(53) il vettore della finestra è l'unica via con cui la condizione (il testo) arriva
ai layer ricorrenti, e arriva a tutti e tre i layer a ogni step. Alla condizione clinica viene data
la stessa via:

$$c = E_{\text{class}}[y] + E_{\text{task}}[\text{task}], \qquad E_{\text{class}} \in \mathbb{R}^{2 \times 80},\; E_{\text{task}} \in \mathbb{R}^{8 \times 80}$$

$c$ entra nelle 80 righe di ingresso $W_{wh^n}$ occupate dalla finestra in tutti e tre i layer, e
quelle righe mantengono i valori pretrained. Ne seguono due cose:

- $c$ è costante lungo una registrazione, perché classe e task valgono per tutta la registrazione.
  Il ritardo di uno step della finestra nel layer 1 ($w_{t-1}$ nell'eq. 52) quindi non ha effetto.
- Entrambi gli embedding partono da zero, quindi prima del fine-tuning la rete calcola la funzione
  originale con finestra vuota ($w_t = 0$). Il gradiente arriva agli embedding perché le righe
  $W_{wh^n}$ che li moltiplicano non sono nulle.

Per questo modello non si usa la classifier-free guidance. La forza della generazione si regola con
il sampling bias delle eq. (61)–(62).

### 3.3 Canali di ingresso: eq. (15)

L'ingresso di Graves è $x_t \in \mathbb{R} \times \mathbb{R} \times \{0, 1\}$: $(dx, dy)$ e un flag
di end-of-stroke. L'ingresso PaHaW ha sei canali:
$[dx, dy, \text{pressure}, \text{azimuth}, \text{altitude}, \text{pen}]$.

- $dx, dy$ hanno lo stesso significato nei due dataset (spostamento della penna per campione,
  standardizzato) ed entrano attraverso le loro righe pretrained di `W_xh` in tutti e tre i layer.
- La riga dell'end-of-stroke (riga 2 di ogni `W_xh`) viene tolta. Il canale pen di PaHaW dice se la
  penna tocca la superficie, che è un'altra grandezza: riusare quella riga le darebbe un significato
  sbagliato.
- Pressure, azimuth, altitude e pen entrano attraverso un nuovo layer lineare per ogni LSTM
  (`extra_input[n]`, 4 → 1600, senza bias), inizializzato a zero, la cui uscita si somma agli
  ingressi dei gate. Equivale a quattro righe in più di `W_xh`, ma lascia i tensori pretrained nella
  loro forma originale e permette un learning rate diverso per i pesi nuovi.

### 3.4 Teste d'uscita

Gli strati pretrained della mixture (`pi`, `mu`, `sd`, `ro`) restano. L'uscita
dell'end-of-stroke (`mixture.eos.*`) viene scartata e sostituita da un logit della penna (`pen`,
1200 → 1). Si aggiunge una testa gaussiana per pressure, azimuth e altitude (`aux`, 1200 → 6: tre
medie e tre log-larghezze). Entrambe le teste nuove partono da zero: probabilità della penna 1/2 e
una normale standard per ogni canale normalizzato.

### 3.5 Riepilogo del trasferimento

`load_pretrained_mdn` (`pretrained.py`) legge il checkpoint con `weights_only=True` e si
ferma se un tensore diverso da quelli dichiarati nuovi resterebbe senza pesi, o se una dimensione
della configurazione è diversa da quella del checkpoint.

| Gruppo | Tensori | Valori | Trattamento |
|---|---|---|---|
| Vettori peephole e bias delle tre LSTM; strati della mixture `pi`, `mu`, `sd`, `ro` | 29 | — | Copiati invariati |
| `W_xh` delle tre LSTM | 3 | — | Copiati senza la riga 2 (end-of-stroke) |
| Totale caricato | 32 | 3,75 M | — |
| `extra_input` (×3), `class_embed`, `task_embed`, `mixture.pen` (2), `mixture.aux` (2) | 9 | 0,03 M | Nuovi, inizializzati a zero |
| `window.*`, `mixture.eos.*` | 8 | — | Scartati |

Il 99,3% dei valori della rete viene dal checkpoint, compresi tutti i pesi ricorrenti. Togliere la
riga 2 di `W_xh` lascia ogni altra riga allineata con il suo ingresso, perché il repository
concatena prima $[dx, dy, \text{eos}]$ e poi gli altri ingressi nello stesso ordine
dell'implementazione PaHaW.

## 4. Verifica numerica

`test_pahaw_mdn.py` confronta l'implementazione PaHaW con il codice del repository,
entrambi caricati con i pesi ufficiali:

- Con end-of-stroke a 0 e finestra vuota nell'originale, i parametri della mixture coincidono entro
  $2{,}4 \times 10^{-6}$.
- Con le protezioni della Sezione 2.3 disattivate, la loss su $(dx, dy)$ coincide con la formula
  originale alla quinta cifra decimale. Con le protezioni attive il valore cambia dove la rete
  prevede componenti molto strette o quasi perfettamente correlate: 5,55 invece di 5,65 nat per step
  su target casuali. Le uscite della rete non cambiano.
- Le uscite sono identiche con e senza padding finale. La rete è causale e il padding sta solo alla
  fine di una registrazione.
- Il gradiente arriva a ogni parametro nuovo, e la condizione cambia l'uscita quando gli embedding
  non sono più nulli.

Campionata con finestra vuota, la rete pretrained disegna anelli simili al corsivo e frammenti di
lettere, non parole leggibili. Ciò che si trasferisce a PaHaW è la dinamica di basso livello del
movimento della penna, non un generatore di testo.

## 5. Differenze rispetto all'articolo

| Aspetto | Articolo | Implementazione | Motivo |
|---|---|---|---|
| Condizione | Testo attraverso una soft window (Sez. 5.1) | Embedding di classe e task attraverso le righe della finestra | PaHaW non ha trascrizioni |
| Ingresso | $(dx, dy, \text{eos})$ | Sei canali; eos tolto | Canali di PaHaW |
| Aggregazione della loss | Somma sugli step, eq. (26) | Media sugli step reali; cap per step; minimi su $\sigma$ e $\lvert\rho\rvert$ | Confrontabilità con gli altri modelli del benchmark; stabilità numerica |
| Ottimizzatore | RMSProp, eq. (38)–(45); clipping delle derivate (Sez. 4.2) | AdamW; pesi pretrained a lr $10^{-4}$ con weight decay 0,01, pesi nuovi a lr $10^{-3}$; clipping della norma del gradiente a 1 | Un solo ottimizzatore e le stesse protezioni per tutti i modelli del benchmark |
| Adaptive weight noise | Usato nel riaddestramento (Sez. 5.2) | Non usato | Fine-tuning con early stopping sui pazienti di validazione |
| Normalizzazione degli offset | Media 0, deviazione standard 1 | Media fissata a 0 ("nessun movimento"), scala comune per dx e dy | Normalizzazione comune a tutto il benchmark |
| Fine della sequenza | Decisa da quando la finestra arriva alla fine del testo | Lunghezza data in generazione, estratta dalle registrazioni reali della stessa classe e task | Non c'è un'uscita di fine registrazione |
