"""
train_common.py: parti del training comuni ai tre modelli (dati del fold, normalizzazione,
sampler, ciclo di training con controlli numerici, early stopping, checkpoint).
"""

from __future__ import annotations

import csv
import math
import os
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.getcwd())
try:
    import dataset_prep as dp
    from create_split import load_fold
except ImportError as error:
    raise SystemExit(
        f"Impossibile importare i moduli condivisi ({error}).\n"
        f"Servono dataset_prep.py e create_split.py in common/ o nella cartella del modello."
    )

import yaml
from torch.utils.data import Sampler

# Cartella del modello = cartella dello script lanciato (train_*.py, generate_synthetic.py).
# Non si usa __file__ perché questo file sta in common/, non nella cartella del modello.
_main_file = getattr(sys.modules.get("__main__"), "__file__", None)
MODEL_DIR = Path(_main_file).resolve().parent if _main_file else Path.cwd()
DIFFINK_DIR = MODEL_DIR


class ModelConfig:
    """Configurazione YAML letta come attributi."""

    def __init__(self, config_dict: dict):
        for key, value in config_dict.items():
            setattr(self, key, value)


def seed_everything(seed: int) -> None:
    """Fissa i seed di torch e random."""
    torch.manual_seed(seed)
    random.seed(seed)


def get_device() -> torch.device:
    """CUDA se disponibile, poi MPS, altrimenti CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_config(path: str | None = None) -> ModelConfig:
    """Legge configs/pahaw_config.yaml della cartella del modello (o il file indicato)."""
    path = Path(path) if path else MODEL_DIR / "configs" / "pahaw_config.yaml"
    with open(path) as f:
        return ModelConfig(yaml.safe_load(f))


def model_path(path: str) -> Path:
    """Percorso relativo -> percorso nella cartella del modello."""
    p = Path(path)
    return p if p.is_absolute() else MODEL_DIR / p


diffink_path = model_path


def run_dir(fold: int, out_root: str = "runs/diffink") -> Path:
    """Cartella dei risultati del fold (<out_root>/fold<k>), creata se manca."""
    folder = Path(out_root) / f"fold{fold}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def add_data_arguments(parser) -> None:
    """Opzioni per lavorare senza i dati grezzi (.svc)."""
    parser.add_argument("--data_file", type=str, default=None,
                        help="file .npz esportato con export_packed_data.py; se dato, i .svc non servono")
    parser.add_argument("--split_file", type=str, default=None,
                        help="split .json esportato insieme al file --data_file")
    parser.add_argument("--pretrained_dir", type=str, default=None,
                        help="cartella con i pesi ufficiali (default: il percorso nel file di configurazione)")


def pretrained_file(config_path: str, pretrained_dir: str | None) -> Path:
    """
    Trova il checkpoint ufficiale.

    Senza --pretrained_dir usa il percorso della configurazione. Con --pretrained_dir
    cerca nell'ordine: il file stesso, <dir>/<percorso della config>, <dir>/<nome file>,
    poi un file con quel nome nelle sottocartelle; se ce n'è più di uno si ferma.
    """
    if not pretrained_dir:
        path = model_path(config_path)
        if not path.exists():
            raise SystemExit(f"Checkpoint non trovato: {path}\n"
                             f"Correggere il percorso in configs/pahaw_config.yaml o usare --pretrained_dir.")
        return path

    root = Path(pretrained_dir)
    relative = Path(config_path.removeprefix("./"))
    if root.is_file():
        return root
    for candidate in (root / relative, root / relative.name):
        if candidate.is_file():
            return candidate

    found = sorted(root.rglob(relative.name)) if root.is_dir() else []
    if len(found) > 1:
        best = [p for p in found if p.parts[-len(relative.parts):] == relative.parts]
        if not best:
            best = [p for p in found if p.parent.name == relative.parent.name]
        found = best if len(best) == 1 else found
    if len(found) == 1:
        return found[0]

    present = sorted(str(p) for p in root.rglob("*") if p.suffix in (".pt", ".pth", ".bin", ".ckpt")) \
        if root.is_dir() else []
    problem = "più di un file con quel nome" if found else "nessun file con quel nome"
    raise SystemExit(
        f"Checkpoint '{relative}' non trovato in --pretrained_dir {root} ({problem}).\n"
        f"File di pesi presenti: {present or ('nessuno' if root.is_dir() else 'la cartella non esiste')}\n"
    )


def load_recordings(data_file: str | None) -> list[dict]:
    """Registrazioni dal file packed oppure dai .svc."""
    if data_file:
        try:
            from packed_data import load_packed, read_meta
        except ImportError:
            raise SystemExit("packed_data.py non trovato (deve stare in common/ o nella cartella del modello).")
        try:
            recordings = load_packed(data_file)
        except ValueError as error:
            raise SystemExit(f"\n{error}")
        made_with = read_meta(data_file)["gap_settings"]
        print(f"Dati: {len(recordings)} sequenze lette da {data_file}")
        print(f"      regole sui gap '{made_with['rules']}', TRIM_AIR_EDGES = {made_with['trim_air_edges']}")
        return recordings

    labels = dp.load_labels(dp.LABELS_CSV)
    recordings, _ = dp.load_recordings(dp.RAW_DATA_DIR, labels)
    print(f"Dati: {len(recordings)} sequenze ricostruite da {dp.RAW_DATA_DIR}")
    return recordings


def report_input_range(dataset, name: str) -> None:
    """Stampa il range degli input normalizzati per scoprire offset anomali (gap non corretti)."""
    x = torch.cat([item["x"] for item in dataset.items], dim=0)
    offsets = x[:, [dp.DX, dp.DY]].abs().flatten()
    q50, q99, q999 = torch.quantile(offsets[:: max(1, len(offsets) // 1_000_000)],
                                    torch.tensor([0.5, 0.99, 0.999])).tolist()
    biggest = x[:, :dp.N_CONTINUOUS].abs().max(dim=0).values.tolist()
    print(f"Range degli input normalizzati ({name}), in deviazioni standard:")
    print(f"   |dx|,|dy|: mediana {q50:.2f}   99% sotto {q99:.2f}   99.9% sotto {q999:.2f}   max {offsets.max():.1f}")
    print("   max per canale: " + ", ".join(f"{n} {v:.1f}" for n, v in zip(dp.CHANNEL_NAMES, biggest)))
    print(f"   offset oltre 20 deviazioni standard: {int((offsets > 20).sum())} in "
          f"{sum(bool((item['x'][:, :2].abs() > 20).any()) for item in dataset.items)} registrazioni su {len(dataset)}")
    if offsets.max() > 60:
        print("   ATTENZIONE: ci sono offset oltre 60 deviazioni standard. Con le regole sui gap attuali "
              "il massimo atteso è circa 50: controllare con check_jumps.py ed esportare di nuovo i dati.")


def clip_offsets(dataset, clip_sigma: float) -> int:
    """Limita dx, dy normalizzati a +-clip_sigma (opzione --clip_sigma, non usata di default)."""
    changed = 0
    for item in dataset.items:
        offsets = item["x"][:, :2]
        changed += int((offsets.abs() > clip_sigma).sum())
        offsets.clamp_(-clip_sigma, clip_sigma)
    return changed


def build_datasets(fold: int, out_dir: Path, max_seq_len: int | None,
                   data_file: str | None = None, split_file: str | None = None,
                   clip_sigma: float | None = None):
    """Dataset di train e val del fold, normalizzati con statistiche calcolate sul solo training."""
    if data_file and not split_file:
        raise SystemExit("--data_file va usato insieme a --split_file (gli ID pseudonimizzati valgono solo in coppia).")

    recordings = load_recordings(data_file)
    split = load_fold(fold, split_file) if split_file else load_fold(fold)

    known = {r["patient_id"] for r in recordings}
    missing = [p for part in ("train", "val", "test") for p in split[part] if p not in known]
    if missing:
        raise SystemExit(f"{len(missing)} pazienti dello split non sono nei dati (es. {missing[:3]}): file di dati e split non corrispondono.")

    stats = dp.fit_norm_stats(recordings, split["train"])
    stats.save(out_dir / "norm_stats.json")

    train_set = dp.PaHaWSequenceDataset(recordings, stats, patient_ids=split["train"], max_len=max_seq_len)
    val_set = dp.PaHaWSequenceDataset(recordings, stats, patient_ids=split["val"], max_len=max_seq_len)

    print(f"Fold {fold}: {len(train_set)} registrazioni di training ({len(split['train'])} pazienti), "
          f"{len(val_set)} di validazione ({len(split['val'])} pazienti)")
    dropped = train_set.n_dropped_too_long + val_set.n_dropped_too_long
    if dropped:
        print(f"   {dropped} registrazioni più lunghe di {max_seq_len} step escluse")

    report_input_range(train_set, "training")
    if clip_sigma is not None:
        changed = clip_offsets(train_set, clip_sigma) + clip_offsets(val_set, clip_sigma)
        print(f"   --clip_sigma {clip_sigma}: limitati {changed} offset")
    return train_set, val_set, stats


class AttentionBudgetSampler(Sampler):
    """
    Batch di lunghezza simile con costo dell'attention limitato.

    Il costo di un batch è B * N^2, con N = frame latenti della registrazione più
    lunga (T/8). Una registrazione che supera il budget da sola forma un batch a sé.
    """

    def __init__(self, lengths: list[int], max_batch_size: int, budget: float, shuffle: bool,
                 bucket_factor: int = 20, seed: int | None = None):
        self.frames = [math.ceil(n / 8) for n in lengths]
        self.max_batch_size, self.budget, self.shuffle = max_batch_size, budget, shuffle
        self.bucket_factor = bucket_factor
        self.rng = random.Random(seed)
        self._count = len(self._make_batches(shuffle=False))

    def _make_batches(self, shuffle: bool) -> list[list[int]]:
        indices = list(range(len(self.frames)))
        if shuffle:
            self.rng.shuffle(indices)
            bucket_size = self.max_batch_size * self.bucket_factor
        else:
            bucket_size = len(indices)

        batches = []
        for start in range(0, len(indices), max(bucket_size, 1)):
            bucket = sorted(indices[start:start + bucket_size], key=lambda i: self.frames[i])
            batch = []
            for i in bucket:
                cost_if_added = (len(batch) + 1) * self.frames[i] ** 2
                if batch and (len(batch) == self.max_batch_size or cost_if_added > self.budget):
                    batches.append(batch)
                    batch = []
                batch.append(i)
            if batch:
                batches.append(batch)
        if shuffle:
            self.rng.shuffle(batches)
        return batches

    def __iter__(self):
        yield from self._make_batches(self.shuffle)

    def __len__(self) -> int:
        return self._count

    def describe(self) -> str:
        batches = self._make_batches(shuffle=False)
        sizes = [len(b) for b in batches]
        worst = max(len(b) * max(self.frames[i] for i in b) ** 2 for b in batches)
        return (f"{len(batches)} batch, da {min(sizes)} a {max(sizes)} registrazioni; "
                f"costo massimo {worst / 1e6:.1f} M (limite {self.budget / 1e6:.1f} M)")


class StepBudgetSampler(Sampler):
    """
    Divide i batch di BucketBatchSampler in modo che B * T resti sotto il budget.

    Serve all'LSTM, la cui memoria nel backward cresce con batch x lunghezza.
    """

    def __init__(self, lengths: list[int], max_batch_size: int, budget: float, shuffle: bool,
                 seed: int | None = None):
        self.lengths, self.budget = lengths, budget
        self.buckets = dp.BucketBatchSampler(lengths, max_batch_size, shuffle=shuffle, seed=seed)
        self._count = sum(1 for _ in self._batches())

    def _split(self, batch: list[int]) -> list[list[int]]:
        batch = sorted(batch, key=lambda i: self.lengths[i])
        pieces, piece = [], []
        for i in batch:
            if piece and (len(piece) + 1) * self.lengths[i] > self.budget:
                pieces.append(piece)
                piece = []
            piece.append(i)
        return pieces + [piece]

    def _batches(self):
        for batch in self.buckets:
            yield from self._split(batch)

    def __iter__(self):
        yield from self._batches()

    def __len__(self) -> int:
        return self._count

    def describe(self) -> str:
        sizes = [len(b) for b in self._batches()]
        return (f"circa {len(sizes)} batch per epoca, da {min(sizes)} a {max(sizes)} registrazioni "
                f"(al massimo {self.budget:,.0f} step per batch)")


class TrainingFailed(SystemExit):
    """Training interrotto per valori non finiti."""


def count_non_finite(state_dict: dict) -> int:
    """Numero di tensori dello state_dict con NaN o infiniti."""
    return sum(1 for v in state_dict.values()
               if torch.is_tensor(v) and v.is_floating_point() and not torch.isfinite(v).all())


def check_checkpoint_is_healthy(state_dict: dict, path) -> None:
    """Si ferma se il checkpoint da caricare contiene NaN o infiniti."""
    bad = count_non_finite(state_dict)
    if bad:
        raise TrainingFailed(
            f"Il checkpoint {path} contiene {bad} tensori con NaN o infiniti: "
            f"il run precedente è fallito. Ripartire con --overwrite."
        )


def describe_batch(batch: dict) -> str:
    """
    Descrizione di un batch (pazienti, task, lunghezze, valori massimi) per i messaggi di errore.
    """
    x, mask = batch["x"], batch["mask"]
    biggest = x[..., :dp.N_CONTINUOUS].abs().amax(dim=(0, 1)).tolist()
    return (f"pazienti {sorted(set(batch['patient_id']))}, task {batch['task'].tolist()}, "
            f"lunghezze {batch['lengths'].tolist()}, lunghezza con padding {x.shape[1]}; "
            f"valori assoluti massimi: " + ", ".join(f"{n} {v:.1f}" for n, v in zip(dp.CHANNEL_NAMES, biggest)))


def check_not_overwriting(last_checkpoint: Path, resume: bool, overwrite: bool) -> None:
    """Non riparte sopra un run esistente se non è indicato --resume o --overwrite."""
    if last_checkpoint.exists() and not resume and not overwrite:
        raise SystemExit(
            f"Esiste già un checkpoint: {last_checkpoint}\n"
            f"   per continuare quel run          : --resume\n"
            f"   per cancellarlo e ricominciare   : --overwrite"
        )


def to_device(batch: dict, device: torch.device) -> dict:
    """Sposta sul device i tensori del batch."""
    return {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in batch.items()}


def train_one_epoch(loss_fn, loader, optimizer, parameters, device, state: dict,
                    accum: int = 1, clip: float = 1.0, lr_warmup_steps: int = 100,
                    limit_batches: int | None = None, explain=None) -> float:
    """
    Un'epoca di training.

    Accumulo dei gradienti, clipping della norma e warm-up lineare del learning rate.
    I batch con loss non finita o gradiente non finito vengono saltati; se sono più
    di un quarto il training si ferma.
    """
    total, count, pending = 0.0, 0, 0
    bad_loss, bad_gradient, seen = 0, 0, 0
    optimizer.zero_grad()

    def update_weights():
        nonlocal bad_gradient
        gradient_size = torch.nn.utils.clip_grad_norm_(parameters, clip)
        if not torch.isfinite(gradient_size):
            bad_gradient += 1
            optimizer.zero_grad()
            return

        factor = min(1.0, (state["step"] + 1) / max(1, lr_warmup_steps))
        for group in optimizer.param_groups:
            group["lr"] = group["base_lr"] * factor

        optimizer.step()
        optimizer.zero_grad()
        state["step"] += 1

    for i, batch in enumerate(loader):
        if limit_batches is not None and i >= limit_batches:
            break
        seen += 1
        batch = to_device(batch, device)
        loss = loss_fn(batch)

        if not torch.isfinite(loss):
            if bad_loss == 0:
                print(f"   prima loss non finita, al batch {i}: {describe_batch(batch)}")
                if explain is not None:
                    print(f"      {explain(batch)}")
            bad_loss += 1
            continue

        (loss / accum).backward()
        total += loss.item()
        count += 1
        pending += 1

        if pending == accum:
            update_weights()
            pending = 0

    if pending:
        update_weights()

    if bad_loss or bad_gradient:
        print(f"   saltati {bad_loss} batch per loss non finita e "
              f"{bad_gradient} aggiornamenti per gradiente non finito, su {seen} batch")
    if seen and (bad_loss + bad_gradient * accum) > 0.25 * seen:
        raise TrainingFailed(
            f"Troppi batch non utilizzabili ({100*(bad_loss + bad_gradient * accum)/seen:.1f}%) "
            f"in questa epoca: training interrotto. I checkpoint già salvati non sono stati toccati."
        )
    return total / max(count, 1)


@torch.no_grad()
def evaluate(loss_fn, loader, device, repeats: int = 1, seed: int = 0,
             limit_batches: int | None = None) -> float:
    """
    Loss media di validazione, con seed fisso.

    La loss dipende da rumore casuale: col seed fisso le epoche sono confrontabili.
    Con repeats > 1 si fa la media su più passate.
    """
    seed_everything(seed)
    total, count = 0.0, 0
    for _ in range(repeats):
        for i, batch in enumerate(loader):
            if limit_batches is not None and i >= limit_batches:
                break
            loss = loss_fn(to_device(batch, device))
            if not torch.isfinite(loss):
                return float("nan")
            total += loss.item()
            count += 1
    return total / count if count else float("nan")


def require_finite(value: float, what: str) -> None:
    """Si ferma se value non è finito."""
    if not math.isfinite(value):
        raise TrainingFailed(
            f"Valore non finito per {what}: {value}. "
            f"Training interrotto; i checkpoint delle epoche precedenti non sono stati toccati."
        )


def make_optimizer(groups: list[dict]) -> torch.optim.Optimizer:
    """
    AdamW sui gruppi di parametri; salva il learning rate di partenza in 'base_lr' (per il warm-up).
    """
    for group in groups:
        group["base_lr"] = group["lr"]
    return torch.optim.AdamW(groups, betas=(0.9, 0.99))


def save_checkpoint(content: dict, path: Path) -> None:
    """Salva prima su un file .tmp e poi lo rinomina, così un'interruzione non lascia file rotti."""
    temporary = Path(str(path) + ".tmp")
    torch.save(content, temporary)
    os.replace(temporary, path)


class EarlyStopper:
    """Early stopping sulla loss di validazione."""

    def __init__(self, patience: int):
        self.patience = patience
        self.best = float("inf")
        self.best_epoch = -1
        self.bad_epochs = 0

    def update(self, value: float, epoch: int) -> bool:
        """
        Aggiorna con la loss di validazione dell'epoca; restituisce True se è la migliore finora.
        """
        if math.isfinite(value) and value < self.best:
            self.best, self.best_epoch, self.bad_epochs = value, epoch, 0
            return True
        self.bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience

    def state_dict(self) -> dict:
        return {"best": self.best, "best_epoch": self.best_epoch, "bad_epochs": self.bad_epochs}

    def load_state_dict(self, state: dict) -> None:
        self.best, self.best_epoch, self.bad_epochs = state["best"], state["best_epoch"], state["bad_epochs"]


class CsvLog:
    """Log CSV con una riga per epoca."""

    def __init__(self, path: Path, columns: list[str], keep_existing: bool):
        self.path, self.columns = path, columns
        if not (keep_existing and path.exists()):
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(columns)

    def write(self, **values) -> None:
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([values.get(c, "") for c in self.columns])


class Timer:
    """Tempo trascorso dall'ultima chiamata."""
    
    def __init__(self):
        self.start = time.time()

    def lap(self) -> float:
        now = time.time()
        elapsed, self.start = now - self.start, now
        return elapsed