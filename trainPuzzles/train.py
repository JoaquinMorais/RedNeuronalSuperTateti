"""Entrenador de gradiente por puzzles tácticos ("mate en N") + CLI.

Flujo:
  1. Genera un set de EVALUACIÓN fijo por nivel (semilla fija) y puzzles de
     ENTRENAMIENTO por nivel (excluyendo el de evaluación).
  2. Mide el BASELINE (red inicial) por nivel antes de tocar los pesos.
  3. Entrena `SuperTatetiNet` con cross-entropy de política sobre las jugadas
     óptimas (+ valor, lado que mueve tiene victoria forzada = +1).
  4. Un nivel se considera DOMINADO cuando el acierto de evaluación >= umbral
     durante `patience` evaluaciones seguidas, habiendo entrenado al menos
     `min-epochs` y sin olvidar los niveles previos (retención). Recién entonces
     sube al siguiente nivel; si no domina, sigue entrenando (sin tope por
     defecto).
  5. Al dominar los niveles pedidos, sigue subiendo de nivel automáticamente
     (generación on-demand) hasta `--max-level` (0 = sin tope). Entre niveles corre
     una fase de REVISIÓN sobre niveles viejos (anti-olvido). Si un nivel nuevo no
     se puede generar, entra en modo revisión permanente hasta que lo pares.
  6. Se puede detener con Ctrl-C o `docker stop` (SIGINT/SIGTERM): finaliza el
     epoch actual, guarda el mejor estado y escribe el reporte + resumen.
  7. Al terminar imprime un resumen que clasifica el aprendizaje por nivel
     (mejoró mucho/poco, sin aprendizaje claro, empeoró poco/mucho) y guarda un
     reporte JSON junto al modelo (también de forma incremental).

Los checkpoints se ordenan así (ver `layout.py`): cada corrida archiva sus artefactos
en `<dir>/runs/NNNN_<tag>/` (modelo final, reporte y `levels/levelN_best.pth`) y el
modelo final se copia siempre a `<dir>/puzzle_canon.pth`, que es el punto de partida
por defecto (`--init`) para reanudar el próximo entrenamiento. `<dir>` depende de
`--arch`: `trainPuzzles/checkpoints_mts` (mlp) o `trainPuzzles/checkpoint_cnn` (cnn).

El modelo se guarda como `state_dict` plano, compatible con `play.py` y
`evolution.modelstore`. No importa ni modifica `mcts.py`, `trainer.py` ni `evolution/`.
"""
from __future__ import annotations

import argparse
import json
import random
import signal
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import SuperTatetiNet, hidden_from_state_dict
from architecture import MLP_HIDDEN

from . import canon, layout
from .model_cnn import SuperTatetiCNN
from .puzzles import Puzzle, harvest


def build_net(arch: str, dropout: float = 0.0, mlp_hidden=None):
    """Factory de arquitecturas: 'mlp' (SuperTatetiNet) o 'cnn' (SuperTatetiCNN).

    Ambas comparten la firma ``forward(x, valid_moves_mask) -> (policy(81), value(1))``
    y un ``state_dict`` estándar. OJO: los checkpoints de una NO son compatibles con la
    otra (formas distintas), así que ``--init`` debe ser de la misma arquitectura.

    ``mlp_hidden`` fija las dimensiones ocultas del MLP (default: ``architecture.MLP_HIDDEN``).
    Se usa para reconstruir redes con un tamaño distinto del default (p. ej. checkpoints
    viejos) al cargarlos, sin editar ``architecture.py``.
    """
    if arch == "cnn":
        return SuperTatetiCNN(dropout=dropout)
    if arch == "mlp":
        hidden = MLP_HIDDEN if mlp_hidden is None else tuple(mlp_hidden)
        return SuperTatetiNet(hidden=hidden, dropout=dropout)
    raise ValueError(f"arquitectura desconocida: {arch!r} (usar 'mlp' o 'cnn')")


# ----------------------------------------------------------------------
# Parada prolija (Ctrl-C / docker stop)
# ----------------------------------------------------------------------
_STOP = threading.Event()


def request_stop(*_args) -> None:
    """Setea el flag de parada (idempotente)."""
    if not _STOP.is_set():
        _STOP.set()
        log_print("Parada solicitada: se finalizará al terminar el epoch actual.")


def _install_signal_handlers() -> None:
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, request_stop)
        except (ValueError, OSError):
            pass  # fuera del hilo principal (p. ej. en tests)


# ----------------------------------------------------------------------
# Clasificación del aprendizaje
# ----------------------------------------------------------------------
IMPROVED_A_LOT = "mejoró mucho"
IMPROVED_LITTLE = "mejoró poco"
NO_CLEAR = "no muestra un aprendizaje claro"
WORSENED_LITTLE = "empeoró poco"
WORSENED_A_LOT = "empeoró mucho"


def classify_delta(delta: float, big: float = 0.20, small: float = 0.05) -> str:
    """Clasifica un cambio de acierto (``final - base``) en una de cinco categorías.

    ``delta >= big`` -> mejoró mucho; ``[small, big)`` -> mejoró poco;
    ``(-small, small)`` -> sin aprendizaje claro; ``(-big, -small]`` -> empeoró poco;
    ``<= -big`` -> empeoró mucho. ``NaN`` -> sin datos.
    """
    if delta != delta:  # NaN
        return "sin datos"
    if delta >= big:
        return IMPROVED_A_LOT
    if delta >= small:
        return IMPROVED_LITTLE
    if delta > -small:
        return NO_CLEAR
    if delta > -big:
        return WORSENED_LITTLE
    return WORSENED_A_LOT


def mastery_step(acc: float, ret: float, threshold: float, ret_threshold: float,
                 epoch: int, min_epochs: int, streak: int) -> int:
    """Actualiza la racha de evaluaciones que cumplen el dominio del nivel.

    Suma 1 si el acierto del nivel y el de retención superan sus umbrales y ya se
    cumplieron los ``min_epochs``; si no, reinicia la racha a 0.
    """
    if acc >= threshold and ret >= ret_threshold and epoch >= min_epochs:
        return streak + 1
    return 0


# ----------------------------------------------------------------------
# Utilidades de datos
# ----------------------------------------------------------------------
def policy_target(best_moves, n: int = 81) -> np.ndarray:
    """Distribución objetivo: uniforme sobre las jugadas óptimas."""
    t = np.zeros(n, dtype=np.float32)
    if best_moves:
        t[list(best_moves)] = 1.0 / len(best_moves)
    return t


def mix_pool(current: list[Puzzle], replay: list[Puzzle], replay_frac: float,
             rng: random.Random) -> list[Puzzle]:
    """Pool de entrenamiento = puzzles del nivel actual + una fracción de los previos."""
    if not replay or replay_frac <= 0:
        return list(current)
    k = int(len(current) * replay_frac)
    k = min(k, len(replay))
    return list(current) + rng.sample(replay, k)


def clone_state(net: SuperTatetiNet) -> dict:
    return {k: v.detach().clone() for k, v in net.state_dict().items()}


# ----------------------------------------------------------------------
# Entrenador
# ----------------------------------------------------------------------
class PuzzleTrainer:
    """Entrena con entrada CANÓNICA (perspectiva de quien mueve) y, por defecto, con una simetría D4
    al azar por muestra en cada epoch (``augment``). Ver ``canon.py``.

    La perspectiva canónica la aplica ``SuperTatetiNet.forward`` (``model.py``), así que acá se le
    pasa el estado CRUDO (``game.get_state()``); ``augment`` (D4) sí se aplica sobre el lote.
    """

    def __init__(self, net: SuperTatetiNet, lr: float = 3e-4, value_coef: float = 0.0,
                 seed: int = 0, augment: bool = True, weight_decay: float = 0.0,
                 lr_schedule: str = "none", lr_final_frac: float = 0.03, lr_decay_epochs: int = 0,
                 max_epochs: int = 0, ema_decay: float = 0.0, rank_coef: float = 0.0,
                 rank_margin: float = 1.0):
        self.net = net
        self.value_coef = value_coef
        self.augment = augment
        self.rank_coef = float(rank_coef)
        self.rank_margin = float(rank_margin)
        self.ema_decay = float(ema_decay)
        self.optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.scheduler = None
        self.ema_state = None
        if ema_decay > 0.0:
            self.ema_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        if lr_schedule == "cosine":
            t_max = lr_decay_epochs
            if t_max <= 0:
                t_max = max_epochs if max_epochs > 0 else 3000
            eta_min = lr * float(lr_final_frac)
            try:
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=t_max, eta_min=eta_min)
            except Exception:
                self.scheduler = None

    def train_epoch(self, puzzles: list[Puzzle], batch_size: int,
                     weights: np.ndarray | None = None) -> tuple[float, float, float]:
        if not puzzles:
            return (0.0, 0.0, 0.0)
        self.net.train()
        n = len(puzzles)
        if weights is not None:
            w = np.asarray(weights, dtype=np.float64)
            if w.shape[0] != n:
                w = np.ones(n, dtype=np.float64)
            p = w / (w.sum() + 1e-12)
            order_base = np.random.choice(n, size=n, replace=True, p=p)
        else:
            order_base = list(range(n))
        order = list(order_base)
        self.rng.shuffle(order)
        losses = []
        for i in range(0, len(order), batch_size):
            idxs = order[i:i + batch_size]
            states_np = np.stack([puzzles[j].state for j in idxs])
            masks_np = np.stack([puzzles[j].mask for j in idxs])
            targets_np = np.stack([policy_target(puzzles[j].best_moves) for j in idxs])
            if self.augment:
                states_np, masks_np, targets_np = canon.augment(states_np, masks_np, targets_np, self.np_rng)
            states = torch.from_numpy(states_np).to(_DEVICE)
            masks = torch.from_numpy(masks_np).to(_DEVICE)
            targets = torch.from_numpy(targets_np).to(_DEVICE)

            pred_policy, pred_value = self.net(states, masks)
            logp = torch.log(pred_policy + 1e-8)
            policy_loss = -(targets * logp).sum(dim=1).mean()
            value_loss = nn.functional.mse_loss(pred_value, torch.ones_like(pred_value))
            loss = policy_loss + self.value_coef * value_loss
            if self.rank_coef > 0.0:
                opt = targets > 0.0
                masked_logp = logp.masked_fill(masks == 0, -1e9)
                best = masked_logp.masked_fill(~opt, -1e9).max(dim=1).values
                decoy = masked_logp.masked_fill(opt, -1e9).max(dim=1).values
                rank = nn.functional.softplus(self.rank_margin + decoy - best).mean()
                loss = loss + self.rank_coef * rank

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.ema_state is not None and self.ema_decay > 0.0:
                with torch.no_grad():
                    for k, p in self.net.state_dict().items():
                        if k in self.ema_state:
                            self.ema_state[k] = self.ema_decay * self.ema_state[k] + (1.0 - self.ema_decay) * p.detach()
            losses.append((loss.item(), policy_loss.item(), value_loss.item()))
        arr = np.asarray(losses, dtype=np.float64)
        return float(arr[:, 0].mean()), float(arr[:, 1].mean()), float(arr[:, 2].mean())

    def step_scheduler(self) -> None:
        if self.scheduler is not None:
            try:
                self.scheduler.step()
            except Exception:
                pass


@torch.no_grad()
def accuracy(net: SuperTatetiNet, puzzles: list[Puzzle], tta: bool = False) -> float:
    """% de puzzles donde la jugada elegida (argmax legal) está entre las óptimas."""
    m = topk_accuracy(net, puzzles, ks=(1,), tta=tta)
    return float(m.get(1, 0.0))


@torch.no_grad()
def topk_accuracy(net: SuperTatetiNet, puzzles: list[Puzzle], ks=(1, 2, 3), tta: bool = False) -> dict:
    """Top-k accuracy. ks=(1,2,3) por defecto."""
    if not puzzles:
        out = {}
        for k in ks:
            out[k] = 0.0
        return out
    net.eval()
    best_list = []
    for p in puzzles:
        best_list.append(set(p.best_moves))
    states_np_full = np.stack([p.state for p in puzzles])
    masks_np_full = np.stack([p.mask for p in puzzles])
    kmax = max(ks) if ks else 1
    res_correct = {k: 0 for k in ks}
    for i in range(0, len(puzzles), 1024):
        chunk_states = states_np_full[i:i + 1024]
        chunk_masks = masks_np_full[i:i + 1024]
        chunk_best = best_list[i:i + 1024]
        B = len(chunk_states)
        if tta and B > 0:
            p_accum = np.zeros((B, 81), dtype=np.float32)
            zeros_t = np.zeros((B, 81), dtype=np.float32)
            for ksym in range(len(canon.SYM_MAPS)):
                s_k, m_k, _ = canon.apply_sym(chunk_states, chunk_masks, zeros_t, ksym)
                pol_k, _ = net(to_device(torch.from_numpy(s_k)), to_device(torch.from_numpy(m_k)))
                pol_k_np = pol_k.detach().cpu().numpy().astype(np.float32)
                p_back = np.zeros_like(pol_k_np)
                inv = canon.IDX81[ksym]
                p_back[:, inv] = pol_k_np
                p_accum += p_back
            policy_np = p_accum / float(len(canon.SYM_MAPS))
        else:
            pol, _ = net(to_device(torch.from_numpy(chunk_states)), to_device(torch.from_numpy(chunk_masks)))
            policy_np = pol.detach().cpu().numpy().astype(np.float32)
        for j in range(B):
            msk = chunk_masks[j]
            p_j = policy_np[j].copy()
            p_j[msk == 0] = -1e18
            idx = np.argsort(-p_j)[:kmax]
            for k in ks:
                topkset = set(idx[:k].tolist())
                if chunk_best[j].intersection(topkset):
                    res_correct[k] += 1
    out = {}
    n = len(puzzles)
    for k in ks:
        out[k] = res_correct[k] / n if n > 0 else 0.0
    return out


@torch.no_grad()
def per_sample_correct(net: SuperTatetiNet, puzzles: list[Puzzle], tta: bool = False) -> np.ndarray:
    if not puzzles:
        return np.array([], dtype=bool)
    m1 = topk_accuracy(net, puzzles, ks=(1,), tta=tta)
    acc1 = m1.get(1, 0.0)
    n = len(puzzles)
    # compute per-sample efficiently: small loop
    correct_arr = np.zeros(n, dtype=bool)
    states_np_full = np.stack([p.state for p in puzzles])
    masks_np_full = np.stack([p.mask for p in puzzles])
    for i in range(0, n, 1024):
        chunk_states = states_np_full[i:i + 1024]
        chunk_masks = masks_np_full[i:i + 1024]
        B = len(chunk_states)
        if tta and B > 0:
            p_accum = np.zeros((B, 81), dtype=np.float32)
            zeros_t = np.zeros((B, 81), dtype=np.float32)
            for ksym in range(len(canon.SYM_MAPS)):
                s_k, m_k, _ = canon.apply_sym(chunk_states, chunk_masks, zeros_t, ksym)
                pol_k, _ = net(to_device(torch.from_numpy(s_k)), to_device(torch.from_numpy(m_k)))
                pol_k_np = pol_k.detach().cpu().numpy().astype(np.float32)
                p_back = np.zeros_like(pol_k_np)
                inv = canon.IDX81[ksym]
                p_back[:, inv] = pol_k_np
                p_accum += p_back
            policy_np = p_accum / float(len(canon.SYM_MAPS))
        else:
            pol, _ = net(to_device(torch.from_numpy(chunk_states)), to_device(torch.from_numpy(chunk_masks)))
            policy_np = pol.detach().cpu().numpy().astype(np.float32)
        for j in range(B):
            msk = chunk_masks[j]
            p_j = policy_np[j].copy()
            p_j[msk == 0] = -1e18
            mv = int(np.argmax(p_j))
            if mv in puzzles[i + j].best_moves:
                correct_arr[i + j] = True
            else:
                correct_arr[i + j] = False
    return correct_arr


@torch.no_grad()
def compute_weights(net: SuperTatetiNet, pool: list[Puzzle], boost: float,
                    warmup_epoch: int, current_epoch: int, mine_every: int) -> np.ndarray:
    if not pool:
        return np.array([], dtype=np.float64)
    if mine_every <= 0 or current_epoch < warmup_epoch or current_epoch % mine_every != 0:
        return np.ones(len(pool), dtype=np.float64)
    c = per_sample_correct(net, pool, tta=False)
    w = np.ones(len(pool), dtype=np.float64)
    w[~c] = float(boost)
    return np.maximum(w, 1.0)


# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
def _ts() -> str:
    return time.strftime("%H:%M:%S")


def log_print(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


def log_phase(tag: str, msg: str) -> None:
    """Log con etiqueta de fase (`[GEN]`, `[TRAIN]`, `[REV]`, `[INFO]`)."""
    log_print(f"{tag} {msg}")


def _bar(frac: float, width: int = 10) -> str:
    """Barra ASCII simple ``[####------]`` para una fracción 0..1."""
    frac = min(max(frac, 0.0), 1.0)
    filled = int(round(frac * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _fmt_dur(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"


# ----------------------------------------------------------------------
# Generación de puzzles
# ----------------------------------------------------------------------
def _eval_seed(args) -> int:
    """Semilla del set de EVALUACIÓN (``--eval-seed``), independiente de la de train.

    Sirve para que el examen no cambie entre reintentos con datos nuevos (``--seed``
    distinto) y las comparaciones sean justas.
    """
    return args.seed if getattr(args, "eval_seed", None) is None else args.eval_seed


def generate_eval_set(n: int, args, exclude: set) -> list[Puzzle]:
    """Genera el set de evaluación fijo del nivel ``n`` y agrega sus claves a ``exclude``."""
    log_phase("[GEN]", f"evaluación mate en {n}...")
    got = harvest({n: args.eval_size}, seed=_eval_seed(args) + 1000 + n,
                  min_turn=args.min_turn, solver_max_nodes=args.solver_max_nodes,
                  games_budget=args.games_budget, max_games=args.max_games,
                  should_stop=_STOP.is_set)
    ps = got.get(n, [])
    for p in ps:
        exclude.add(p.key)
    log_phase("[GEN]", f"  evaluación mate en {n}: {len(ps)}/{args.eval_size}")
    if len(ps) < args.eval_size:
        log_phase("[GEN]", f"  aviso: evaluación {n} incompleta ({len(ps)}/{args.eval_size}); "
                           f"¿subir --max-games?")
    return ps


def generate_train_set(n: int, args, exclude: set) -> list[Puzzle]:
    """Genera los puzzles de entrenamiento del nivel ``n``."""
    target = args.puzzles_per_level
    start = time.time()

    def prog(counts, games, n=n, target=target, start=start):
        got_n = counts[n]
        elapsed = max(time.time() - start, 1e-6)
        rate = got_n / elapsed
        pct = got_n / target if target else 1.0
        eta = (target - got_n) / rate if rate > 0 else 0.0
        log_phase("[GEN]", f"  mate {n}: {got_n}/{target} ({pct * 100:4.1f}%) {_bar(pct)} "
                  f"{rate:.1f} puz/s | partidas {games} | ETA {_fmt_dur(eta)}")

    log_phase("[GEN]", f"entrenamiento mate en {n}...")
    got = harvest({n: target}, seed=args.seed + 2000 + n, exclude=exclude,
                  min_turn=args.min_turn, solver_max_nodes=args.solver_max_nodes,
                  games_budget=args.games_budget, max_games=args.max_games,
                  log=prog, log_interval=2.0, should_stop=_STOP.is_set)
    ps = got.get(n, [])
    log_phase("[GEN]", f"  entrenamiento mate en {n}: {len(ps)}/{target}")
    if len(ps) < target:
        log_phase("[GEN]", f"  aviso: no se alcanzó el target de mate {n} ({len(ps)}/{target}); "
                           f"¿subir --max-games?")
    return ps


def load_net(init_path: Path, dropout: float = 0.0, arch: str = "mlp") -> SuperTatetiNet:
    if init_path.exists():
        state = torch.load(init_path, weights_only=True, map_location="cpu")
        mlp_hidden = None
        if arch == "mlp":
            try:
                mlp_hidden = hidden_from_state_dict(state)
            except ValueError:
                mlp_hidden = None
        net = build_net(arch, dropout=dropout, mlp_hidden=mlp_hidden)
        try:
            net.load_state_dict(state)
            log_print(f"Red inicializada desde {init_path}")
        except RuntimeError as exc:
            log_print(f"!! No se pudo inicializar {arch} desde {init_path} "
                      f"(arquitectura incompatible): se entrena desde una red aleatoria")
            log_print(f"   detalle: {exc}")
            net = build_net(arch, dropout=dropout)
    else:
        net = build_net(arch, dropout=dropout)
        log_print(f"No existe {init_path}: se entrena desde una red aleatoria")
    return net


# ----------------------------------------------------------------------
# Dispositivo (CPU/GPU)
# ----------------------------------------------------------------------
# Device global del proceso: hay una sola red y un solo hilo de entrenamiento,
# así que centralizarlo evita enhebrar `device` por cada firma. Se resuelve en
# `run()` a partir de `--device`. En Linux se usa CPU (auto sin CUDA -> cpu);
# en Windows con Docker+GPU se pasa `--device cuda`.
_DEVICE = torch.device("cpu")


def resolve_device(name: str = "auto") -> torch.device:
    """Resuelve 'auto' | 'cpu' | 'cuda' a un ``torch.device``.

    ``auto`` usa CUDA si está disponible (imagen GPU) y CPU en caso contrario.
    """
    name = (name or "auto").lower()
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit(
                "--device cuda pedido pero CUDA no está disponible. "
                "Construí la imagen GPU (docker-compose.gpu.yml) y pasá la GPU al contenedor."
            )
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_device(x):
    """Mueve un tensor al device de entrenamiento (no-op si ya está en él)."""
    return x.to(_DEVICE)


def scheduled_levels(args) -> tuple[int, int]:
    """Normaliza (start, levels) desde --start-level / --levels.

    ``start`` es el nivel donde arranca el currículo y ``levels`` el mayor nivel
    pre-generado; se garantiza ``levels >= start`` para no dejar afuera el nivel
    inicial cuando arranca por encima de 1.
    """
    start = max(1, getattr(args, "start_level", 1))
    levels = max(args.levels, start)
    return start, levels


def build_sets(args) -> tuple[dict[int, list[Puzzle]], dict[int, list[Puzzle]]]:
    """Genera los sets iniciales (evaluación y entrenamiento).

    La evaluación se genera para 1..levels (barata y necesaria para medir retención
    de niveles previos); el entrenamiento solo desde start..levels (los niveles
    saltados son costosos de generar y no se entrenan en esta corrida).
    """
    start, levels = scheduled_levels(args)
    exclude: set = set()
    eval_sets: dict[int, list[Puzzle]] = {}
    train_sets: dict[int, list[Puzzle]] = {}
    log_phase("[GEN]", f"pre-generando datasets de los niveles {start}..{levels} "
                       f"(eval 1..{levels}) (esto ocurre antes de entrenar)")
    for n in range(1, levels + 1):
        eval_sets[n] = generate_eval_set(n, args, exclude)
    for n in range(start, levels + 1):
        train_sets[n] = generate_train_set(n, args, exclude)
    return eval_sets, train_sets


def save_net(net: SuperTatetiNet, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in net.state_dict().items()}, path)
    log_print(f"Modelo guardado en {path}")


# ----------------------------------------------------------------------
# Entrenamiento de un nivel y fase de revisión
# ----------------------------------------------------------------------
def train_level(net: SuperTatetiNet, trainer: PuzzleTrainer, args, level: int,
                pool: list[Puzzle], evalset: list[Puzzle], retention_eval: list[Puzzle],
                base: float, ret_threshold: float) -> dict:
    """Entrena un nivel hasta dominarlo, agotar el tope o recibir parada."""
    best_acc = -1.0
    best_acc2 = 0.0
    best_acc3 = 0.0
    best_state = None
    best_epoch = 0
    epoch = 0
    probe = random.Random(level).sample(pool, min(1000, len(pool)))   # muestra fija de entrenamiento: mide la memorización
    streak = 0
    plateau = 0
    mastered = False
    stopped = False
    capped = False
    plateau_stopped = False

    while True:
        if _STOP.is_set():
            stopped = True
            log_phase("[TRAIN]", f"  parada solicitada: nivel {level} interrumpido tras {epoch} epochs.")
            break
        if args.max_epochs and epoch >= args.max_epochs:
            log_phase("[TRAIN]", f"  aviso: tope de {args.max_epochs} epochs en el nivel {level} "
                       f"sin dominar (mejor {best_acc * 100:.1f}%).")
            capped = True
            break
        epoch += 1
        t0 = time.time()
        w_pool = compute_weights(net, pool, args.mine_boost,
                                 args.mine_warmup, epoch, args.mine_every) if getattr(args, 'hard_mining', False) else None
        loss, ploss, vloss = trainer.train_epoch(pool, args.batch_size, weights=w_pool)
        trainer.step_scheduler()
        do_eval = (args.eval_every <= 0) or (epoch % args.eval_every == 0)
        cand_is_ema = False
        if do_eval:
            m_eval = topk_accuracy(net, evalset, ks=(1,2,3), tta=getattr(args, 'tta', False))
            m_ret = topk_accuracy(net, retention_eval, ks=(1,), tta=getattr(args, 'tta', False))
            m_tr = topk_accuracy(net, probe, ks=(1,), tta=getattr(args, 'tta', False))
            acc1, acc2, acc3 = m_eval.get(1,0.0), m_eval.get(2,0.0), m_eval.get(3,0.0)
            ret1 = m_ret.get(1,0.0)
            tr1 = m_tr.get(1,0.0)
            if getattr(args, 'ema', False) and getattr(trainer, 'ema_state', None) is not None:
                state_bak = {k: v.detach().clone() for k, v in net.state_dict().items()}
                ema_sd = trainer.ema_state
                net.load_state_dict({k: ema_sd[k].detach().clone().to(v.device, v.dtype) for k,v in state_bak.items()})
                m_eval_e = topk_accuracy(net, evalset, ks=(1,2,3), tta=getattr(args, 'tta', False))
                m_ret_e = topk_accuracy(net, retention_eval, ks=(1,), tta=getattr(args, 'tta', False))
                acc1_e = m_eval_e.get(1,0.0)
                net.load_state_dict(state_bak)
                if acc1_e > acc1:
                    acc1_cand, acc2_cand, acc3_cand, ret1_cand = acc1_e, m_eval_e.get(2,0.0), m_eval_e.get(3,0.0), m_ret_e.get(1,0.0)
                    cand_is_ema = True
                else:
                    acc1_cand, acc2_cand, acc3_cand, ret1_cand = acc1, acc2, acc3, ret1
                    cand_is_ema = False
            else:
                acc1_cand, acc2_cand, acc3_cand, ret1_cand, cand_is_ema = acc1, acc2, acc3, ret1, False
        else:
            acc1_cand = best_acc if best_acc >= 0 else accuracy(net, evalset, tta=getattr(args, 'tta', False))
            acc2_cand = acc3_cand = 0.0; ret1_cand = 0.0; tr1 = 0.0; cand_is_ema = False
        dt = time.time() - t0

        # El seguimiento del mejor, la racha de dominio y el estancamiento solo
        # se actualizan en epochs con evaluación completa: en los demás, la retención
        # queda en 0.0 y reiniciaría la racha (evita que con --eval-every > 1 un nivel
        # nunca se marque como dominado).
        improved = False
        if do_eval:
            improved = acc1_cand > best_acc
            if improved:
                best_acc = acc1_cand
                best_acc2 = acc2_cand
                best_acc3 = acc3_cand
                if cand_is_ema and getattr(trainer, 'ema_state', None) is not None:
                    best_state = {k: v.detach().clone() for k, v in trainer.ema_state.items()}
                else:
                    best_state = clone_state(net)
                best_epoch = epoch
                plateau = 0
            else:
                plateau += 1

            streak = mastery_step(acc1_cand, ret1_cand, args.threshold, ret_threshold,
                                  epoch, args.min_epochs, streak)
            mastered = streak >= args.patience

            if plateau and args.plateau_patience and plateau % args.plateau_patience == 0:
                log_phase("[TRAIN]", f"  aviso: nivel {level} sin mejorar el mejor acierto desde hace "
                           f"{plateau} epochs (mejor {best_acc * 100:.1f}%).")

        every = args.log_every
        if every <= 0 or epoch % every == 0 or improved or mastered:
            status = "MASTERED" if mastered else f"meta {args.threshold * 100:.0f}%"
            log_phase("[TRAIN]",
                      f"  N{level} ep {epoch:3d} | loss {loss:.3f} (P {ploss:.3f} V {vloss:.3f}) | "
                      f"T1 {acc1_cand*100:5.1f}% T2 {acc2_cand*100:5.1f}% T3 {acc3_cand*100:5.1f}% base {base*100:5.1f}% d {(acc1_cand-base)*100:+5.1f} | "
                      f"train {tr1*100:5.1f}% brecha {(tr1-acc1_cand)*100:+5.1f} | "
                      f"mejor {best_acc*100:5.1f}% | ret {ret1_cand*100:5.1f}% | "
                      f"{_bar(acc1_cand/args.threshold if args.threshold else 1.0)} {dt:.1f}s | {status}")

        if mastered:
            log_phase("[TRAIN]", f"  nivel {level} DOMINADO: acierto T1 {acc1_cand * 100:.1f}% "
                       f"(>= {args.threshold * 100:.0f}% durante {args.patience} evals, "
                       f"{epoch} epochs).")
            break

        if args.plateau_stop and plateau >= args.plateau_stop:
            log_phase("[TRAIN]", f"  nivel {level}: {plateau} epochs sin mejorar el mejor acierto "
                       f"({best_acc * 100:.1f}% en el epoch {best_epoch}); se corta por "
                       f"--plateau-stop {args.plateau_stop}.")
            plateau_stopped = True
            break

    return {"epochs": epoch, "best_acc": best_acc, "best_acc2": best_acc2,
            "best_acc3": best_acc3, "best_state": best_state,
            "best_epoch": best_epoch, "mastered": mastered, "stopped": stopped,
            "capped": capped, "plateau_stopped": plateau_stopped}


def revision_phase(net: SuperTatetiNet, trainer: PuzzleTrainer, args,
                   replay_by_level: dict[int, list[Puzzle]], learned: list[int],
                   eval_sets: dict[int, list[Puzzle]], rng: random.Random) -> dict[int, float]:
    """Entrena unas epochs sobre una mezcla de niveles viejos y mide la retención."""
    if not learned:
        return {}
    per = max(1, args.revision_puzzles // len(learned))
    pool: list[Puzzle] = []
    for lvl in learned:
        lvl_pool = replay_by_level.get(lvl, [])
        if lvl_pool:
            pool += rng.sample(lvl_pool, min(per, len(lvl_pool)))
    if not pool:
        return {}
    log_phase("[REV]", f"revisión de niveles {learned} ({len(pool)} puzzles, "
              f"{args.revision_epochs} epochs)")
    ret_accs: dict[int, float] = {}
    for epoch in range(1, args.revision_epochs + 1):
        if _STOP.is_set():
            log_phase("[REV]", "  revisión interrumpida por parada solicitada.")
            break
        loss, ploss, vloss = trainer.train_epoch(pool, args.batch_size)
        ret_accs = {lvl: accuracy(net, eval_sets[lvl]) for lvl in learned}
        acc_txt = " ".join(f"m{lvl}:{a * 100:.0f}%" for lvl, a in ret_accs.items())
        log_phase("[REV]", f"  Rev ep {epoch}/{args.revision_epochs} | loss {loss:.3f} | "
                  f"retención {acc_txt}")
    return ret_accs


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m trainPuzzles",
        description="Entrena SuperTatetiNet con puzzles de mate en N (currículo).")
    p.add_argument("--init", default=None,
                   help=".pth de partida (default: el canon de la arquitectura --arch, el "
                        "último modelo terminado; si no existe, red aleatoria)")
    p.add_argument("--out", default=None,
                   help=".pth de salida (default: el canon de la arquitectura --arch). Cada "
                        "corrida además archiva su modelo/reporte en <dir>/runs/NNNN_<run-name>/")
    p.add_argument("--run-name", dest="run_name", default="puzzle",
                   help="etiqueta de la corrida archivada en <dir>/runs/ (default: puzzle)")
    p.add_argument("--no-canon", dest="no_canon", action="store_true",
                   help="no copiar el modelo final a <dir>/puzzle_canon.pth "
                        "(útil para experimentos A/B que no deben pisar la cadena)")
    p.add_argument("--report", default=None,
                   help="reporte JSON del canon (default: <out>_report.json, junto al modelo)")
    p.add_argument("--levels", type=int, default=4,
                   help="niveles iniciales pre-generados (1..N)")
    p.add_argument("--start-level", dest="start_level", type=int, default=1,
                   help="nivel donde arranca el currículo (default 1). Con >1 se salta el "
                        "reentrenamiento de los niveles previos (p. ej. --start-level 2 arranca "
                        "en mate-2 calentando desde --init); se generan igual sus sets de "
                        "evaluación para medir retención")
    p.add_argument("--max-level", type=int, default=0,
                   help="nivel máximo a alcanzar; 0 = sin tope (sube de nivel indefinidamente)")
    p.add_argument("--stop-at-level", dest="stop_at_level", type=int, default=0,
                   help="termina el currículo tras procesar este nivel (dominado o no) y guarda "
                        "el reporte; 0 = off. Sirve para que retrain_loop evalúe el dominio del "
                        "nivel objetivo sin seguir a niveles superiores")
    p.add_argument("--min-level-puzzles", type=int, default=200,
                   help="mínimo de puzzles para considerar entrenable un nivel nuevo")
    p.add_argument("--threshold", type=float, default=0.95,
                   help="acierto mínimo en evaluación para considerar el nivel dominado (default 0.95)")
    p.add_argument("--retention-threshold", type=float, default=None,
                   help="acierto mínimo en la unión de niveles previos (default: = --threshold)")
    p.add_argument("--patience", type=int, default=2,
                   help="evaluaciones consecutivas sobre el umbral para dominar (default 2)")
    p.add_argument("--min-epochs", type=int, default=15,
                   help="epochs mínimos en un nivel antes de poder avanzar (default 15)")
    p.add_argument("--max-epochs", type=int, default=0,
                   help="tope de seguridad de epochs por nivel; 0 = sin tope, entrena hasta dominar")
    p.add_argument("--plateau-patience", type=int, default=10,
                   help="avisa si pasan estos epochs sin mejorar el mejor acierto (default 10)")
    p.add_argument("--plateau-stop", type=int, default=0,
                   help="corta el nivel y la corrida si pasan estos epochs sin mejorar el mejor "
                        "acierto (0 = off, entrena sin tope)")
    p.add_argument("--log-every", type=int, default=10,
                    help="imprime la línea por epoch 1 de cada N epochs (<=0 = todas); "
                         "los hitos (nuevo mejor, dominado, plateau, parada) se imprimen siempre")
    p.add_argument("--eval-every", type=int, default=1,
                    help="evalúa/selecciona mejor cada N epochs (default 1)")
    p.add_argument("--revision-epochs", type=int, default=6,
                    help="epochs de revisión de niveles viejos tras dominar cada nivel (default 6)")
    p.add_argument("--revision-puzzles", type=int, default=500,
                    help="puzzles por fase de revisión (repartidos entre los niveles viejos)")
    p.add_argument("--big", type=float, default=0.20,
                    help="cambio (en acierto) para 'mejoró/empeoró mucho' (default 0.20)")
    p.add_argument("--small", type=float, default=0.05,
                    help="cambio (en acierto) para 'mejoró/empeoró poco' (default 0.05)")
    p.add_argument("--puzzles-per-level", type=int, default=25000,
                    help="puzzles de entrenamiento por nivel (con pocos la red los memoriza)")
    p.add_argument("--eval-size", type=int, default=1000, help="puzzles del set de evaluación por nivel")
    p.add_argument("--no-augment", dest="no_augment", action="store_true",
                    help="desactiva las simetrías D4 (por defecto cada muestra se rota/refleja al azar)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-schedule", choices=["none", "cosine"], default="none",
                    help="programa de LR (default none)")
    p.add_argument("--lr-final-frac", type=float, default=0.03,
                    help="eta_min = lr * lr-final-frac para cosine")
    p.add_argument("--lr-decay-epochs", type=int, default=0,
                    help="T_max para cosine; 0 = usar max-epochs o 3000")
    p.add_argument("--tta", action="store_true", help="usar TTA (8 simetrías) en evaluación")
    p.add_argument("--arch", choices=("mlp", "cnn"), default="mlp",
                    help="arquitectura: 'mlp' (SuperTatetiNet, checkpoints_mts) o "
                         "'cnn' (SuperTatetiCNN, checkpoint_cnn). Define el directorio "
                         "por defecto de --init/--out")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                    help="dispositivo de cómputo: 'auto' usa CUDA si está disponible (default)")
    p.add_argument("--dropout", type=float, default=0.0,
                    help="dropout funcional en las capas compartidas (0 = off; se aplica solo entrenando)")
    p.add_argument("--weight-decay", type=float, default=0.0,
                    help="penalización L2 de Adam (weight decay; 0 = off)")
    p.add_argument("--value-coef", type=float, default=0.0,
                    help="peso de la pérdida de valor (0 = off: todos los puzzles son ganados, aprendería una constante)")
    p.add_argument("--replay-frac", type=float, default=0.30,
                    help="fracción de puzzles de niveles previos en cada nivel")
    p.add_argument("--min-turn", type=int, default=16,
                    help="solo se resuelven posiciones con turno >= este valor")
    p.add_argument("--solver-max-nodes", type=int, default=120000,
                    help="presupuesto de nodos del solver por posición (0 = sin límite)")
    p.add_argument("--games-budget", type=int, default=0,
                    help="tope duro de partidas aleatorias por generación; 0 = sin tope "
                         "(junta el target usando --max-games como seguridad)")
    p.add_argument("--max-games", type=int, default=200000,
                    help="tope de seguridad de partidas cuando --games-budget=0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-seed", type=int, default=None,
                    help="semilla del set de evaluación (default: = --seed); fijala para comparar "
                         "entre reintentos con datos de train distintos")
    p.add_argument("--hard-mining", action="store_true", help="activar hard-example mining")
    p.add_argument("--mine-every", type=int, default=10)
    p.add_argument("--mine-boost", type=float, default=3.0)
    p.add_argument("--mine-warmup", type=int, default=25)
    p.add_argument("--ema", action="store_true", help="activar EMA de pesos")
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--rank-coef", type=float, default=0.0, help="coeficiente de ranking loss")
    p.add_argument("--rank-margin", type=float, default=1.0, help="margen de ranking loss")
    args = p.parse_args(argv)
    arch = getattr(args, "arch", "mlp")
    if args.init is None:
        args.init = str(layout.default_canon(arch))
    if args.out is None:
        args.out = str(layout.default_canon(arch))
    return args


def _log_config(args, init: Path, out: Path, run_dir: Path, canon_pth: Path) -> None:
    ret_threshold = args.threshold if args.retention_threshold is None else args.retention_threshold
    log_phase("[INFO]", "Configuración:")
    log_phase("[INFO]", f"  arch={getattr(args, 'arch', 'mlp')}  init={init}  out={out}  corrida={run_dir}  "
              f"canon={'off' if getattr(args, 'no_canon', False) else canon_pth}  "
              f"niveles iniciales={args.levels}  "
              f"nivel inicial={getattr(args, 'start_level', 1)}  "
              f"max-level={'sin tope' if not args.max_level else args.max_level}")
    log_phase("[INFO]", f"  umbral={args.threshold * 100:.0f}%  retención={ret_threshold * 100:.0f}%"
              f"  patience={args.patience}  min-epochs={args.min_epochs}  "
              f"max-epochs={'sin tope' if not args.max_epochs else args.max_epochs}  "
              f"plateau-stop={'off' if not args.plateau_stop else args.plateau_stop}  "
              f"log-every={args.log_every}")
    log_phase("[INFO]", f"  entrada={canon.INPUT_ENCODING} (perspectiva de quien mueve)  "
               f"simetrías D4={'no' if getattr(args, 'no_augment', False) else 'sí'}  value-coef={args.value_coef}")
    log_phase("[INFO]", f"  puzzles/nivel={args.puzzles_per_level}  eval/nivel={args.eval_size}  "
               f"batch={args.batch_size}  lr={args.lr} ({getattr(args,'lr_schedule','none')}) dropout={args.dropout}  "
               f"weight-decay={args.weight_decay}  replay={args.replay_frac}  "
               f"revisión={args.revision_epochs}ep/{args.revision_puzzles}puz  tta={getattr(args,'tta',False)}  "
               f"hard={getattr(args,'hard_mining',False)} ema={getattr(args,'ema',False)} rank={getattr(args,'rank_coef',0.0)}")
    log_phase("[INFO]", f"  generación: games-budget={'sin tope' if args.games_budget <= 0 else args.games_budget}"
              f"  max-games={args.max_games}  min-turn={args.min_turn}  "
              f"solver-max-nodes={'sin límite' if not args.solver_max_nodes else args.solver_max_nodes}  "
              f"seed={args.seed}  eval-seed={_eval_seed(args)}")


# ----------------------------------------------------------------------
# Bucle principal
# ----------------------------------------------------------------------
def run(args) -> int:
    _STOP.clear()
    _install_signal_handlers()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out = Path(args.out)
    init = Path(args.init)
    canon_pth, canon_report = layout.canon_paths(out)
    run_dir = layout.next_run_dir(out, args.run_name)
    run_pth, run_report, levels_dir = layout.run_paths(run_dir, args.run_name)
    no_canon = getattr(args, "no_canon", False)
    report_path = Path(args.report) if args.report else canon_report
    ret_threshold = args.threshold if args.retention_threshold is None else args.retention_threshold

    start_level, args.levels = scheduled_levels(args)

    global _DEVICE
    _DEVICE = resolve_device(getattr(args, "device", "auto"))
    log_print(f"Dispositivo de cómputo: {_DEVICE}"
              + (f" ({torch.cuda.get_device_name(0)})" if _DEVICE.type == "cuda" else ""))

    _log_config(args, init, out, run_dir, canon_pth)

    exclude: set = set()
    eval_sets, train_sets = build_sets(args)
    arch = getattr(args, "arch", "mlp")
    net = load_net(init, dropout=args.dropout, arch=arch)
    net.to(_DEVICE)

    initial_state = clone_state(net)
    baseline_hidden = None
    if arch == "mlp":
        try:
            baseline_hidden = hidden_from_state_dict(initial_state)
        except ValueError:
            baseline_hidden = None
    baseline_net = build_net(arch, mlp_hidden=baseline_hidden)
    baseline_net.load_state_dict(initial_state)
    baseline_net.to(_DEVICE)
    baselines = {n: accuracy(baseline_net, eval_sets.get(n, []), tta=getattr(args, 'tta', False))
                 for n in range(1, args.levels + 1)}
    log_phase("[INFO]", "Baseline (red inicial) por nivel: " +
              ", ".join(f"mate {n}: {baselines[n] * 100:.1f}%" for n in baselines))

    trainer = PuzzleTrainer(net, lr=args.lr, value_coef=args.value_coef, seed=args.seed,
                            augment=not getattr(args, "no_augment", False),
                            weight_decay=args.weight_decay,
                            lr_schedule=getattr(args, 'lr_schedule', 'none'),
                            lr_final_frac=getattr(args, 'lr_final_frac', 0.03),
                            lr_decay_epochs=getattr(args, 'lr_decay_epochs', 0),
                            max_epochs=args.max_epochs,
                            ema_decay=getattr(args, 'ema_decay', 0.0),
                            rank_coef=getattr(args, 'rank_coef', 0.0),
                            rank_margin=getattr(args, 'rank_margin', 1.0))
    rng = random.Random(args.seed)

    replay: list[Puzzle] = []
    replay_by_level: dict[int, list[Puzzle]] = {}
    summaries: list[dict] = []
    learned: list[int] = []

    level = start_level
    revision_only = False

    def write_incremental():
        report = build_report(args, init, out, report_path, summaries, ret_threshold,
                              run_dir=run_dir)
        write_report(report, report_path, announce=False)
        write_report(report, run_report, announce=False)
        return report

    def save_final() -> None:
        """Guarda el modelo final en `--out`, la carpeta de la corrida y el canon.

        Con `--no-canon` el canon no se pisa, aunque `--out` apunte a él.
        """
        dests = [out, run_pth]
        if no_canon:
            dests = [d for d in dests if d != canon_pth]
        else:
            dests.append(canon_pth)
        saved: list[Path] = []
        for dest in dests:
            if dest in saved:
                continue
            save_net(net, dest)
            saved.append(dest)

    while not _STOP.is_set():
        # 1) Asegurar que existan los sets del nivel (generación on-demand).
        if level not in train_sets:
            if args.max_level and level > args.max_level:
                log_phase("[INFO]", f"se alcanzó --max-level {args.max_level}: solo revisión de "
                          f"niveles viejos hasta que lo pares.")
                revision_only = True
            else:
                log_phase("[GEN]", f"generando datasets del nivel {level} (mate en {level})...")
                ev = generate_eval_set(level, args, exclude)
                tr = generate_train_set(level, args, exclude)
                if len(tr) < args.min_level_puzzles or len(ev) < max(1, args.eval_size // 4):
                    log_phase("[INFO]", f"nivel {level}: puzzles insuficientes "
                              f"({len(tr)} train / {len(ev)} eval < mínimo "
                              f"{args.min_level_puzzles}); se pasa a modo revisión.")
                    revision_only = True
                else:
                    eval_sets[level] = ev
                    train_sets[level] = tr
                    baselines[level] = accuracy(baseline_net, ev, tta=getattr(args, 'tta', False))

        # 2) Modo revisión (o parada): repite fases de revisión hasta que lo pares.
        if revision_only or _STOP.is_set():
            if _STOP.is_set():
                break
            ret_accs = revision_phase(net, trainer, args, replay_by_level, learned, eval_sets, rng)
            write_incremental()
            if not ret_accs and not _STOP.is_set():
                log_phase("[INFO]", "no hay puzzles de revisión disponibles; se detiene.")
                break
            continue

        current = train_sets.get(level, [])
        evalset = eval_sets.get(level, [])
        retention_eval = [p for k in range(1, level + 1) for p in eval_sets.get(k, [])]
        base = baselines.get(level, 0.0)
        log_phase("[TRAIN]", f"===== Nivel {level} (mate en {level}) ====="
                  + (f" (hasta {args.max_level})" if args.max_level else "")
                  + f"  train={len(current)} eval={len(evalset)} "
                  f"umbral={args.threshold * 100:.0f}% retención={ret_threshold * 100:.0f}% "
                  f"min-epochs={args.min_epochs} patience={args.patience}")

        if not current or not evalset:
            log_phase("[INFO]", f"faltan puzzles para el nivel {level}; se omite.")
            summaries.append({"level": level, "baseline": base, "final": float("nan"),
                              "delta": float("nan"), "epochs": 0, "mastered": False,
                              "reached": False, "best": float("nan"), "best_epoch": 0})
            level += 1
            continue

        pool = mix_pool(current, replay, args.replay_frac, rng)
        res = train_level(net, trainer, args, level, pool, evalset, retention_eval,
                          base, ret_threshold)

        if res["best_state"] is not None:
            net.load_state_dict(res["best_state"])
        final_tk = topk_accuracy(net, evalset, ks=(1, 2, 3), tta=getattr(args, 'tta', False))
        final_acc = final_tk.get(1, 0.0)
        summaries.append({"level": level, "baseline": base, "final": final_acc,
                          "delta": final_acc - base, "epochs": res["epochs"],
                          "mastered": res["mastered"], "reached": True,
                          "best": res["best_acc"], "best_epoch": res["best_epoch"],
                          "best_top2": res.get("best_acc2"), "best_top3": res.get("best_acc3"),
                          "final_top2": final_tk.get(2), "final_top3": final_tk.get(3)})
        learned.append(level)
        replay.extend(current)
        replay_by_level.setdefault(level, []).extend(current)
        save_net(net, levels_dir / f"level{level}_best.pth")
        if not no_canon:
            save_net(net, canon_pth)          # canon reanudable aunque se corte acá
        write_incremental()
        dom = "dominado" if res["mastered"] else "no dominado"
        log_phase("[INFO]", f"resumen parcial: mate {level} -> {final_acc * 100:.1f}% "
                  f"(base {base * 100:.1f}%, {((final_acc - base) * 100):+.1f}) | {dom}")

        # 3) Revisión anti-olvido de niveles viejos.
        revision_phase(net, trainer, args, replay_by_level, learned, eval_sets, rng)
        write_incremental()

        if res["stopped"] or res["capped"] or res["plateau_stopped"]:
            if res["capped"]:
                log_phase("[INFO]", "tope de epochs alcanzado sin dominar: se detiene el currículo.")
            if res["plateau_stopped"]:
                log_phase("[INFO]", "estancamiento (--plateau-stop): se detiene el currículo "
                          "para reintentar con más datos.")
            break
        if args.stop_at_level and level >= args.stop_at_level:
            log_phase("[TRAIN]", f"nivel {level} finalizado; --stop-at-level {args.stop_at_level}"
                                 f" alcanzado: se detiene el currículo.")
            break
        if not _STOP.is_set():
            log_phase("[TRAIN]", f"nivel {level} finalizado; se continúa con el nivel {level + 1}.")
        level += 1

    save_final()
    report = build_report(args, init, out, report_path, summaries, ret_threshold,
                          run_dir=run_dir)
    print_report(report)
    write_report(report, report_path, announce=True)
    write_report(report, run_report, announce=False)
    if _STOP.is_set():
        log_phase("[INFO]", "entrenamiento detenido por el usuario. Resumen y reporte guardados.")
    return 0


# ----------------------------------------------------------------------
# Reporte
# ----------------------------------------------------------------------
def build_report(args, init: Path, out: Path, report_path: Path,
                 summaries: list[dict], ret_threshold: float,
                 run_dir: Path | None = None) -> dict:
    reached = [s for s in summaries if s.get("reached")]
    deltas = [s["delta"] for s in reached if s["delta"] == s["delta"]]
    avg = float(sum(deltas) / len(deltas)) if deltas else float("nan")
    verdict = classify_delta(avg, args.big, args.small) if deltas else "sin datos"
    mastered = sum(1 for s in reached if s["mastered"])
    for s in summaries:
        s["verdict"] = classify_delta(s["delta"], args.big, args.small)
    return {
        "input_encoding": canon.INPUT_ENCODING,
        "init": str(init),
        "out": str(out),
        "report": str(report_path),
        "run_dir": None if run_dir is None else str(run_dir),
        "run_name": getattr(args, "run_name", None),
        "levels_requested": args.levels,
        "start_level": getattr(args, "start_level", 1),
        "max_level": args.max_level,
        "threshold": args.threshold,
        "retention_threshold": ret_threshold,
        "patience": args.patience,
        "min_epochs": args.min_epochs,
        "big": args.big,
        "small": args.small,
        "levels": summaries,
        "avg_delta": avg,
        "verdict": verdict,
        "levels_reached": len(reached),
        "levels_mastered": mastered,
    }


def print_report(report: dict) -> None:
    log_print("=" * 60)
    log_print("RESUMEN DEL APRENDIZAJE")
    log_print("=" * 60)
    log_print(f"  {'nivel':>5}  {'base':>7}  {'final':>7}  {'cambio':>7}  "
              f"{'epochs':>6}  {'dominio':>7}  clasificación")
    for s in report["levels"]:
        base = "s/d" if s["baseline"] != s["baseline"] else f"{s['baseline'] * 100:.1f}%"
        final = "s/d" if s["final"] != s["final"] else f"{s['final'] * 100:.1f}%"
        delta = "s/d" if s["delta"] != s["delta"] else f"{s['delta'] * 100:+.1f}"
        dominio = "sí" if s["mastered"] else ("no" if s.get("reached") else "no entrenado")
        log_print(f"  {s['level']:>5}  {base:>7}  {final:>7}  {delta:>7}  "
                  f"{s['epochs']:>6}  {dominio:>7}  {s['verdict']}")
        if s.get("final_top2") is not None:
            log_print(f"  {'':>5}  top-2 {s['final_top2'] * 100:5.1f}%  "
                      f"top-3 {s['final_top3'] * 100:5.1f}%  (mejor T2 "
                      f"{(s.get('best_top2') or 0) * 100:5.1f}%  T3 "
                      f"{(s.get('best_top3') or 0) * 100:5.1f}%)")
    avg = report["avg_delta"]
    avg_txt = "s/d" if avg != avg else f"{avg * 100:+.1f} pts"
    log_print("-" * 60)
    log_print(f"VEREDICTO GLOBAL: {report['verdict'].upper()} "
              f"(promedio {avg_txt}; dominó {report['levels_mastered']}/"
              f"{report['levels_reached']} niveles alcanzados).")


def write_report(report: dict, path: Path, announce: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    if announce:
        log_print(f"Reporte guardado en {path}")


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())