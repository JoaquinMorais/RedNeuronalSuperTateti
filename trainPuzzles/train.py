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

from model import SuperTatetiNet

from . import canon
from .puzzles import Puzzle, harvest


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
    al azar por muestra en cada epoch (``augment``). Ver ``canon.py``."""

    def __init__(self, net: SuperTatetiNet, lr: float = 3e-4, value_coef: float = 0.0,
                 seed: int = 0, augment: bool = True):
        self.net = net
        self.value_coef = value_coef
        self.augment = augment
        self.optimizer = optim.Adam(net.parameters(), lr=lr)
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    def train_epoch(self, puzzles: list[Puzzle], batch_size: int) -> tuple[float, float, float]:
        if not puzzles:
            return (0.0, 0.0, 0.0)
        self.net.train()
        order = list(range(len(puzzles)))
        self.rng.shuffle(order)
        losses = []
        for i in range(0, len(order), batch_size):
            idxs = order[i:i + batch_size]
            states_np = canon.canonical_batch(np.stack([puzzles[j].state for j in idxs]))
            masks_np = np.stack([puzzles[j].mask for j in idxs])
            targets_np = np.stack([policy_target(puzzles[j].best_moves) for j in idxs])
            if self.augment:
                states_np, masks_np, targets_np = canon.augment(states_np, masks_np, targets_np, self.np_rng)
            states = torch.from_numpy(states_np)
            masks = torch.from_numpy(masks_np)
            targets = torch.from_numpy(targets_np)

            pred_policy, pred_value = self.net(states, masks)
            policy_loss = -(targets * torch.log(pred_policy + 1e-8)).sum(dim=1).mean()
            value_loss = nn.functional.mse_loss(pred_value, torch.ones_like(pred_value))
            loss = policy_loss + self.value_coef * value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append((loss.item(), policy_loss.item(), value_loss.item()))
        arr = np.asarray(losses, dtype=np.float64)
        return float(arr[:, 0].mean()), float(arr[:, 1].mean()), float(arr[:, 2].mean())


@torch.no_grad()
def accuracy(net: SuperTatetiNet, puzzles: list[Puzzle]) -> float:
    """% de puzzles donde la jugada elegida (argmax legal) está entre las óptimas."""
    if not puzzles:
        return 0.0
    net.eval()
    correct = 0
    for i in range(0, len(puzzles), 1024):                      # por lotes: mucho más rápido que uno a uno
        chunk = puzzles[i:i + 1024]
        states = torch.from_numpy(canon.canonical_batch(np.stack([p.state for p in chunk])))
        masks = torch.from_numpy(np.stack([p.mask for p in chunk]))
        policy, _ = net(states, masks)
        moves = torch.argmax(policy.masked_fill(masks == 0, -1.0), dim=-1).tolist()
        correct += sum(1 for mv, p in zip(moves, chunk) if mv in p.best_moves)
    return correct / len(puzzles)


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
def generate_eval_set(n: int, args, exclude: set) -> list[Puzzle]:
    """Genera el set de evaluación fijo del nivel ``n`` y agrega sus claves a ``exclude``."""
    log_phase("[GEN]", f"evaluación mate en {n}...")
    got = harvest({n: args.eval_size}, seed=args.seed + 1000 + n,
                  min_turn=args.min_turn, solver_max_nodes=args.solver_max_nodes,
                  games_budget=args.games_budget, should_stop=_STOP.is_set)
    ps = got.get(n, [])
    for p in ps:
        exclude.add(p.key)
    log_phase("[GEN]", f"  evaluación mate en {n}: {len(ps)}/{args.eval_size}")
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
                  games_budget=args.games_budget, log=prog, log_interval=2.0,
                  should_stop=_STOP.is_set)
    ps = got.get(n, [])
    log_phase("[GEN]", f"  entrenamiento mate en {n}: {len(ps)}/{target}")
    return ps


def build_sets(args) -> tuple[dict[int, list[Puzzle]], dict[int, list[Puzzle]]]:
    """Genera los sets iniciales (evaluación y entrenamiento) para 1..--levels."""
    exclude: set = set()
    eval_sets: dict[int, list[Puzzle]] = {}
    train_sets: dict[int, list[Puzzle]] = {}
    log_phase("[GEN]", f"pre-generando datasets de los niveles 1..{args.levels} "
                       f"(esto ocurre antes de entrenar)")
    for n in range(1, args.levels + 1):
        eval_sets[n] = generate_eval_set(n, args, exclude)
    for n in range(1, args.levels + 1):
        train_sets[n] = generate_train_set(n, args, exclude)
    return eval_sets, train_sets


def load_net(init_path: Path) -> SuperTatetiNet:
    net = SuperTatetiNet()
    if init_path.exists():
        state = torch.load(init_path, weights_only=True, map_location="cpu")
        net.load_state_dict(state)
        log_print(f"Red inicializada desde {init_path}")
    else:
        log_print(f"No existe {init_path}: se entrena desde una red aleatoria")
    return net


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
    best_state = None
    best_epoch = 0
    epoch = 0
    probe = random.Random(level).sample(pool, min(1000, len(pool)))   # muestra fija de entrenamiento: mide la memorización
    streak = 0
    plateau = 0
    mastered = False
    stopped = False
    capped = False

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
        loss, ploss, vloss = trainer.train_epoch(pool, args.batch_size)
        acc = accuracy(net, evalset)
        ret = accuracy(net, retention_eval)
        tr_acc = accuracy(net, probe)
        dt = time.time() - t0

        if acc > best_acc:
            best_acc = acc
            best_state = clone_state(net)
            best_epoch = epoch
            plateau = 0
        else:
            plateau += 1

        streak = mastery_step(acc, ret, args.threshold, ret_threshold,
                              epoch, args.min_epochs, streak)
        mastered = streak >= args.patience

        status = "MASTERED" if mastered else f"meta {args.threshold * 100:.0f}%"
        log_phase("[TRAIN]",
                  f"  N{level} ep {epoch:3d} | loss {loss:.3f} (P {ploss:.3f} V {vloss:.3f}) | "
                  f"acierto {acc * 100:5.1f}% base {base * 100:5.1f}% d {(acc - base) * 100:+5.1f} | "
                  f"train {tr_acc * 100:5.1f}% brecha {(tr_acc - acc) * 100:+5.1f} | "
                  f"mejor {best_acc * 100:5.1f}% | ret {ret * 100:5.1f}% | "
                  f"{_bar(acc / args.threshold if args.threshold else 1.0)} {dt:.1f}s | {status}")

        if plateau and args.plateau_patience and plateau % args.plateau_patience == 0:
            log_phase("[TRAIN]", f"  aviso: nivel {level} sin mejorar el mejor acierto desde hace "
                       f"{plateau} epochs (mejor {best_acc * 100:.1f}%).")

        if mastered:
            log_phase("[TRAIN]", f"  nivel {level} DOMINADO: acierto {acc * 100:.1f}% "
                       f"(>= {args.threshold * 100:.0f}% durante {args.patience} evals, "
                       f"{epoch} epochs).")
            break

    return {"epochs": epoch, "best_acc": best_acc, "best_state": best_state,
            "best_epoch": best_epoch, "mastered": mastered, "stopped": stopped,
            "capped": capped}


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
    p.add_argument("--init", default="model.pth", help=".pth de partida (si no existe, red aleatoria)")
    p.add_argument("--out", default="trainPuzzles/checkpoints/puzzle_net.pth",
                   help=".pth de salida (dentro de trainPuzzles/ por defecto)")
    p.add_argument("--report", default=None,
                   help="reporte JSON (default: <out>_report.json, junto al modelo)")
    p.add_argument("--levels", type=int, default=4,
                   help="niveles iniciales pre-generados (1..N)")
    p.add_argument("--max-level", type=int, default=0,
                   help="nivel máximo a alcanzar; 0 = sin tope (sube de nivel indefinidamente)")
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
    p.add_argument("--value-coef", type=float, default=0.0,
                   help="peso de la pérdida de valor (0 = off: todos los puzzles son ganados, aprendería una constante)")
    p.add_argument("--replay-frac", type=float, default=0.30,
                   help="fracción de puzzles de niveles previos en cada nivel")
    p.add_argument("--min-turn", type=int, default=16,
                   help="solo se resuelven posiciones con turno >= este valor")
    p.add_argument("--solver-max-nodes", type=int, default=120000,
                   help="presupuesto de nodos del solver por posición (0 = sin límite)")
    p.add_argument("--games-budget", type=int, default=8000,
                   help="tope de partidas aleatorias por generación")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def _log_config(args, init: Path, out: Path) -> None:
    ret_threshold = args.threshold if args.retention_threshold is None else args.retention_threshold
    log_phase("[INFO]", "Configuración:")
    log_phase("[INFO]", f"  init={init}  out={out}  niveles iniciales={args.levels}  "
              f"max-level={'sin tope' if not args.max_level else args.max_level}")
    log_phase("[INFO]", f"  umbral={args.threshold * 100:.0f}%  retención={ret_threshold * 100:.0f}%"
              f"  patience={args.patience}  min-epochs={args.min_epochs}  "
              f"max-epochs={'sin tope' if not args.max_epochs else args.max_epochs}")
    log_phase("[INFO]", f"  entrada={canon.INPUT_ENCODING} (perspectiva de quien mueve)  "
              f"simetrías D4={'no' if getattr(args, 'no_augment', False) else 'sí'}  value-coef={args.value_coef}")
    log_phase("[INFO]", f"  puzzles/nivel={args.puzzles_per_level}  eval/nivel={args.eval_size}  "
              f"batch={args.batch_size}  lr={args.lr}  replay={args.replay_frac}  "
              f"revisión={args.revision_epochs}ep/{args.revision_puzzles}puz")


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
    report_path = Path(args.report) if args.report else out.with_name(f"{out.stem}_report.json")
    ret_threshold = args.threshold if args.retention_threshold is None else args.retention_threshold

    _log_config(args, init, out)

    exclude: set = set()
    eval_sets, train_sets = build_sets(args)
    net = load_net(init)

    initial_state = clone_state(net)
    baseline_net = SuperTatetiNet()
    baseline_net.load_state_dict(initial_state)
    baselines = {n: accuracy(baseline_net, eval_sets.get(n, []))
                 for n in range(1, args.levels + 1)}
    log_phase("[INFO]", "Baseline (red inicial) por nivel: " +
              ", ".join(f"mate {n}: {baselines[n] * 100:.1f}%" for n in baselines))

    trainer = PuzzleTrainer(net, lr=args.lr, value_coef=args.value_coef, seed=args.seed,
                            augment=not getattr(args, "no_augment", False))
    rng = random.Random(args.seed)

    replay: list[Puzzle] = []
    replay_by_level: dict[int, list[Puzzle]] = {}
    summaries: list[dict] = []
    learned: list[int] = []

    level = 1
    revision_only = False

    def write_incremental():
        report = build_report(args, init, out, report_path, summaries, ret_threshold)
        write_report(report, report_path, announce=False)
        return report

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
                    baselines[level] = accuracy(baseline_net, ev)

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
        final_acc = accuracy(net, evalset)
        summaries.append({"level": level, "baseline": base, "final": final_acc,
                          "delta": final_acc - base, "epochs": res["epochs"],
                          "mastered": res["mastered"], "reached": True,
                          "best": res["best_acc"], "best_epoch": res["best_epoch"]})
        learned.append(level)
        replay.extend(current)
        replay_by_level.setdefault(level, []).extend(current)
        save_net(net, out.with_name(f"{out.stem}_level{level}_best{out.suffix}"))
        write_incremental()
        dom = "dominado" if res["mastered"] else "no dominado"
        log_phase("[INFO]", f"resumen parcial: mate {level} -> {final_acc * 100:.1f}% "
                  f"(base {base * 100:.1f}%, {((final_acc - base) * 100):+.1f}) | {dom}")

        # 3) Revisión anti-olvido de niveles viejos.
        revision_phase(net, trainer, args, replay_by_level, learned, eval_sets, rng)
        write_incremental()

        if res["stopped"] or res["capped"]:
            if res["capped"]:
                log_phase("[INFO]", "tope de epochs alcanzado sin dominar: se detiene el currículo.")
            break
        if not _STOP.is_set():
            log_phase("[TRAIN]", f"nivel {level} finalizado; se continúa con el nivel {level + 1}.")
        level += 1

    save_net(net, out)
    report = build_report(args, init, out, report_path, summaries, ret_threshold)
    print_report(report)
    write_report(report, report_path, announce=True)
    if _STOP.is_set():
        log_phase("[INFO]", "entrenamiento detenido por el usuario. Resumen y reporte guardados.")
    return 0


# ----------------------------------------------------------------------
# Reporte
# ----------------------------------------------------------------------
def build_report(args, init: Path, out: Path, report_path: Path,
                 summaries: list[dict], ret_threshold: float) -> dict:
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
        "levels_requested": args.levels,
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