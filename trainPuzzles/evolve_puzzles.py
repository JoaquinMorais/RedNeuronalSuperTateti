"""Entrenador evolutivo memético de puzzles tácticos ("mate en N").

Combina dos ideas:
  * **Gradiente** (búsqueda local): cada individuo entrena unos epochs sobre una
    muestra del pool de puzzles y conserva su mejor estado según validación.
  * **Evolución**: una población de redes; los mejores sobreviven intactos y el
    resto se recrea mutando a los supervivientes. La selección mantiene diversidad
    y frena el sobreajuste (que es el problema del entrenamiento de un solo linaje).

No depende de `evolution/`: la mutación y la configuración viven en
`trainPuzzles/evo_config.py`. No usa base de datos, ni colores, ni torneos.

Flujo por generación:
  1. Búsqueda local de cada individuo (con su propia bootstrap del pool).
  2. Evaluación del acierto en puzzles (validación).
  3. Selección: los ``survivors`` mejores pasan intactos; el resto se crea mutando.
  4. Guardado del mejor histórico + reporte JSON incremental.

Se detiene prolijamente con Ctrl-C / SIGTERM reutilizando la señal de `train`.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from model import SuperTatetiNet

from . import train as T
from .evo_config import EvoConfig, load_config, policy_tuple, save_config, validate
from .puzzles import Puzzle, harvest

# Capas cuyos desvíos se usan como referencia para detectar explosión de pesos.
_REF_LAYERS = [("shared.0", 100), ("shared.2", 256), ("shared.4", 256), ("policy_head", 128)]


# ----------------------------------------------------------------------
# Mutación de pesos (propia, autocontenida)
# ----------------------------------------------------------------------
def _is_mutable(name: str, mutate_value_head: bool) -> bool:
    if name.startswith("value_head.") and not mutate_value_head:
        return False
    return True


def normalize_state_dict(sd: dict, threshold: float = 4.0) -> tuple[dict, bool]:
    """Reescala todas las capas si algún linaje hizo crecer sus pesos de más.

    Una ReLU es homogénea positiva: escalar los pesos de una capa y compensar en la
    siguiente deja la jugada idéntica. Si algún tensor de referencia supera
    ``threshold`` veces su desvío natural, TODAS las capas vuelven a su desvío
    natural. La cabeza de valor no se toca. ``threshold <= 0`` lo desactiva.
    """
    if threshold <= 0:
        return sd, False
    d = {k: (torch.nan_to_num(v.detach().double(), nan=0.0, posinf=0.0, neginf=0.0)
             if torch.is_floating_point(v) else v) for k, v in sd.items()}
    stds = {n: float(d[f"{n}.weight"].std()) for n, _ in _REF_LAYERS}
    refs = {n: 1.0 / (fan_in ** 0.5) for n, fan_in in _REF_LAYERS}
    broken = any(not torch.isfinite(v).all() for v in sd.values() if torch.is_floating_point(v))
    if not broken and max(stds[n] / refs[n] for n, _ in _REF_LAYERS) <= threshold:
        return sd, False
    out = dict(d)
    scale = 1.0
    for n, _ in _REF_LAYERS:
        m = refs[n] / max(stds[n], 1e-300)
        scale *= m
        out[f"{n}.weight"] = d[f"{n}.weight"] * m
        out[f"{n}.bias"] = d[f"{n}.bias"] * scale
    return {k: (v.float() if torch.is_floating_point(v) else v) for k, v in out.items()}, True


def mutate_state_dict(parent: dict, level, seed: int, mutate_value_head: bool = False,
                      norm_threshold: float = 0.0) -> dict:
    """Copia INDEPENDIENTE del padre con mutación gaussiana (el padre no se toca)."""
    parent, _ = normalize_state_dict(parent, norm_threshold)
    gen = torch.Generator().manual_seed(seed)
    child = {k: v.detach().clone() for k, v in parent.items()}
    for name in sorted(child):
        t = child[name]
        if not _is_mutable(name, mutate_value_head) or not torch.is_floating_point(t):
            continue
        scale = max(float(t.std()) if t.numel() > 1 else abs(float(t)), 1e-3)
        noise = torch.randn(t.shape, generator=gen) * (level.sigma * scale)
        mask = (torch.rand(t.shape, generator=gen) < level.rate).to(t.dtype)
        child[name] = t + noise * mask
    return child


def pick_level(probs: dict, rng: random.Random) -> str:
    """Elige un nivel de mutación según sus probabilidades."""
    r = rng.random()
    acc = 0.0
    for name in ("small", "medium", "large"):
        acc += probs[name]
        if r < acc:
            return name
    return "large"


# ----------------------------------------------------------------------
# Individuos
# ----------------------------------------------------------------------
@dataclass
class Individual:
    sd: dict
    id: int
    parent: int | None
    gen: int
    level: str
    fitness: float = -1.0


def initial_population(cfg: EvoConfig, seed_sd: dict, rng: random.Random) -> list[Individual]:
    """Población inicial: el modelo semilla intacto + variantes mutadas."""
    pop = [Individual(sd=seed_sd, id=0, parent=None, gen=0, level="seed")]
    for i in range(1, cfg.population):
        lvl_name = pick_level(cfg.mutation_probs, rng)
        child = mutate_state_dict(seed_sd, cfg.mutation_levels[lvl_name],
                                  seed=rng.randrange(1 << 31),
                                  mutate_value_head=cfg.mutate_value_head,
                                  norm_threshold=cfg.mutation_norm_threshold)
        pop.append(Individual(sd=child, id=i, parent=0, gen=0, level=lvl_name))
    return pop


# ----------------------------------------------------------------------
# Datos
# ----------------------------------------------------------------------
def _harvest_one(n: int, target: int, seed: int, cfg: EvoConfig, exclude: set) -> list[Puzzle]:
    got = harvest({n: target}, seed=seed, exclude=exclude, min_turn=cfg.min_turn,
                  solver_max_nodes=cfg.solver_max_nodes, games_budget=cfg.games_budget,
                  should_stop=T._STOP.is_set)
    return got.get(n, [])


def build_data(cfg: EvoConfig) -> tuple[dict, dict, dict]:
    """Genera (eval_pool, test, train) por nivel. Test y eval nunca se usan para entrenar.

    ``eval_pool`` es un pool grande de validación del que se sortea, por generación,
    el subconjunto de selección y el de early-stop: al rotar evita que la población
    memorice un set fijo (el fallo del maratón anterior).
    """
    exclude: set = set()
    evalset: dict[int, list[Puzzle]] = {}
    testset: dict[int, list[Puzzle]] = {}
    trainset: dict[int, list[Puzzle]] = {}

    T.log_phase("[GEN]", f"generando pool de validación mate en {list(cfg.levels)}...")
    for n in cfg.levels:
        ps = _harvest_one(n, cfg.eval_pool_size, cfg.seed + 1000 + n, cfg, exclude)
        evalset[n] = ps
        exclude.update(p.key for p in ps)
        T.log_phase("[GEN]", f"  eval mate {n}: {len(ps)}/{cfg.eval_pool_size}")
    if T._STOP.is_set():
        return evalset, testset, trainset

    T.log_phase("[GEN]", f"generando test fresco mate en {list(cfg.levels)}...")
    for n in cfg.levels:
        ps = _harvest_one(n, cfg.test_size, cfg.seed + 9000 + n, cfg, exclude)
        testset[n] = ps
        exclude.update(p.key for p in ps)
        T.log_phase("[GEN]", f"  test mate {n}: {len(ps)}/{cfg.test_size}")
    if T._STOP.is_set():
        return evalset, testset, trainset

    T.log_phase("[GEN]", f"generando entrenamiento mate en {list(cfg.levels)}...")
    for n in cfg.levels:
        ps = _harvest_one(n, cfg.puzzles_per_level, cfg.seed + 2000 + n, cfg, exclude)
        trainset[n] = ps
        exclude.update(p.key for p in ps)
        T.log_phase("[GEN]", f"  train mate {n}: {len(ps)}/{cfg.puzzles_per_level}")
    return evalset, testset, trainset


def fitness_of(net: SuperTatetiNet, sets: dict, levels) -> float:
    """Acierto medio (mate en N) sobre los sets indicados."""
    accs = [T.accuracy(net, sets.get(n, [])) for n in levels if sets.get(n)]
    return float(np.mean(accs)) if accs else 0.0


def sample_eval(pool: dict, size: int, rng: random.Random,
                exclude: set | None = None) -> dict:
    """Sortea ``size`` puzzles por nivel del ``eval_pool`` (validación rotativa).

    Si ``exclude`` trae las claves ya usadas (p. ej. por el set de selección), los
    sorteos nuevos evitan solaparse con ellas. El resultado cambia en cada llamada,
    así que la población no puede memorizar un set fijo.
    """
    if size <= 0:
        return {}
    taken = exclude if exclude is not None else set()
    out: dict[int, list[Puzzle]] = {}
    for n, ps in pool.items():
        if not ps:
            continue
        k = min(size, len(ps))
        out[n] = []
        attempts = 0
        while len(out[n]) < k and attempts < 10 * len(ps):
            p = ps[rng.randrange(len(ps))]
            attempts += 1
            if p.key in taken:
                continue
            taken.add(p.key)
            out[n].append(p)
    return out


def bootstrap(pool: list[Puzzle], frac: float, rng: random.Random) -> list[Puzzle]:
    """Muestra con reemplazo una fracción del pool (bagging por individuo)."""
    if not pool:
        return []
    k = max(1, int(round(len(pool) * frac)))
    return [pool[rng.randrange(len(pool))] for _ in range(k)]


# ----------------------------------------------------------------------
# Búsqueda local (gradiente)
# ----------------------------------------------------------------------
def local_search(sd: dict, cfg: EvoConfig, pool: list[Puzzle], stop_sets: dict,
                 rng: random.Random) -> tuple[dict, float, int]:
    """Entrena un individuo ``epochs_per_gen`` epochs y devuelve su mejor estado.

    Evalúa en ``stop_sets`` (un conjunto de early-stop **disjunto** del de selección)
    tras cada epoch y guarda el mejor: si el gradiente empieza a sobreajustar, se
    queda con el punto óptimo. El ``fitness`` de competencia lo calcula luego ``run``
    sobre el set de selección, no sobre este.
    """
    if cfg.epochs_per_gen <= 0 or not pool:
        net = SuperTatetiNet()
        net.load_state_dict(sd)
        return sd, fitness_of(net, stop_sets, cfg.levels), 0

    net = SuperTatetiNet()
    net.load_state_dict(sd)
    best_sd = {k: v.detach().clone() for k, v in sd.items()}
    best_acc = fitness_of(net, stop_sets, cfg.levels)
    best_ep = 0
    trainer = T.PuzzleTrainer(net, lr=cfg.lr, value_coef=cfg.value_coef, seed=rng.randrange(1 << 31))
    for ep in range(1, cfg.epochs_per_gen + 1):
        if T._STOP.is_set():
            break
        batch = bootstrap(pool, cfg.bootstrap_frac, rng)
        trainer.train_epoch(batch, cfg.batch_size)
        acc = fitness_of(net, stop_sets, cfg.levels)
        if acc > best_acc:
            best_acc = acc
            best_sd = T.clone_state(net)
            best_ep = ep
    return best_sd, best_acc, best_ep


def next_generation(ranked: list[Individual], cfg: EvoConfig, gen: int,
                    rng: random.Random) -> list[Individual]:
    """Supervivientes intactos + hijos mutados con paternidad por puesto.

    Garantiza al menos ``min_large_frac`` de los hijos con mutación ``large``
    (inmigración) para frenar el colapso de diversidad de la población.
    """
    survivors = ranked[:cfg.survivors]
    nxt = [replace(s, gen=gen) for s in survivors]
    next_id = max(p.id for p in ranked) + 1
    policy = policy_tuple(cfg)

    plan: list[tuple[Individual, str]] = []
    for rank, surv in enumerate(survivors):
        for _ in range(policy[rank]):
            plan.append((surv, pick_level(cfg.mutation_probs, rng)))

    n_large = sum(1 for _, lvl in plan if lvl == "large")
    wanted_large = int(np.ceil(cfg.min_large_frac * len(plan)))
    for i in range(len(plan)):
        if n_large >= wanted_large:
            break
        if plan[i][1] != "large":
            plan[i] = (plan[i][0], "large")
            n_large += 1

    for surv, lvl_name in plan:
        child_sd = mutate_state_dict(surv.sd, cfg.mutation_levels[lvl_name],
                                     seed=rng.randrange(1 << 31),
                                     mutate_value_head=cfg.mutate_value_head,
                                     norm_threshold=cfg.mutation_norm_threshold)
        nxt.append(Individual(sd=child_sd, id=next_id, parent=surv.id, gen=gen, level=lvl_name))
        next_id += 1
    return nxt


# ----------------------------------------------------------------------
# Reporte
# ----------------------------------------------------------------------
def build_report(cfg: EvoConfig, best: dict, history: list[dict], test_accs: dict,
                 best_test: dict | None = None) -> dict:
    report = {
        "config": cfg.snapshot(),
        "best_fitness": best["fitness"],
        "best_gen": best["gen"],
        "best_id": best["id"],
        "best_lineage": best["lineage"],
        "generations_run": len(history),
        "test_accuracy": {str(k): v for k, v in test_accs.items()},
        "history": history,
    }
    if best_test is not None:
        report["best_test_fitness"] = best_test["fitness"]
        report["best_test_gen"] = best_test["gen"]
        report["best_test_id"] = best_test["id"]
    return report


def write_report(report: dict, path: Path, announce: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    if announce:
        T.log_print(f"Reporte guardado en {path}")


def print_summary(cfg: EvoConfig, report: dict) -> None:
    T.log_print("=" * 60)
    T.log_print("RESUMEN EVOLUTIVO")
    T.log_print("=" * 60)
    T.log_print(f"  generaciones: {report['generations_run']}  población: {cfg.population}  "
                f"supervivientes: {cfg.survivors}")
    T.log_print(f"  mejor acierto (validación rotativa): {report['best_fitness'] * 100:.1f}% "
                f"(gen {report['best_gen']}, id {report['best_id']})")
    if report["test_accuracy"]:
        last_gen = max(int(k) for k in report["test_accuracy"])
        T.log_print(f"  acierto en test fresco (gen {last_gen}): "
                    f"{report['test_accuracy'][str(last_gen)] * 100:.1f}%")
    if report.get("best_test_id") is not None:
        T.log_print(f"  mejor test honesto: {report['best_test_fitness'] * 100:.1f}% "
                    f"(gen {report['best_test_gen']}, id {report['best_test_id']})")
    T.log_print(f"  modelo (mejor val): {cfg.out}")
    if cfg.out_test is not None:
        T.log_print(f"  modelo (mejor test): {cfg.out_test}")


# ----------------------------------------------------------------------
# Bucle principal
# ----------------------------------------------------------------------
def run(cfg: EvoConfig) -> int:
    validate(cfg)
    T._STOP.clear()
    T._install_signal_handlers()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    out = Path(cfg.out)
    report_path = Path(cfg.report) if cfg.report else out.with_name(f"{out.stem}_report.json")
    rng = random.Random(cfg.seed)

    if not Path(cfg.seed_model).exists():
        raise FileNotFoundError(f"No existe el modelo semilla: {cfg.seed_model}")
    seed_sd = torch.load(cfg.seed_model, weights_only=True, map_location="cpu")

    T.log_phase("[INFO]", "Configuración evolutiva:")
    T.log_phase("[INFO]", f"  semilla={cfg.seed_model}  out={out}  reporte={report_path}")
    T.log_phase("[INFO]", f"  población={cfg.population}  supervivientes={cfg.survivors}  "
               f"hijos={cfg.children}  generaciones={cfg.generations}")
    T.log_phase("[INFO]", f"  epochs/gen={cfg.epochs_per_gen}  lr={cfg.lr}  batch={cfg.batch_size}  "
               f"bootstrap={cfg.bootstrap_frac}  niveles={list(cfg.levels)}")
    T.log_phase("[INFO]", f"  mutación (probs)={cfg.mutation_probs}  "
               f"norm_threshold={cfg.mutation_norm_threshold}")

    eval_pool, testset, trainset = build_data(cfg)
    pool = [p for n in cfg.levels for p in trainset.get(n, [])]
    if not pool:
        T.log_phase("[INFO]", "no se generaron puzzles de entrenamiento; se detiene.")
        return 1
    T.log_phase("[INFO]", f"pool de entrenamiento: {len(pool)} puzzles | "
               f"pool de validación: {sum(len(eval_pool.get(n, [])) for n in cfg.levels)} | "
               f"test: {sum(len(testset.get(n, [])) for n in cfg.levels)}")
    T.log_phase("[INFO]", f"  selección: {cfg.eval_size}/gen (rotativo)  "
               f"early-stop: {cfg.early_stop_size}/gen (disjunto)  test_every={cfg.test_every}  "
               f"min_large_frac={cfg.min_large_frac}  patience={cfg.patience}")

    pop = initial_population(cfg, seed_sd, rng)
    registry: dict[int, Individual] = {p.id: p for p in pop}
    best = {"fitness": -1.0, "gen": 0, "id": None, "sd": None, "lineage": []}
    best_test = {"fitness": -1.0, "gen": 0, "id": None, "sd": None}
    history: list[dict] = []
    test_accs: dict[int, float] = {}
    best_test_seen = -1.0
    gens_since_test_improved = 0

    for gen in range(1, cfg.generations + 1):
        if T._STOP.is_set():
            T.log_phase("[EVO]", "parada solicitada: se corta antes de la generación.")
            break
        T.log_phase("[EVO]", f"===== Generación {gen}/{cfg.generations} =====")
        t0 = time.time()

        gen_eval = sample_eval(eval_pool, cfg.eval_size, rng)
        stop_set = sample_eval(eval_pool, cfg.early_stop_size, rng,
                               exclude={p.key for ps in gen_eval.values() for p in ps})

        for ind in pop:
            if T._STOP.is_set():
                break
            sd, _, ep = local_search(ind.sd, cfg, pool, stop_set, rng)
            ind.sd = sd
            ind.fitness = fitness_of(_net_from(sd), gen_eval, cfg.levels)
            T.log_phase("[TRAIN]", f"  gen {gen} | id {ind.id:3d} | padre {str(ind.parent):>4} | "
                       f"mut {ind.level:<6} | acierto {ind.fitness * 100:5.1f}% | +{ep} ep")

        if T._STOP.is_set():
            ranked = sorted(pop, key=lambda x: x.fitness, reverse=True)
            row = _gen_row(gen, ranked, best)
            history.append(row)
            write_report(build_report(cfg, best, history, test_accs, best_test), report_path)
            break

        ranked = sorted(pop, key=lambda x: x.fitness, reverse=True)
        champ = ranked[0]

        if champ.fitness > best["fitness"]:
            best = {"fitness": champ.fitness, "gen": gen, "id": champ.id, "sd": champ.sd,
                    "lineage": _lineage(registry, champ)}
            T.save_net(_net_from(champ.sd), out)

        champ_test = None
        if gen % cfg.test_every == 0 and testset:
            test_accs[gen] = fitness_of(_net_from(best["sd"]), testset, cfg.levels)
            T.log_phase("[EVO]", f"  test fresco del mejor HISTÓRICO gen {gen}: {test_accs[gen] * 100:.1f}% "
                       f"(validación rotativa {best['fitness'] * 100:.1f}%; es siempre el mismo modelo "
                       f"hasta que aparezca uno mejor)")
            champ_test = fitness_of(_net_from(champ.sd), testset, cfg.levels)
            T.log_phase("[EVO]", f"  test fresco del CAMPEÓN de esta gen: {champ_test * 100:.1f}% "
                       f"(brecha contra su validación: {(champ.fitness - champ_test) * 100:+.1f})")
            if test_accs[gen] > best_test_seen + 1e-9:
                best_test_seen = test_accs[gen]
                best_test = {"fitness": test_accs[gen], "gen": gen, "id": best["id"],
                             "sd": best["sd"]}
                if cfg.out_test is not None:
                    T.save_net(_net_from(best_test["sd"]), Path(cfg.out_test))
                gens_since_test_improved = 0
            else:
                gens_since_test_improved += 1

        row = _gen_row(gen, ranked, best)
        row["test_accuracy"] = test_accs.get(gen)
        row["champion_test_accuracy"] = champ_test
        row["seconds"] = round(time.time() - t0, 1)
        history.append(row)
        write_report(build_report(cfg, best, history, test_accs, best_test), report_path)

        T.log_phase("[EVO]", f"  gen {gen} | mejor {champ.fitness * 100:5.1f}% | "
                   f"histórico {best['fitness'] * 100:5.1f}% (gen {best['gen']}) | "
                   f"media {np.mean([p.fitness for p in pop]) * 100:5.1f}% | "
                   f"diversidad {row['diversity']} | {row['seconds']:.1f}s")

        if cfg.patience > 0 and gens_since_test_improved >= cfg.patience:
            T.log_phase("[EVO]", f"  sin mejora en test por {cfg.patience} generaciones: se corta.")
            break

        pop = next_generation(ranked, cfg, gen + 1, rng)
        registry.update({p.id: p for p in pop})

    if best["sd"] is not None:
        T.save_net(_net_from(best["sd"]), out)
    report = build_report(cfg, best, history, test_accs, best_test)
    print_summary(cfg, report)
    write_report(report, report_path, announce=True)
    if T._STOP.is_set():
        T.log_phase("[INFO]", "entrenamiento evolutivo detenido por el usuario. Reporte guardado.")
    return 0


def _net_from(sd: dict) -> SuperTatetiNet:
    net = SuperTatetiNet()
    net.load_state_dict(sd)
    return net


def _lineage(registry: dict, ind: Individual) -> list:
    """Camino de ids desde la raíz hasta ``ind`` (id -> padre)."""
    path = []
    cur = ind
    while cur is not None:
        path.append({"id": cur.id, "gen": cur.gen, "parent": cur.parent, "level": cur.level,
                     "fitness": round(cur.fitness, 4)})
        cur = registry.get(cur.parent) if cur.parent is not None else None
    return path[::-1]


def _gen_row(gen: int, ranked: list[Individual], best: dict) -> dict:
    top = ranked[:10]
    return {
        "gen": gen,
        "best_fitness": ranked[0].fitness,
        "mean_fitness": float(np.mean([p.fitness for p in ranked])),
        "diversity": len({p.parent for p in top if p.parent is not None}),
        "top": [{"id": p.id, "parent": p.parent, "level": p.level,
                 "fitness": round(p.fitness, 4)} for p in top],
        "historical_best": best["fitness"],
    }


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> EvoConfig:
    p = argparse.ArgumentParser(
        prog="python -m trainPuzzles.evolve_puzzles",
        description="Entrena SuperTatetiNet con un ciclo evolutivo memético sobre puzzles de mate en N.")
    p.add_argument("--config", default=None, help="JSON de configuración (los demás flags lo sobrescriben)")
    p.add_argument("--seed-model", dest="seed_model", default=argparse.SUPPRESS)
    p.add_argument("--out", default=argparse.SUPPRESS)
    p.add_argument("--out-test", dest="out_test", default=argparse.SUPPRESS)
    p.add_argument("--report", default=argparse.SUPPRESS)
    p.add_argument("--population", type=int, default=argparse.SUPPRESS)
    p.add_argument("--survivors", type=int, default=argparse.SUPPRESS)
    p.add_argument("--generations", type=int, default=argparse.SUPPRESS)
    p.add_argument("--epochs-per-gen", dest="epochs_per_gen", type=int, default=argparse.SUPPRESS)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=argparse.SUPPRESS)
    p.add_argument("--lr", type=float, default=argparse.SUPPRESS)
    p.add_argument("--value-coef", dest="value_coef", type=float, default=argparse.SUPPRESS)
    p.add_argument("--bootstrap-frac", dest="bootstrap_frac", type=float, default=argparse.SUPPRESS)
    p.add_argument("--mutation-probs", dest="mutation_probs", default=argparse.SUPPRESS,
                   help='"small:0.5,medium:0.35,large:0.15"')
    p.add_argument("--mutate-value-head", dest="mutate_value_head", action="store_true",
                   default=argparse.SUPPRESS)
    p.add_argument("--mutation-norm-threshold", dest="mutation_norm_threshold", type=float,
                   default=argparse.SUPPRESS)
    p.add_argument("--levels", type=int, nargs="+", default=argparse.SUPPRESS)
    p.add_argument("--puzzles-per-level", dest="puzzles_per_level", type=int, default=argparse.SUPPRESS)
    p.add_argument("--eval-pool-size", dest="eval_pool_size", type=int, default=argparse.SUPPRESS)
    p.add_argument("--eval-size", dest="eval_size", type=int, default=argparse.SUPPRESS)
    p.add_argument("--early-stop-size", dest="early_stop_size", type=int, default=argparse.SUPPRESS)
    p.add_argument("--test-size", dest="test_size", type=int, default=argparse.SUPPRESS)
    p.add_argument("--test-every", dest="test_every", type=int, default=argparse.SUPPRESS)
    p.add_argument("--min-large-frac", dest="min_large_frac", type=float, default=argparse.SUPPRESS)
    p.add_argument("--patience", type=int, default=argparse.SUPPRESS)
    p.add_argument("--min-turn", dest="min_turn", type=int, default=argparse.SUPPRESS)
    p.add_argument("--solver-max-nodes", dest="solver_max_nodes", type=int, default=argparse.SUPPRESS)
    p.add_argument("--games-budget", dest="games_budget", type=int, default=argparse.SUPPRESS)
    p.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    args = p.parse_args(argv)

    cfg = load_config(Path(args.config)) if args.config else EvoConfig()
    for key, value in vars(args).items():
        if key == "config":
            continue
        setattr(cfg, key, value)
    if "mutation_probs" in vars(args):
        cfg.mutation_probs = {k: float(v) for k, v in
                              (pz.split(":") for pz in cfg.mutation_probs.split(","))}
    if isinstance(cfg.levels, list):
        cfg.levels = tuple(cfg.levels)
    return cfg


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())