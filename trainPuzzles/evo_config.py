"""Configuración del entrenador evolutivo de puzzles (autónoma, sin `evolution/`).

Todo lo que necesita ``evolve_puzzles`` vive acá: estructura de los individuos,
niveles de mutación y parámetros del ciclo evolutivo. No hay colores, ni base de
datos, ni conceptos de liga/torneo: solo una población de redes que entrena por
gradiente y compite por acierto en puzzles de mate en N.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class MutationLevel:
    """Perturbación de pesos: cada peso muta con probabilidad ``rate`` sumándole
    ruido gaussiano de desvío ``sigma`` x (desvío actual de ese tensor)."""
    rate: float
    sigma: float


DEFAULT_MUTATION_LEVELS = {
    "small": MutationLevel(rate=0.10, sigma=0.05),   # refinar
    "medium": MutationLevel(rate=0.30, sigma=0.20),  # explorar variantes
    "large": MutationLevel(rate=0.60, sigma=0.60),   # buscar estrategias nuevas
}
# Probabilidad de que un hijo reciba cada nivel de mutación.
DEFAULT_MUTATION_PROBS = {"small": 0.50, "medium": 0.35, "large": 0.15}


def children_per_rank(survivors: int, children: int) -> tuple[int, ...]:
    """Reparte ``children`` hijos entre ``survivors`` puestos (el 1.º primero).

    Reparto parejo con el resto para los mejores puestos. Con 10 supervivientes y
    20 hijos da 2 por puesto.
    """
    if survivors <= 0:
        raise ValueError("survivors debe ser >= 1")
    if children < 0:
        raise ValueError("children debe ser >= 0")
    base, rem = divmod(children, survivors)
    return tuple(base + (1 if i < rem else 0) for i in range(survivors))


@dataclass
class EvoConfig:
    """Parámetros del ciclo evolutivo memético (gradiente + mutación + selección)."""
    # --- modelo y salida ---
    seed_model: Path = Path("trainPuzzles/my_model.pth")
    out: Path = Path("trainPuzzles/checkpoints/puzzle_canon.pth")  # canon: último terminado
    out_test: Path | None = None              # checkpoint elegido por el test honesto (diagnóstico)
    report: Path | None = None
    run_name: str = "evo"                     # etiqueta de la corrida archivada en checkpoints/runs/
    no_canon: bool = False                    # no copiar a puzzle_canon.pth (experimentos A/B)

    # --- población ---
    population: int = 30
    survivors: int = 10
    generations: int = 50
    parent_policy: tuple | None = None       # hijos por puesto; None = reparto parejo

    # --- búsqueda local (gradiente) ---
    epochs_per_gen: int = 10                 # epochs de gradiente por individuo y generación
    batch_size: int = 64
    lr: float = 3e-4
    value_coef: float = 0.0                  # 0 = off (todos los puzzles son ganados: la cabeza de valor aprendería una constante)
    bootstrap_frac: float = 1.0              # fracción del pool muestreada con reemplazo

    # --- mutación ---
    mutation_levels: dict = field(default_factory=lambda: dict(DEFAULT_MUTATION_LEVELS))
    mutation_probs: dict = field(default_factory=lambda: dict(DEFAULT_MUTATION_PROBS))
    mutate_value_head: bool = False
    mutation_norm_threshold: float = 4.0     # reescala si un linaje explota (0 = desactivado)

    # --- datos (puzzles mate en N) ---
    levels: tuple = (1,)
    puzzles_per_level: int = 25000           # con pocos puzzles la red los memoriza (train 99 %, validación 62 %)
    eval_pool_size: int = 4000               # pool grande de validación (del que se sortea cada gen)
    eval_size: int = 1000                    # subconjunto de selección, fresco por generación
    early_stop_size: int = 1000              # subconjunto de early-stop, disjunto del de selección
    test_size: int = 2000
    test_every: int = 5                      # cada cuántas generaciones se mide en el test fresco
    min_turn: int = 16
    solver_max_nodes: int = 120000
    games_budget: int = 8000

    # --- evolución ---
    min_large_frac: float = 0.25             # fracción mínima de hijos con mutación "large" (inmigración)
    patience: int = 12                       # corta si el test honesto no mejora en N generaciones (0 = off)

    seed: int = 0

    # ------------------------------------------------------------------
    def snapshot(self) -> dict:
        """Config serializable (para el reporte y para guardar/leer JSON)."""
        return {
            "seed_model": str(self.seed_model),
            "out": str(self.out),
            "out_test": None if self.out_test is None else str(self.out_test),
            "report": None if self.report is None else str(self.report),
            "run_name": self.run_name,
            "no_canon": self.no_canon,
            "population": self.population,
            "survivors": self.survivors,
            "generations": self.generations,
            "parent_policy": None if self.parent_policy is None else list(self.parent_policy),
            "epochs_per_gen": self.epochs_per_gen,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "value_coef": self.value_coef,
            "bootstrap_frac": self.bootstrap_frac,
            "mutation_levels": {k: vars(v) for k, v in self.mutation_levels.items()},
            "mutation_probs": dict(self.mutation_probs),
            "mutate_value_head": self.mutate_value_head,
            "mutation_norm_threshold": self.mutation_norm_threshold,
            "levels": list(self.levels),
            "puzzles_per_level": self.puzzles_per_level,
            "eval_pool_size": self.eval_pool_size,
            "eval_size": self.eval_size,
            "early_stop_size": self.early_stop_size,
            "test_size": self.test_size,
            "test_every": self.test_every,
            "min_turn": self.min_turn,
            "solver_max_nodes": self.solver_max_nodes,
            "games_budget": self.games_budget,
            "min_large_frac": self.min_large_frac,
            "patience": self.patience,
            "seed": self.seed,
        }

    @property
    def children(self) -> int:
        return self.population - self.survivors


def from_dict(data: dict) -> EvoConfig:
    """Reconstruye una ``EvoConfig`` desde un dict (JSON)."""
    kw = dict(data)
    if kw.get("seed_model") is not None:
        kw["seed_model"] = Path(kw["seed_model"])
    if kw.get("out") is not None:
        kw["out"] = Path(kw["out"])
    kw["out_test"] = Path(kw["out_test"]) if kw.get("out_test") else None
    kw["report"] = Path(kw["report"]) if kw.get("report") else None
    if kw.get("parent_policy") is not None:
        kw["parent_policy"] = tuple(kw["parent_policy"])
    if kw.get("levels") is not None:
        kw["levels"] = tuple(int(x) for x in kw["levels"])
    if kw.get("mutation_levels"):
        kw["mutation_levels"] = {k: MutationLevel(**v) for k, v in kw["mutation_levels"].items()}
    return EvoConfig(**kw)


def load_config(path: Path) -> EvoConfig:
    """Carga una configuración desde un JSON (los campos ausentes usan el default)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    base = EvoConfig().snapshot()
    base.update(data)
    return from_dict(base)


def save_config(cfg: EvoConfig, path: Path) -> None:
    """Guarda la configuración efectiva como JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg.snapshot(), indent=2, ensure_ascii=False), encoding="utf-8")


def validate(cfg: EvoConfig) -> None:
    """Valida los invariantes de la configuración (lanza ``ValueError``)."""
    if cfg.population < 2:
        raise ValueError("population debe ser >= 2")
    if not 1 <= cfg.survivors < cfg.population:
        raise ValueError("survivors debe cumplir 1 <= survivors < population")
    if cfg.generations < 1:
        raise ValueError("generations debe ser >= 1")
    if cfg.epochs_per_gen < 0:
        raise ValueError("epochs_per_gen debe ser >= 0")
    if cfg.batch_size < 1:
        raise ValueError("batch_size debe ser >= 1")
    if not 0.0 <= cfg.bootstrap_frac <= 1.0:
        raise ValueError("bootstrap_frac debe estar en [0, 1]")
    if set(cfg.mutation_levels) != {"small", "medium", "large"}:
        raise ValueError("mutation_levels debe definir small/medium/large")
    if set(cfg.mutation_probs) != {"small", "medium", "large"}:
        raise ValueError("mutation_probs debe definir small/medium/large")
    if abs(sum(cfg.mutation_probs.values()) - 1.0) > 1e-6:
        raise ValueError("mutation_probs debe sumar 1")
    for name, lvl in cfg.mutation_levels.items():
        if lvl.rate < 0 or lvl.sigma < 0:
            raise ValueError(f"mutation_levels[{name}] no admite valores negativos")
    if not cfg.levels:
        raise ValueError("levels no puede estar vacío")
    if any(int(n) < 1 for n in cfg.levels):
        raise ValueError("levels debe contener enteros >= 1")
    if cfg.test_every < 1:
        raise ValueError("test_every debe ser >= 1")
    if not 0.0 <= cfg.min_large_frac <= 1.0:
        raise ValueError("min_large_frac debe estar en [0, 1]")
    if cfg.patience < 0:
        raise ValueError("patience debe ser >= 0 (0 = desactivado)")
    if cfg.eval_pool_size < cfg.eval_size + cfg.early_stop_size:
        raise ValueError("eval_pool_size debe ser >= eval_size + early_stop_size")
    if cfg.mutation_norm_threshold < 0:
        raise ValueError("mutation_norm_threshold debe ser >= 0 (0 = desactivado)")
    if cfg.parent_policy is not None:
        if len(cfg.parent_policy) > cfg.survivors:
            raise ValueError("parent_policy no puede tener más puestos que survivors")
        if sum(cfg.parent_policy) != cfg.children:
            raise ValueError(f"parent_policy debe sumar {cfg.children} (recibido {sum(cfg.parent_policy)})")
        if any(int(n) < 0 for n in cfg.parent_policy):
            raise ValueError("parent_policy no admite valores negativos")


def policy_tuple(cfg: EvoConfig) -> tuple[int, ...]:
    """Hijos por puesto efectivos (usando ``parent_policy`` o reparto parejo)."""
    if cfg.parent_policy is not None:
        return tuple(cfg.parent_policy) + (0,) * (cfg.survivors - len(cfg.parent_policy))
    return children_per_rank(cfg.survivors, cfg.children)