"""Configuración por variables de entorno (+ archivo JSON opcional para mutaciones/colores)."""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .colors import DEFAULT_TIERS, ColorTier
from .rules import CHILDREN, POPULATION


@dataclass(frozen=True)
class MutationLevel:
    """Una mutación perturba cada peso con probabilidad ``rate`` sumándole ruido
    gaussiano de desvío ``sigma`` x (desvío de ese tensor)."""
    rate: float
    sigma: float


DEFAULT_MUTATION_LEVELS = {
    "small": MutationLevel(rate=0.10, sigma=0.05),    # refinar
    "medium": MutationLevel(rate=0.30, sigma=0.20),   # explorar variantes
    "large": MutationLevel(rate=0.60, sigma=0.60),    # estrategias nuevas
}
# Probabilidad de que un descendiente reciba cada nivel de mutación
DEFAULT_LEVEL_PROBS = {"small": 0.50, "medium": 0.35, "large": 0.15}


@dataclass(frozen=True)
class ParentPolicy:
    """Hijos por puesto de la TABLA ANUAL (Copa + Liga + Recopa); el primer valor es el 1.º.

    Por defecto: 1.º 5 hijos, 2.º 3, 3.º 2, 4.º 2, 5.º 1, 6.º y 7.º ninguno (suman 13).
    Los 7 primeros siempre se conservan intactos. Debe sumar exactamente 13.
    """
    children_per_rank: tuple = (5, 3, 2, 2, 1, 0, 0)


@dataclass(frozen=True)
class Config:
    database_url: str = "postgresql://tateti:tateti@localhost:5432/tateti"
    models_dir: Path = Path("models")
    seed: int | None = None
    max_seasons: int = 0                 # 0 = sin límite
    initial_population: str = "auto"     # auto | seed | random
    seed_model_path: Path = Path("model.pth")
    mutation_levels: dict = field(default_factory=lambda: dict(DEFAULT_MUTATION_LEVELS))
    level_probs: dict = field(default_factory=lambda: dict(DEFAULT_LEVEL_PROBS))
    mutate_value_head: bool = False      # la cabeza de valor no interviene en la jugada
    mutation_norm_threshold: float = 4.0  # reescala (sin cambiar la jugada) los pesos de un padre que crecieron > N veces; 0 = no
    color_tiers: dict = field(default_factory=lambda: dict(DEFAULT_TIERS))
    parent_policy: ParentPolicy = field(default_factory=ParentPolicy)
    retention_seasons: int = 10          # temporadas que se guarda el .pth de una IA expulsada (0 = nunca borrar)
    protect_every: int = 1               # 1 = se conservan TODOS los campeones/1.º de tabla anual; K = solo los de temporadas múltiplo de K
    compact_after: int = 0               # >0: en temporadas más viejas que N se borran las jugadas y el historial de fechas de la BD
    champions_top: int = 30              # filas del ranking de campeones
    bench_every: int = 5                 # cada cuántas temporadas se mide contra los bots fijos (0 = nunca)
    bench_games: int = 120               # partidas del 1.º de la tabla anual contra cada bot
    bench_pop_games: int = 8             # partidas de CADA IA de la población contra cada bot (0 = solo el 1.º)
    output: str = "full"                 # full | fecha | temporada (cuánto imprime el entrenamiento)
    play_host: str = "0.0.0.0"
    play_port: int = 8765
    db_retries: int = 8
    db_retry_base_delay: float = 0.5
    color: bool = False

    def snapshot(self) -> dict:
        """Parte de la configuración que influye en el resultado (se guarda con la temporada)."""
        return {
            "mutation_levels": {k: vars(v) for k, v in self.mutation_levels.items()},
            "level_probs": self.level_probs,
            "mutate_value_head": self.mutate_value_head,
            "mutation_norm_threshold": self.mutation_norm_threshold,
            "color_tiers": {k: vars(v) for k, v in self.color_tiers.items()},
            "parent_policy": {"children_per_rank": list(self.parent_policy.children_per_rank)},
        }


def _use_color(stream=None) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    stream = stream or sys.stdout
    return bool(getattr(stream, "isatty", lambda: False)()) and os.environ.get("TERM") != "dumb"


def load_config(env: dict | None = None) -> Config:
    env = os.environ if env is None else env
    kw: dict = {}
    if "DATABASE_URL" in env:
        kw["database_url"] = env["DATABASE_URL"]
    if "MODELS_DIR" in env:
        kw["models_dir"] = Path(env["MODELS_DIR"])
    if env.get("SEED"):
        kw["seed"] = int(env["SEED"])
    if "MAX_SEASONS" in env:
        kw["max_seasons"] = int(env["MAX_SEASONS"])
    if "INITIAL_POPULATION" in env:
        kw["initial_population"] = env["INITIAL_POPULATION"]
    if "SEED_MODEL_PATH" in env:
        kw["seed_model_path"] = Path(env["SEED_MODEL_PATH"])

    if "MODEL_RETENTION_SEASONS" in env:
        kw["retention_seasons"] = int(env["MODEL_RETENTION_SEASONS"])
    if "MODEL_PROTECT_EVERY" in env:
        kw["protect_every"] = int(env["MODEL_PROTECT_EVERY"])
    if "DB_COMPACT_AFTER" in env:
        kw["compact_after"] = int(env["DB_COMPACT_AFTER"])
    if "CHAMPIONS_TOP" in env:
        kw["champions_top"] = int(env["CHAMPIONS_TOP"])
    for var, field in (("BENCH_EVERY", "bench_every"), ("BENCH_GAMES", "bench_games"), ("BENCH_POP_GAMES", "bench_pop_games")):
        if var in env:
            kw[field] = int(env[var])
    if "MUTATION_NORM_THRESHOLD" in env:
        kw["mutation_norm_threshold"] = float(env["MUTATION_NORM_THRESHOLD"])
    if "OUTPUT" in env:
        kw["output"] = env["OUTPUT"].strip().lower()
    if "PLAY_PORT" in env:
        kw["play_port"] = int(env["PLAY_PORT"])
    if "PLAY_HOST" in env:
        kw["play_host"] = env["PLAY_HOST"]

    path = env.get("EVO_CONFIG_FILE")
    if path:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if "mutation_levels" in data:
            kw["mutation_levels"] = {**DEFAULT_MUTATION_LEVELS,
                                     **{k: MutationLevel(**v) for k, v in data["mutation_levels"].items()}}
        if "level_probs" in data:
            kw["level_probs"] = {**DEFAULT_LEVEL_PROBS, **data["level_probs"]}
        if "mutate_value_head" in data:
            kw["mutate_value_head"] = bool(data["mutate_value_head"])
        if "color_tiers" in data:
            kw["color_tiers"] = {**DEFAULT_TIERS, **{k: ColorTier(**v) for k, v in data["color_tiers"].items()}}
        if "parent_policy" in data:
            kw["parent_policy"] = ParentPolicy(children_per_rank=tuple(data["parent_policy"]["children_per_rank"]))
    # Atajo por variable de entorno para la distribución de niveles: "small:0.5,medium:0.35,large:0.15"
    if env.get("MUTATION_PROBS"):
        kw["level_probs"] = {k: float(v) for k, v in (p.split(":") for p in env["MUTATION_PROBS"].split(","))}
    cfg = Config(color=_use_color(), **kw)
    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    if set(cfg.level_probs) != {"small", "medium", "large"} or abs(sum(cfg.level_probs.values()) - 1) > 1e-6:
        raise ValueError("level_probs debe tener small/medium/large y sumar 1")
    if set(cfg.mutation_levels) != {"small", "medium", "large"}:
        raise ValueError("mutation_levels debe definir small/medium/large")
    if cfg.output not in ("full", "fecha", "temporada"):
        raise ValueError("OUTPUT debe ser full, fecha o temporada")
    if min(cfg.bench_every, cfg.bench_games, cfg.bench_pop_games) < 0:
        raise ValueError("BENCH_EVERY, BENCH_GAMES y BENCH_POP_GAMES deben ser >= 0")
    if cfg.mutation_norm_threshold < 0:
        raise ValueError("MUTATION_NORM_THRESHOLD debe ser >= 0 (0 = desactivado)")
    if cfg.protect_every < 1 or cfg.compact_after < 0:
        raise ValueError("MODEL_PROTECT_EVERY debe ser >= 1 y DB_COMPACT_AFTER >= 0")
    if cfg.retention_seasons < 0 or cfg.champions_top < 1:
        raise ValueError("MODEL_RETENTION_SEASONS debe ser >= 0 y CHAMPIONS_TOP >= 1")
    cpr = cfg.parent_policy.children_per_rank
    if (not all(isinstance(n, int) and n >= 0 for n in cpr) or len(cpr) > POPULATION or sum(cpr) != CHILDREN):
        raise ValueError(f"parent_policy.children_per_rank debe sumar exactamente {CHILDREN} (recibido {list(cpr)})")
    if cfg.initial_population not in ("auto", "seed", "random"):
        raise ValueError("INITIAL_POPULATION debe ser auto, seed o random")