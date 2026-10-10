"""Configuración por variables de entorno (+ archivo JSON opcional para mutaciones/colores)."""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .colors import DEFAULT_TIERS, ColorTier


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
    """Selección de padres para los 13 descendientes."""
    power: float = 2.0                 # peso = max(puntos_totales, 0) ** power
    min_weight: float = 1e-6           # evita pesos 0 (siempre factible completar 13)
    max_children_per_parent: int = 5
    min_children_from_non_survivors: int = 2  # descendencia de IA fuera del top 7


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
    color_tiers: dict = field(default_factory=lambda: dict(DEFAULT_TIERS))
    parent_policy: ParentPolicy = field(default_factory=ParentPolicy)
    db_retries: int = 8
    db_retry_base_delay: float = 0.5
    color: bool = False

    def snapshot(self) -> dict:
        """Parte de la configuración que influye en el resultado (se guarda con la temporada)."""
        return {
            "mutation_levels": {k: vars(v) for k, v in self.mutation_levels.items()},
            "level_probs": self.level_probs,
            "mutate_value_head": self.mutate_value_head,
            "color_tiers": {k: vars(v) for k, v in self.color_tiers.items()},
            "parent_policy": vars(self.parent_policy),
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
            kw["parent_policy"] = ParentPolicy(**data["parent_policy"])
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
    if cfg.initial_population not in ("auto", "seed", "random"):
        raise ValueError("INITIAL_POPULATION debe ser auto, seed o random")
