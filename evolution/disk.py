"""¿Qué ocupa espacio? Modelos (por categoría), archivos huérfanos y base de datos."""
from __future__ import annotations

import re
from pathlib import Path

from . import prune
from .config import Config

_FILE = re.compile(r"^[0-9a-f]{6}\.(pth|json)$")


def _mb(n: float) -> str:
    return f"{n / 1e6:.1f} MB"


def referenced_files(conn) -> set[str]:
    """Nombres (relativos a models/) que la base de datos conoce: modelos, sus .json y los históricos."""
    names: set[str] = set()
    for r in conn.execute("SELECT model_file FROM individuals"):
        names.add(r["model_file"]); names.add(Path(r["model_file"]).with_suffix(".json").as_posix())
    for r in conn.execute("SELECT model_file FROM historical_best"):
        names.add(r["model_file"]); names.add(Path(r["model_file"]).with_suffix(".json").as_posix())
    return names


def find_orphans(conn, models_dir: Path, tmp_min_age_s: float = 3600) -> list[Path]:
    """Archivos de modelo (6 hex + .pth/.json, o históricos) que NINGUNA fila de la base de datos usa:
    sobras de reinicios de la base (`down -v`) o de escrituras cortadas. Los .tmp solo si tienen > 1 h."""
    import time
    known = referenced_files(conn)
    out: list[Path] = []
    if models_dir.exists():
        for p in models_dir.iterdir():
            if p.is_file() and _FILE.match(p.name) and p.name not in known:
                out.append(p)
            elif p.is_file() and p.suffix == ".tmp" and time.time() - p.stat().st_mtime > tmp_min_age_s:
                out.append(p)
        hist = models_dir / "historical"
        if hist.exists():
            for p in hist.iterdir():
                if p.is_file() and f"historical/{p.name}" not in known:
                    out.append(p)
    return sorted(out)


def delete_orphans(paths: list[Path], models_dir: Path) -> int:
    freed = 0
    for p in paths:
        if p.parent in (models_dir, models_dir / "historical") and p.exists():
            freed += p.stat().st_size
            p.unlink()
    return freed


def collect(conn, cfg: Config) -> dict:
    """Categorías de los .pth guardados según las reglas de limpieza (ver prune.py)."""
    completed = conn.execute("SELECT max(number) AS n FROM seasons WHERE status='completed'").fetchone()["n"] or 0
    rows = conn.execute(
        f"""WITH last AS (SELECT p.individual_id, max(s.number) AS last_season FROM season_participants p
                          JOIN seasons s ON s.id = p.season_id GROUP BY p.individual_id)
            SELECT i.model_file, l.last_season, s.status,
                   (s.status = 'completed' AND NOT EXISTS (SELECT 1 FROM evolution_plan e
                                                           WHERE e.season_id = s.id AND e.individual_id = i.id)) AS expelled,
                   {prune.PROTECTED_SQL} AS protected
            FROM individuals i LEFT JOIN last l ON l.individual_id = i.id LEFT JOIN seasons s ON s.number = l.last_season
            WHERE i.model_sha256 IS NOT NULL""", {"every": cfg.protect_every}).fetchall()
    cats = {"compitiendo": [0, 0], "conservadas (campeones / 1.º)": [0, 0], "en ventana (se borran luego)": [0, 0],
            "borrables ya": [0, 0], "sin competir aún": [0, 0]}
    for r in rows:
        path = cfg.models_dir / r["model_file"]
        if not path.exists():
            continue
        size = path.stat().st_size
        if r["last_season"] is None:
            k = "sin competir aún"
        elif not r["expelled"]:
            k = "compitiendo"
        elif r["protected"]:
            k = "conservadas (campeones / 1.º)"
        elif cfg.retention_seasons > 0 and completed - r["last_season"] >= cfg.retention_seasons:
            k = "borrables ya"
        else:
            k = "en ventana (se borran luego)"
        cats[k][0] += 1
        cats[k][1] += size
    hist = cfg.models_dir / "historical"
    hist_files = [p for p in hist.glob("*.pth")] if hist.exists() else []
    orphans = find_orphans(conn, cfg.models_dir)
    tables = conn.execute(
        """SELECT relname, pg_total_relation_size(c.oid) AS bytes FROM pg_class c
           JOIN pg_namespace n ON n.oid = c.relnamespace
           WHERE n.nspname = current_schema() AND c.relkind = 'r' ORDER BY bytes DESC LIMIT 4""").fetchall()
    return {
        "cats": cats, "historical": (len(hist_files), sum(p.stat().st_size for p in hist_files)),
        "orphans": [(p, p.stat().st_size) for p in orphans],
        "db_bytes": conn.execute("SELECT pg_database_size(current_database()) AS b").fetchone()["b"],
        "tables": [(r["relname"], r["bytes"]) for r in tables],
        "seasons": conn.execute("SELECT count(*) AS n FROM seasons WHERE status='completed'").fetchone()["n"],
    }


def render(out, d: dict, cfg: Config) -> None:
    out.banner("ESPACIO EN DISCO")
    total_n = sum(c[0] for c in d["cats"].values())
    total_b = sum(c[1] for c in d["cats"].values())
    out.section(f"Modelos ({cfg.models_dir}): {total_n} archivos .pth · {_mb(total_b)}")
    for name, (n, b) in d["cats"].items():
        if n:
            out.emit(f"  {name:<32} {n:>6} archivos  {_mb(b):>10}")
    n, b = d["historical"]
    out.emit(f"  {'históricos (historical/)':<32} {n:>6} archivos  {_mb(b):>10}   (nunca se borran)")
    out.emit(out.dim(f"  Reglas: se borra el .pth de IA expulsadas hace {cfg.retention_seasons}+ temporadas "
                     f"({'desactivado' if cfg.retention_seasons <= 0 else 'MODEL_RETENTION_SEASONS'}); campeones/1.º se conservan "
                     f"{'todos' if cfg.protect_every <= 1 else 'solo los de temporadas múltiplo de ' + str(cfg.protect_every)} (MODEL_PROTECT_EVERY)."))
    if d["orphans"]:
        ob = sum(b for _, b in d["orphans"])
        out.section(f"Archivos HUÉRFANOS: {len(d['orphans'])} · {_mb(ob)}")
        out.emit("  Existen en models/ pero ninguna IA de la base de datos actual los usa (típico tras `docker compose down -v`).")
        for p, b in d["orphans"][:5]:
            out.emit(f"    {p.name}  {_mb(b)}")
        if len(d["orphans"]) > 5:
            out.emit(f"    ... y {len(d['orphans']) - 5} más")
        out.emit("  Borrarlos: `python -m evolution prune --orphans --yes`  (sin --yes solo los lista).")
    else:
        out.emit("  Huérfanos: ninguno.")
    out.section(f"Base de datos: {_mb(d['db_bytes'])} tras {d['seasons']} temporadas")
    for name, b in d["tables"]:
        out.emit(f"  {name:<22} {_mb(b):>10}")
    if d["seasons"]:
        out.emit(out.dim(f"  ≈ {d['db_bytes'] / d['seasons'] / 1e6:.2f} MB por temporada. DB_COMPACT_AFTER=N borra jugadas/historial de temporadas viejas."))
    out.section("Fuera de este programa")
    out.emit("  Logs de Docker:  docker system df   ·   sudo du -sh /var/lib/docker/containers/*/*-json.log")
    out.emit("  Git:             du -sh .git   ·   git count-objects -vH   ·   git status --short | head")