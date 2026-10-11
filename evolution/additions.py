"""Añadir IA a mano a la PRÓXIMA temporada (21+ equipos solo en esa temporada).

Flujo: ``register_additions`` valida el .pth, le asigna un color único, lo guarda en ``models/`` como
``<color>.pth`` (+ .json) y lo anota para la temporada siguiente a la última creada. Cuando el
entrenamiento crea esa temporada, suma estas IA a las 20 de la generación. Después la composición
vuelve a ser de 20 (7 supervivientes + 13 descendientes), vengan o no de las añadidas.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import colors as colormod
from . import modelstore, repo, rules
from .config import Config
from .rng import derive_rng

ADD_LOCK_KEY = 0x7A7E7E2   # serializa `addai` con la creación de temporadas (distinto del bloqueo de entrenamiento)


class AdditionError(Exception):
    """Error que se le muestra a la persona."""


def pending_ids(conn, season_number: int) -> list[int]:
    return [r["individual_id"] for r in conn.execute(
        "SELECT individual_id FROM season_additions WHERE season_number=%s ORDER BY id", (season_number,))]


def list_pending(conn) -> list[dict]:
    """IA añadidas cuya temporada todavía no fue creada."""
    return conn.execute(
        """SELECT a.season_number, i.color, a.source, a.added_at FROM season_additions a
           JOIN individuals i ON i.id = a.individual_id
           WHERE NOT EXISTS (SELECT 1 FROM seasons s WHERE s.number = a.season_number)
           ORDER BY a.season_number, a.id""").fetchall()


def _resolve_source(cfg: Config, raw: str) -> Path:
    p = Path(raw.strip().strip('"').strip("'")).expanduser()
    for cand in (p, cfg.models_dir / p):
        if cand.is_file():
            return cand.resolve()
    raise AdditionError(f"No se encuentra el archivo '{raw}' (se buscó tal cual y dentro de {cfg.models_dir}).")


def register_additions(db, cfg: Config, raw_paths: list[str]) -> dict:
    """Valida y registra IA para la próxima temporada. Devuelve {target, added, total_extra, format}."""
    if not raw_paths:
        raise AdditionError("Indicá al menos un archivo .pth.")
    loaded = []
    seen: set[str] = set()
    for raw in raw_paths:
        src = _resolve_source(cfg, raw)
        try:
            sd = modelstore.load_state_dict(src)   # verifica que sea un state_dict de SuperTatetiNet
        except Exception as e:  # noqa: BLE001
            raise AdditionError(f"{src.name} no es un modelo válido de SuperTatetiNet: {e}") from e
        digest = modelstore.weights_hash(sd)
        if digest in seen:
            raise AdditionError(f"{src.name} está repetido en el pedido (mismos pesos).")
        seen.add(digest)
        loaded.append((src, digest))

    def tx(conn):
        conn.execute("SELECT pg_advisory_xact_lock(%s)", (ADD_LOCK_KEY,))
        last = conn.execute("SELECT max(number) AS n FROM seasons").fetchone()["n"] or 0
        target = last + 1                       # temporada siguiente a la última creada
        already = conn.execute("SELECT count(*) AS n FROM season_additions WHERE season_number=%s", (target,)).fetchone()["n"]
        if already + len(loaded) > rules.MAX_EXTRA_TEAMS:
            raise AdditionError(f"Se pueden añadir como máximo {rules.MAX_EXTRA_TEAMS} IA por temporada "
                                f"(ya hay {already} para la temporada {target}).")
        dup = conn.execute(
            """SELECT i.color FROM season_additions a JOIN individuals i ON i.id=a.individual_id
               WHERE a.season_number=%s AND i.mutation_meta->>'source_weights_sha256' = ANY(%s)""",
            (target, [d for _, d in loaded])).fetchall()
        if dup:
            raise AdditionError(f"Esa IA ya está añadida a la temporada {target}: {dup[0]['color']}.")
        master = repo.get_master_seed(conn, cfg.seed)
        taken = repo.all_colors(conn)
        added = []
        for src, digest in loaded:
            c = colormod.spread_colors(1, derive_rng(master, "import", digest[:16]), taken)[0]
            taken.add(c.color)
            iid = repo.insert_individual(
                conn, color=c.color, origin_season=target, parent_id=None, kind="imported",
                mutation_meta={"source": str(src), "source_weights_sha256": digest}, color_meta=c.meta)
            conn.execute("INSERT INTO season_additions (season_number, individual_id, source) VALUES (%s,%s,%s)",
                         (target, iid, str(src)))
            added.append({"color": c.color, "source": str(src)})
        total = already + len(loaded)
        return {"target": target, "added": added, "total_extra": total}

    res = db.run(tx)
    repo.materialize_models(db, cfg)   # escribe models/<color>.pth + .json (sin sobrescribir)
    res["format"] = rules.season_format(rules.POPULATION + res["total_extra"])
    return res


def cancel_addition(db, cfg: Config, color: str) -> dict:
    """Quita una IA añadida cuya temporada todavía no se creó (borra su fila y sus archivos)."""
    color = color if color.startswith("#") else "#" + color

    def tx(conn):
        conn.execute("SELECT pg_advisory_xact_lock(%s)", (ADD_LOCK_KEY,))
        row = conn.execute(
            """SELECT a.id, a.individual_id, a.season_number, i.model_file FROM season_additions a
               JOIN individuals i ON i.id = a.individual_id WHERE i.color = %s""", (color.lower(),)).fetchone()
        if row is None:
            raise AdditionError(f"{color} no está entre las IA añadidas pendientes.")
        if conn.execute("SELECT 1 FROM seasons WHERE number=%s", (row["season_number"],)).fetchone():
            raise AdditionError(f"{color} ya forma parte de la temporada {row['season_number']}: no se puede quitar.")
        conn.execute("DELETE FROM season_additions WHERE id=%s", (row["id"],))
        conn.execute("DELETE FROM individuals WHERE id=%s", (row["individual_id"],))
        return row

    row = db.run(tx)
    path = cfg.models_dir / row["model_file"]
    for f in (path, modelstore.sidecar_path(path)):
        if f.parent == cfg.models_dir and f.exists():
            f.unlink()
    return {"color": color.lower(), "season": row["season_number"]}