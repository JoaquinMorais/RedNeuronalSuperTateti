"""Limpieza de modelos antiguos: se borra el .pth (peso en disco) y se conserva el .json.

Se borra el .pth de una IA solo si se cumplen TODAS estas condiciones:
  1. Fue expulsada: su última temporada terminó y no pasó a la siguiente generación.
  2. Pasaron al menos ``retention_seasons`` temporadas completas desde esa expulsión.
  3. No fue 1.ª de la tabla anual ni ganó una Liga, una Copa o una Recopa (en temporadas que cuentan: ver
     ``MODEL_PROTECT_EVERY``; con 1, todas).
Las IA que siguen compitiendo y los modelos de ``models/historical`` no se tocan nunca.
La fila de la base de datos y el .json (arquitectura, hash, linaje) se conservan.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from . import modelstore
from .config import Config

# Una red está protegida si fue la 1.ª RED de la tabla anual o ganó Liga/Copa/Recopa en una temporada que cuenta
# (todas si protect_every = 1; solo las múltiplo de K si K > 1: así los modelos conservados no crecen sin límite).
PROTECTED_SQL = """(
    EXISTS (SELECT 1 FROM season_participants q JOIN seasons sq ON sq.id = q.season_id
            WHERE q.individual_id = i.id
              AND q.final_position = (SELECT min(q2.final_position) FROM season_participants q2
                                      JOIN individuals i2 ON i2.id = q2.individual_id
                                      WHERE q2.season_id = q.season_id AND i2.ia_type = 'neuronal')
              AND (%(every)s <= 1 OR MOD(sq.number, %(every)s) = 0))
    OR EXISTS (SELECT 1 FROM seasons z WHERE i.id IN (z.league_champion_id, z.cup_champion_id, z.recopa_champion_id)
              AND (%(every)s <= 1 OR MOD(z.number, %(every)s) = 0))
)"""

CANDIDATES_SQL = """
WITH last AS (
    SELECT p.individual_id, max(s.number) AS last_season
    FROM season_participants p JOIN seasons s ON s.id = p.season_id
    GROUP BY p.individual_id
)
SELECT i.id, i.color, i.model_file, i.model_sha256, l.last_season
FROM last l
JOIN seasons s ON s.number = l.last_season
JOIN individuals i ON i.id = l.individual_id
WHERE s.status = 'completed'
  AND NOT EXISTS (SELECT 1 FROM evolution_plan e WHERE e.season_id = s.id AND e.individual_id = i.id)   -- expulsada
  AND l.last_season <= %(completed)s - %(retention)s                                                    -- pasó el plazo
  AND i.ia_type = 'neuronal' AND i.model_sha256 IS NOT NULL AND i.model_pruned_at IS NULL
  AND NOT """ + PROTECTED_SQL + """
ORDER BY l.last_season, i.id
"""


def _drop_weights(path: Path) -> int:
    """Borra el .pth y anota en el .json que los pesos fueron eliminados. Devuelve bytes liberados."""
    freed = path.stat().st_size if path.exists() else 0
    side = modelstore.sidecar_path(path)
    if side.exists():
        info = json.loads(side.read_text(encoding="utf-8"))
        info["weights_deleted"] = True
        info["pruned_at"] = datetime.now(timezone.utc).isoformat()
        tmp = side.with_name(side.name + ".tmp")
        tmp.write_text(json.dumps(info, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, side)
    if path.exists():
        path.unlink()
    return freed


def prune_models(db, cfg: Config, completed_season: int, dry_run: bool = False) -> list[dict]:
    """Borra los .pth que cumplan las condiciones. Idempotente (si se corta, se repite sin problema)."""
    if cfg.retention_seasons <= 0:
        return []
    rows = db.run(lambda c: c.execute(
        CANDIDATES_SQL, {"completed": completed_season, "retention": cfg.retention_seasons,
                         "every": cfg.protect_every}).fetchall())
    done = []
    for r in rows:
        path = cfg.models_dir / r["model_file"]
        if path.parent != cfg.models_dir:  # seguridad: solo archivos directos de models/
            continue
        freed = path.stat().st_size if (dry_run and path.exists()) else 0
        if not dry_run:
            freed = _drop_weights(path)
            db.run(lambda c, i=r["id"]: c.execute(
                "UPDATE individuals SET model_pruned_at=now() WHERE id=%s AND model_pruned_at IS NULL", (i,)))
        done.append({"color": r["color"], "last_season": r["last_season"], "file": r["model_file"], "bytes": freed})
    return done


def compact_database(db, completed_season: int, after: int) -> dict:
    """Libera espacio en la base de datos en temporadas viejas (más de ``after`` temporadas atrás):
    borra las jugadas guardadas de cada partida y el historial fecha a fecha de la Liga (queda la última
    fecha, que usan las vistas). Los resultados, tablas finales, linajes y puntajes NO se tocan."""
    if after <= 0:
        return {}
    limit = completed_season - after

    def tx(conn):
        moves = conn.execute(
            """UPDATE matches SET moves = NULL WHERE moves IS NOT NULL
               AND season_id IN (SELECT id FROM seasons WHERE status='completed' AND number <= %s)""", (limit,)).rowcount
        hist = conn.execute(
            """DELETE FROM league_history h USING seasons s WHERE h.season_id = s.id AND s.status='completed'
               AND s.number <= %s AND h.matchday < s.league_matchdays""", (limit,)).rowcount
        return {"moves_cleared": moves, "history_rows_deleted": hist}
    return db.run(tx)