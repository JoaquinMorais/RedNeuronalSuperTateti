"""Creación de la siguiente generación: 7 supervivientes intactos + 13 descendientes mutados."""
from __future__ import annotations

from pathlib import Path

from . import colors as colormod
from . import modelstore, mutation, rules
from .competitions import SeasonInfo
from .config import Config
from .repo import all_colors, insert_individual, mutation_meta_for
from .rng import derive_rng, derive_seed


def plan_evolution(conn, cfg: Config, s: SeasonInfo) -> list[dict]:
    """Decide y PERSISTE la composición de la próxima generación (idempotente).

    Los descendientes se registran (identidad, color único, padre, nivel y semilla de
    mutación) pero sus archivos se escriben después con ``repo.materialize_models``.
    Los padres no se modifican: cada hijo se genera como copia independiente.
    """
    season = conn.execute("SELECT stage FROM seasons WHERE id=%s", (s.id,)).fetchone()
    if season["stage"] != "evolution" or conn.execute(
            "SELECT 1 FROM evolution_plan WHERE season_id=%s LIMIT 1", (s.id,)).fetchone():
        return []
    rows = conn.execute("SELECT individual_id, total_score, final_position FROM season_participants "
                        "WHERE season_id=%s ORDER BY final_position", (s.id,)).fetchall()
    ranked = [s.color_of[r["individual_id"]] for r in rows]
    totals = {s.color_of[r["individual_id"]]: r["total_score"] for r in rows}
    survivors = ranked[:rules.SURVIVORS]

    pol = cfg.parent_policy
    parents = mutation.select_parents(
        ranked, totals, set(survivors), derive_rng(s.seed, "parents"),
        pol.power, pol.min_weight, pol.max_children_per_parent, pol.min_children_from_non_survivors)
    parents.sort(key=ranked.index)  # orden estable: del mejor padre al peor

    for pos, c in enumerate(survivors, start=1):
        conn.execute("INSERT INTO evolution_plan (season_id, position, role, individual_id) VALUES (%s,%s,'survivor',%s)",
                     (s.id, pos, s.id_of[c]))

    taken = all_colors(conn)
    children = []
    for k, parent_color in enumerate(parents, start=1):
        seed = derive_seed(s.seed, "child", k)
        level = mutation.pick_level(cfg.level_probs, derive_rng(seed, "level"))
        cc = colormod.mutate_color(parent_color, level, derive_rng(seed, "color"), taken, cfg.color_tiers)
        taken.add(cc.color)
        cid = insert_individual(
            conn, color=cc.color, origin_season=s.number + 1, parent_id=s.id_of[parent_color], kind="mutant",
            level=level, seed=seed, mutation_meta=mutation_meta_for(cfg, level), color_meta=cc.meta)
        conn.execute(
            """INSERT INTO evolution_plan (season_id, position, role, individual_id, parent_id, mutation_level, mutation_seed)
               VALUES (%s,%s,'child',%s,%s,%s,%s)""",
            (s.id, rules.SURVIVORS + k, cid, s.id_of[parent_color], level, seed))
        children.append({"color": cc.color, "parent": parent_color, "level": level, "seed": seed})
    return [{"type": "evolution_planned", "survivors": survivors, "children": children}]


def record_historical_best(db, cfg: Config, s: SeasonInfo) -> dict | None:
    """Guarda un modelo histórico del mejor individuo de la temporada si supera al
    mejor registrado hasta ahora (por puntos totales; ante igualdad se conserva el anterior).
    Es una COPIA nueva en ``models/historical``: no sobrescribe nada."""
    def read(conn):
        if conn.execute("SELECT 1 FROM historical_best WHERE season_id=%s", (s.id,)).fetchone():
            return None
        champ = conn.execute(
            """SELECT p.individual_id, p.total_score, i.model_file, i.model_sha256 FROM season_participants p
               JOIN individuals i ON i.id=p.individual_id WHERE p.season_id=%s AND p.final_position=1""", (s.id,)).fetchone()
        best = conn.execute("SELECT max(total_score) AS m FROM historical_best").fetchone()["m"]
        return champ if best is None or champ["total_score"] > best else None

    champ = db.run(read)
    if champ is None:
        return None
    color = s.color_of[champ["individual_id"]]
    src = cfg.models_dir / champ["model_file"]
    dst_rel = f"historical/season{s.number:04d}_{color.lstrip('#')}.pth"
    dst = cfg.models_dir / dst_rel
    if not dst.exists():
        modelstore.copy_model(src, dst)
    if modelstore.weights_hash(modelstore.load_state_dict(dst)) != champ["model_sha256"]:
        raise modelstore.ModelIntegrityError(f"El modelo histórico {dst} no coincide con el original")
    db.run(lambda c: c.execute(
        "INSERT INTO historical_best (season_id, individual_id, total_score, model_file) VALUES (%s,%s,%s,%s) ON CONFLICT DO NOTHING",
        (s.id, champ["individual_id"], champ["total_score"], dst_rel)))
    return {"color": color, "total": champ["total_score"], "file": dst_rel}


def complete_season(conn, s: SeasonInfo) -> None:
    conn.execute("UPDATE seasons SET stage='completed', status='completed', completed_at=now() "
                 "WHERE id=%s AND status<>'completed'", (s.id,))
