"""Acceso a datos: creación de temporadas/calendarios, individuos y archivos de modelos.

Todas las funciones que reciben ``conn`` se ejecutan dentro de una transacción
(``Database.run``) y son idempotentes.
"""
from __future__ import annotations

import json
import secrets
from pathlib import Path

import torch

from model import SuperTatetiNet

from . import colors as colormod
from . import modelstore, mutation, rules, schedule
from .config import Config, MutationLevel
from .rng import derive_rng, derive_seed



# ---------------------------------------------------------------------------
# meta / semilla maestra
# ---------------------------------------------------------------------------
def get_master_seed(conn, requested: int | None) -> int:
    row = conn.execute("SELECT value FROM meta WHERE key='master_seed'").fetchone()
    if row:
        return int(row["value"])
    seed = requested if requested is not None else secrets.randbits(62)
    conn.execute("INSERT INTO meta(key, value) VALUES ('master_seed', %s) ON CONFLICT (key) DO NOTHING", (str(seed),))
    return int(conn.execute("SELECT value FROM meta WHERE key='master_seed'").fetchone()["value"])


# ---------------------------------------------------------------------------
# individuos
# ---------------------------------------------------------------------------
def all_colors(conn) -> set[str]:
    return {r["color"] for r in conn.execute("SELECT color FROM individuals")}


def insert_individual(conn, *, color, origin_season, parent_id, kind, level=None, seed=None,
                      mutation_meta=None, color_meta=None) -> int:
    row = conn.execute(
        """INSERT INTO individuals (color, origin_season, parent_id, kind, model_file, architecture,
                                    architecture_sig, mutation_level, mutation_seed, mutation_meta,
                                    color_mutation_meta)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb) RETURNING id""",
        (color, origin_season, parent_id, kind, modelstore.file_name_for(color),
         modelstore.ARCHITECTURE_NAME, modelstore.architecture_signature(), level, seed,
         json.dumps(mutation_meta or {}), json.dumps(color_meta or {})),
    ).fetchone()
    return row["id"]


def mutation_meta_for(cfg: Config, level: str) -> dict:
    lv = cfg.mutation_levels[level]
    return {"rate": lv.rate, "sigma": lv.sigma, "mutate_value_head": cfg.mutate_value_head,
            "norm_threshold": cfg.mutation_norm_threshold}


def bootstrap_population(conn, cfg: Config, master_seed: int, seed_hash: str | None) -> list[int]:
    """Crea los 20 individuos de la primera temporada (sin archivos todavía)."""
    taken = all_colors(conn)
    ids: list[int] = []
    use_seed = cfg.initial_population == "seed" or (cfg.initial_population == "auto" and seed_hash is not None)
    if use_seed and seed_hash is None:
        raise FileNotFoundError(f"INITIAL_POPULATION=seed pero no existe {cfg.seed_model_path}")
    # Colores de la generación inicial: repartidos por todo el círculo cromático para distinguirse a simple vista
    palette = colormod.spread_colors(rules.POPULATION, derive_rng(master_seed, "founder_colors"), taken)
    if use_seed:
        c = palette[0]
        founder = insert_individual(
            conn, color=c.color, origin_season=1, parent_id=None, kind="founder_seed",
            mutation_meta={"source": str(cfg.seed_model_path), "source_weights_sha256": seed_hash},
            color_meta=c.meta)
        ids.append(founder)
        # 19 descendientes mutados del fundador (el fundador queda intacto). El linaje es el parent_id;
        # el color de la generación inicial NO imita al del fundador: así se distinguen.
        for k in range(1, rules.POPULATION):
            seed = derive_seed(master_seed, "founder_child", k)
            level = mutation.pick_level(cfg.level_probs, derive_rng(seed, "level"))
            ids.append(insert_individual(
                conn, color=palette[k].color, origin_season=1, parent_id=founder, kind="mutant", level=level,
                seed=seed, mutation_meta=mutation_meta_for(cfg, level), color_meta=palette[k].meta))
    else:
        for k in range(rules.POPULATION):
            seed = derive_seed(master_seed, "founder_random", k)
            ids.append(insert_individual(
                conn, color=palette[k].color, origin_season=1, parent_id=None, kind="founder_random", seed=seed,
                mutation_meta={"init": "torch_default_init"}, color_meta=palette[k].meta))
    return ids


def materialize_models(db, cfg: Config) -> list[int]:
    """Escribe en disco los modelos que existen en la base de datos pero aún no tienen archivo.

    La creación es determinista (semilla + padre + nivel guardados), por lo que si el
    proceso se interrumpe entre escribir el archivo y anotar el hash, al reanudar se
    regenera, se compara con lo que hay en disco y NUNCA se sobrescribe.
    Devuelve los ids materializados.
    """
    rows = db.run(lambda c: c.execute("SELECT * FROM individuals WHERE model_sha256 IS NULL ORDER BY id").fetchall())
    done: list[int] = []
    digests: dict[int, str] = {}
    for r in rows:
        meta = r["mutation_meta"]
        if r["kind"] == "founder_seed":
            sd = modelstore.load_state_dict(Path(meta["source"]), meta["source_weights_sha256"])
        elif r["kind"] == "imported":
            src = Path(meta["source"])
            if not src.exists():
                raise FileNotFoundError(f"No se encuentra el archivo de la IA añadida: {src}")
            sd = modelstore.load_state_dict(src, meta["source_weights_sha256"])
        elif r["kind"] == "founder_random":
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(r["mutation_seed"])
                sd = SuperTatetiNet().state_dict()
        else:
            parent = db.run(lambda c: c.execute("SELECT * FROM individuals WHERE id=%s", (r["parent_id"],)).fetchone())
            psd = modelstore.load_state_dict(cfg.models_dir / parent["model_file"], parent["model_sha256"] or digests.get(parent["id"]))
            sd = mutation.mutate_state_dict(psd, MutationLevel(meta["rate"], meta["sigma"]),
                                            r["mutation_seed"], meta["mutate_value_head"],
                                            meta.get("norm_threshold", 0.0))   # filas anteriores a la corrección: sin reescalado
        digest = modelstore.weights_hash(sd)
        path = cfg.models_dir / r["model_file"]
        if path.exists():
            if modelstore.weights_hash(modelstore.load_state_dict(path)) != digest:
                raise modelstore.ModelIntegrityError(f"{path} existe pero no coincide con lo esperado; no se sobrescribe")
        else:
            modelstore.save_model(path, sd, {
                "color": r["color"], "origin_season": r["origin_season"], "kind": r["kind"],
                "model_version": r["model_version"], "parent_id": r["parent_id"],
                "mutation_level": r["mutation_level"], "mutation_seed": r["mutation_seed"],
                "mutation_meta": meta,
            })
        digests[r["id"]] = digest
        db.run(lambda c: c.execute("UPDATE individuals SET model_sha256=%s WHERE id=%s AND model_sha256 IS NULL",
                                   (digest, r["id"])))
        done.append(r["id"])
    return done


# ---------------------------------------------------------------------------
# temporadas y calendarios
# ---------------------------------------------------------------------------
def create_season(conn, cfg: Config, number: int, seed: int, individual_ids: list[int]) -> int:
    """Crea la temporada con TODOS sus calendarios (Liga, grupos y cuadro de Copa) en una
    única transacción, antes de jugar ninguna partida. Idempotente por ``number``."""
    existing = conn.execute("SELECT id FROM seasons WHERE number=%s", (number,)).fetchone()
    if existing:
        return existing["id"]
    n = len(individual_ids)
    assert n == len(set(individual_ids))
    fmt = rules.season_format(n)       # 20 = normal; 21+ = IA añadidas (Liga más larga, grupos más grandes)
    rows = conn.execute("SELECT id, color FROM individuals WHERE id = ANY(%s)", (individual_ids,)).fetchall()
    id_by_color = {r["color"]: r["id"] for r in rows}
    colors = list(id_by_color)

    # Sorteo reproducible de posiciones de calendario y de grupos
    slot_order = rules.shuffled_canonical(colors, derive_rng(seed, "slots"))
    slot_of = {c: i + 1 for i, c in enumerate(slot_order)}
    groups = schedule.draw_groups(colors, derive_rng(seed, "groups"))
    group_of = {c: g for g, members in groups.items() for c in members}

    sid = conn.execute(
        """INSERT INTO seasons (number, seed, draw_meta, config, n_participants, league_matchdays)
           VALUES (%s,%s,%s::jsonb,%s::jsonb,%s,%s) RETURNING id""",
        (number, seed, json.dumps({"slots": slot_of, "groups": groups, "format": fmt}), json.dumps(cfg.snapshot()),
         n, fmt["league_matchdays"]),
    ).fetchone()["id"]

    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO season_participants (season_id, individual_id, slot, cup_group) VALUES (%s,%s,%s,%s)",
            [(sid, id_by_color[c], slot_of[c], group_of[c]) for c in colors])

        slot_to_id = {slot_of[c]: id_by_color[c] for c in colors}
        match_rows = []
        half = fmt["league_matchdays"] // 2
        for md, pairs in enumerate(schedule.league_fixtures(n), start=1):
            for xs, os_ in pairs:
                match_rows.append((sid, "league", md, md, None, None, 1 if md <= half else 2,
                                   slot_to_id[xs], slot_to_id[os_]))
        for g, members in groups.items():
            fixtures = schedule.group_fixtures(len(members))
            for rnd_idx, (md, pairs) in enumerate(fixtures):
                for xi, oi in pairs:
                    match_rows.append((sid, "cup_group", md, rnd_idx + 1, g, None, 1 if rnd_idx < len(fixtures) // 2 else 2,
                                       id_by_color[members[xi]], id_by_color[members[oi]]))
        cur.executemany(
            """INSERT INTO matches (season_id, competition, matchday, round_no, group_name, tie_id, seq,
                                    x_individual_id, o_individual_id) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            match_rows)
        cur.executemany(
            """INSERT INTO cup_ties (season_id, round, tie_no, matchday, slot_a, slot_b)
               VALUES (%s,%s,%s,%s,%s,%s)""",
            [(sid, t["round"], t["tie_no"], t["matchday"], t["slot_a"], t["slot_b"])
             for t in schedule.knockout_bracket()])
    return sid


def get_incomplete_season(conn):
    return conn.execute("SELECT * FROM seasons WHERE status='in_progress' ORDER BY number LIMIT 1").fetchone()


def next_generation_ids(conn) -> tuple[int, list[int]] | None:
    """(número de la nueva temporada, ids de la composición planificada) o None si no hay temporadas."""
    last = conn.execute("SELECT * FROM seasons WHERE status='completed' ORDER BY number DESC LIMIT 1").fetchone()
    if last is None:
        return None
    ids = [r["individual_id"] for r in conn.execute(
        "SELECT individual_id FROM evolution_plan WHERE season_id=%s ORDER BY position", (last["id"],))]
    return last["number"] + 1, ids