"""Añadir IA a mano a la PRÓXIMA temporada: redes (.pth) y bots estáticos (``models_statics/``).

* Una red (.pth): recibe un color nuevo y único y se guarda en ``models/`` (kind 'imported').
* Un estático (por NOMBRE): usa su COLOR1 como identidad (único en toda la historia) y no tiene pesos.
Ambos suman cupos extra solo en esa temporada. Después la composición vuelve a 20 (7 supervivientes + 13
descendientes), salvo los estáticos que sigan por terminar entre los primeros 7 + (nº de estáticos).
"""
from __future__ import annotations

from pathlib import Path

from . import colors as colormod
from . import modelstore, repo, rules, statics
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
        """SELECT a.season_number, i.color, i.ia_type, i.static_name, i.static_color2, a.source, a.added_at
           FROM season_additions a JOIN individuals i ON i.id = a.individual_id
           WHERE NOT EXISTS (SELECT 1 FROM seasons s WHERE s.number = a.season_number)
           ORDER BY a.season_number, a.id""").fetchall()


def _is_path(token: str) -> bool:
    t = token.strip().strip('"').strip("'")
    return t.lower().endswith(".pth") or Path(t).expanduser().is_file()


def _resolve_source(cfg: Config, raw: str) -> Path:
    p = Path(raw.strip().strip('"').strip("'")).expanduser()
    for cand in (p, cfg.models_dir / p):
        if cand.is_file():
            return cand.resolve()
    raise AdditionError(f"No se encuentra el archivo '{raw}' (se buscó tal cual y dentro de {cfg.models_dir}).")


def _competing_statics(conn) -> set[int]:
    """Estáticos que ya están (o seguirán) en la última temporada creada."""
    last = conn.execute("SELECT id, status FROM seasons ORDER BY number DESC LIMIT 1").fetchone()
    if last is None:
        return set()
    if last["status"] == "in_progress":
        q = "SELECT p.individual_id AS i FROM season_participants p JOIN individuals x ON x.id=p.individual_id WHERE p.season_id=%s AND x.ia_type='estatico'"
    else:   # terminada: los que siguen están en su plan de la siguiente generación
        q = "SELECT p.individual_id AS i FROM evolution_plan p JOIN individuals x ON x.id=p.individual_id WHERE p.season_id=%s AND x.ia_type='estatico'"
    return {r["i"] for r in conn.execute(q, (last["id"],))}


def register_additions(db, cfg: Config, tokens: list[str]) -> dict:
    """Valida y registra IA para la próxima temporada. ``tokens``: rutas .pth y/o NOMBRES de bots estáticos.
    Devuelve {target, added, total_extra, format}."""
    if not tokens:
        raise AdditionError("Indicá al menos un archivo .pth o el nombre de un bot estático.")
    neural, static_req = [], []
    for tok in tokens:
        (neural if _is_path(tok) else static_req).append(tok)

    loaded, seen = [], set()
    for raw in neural:
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

    bots, seen_slugs = [], set()
    if static_req:
        try:
            statics.check_unique(cfg.statics_dir)
            for tok in static_req:
                slug, cls = statics.find(cfg.statics_dir, tok)
                if slug in seen_slugs:
                    raise AdditionError(f"El bot estático '{cls.NAME}' está repetido en el pedido.")
                seen_slugs.add(slug)
                bots.append((slug, cls, statics.code_hash(cfg.statics_dir, slug)))
        except statics.StaticError as e:
            raise AdditionError(str(e)) from e

    def tx(conn):
        conn.execute("SELECT pg_advisory_xact_lock(%s)", (ADD_LOCK_KEY,))
        last = conn.execute("SELECT max(number) AS n FROM seasons").fetchone()["n"] or 0
        target = last + 1                       # temporada siguiente a la última creada
        already = conn.execute("SELECT count(*) AS n FROM season_additions WHERE season_number=%s", (target,)).fetchone()["n"]
        competing = _competing_statics(conn)
        if already + len(loaded) + len(bots) + len(competing) > rules.MAX_EXTRA_TEAMS:
            raise AdditionError(
                f"Una temporada admite como máximo {rules.MAX_EXTRA_TEAMS} equipos extra (redes añadidas + estáticos): "
                f"ya hay {already} añadidos para la temporada {target} y {len(competing)} estático(s) compitiendo.")
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
            added.append({"type": "neuronal", "color": c.color, "source": str(src)})
        for slug, cls, chash in bots:
            added.append(_register_static(conn, target, slug, cls, chash, competing))
        return {"target": target, "added": added, "total_extra": already + len(loaded) + len(bots)}

    res = db.run(tx)
    repo.materialize_models(db, cfg)   # escribe models/<color>.pth + .json de las redes añadidas (sin sobrescribir)
    res["format"] = rules.season_format(rules.POPULATION + res["total_extra"])
    return res


def _register_static(conn, target: int, slug: str, cls, chash: str, competing: set[int]) -> dict:
    name = cls.NAME
    row = conn.execute("SELECT * FROM individuals WHERE ia_type='estatico' AND static_slug=%s", (slug,)).fetchone()
    other = conn.execute(
        """SELECT color, static_name, static_slug, ia_type FROM individuals
           WHERE color=%s OR (ia_type='estatico' AND lower(static_name)=lower(%s))""", (cls.COLOR1, name)).fetchall()
    for o in other:
        if row is not None and o["color"] == row["color"]:
            continue
        if o["ia_type"] == "estatico":
            raise AdditionError(f"El nombre o el color de '{name}' ya lo usa el bot estático '{o['static_name']}' ({o['static_slug']}).")
        raise AdditionError(f"El COLOR1 {cls.COLOR1} de '{name}' ya es la identidad de una red neuronal: elegí otro color.")
    if row is None:
        iid = repo.insert_individual(
            conn, color=cls.COLOR1, origin_season=target, parent_id=None, kind="static", ia_type="estatico",
            static_name=name, static_color2=cls.COLOR2, static_slug=slug, static_code_sha256=chash)
    else:
        iid = row["id"]
        if row["color"] != cls.COLOR1:
            raise AdditionError(f"'{name}' cambió su COLOR1 ({row['color']} -> {cls.COLOR1}): ese color es su identidad en la "
                                f"base de datos; devolvelo al valor original o creá un bot nuevo.")
        if iid in competing:
            raise AdditionError(f"'{name}' ya compite en la última temporada creada (o sigue en la próxima).")
        if conn.execute("SELECT 1 FROM season_additions WHERE season_number=%s AND individual_id=%s", (target, iid)).fetchone():
            raise AdditionError(f"'{name}' ya está añadido a la temporada {target}.")
        conn.execute("UPDATE individuals SET static_name=%s, static_color2=%s, static_code_sha256=%s, model_sha256=%s WHERE id=%s",
                     (name, cls.COLOR2, chash, chash, iid))
    conn.execute("INSERT INTO season_additions (season_number, individual_id, source) VALUES (%s,%s,%s)",
                 (target, iid, f"static:{slug}"))
    return {"type": "estatico", "color": cls.COLOR1, "name": name, "color2": cls.COLOR2, "source": f"static:{slug}"}


def cancel_addition(db, cfg: Config, token: str) -> dict:
    """Quita una IA añadida cuya temporada todavía no se creó. ``token``: color (#rrggbb) o nombre de un estático."""
    raw = token.strip()

    def tx(conn):
        conn.execute("SELECT pg_advisory_xact_lock(%s)", (ADD_LOCK_KEY,))
        if raw.startswith("#") or (len(raw) == 6 and all(c in "0123456789abcdefABCDEF" for c in raw)):
            col = raw.lower() if raw.startswith("#") else "#" + raw.lower()
            cand = conn.execute(
                """SELECT a.id, a.individual_id, a.season_number, i.model_file, i.ia_type, i.color FROM season_additions a
                   JOIN individuals i ON i.id = a.individual_id WHERE i.color = %s ORDER BY a.id DESC""", (col,)).fetchall()
        else:
            key = statics.normalize(raw)
            cand = [r for r in conn.execute(
                """SELECT a.id, a.individual_id, a.season_number, i.model_file, i.ia_type, i.color, i.static_name FROM season_additions a
                   JOIN individuals i ON i.id = a.individual_id WHERE i.ia_type='estatico' ORDER BY a.id DESC""")
                if statics.normalize(r["static_name"]) == key]
        pend = [r for r in cand if conn.execute("SELECT 1 FROM seasons WHERE number=%s", (r["season_number"],)).fetchone() is None]
        if not cand:
            raise AdditionError(f"{raw} no está entre las IA añadidas pendientes.")
        if not pend:
            raise AdditionError(f"{raw} ya forma parte de la temporada {cand[0]['season_number']}: no se puede quitar.")
        row = pend[0]
        conn.execute("DELETE FROM season_additions WHERE id=%s", (row["id"],))
        # la fila del individuo solo se borra si nunca compitió ni se usó (un estático con historia se conserva)
        used = conn.execute(
            """SELECT 1 FROM season_participants WHERE individual_id=%(i)s UNION ALL
               SELECT 1 FROM evolution_plan WHERE individual_id=%(i)s UNION ALL
               SELECT 1 FROM season_additions WHERE individual_id=%(i)s UNION ALL
               SELECT 1 FROM individuals WHERE parent_id=%(i)s""", {"i": row["individual_id"]}).fetchone()
        deleted = False
        if used is None:
            conn.execute("DELETE FROM individuals WHERE id=%s", (row["individual_id"],))
            deleted = True
        return {**row, "deleted": deleted}

    row = db.run(tx)
    if row["deleted"] and row["ia_type"] == "neuronal":
        path = cfg.models_dir / row["model_file"]
        for f in (path, modelstore.sidecar_path(path)):
            if f.parent == cfg.models_dir and f.exists():
                f.unlink()
    return {"color": row["color"], "season": row["season_number"], "type": row["ia_type"],
            "name": row.get("static_name")}