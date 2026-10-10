"""Orquestador: crea/reanuda la temporada y la juega fecha a fecha."""
from __future__ import annotations

import logging
import time
from pathlib import Path

from . import competitions as comp
from . import evolve, modelstore, repo, reports, rules
from .config import Config
from .db import Database
from .display import ROUND_NAME, Display
from .engine import play_match
from .rng import derive_seed

log = logging.getLogger("evolution.runner")
_KO_ROUND_BY_MATCHDAY = {v: k for k, v in rules.KNOCKOUT_MATCHDAYS.items()}


class StopFlag:
    """Se activa con SIGTERM/SIGINT; se consulta entre partidas."""
    def __init__(self):
        self._set = False

    def set(self, *_args) -> None:
        self._set = True

    def is_set(self) -> bool:
        return self._set


class Interrupted(Exception):
    pass


class Trainer:
    def __init__(self, cfg: Config, db: Database, out: Display, stop: StopFlag | None = None,
                 on_match_played=None, on_matchday_closed=None):
        self.cfg, self.db, self.out = cfg, db, out
        self.stop = stop or StopFlag()
        self.on_match_played = on_match_played      # ganchos para pruebas de interrupción
        self.on_matchday_closed = on_matchday_closed
        self.nets: dict[int, object] = {}
        self.session_t0 = time.perf_counter()

    # ------------------------------------------------------------------
    def run(self) -> int:
        self.db.connect_with_retries()
        self.db.init_schema()
        master = self.db.run(lambda c: repo.get_master_seed(c, self.cfg.seed))
        if self.cfg.seed is not None and self.cfg.seed != master:
            self.out.emit(f"AVISO: SEED={self.cfg.seed} ignorada; la base de datos ya tiene la semilla maestra {master}.")
        completed = 0
        try:
            while True:
                if self.stop.is_set():
                    raise Interrupted()
                season = self.db.run(repo.get_incomplete_season)
                resumed = season is not None
                if season is None:
                    if self.cfg.max_seasons and completed >= self.cfg.max_seasons:
                        break
                    season = self._create_next_season(master)
                self._run_season(season, master, resumed)
                completed += 1
                if self.cfg.max_seasons and completed >= self.cfg.max_seasons:
                    break
        except Interrupted:
            st = self.db.run(repo.get_incomplete_season)
            where = f"temporada {st['number']}, última fecha cerrada {st['current_matchday']}/38" if st else "entre temporadas"
            self.out.emit(f"\nDetención solicitada: estado guardado ({where}). Se reanudará al reiniciar.")
        finally:
            self.db.close()
        return 0

    def _check_stop(self) -> None:
        if self.stop.is_set():
            raise Interrupted()

    # ------------------------------------------------------------------
    def _create_next_season(self, master: int) -> dict:
        cfg = self.cfg
        nxt = self.db.run(repo.next_generation_ids)
        seed_hash = None
        if nxt is None and cfg.initial_population in ("auto", "seed") and Path(cfg.seed_model_path).exists():
            seed_hash = modelstore.weights_hash(modelstore.load_state_dict(cfg.seed_model_path))

        def tx(conn):
            if nxt is None:
                number = 1
                if conn.execute("SELECT 1 FROM seasons WHERE number=1").fetchone() is None:
                    ids = repo.bootstrap_population(conn, cfg, master, seed_hash)
                    repo.create_season(conn, cfg, number, derive_seed(master, "season", number), ids)
            else:
                number, ids = nxt
                repo.create_season(conn, cfg, number, derive_seed(master, "season", number), ids)
            return conn.execute("SELECT * FROM seasons WHERE number=%s", (number,)).fetchone()

        return self.db.run(tx)

    def _load_nets(self, info: comp.SeasonInfo) -> None:
        rows = self.db.run(lambda c: c.execute(
            "SELECT id, model_file, model_sha256 FROM individuals WHERE id = ANY(%s)", (list(info.color_of),)).fetchall())
        for r in rows:
            if r["model_sha256"] is None:
                raise RuntimeError(f"El modelo {r['model_file']} no está materializado")
            self.nets[r["id"]] = modelstore.load_net(self.cfg.models_dir / r["model_file"], r["model_sha256"])

    # ------------------------------------------------------------------
    def _run_season(self, season_row: dict, master: int, resumed: bool) -> None:
        recovered = self.db.run(lambda c: comp.reset_running(c, season_row["id"]))
        repo.materialize_models(self.db, self.cfg)
        info = self.db.run(lambda c: comp.load_season_info(c, season_row))
        self._load_nets(info)
        season = self.db.run(lambda c: c.execute("SELECT * FROM seasons WHERE id=%s", (info.id,)).fetchone())
        self.out.titles = self.db.run(lambda c: reports.titles_through(c, info.number - 1))  # campeones de temporadas anteriores
        prog = self.db.run(lambda c: comp.progress(c, info.id))
        self.out.header(season=season, master_seed=master, cfg=self.cfg, resumed=resumed, progress=prog,
                        db_status=self._db_status())
        if recovered:
            self.out.emit(f"  Recuperadas {recovered} partida(s) que estaban en ejecución: se repiten desde el principio "
                          f"(determinista: mismo resultado; nunca se cuentan dos veces).")
        handlers = {
            "league_cup": self._stage_league_cup, "recopa": self._stage_recopa,
            "scoring": self._stage_scoring, "evolution": self._stage_evolution,
        }
        while True:
            self._check_stop()
            stage = self.db.run(lambda c: c.execute("SELECT stage FROM seasons WHERE id=%s", (info.id,)).fetchone()["stage"])
            if stage == "completed":
                return
            handlers[stage](info)
            new = self.db.run(lambda c: c.execute("SELECT stage FROM seasons WHERE id=%s", (info.id,)).fetchone()["stage"])
            if new == stage:
                raise RuntimeError(f"La temporada no avanzó de la etapa '{stage}'")

    def _db_status(self) -> str:
        r = self.db.run(lambda c: c.execute("SELECT current_database() AS db, version() AS v").fetchone())
        return f"conectada a '{r['db']}' ({r['v'].split(',')[0]}); esquema OK; bloqueo de instancia única adquirido"

    # -- etapas --------------------------------------------------------
    def _stage_league_cup(self, info: comp.SeasonInfo) -> None:
        start = self.db.run(lambda c: c.execute("SELECT current_matchday FROM seasons WHERE id=%s", (info.id,)).fetchone())["current_matchday"]
        for md in range(start + 1, rules.LEAGUE_MATCHDAYS + 1):
            self._play_matchday(info, md)
        for ev in self.db.run(lambda c: comp.finish_league(c, info)):
            self.out.section("Liga terminada: clasificación definitiva (con desempates)")
            for pos, color in enumerate(ev["order"], start=1):
                self.out.emit(f"  {pos:>2}. {self.out.ia(color)}  {ev['points'][color]} pts  -> {rules.LEAGUE_REWARD[pos - 1]:+d} puntos evolutivos")
            if ev["draws"]:
                self.out.emit(self.out.dim(f"  Desempates por sorteo reproducible: {', '.join(ev['draws'])}"))

    def _stage_recopa(self, info: comp.SeasonInfo) -> None:
        for ev in self.db.run(lambda c: comp.setup_recopa(c, info)):
            self.out.event(ev)
        self._play_pending(info, comp.MATCHDAY_RECOPA, label_info=lambda m: f"#{m['seq']}")
        for ev in self.db.run(lambda c: comp.finish_recopa(c, info)):
            self.out.event(ev)

    def _stage_scoring(self, info: comp.SeasonInfo) -> None:
        self.db.run(lambda c: comp.compute_global(c, info))

    def _stage_evolution(self, info: comp.SeasonInfo) -> None:
        self.db.run(lambda c: evolve.plan_evolution(c, self.cfg, info))
        repo.materialize_models(self.db, self.cfg)  # escribe los archivos de los 13 descendientes
        hist = evolve.record_historical_best(self.db, self.cfg, info)
        self.db.run(lambda c: evolve.complete_season(c, info))
        self.out.titles = self.db.run(lambda c: reports.titles_through(c, info.number))  # ya incluye a los campeones de esta temporada
        self.out.final_report(self._final_report_data(info, hist))

    # -- fechas ---------------------------------------------------------
    def _match_info(self, m: dict) -> str:
        if m["competition"] == "league":
            return f"F{m['matchday']}"
        if m["competition"] == "cup_group":
            return f"{m['group_name']}·R{m['round_no']}"
        if m["competition"] == "cup_ko":
            leg = {1: "ida", 2: "vuelta", 3: "desempate"}[m["seq"]]
            return f"{ROUND_NAME[_KO_ROUND_BY_MATCHDAY[m['matchday']]]}·{leg}"
        return f"#{m['seq']}"

    def _play_pending(self, info: comp.SeasonInfo, md: int, label_info=None) -> float:
        """Juega todo lo pendiente hasta la fecha ``md`` (incluidas recuperaciones de fechas
        anteriores), avanzando la Copa entre medias. Devuelve ms de partidas jugadas."""
        game_ms = 0.0
        while True:
            pending = self.db.run(lambda c: comp.fetch_pending(c, info.id, md))
            for m in pending:
                self._check_stop()
                if not self.db.run(lambda c, m=m: comp.claim_match(c, m["id"])):
                    continue
                outcome = play_match(self.nets[m["x_individual_id"]], self.nets[m["o_individual_id"]])
                if self.on_match_played:
                    self.on_match_played(m, outcome)
                if self.db.run(lambda c, m=m, o=outcome: comp.record_result(c, m, o)):
                    game_ms += outcome.duration_ms
                    prog = self.db.run(lambda c: comp.progress(c, info.id))
                    info_txt = label_info(m) if label_info else self._match_info(m)
                    self.out.emit(self.out.match_line(m, info.color_of, outcome, prog, info_txt))
            events = self.db.run(lambda c: comp.advance_cup(c, info))
            for ev in events:
                self.out.event(ev)
            if not pending and not events:
                return game_ms

    def _play_matchday(self, info: comp.SeasonInfo, md: int) -> None:
        t0 = time.perf_counter()
        n = len(self.db.run(lambda c: comp.fetch_pending(c, info.id, md)))
        self.out.matchday_header(md, n)
        game_ms = self._play_pending(info, md)
        self.db.run(lambda c: comp.close_matchday(c, info, md))
        rows = self.db.run(lambda c: comp.league_table_snapshot(c, info, md))
        self.out.league_table(rows, md)
        prog = self.db.run(lambda c: comp.progress(c, info.id))
        self.out.matchday_footer(md, (time.perf_counter() - t0) * 1000, game_ms, prog, prog["sim_ms"])
        if self.on_matchday_closed:
            self.on_matchday_closed(md)

    # -- informe final --------------------------------------------------
    def _final_report_data(self, info: comp.SeasonInfo, hist: dict | None) -> dict:
        def q(conn):
            s = conn.execute("SELECT * FROM seasons WHERE id=%s", (info.id,)).fetchone()
            parts = conn.execute("SELECT * FROM season_participants WHERE season_id=%s", (info.id,)).fetchall()
            plan = conn.execute("""SELECT e.*, i.color, pi.color AS parent_color FROM evolution_plan e
                                   JOIN individuals i ON i.id=e.individual_id
                                   LEFT JOIN individuals pi ON pi.id=e.parent_id
                                   WHERE e.season_id=%s ORDER BY e.position""", (info.id,)).fetchall()
            return s, parts, plan
        s, parts, plan = self.db.run(q)
        add = lambda r: {**r, "color": info.color_of[r["individual_id"]]}  # noqa: E731
        parts = [add(r) for r in parts]
        prog = self.db.run(lambda c: comp.progress(c, info.id))
        return {
            "number": info.number,
            "league": sorted(parts, key=lambda r: r["league_position"]),
            "global": sorted(parts, key=lambda r: r["final_position"]),
            "cup_champion": info.color_of[s["cup_champion_id"]], "cup_runner_up": info.color_of[s["cup_runner_up_id"]],
            "recopa_champion": info.color_of[s["recopa_champion_id"]],
            "survivors": [p["color"] for p in plan if p["role"] == "survivor"],
            "children": [{"color": p["color"], "parent": p["parent_color"], "level": p["mutation_level"]}
                         for p in plan if p["role"] == "child"],
            "historical": hist, "sim_ms": prog["sim_ms"],
            "wall_ms": (time.perf_counter() - self.session_t0) * 1000,
        }
