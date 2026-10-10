import hashlib
import io
import os
import uuid
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

from evolution.config import Config
from evolution.db import Database
from evolution.display import Display
from evolution.runner import Trainer

ROOT = Path(__file__).resolve().parent.parent
TEST_URL = os.environ.get("TEST_DATABASE_URL")


class Env:
    """Un esquema PostgreSQL aislado + un directorio de modelos propio."""

    def __init__(self, tmp_path: Path, seed=42, initial="seed"):
        self.schema = "t_" + uuid.uuid4().hex[:10]
        self.models = tmp_path / "models"
        self.seed, self.initial = seed, initial
        self.admin = psycopg.connect(TEST_URL, autocommit=True, row_factory=dict_row)
        self.admin.execute(f'CREATE SCHEMA "{self.schema}"')
        self.out = io.StringIO()

    def cfg(self, **kw) -> Config:
        base = dict(database_url=TEST_URL, models_dir=self.models, seed=self.seed, max_seasons=1,
                    initial_population=self.initial, seed_model_path=ROOT / "model.pth", db_retry_base_delay=0.05)
        base.update(kw)
        return Config(**base)

    def db(self) -> Database:
        return Database(TEST_URL, retries=5, base_delay=0.05, schema=self.schema)

    def trainer(self, db=None, stop=None, **kw) -> Trainer:
        hooks = {k: kw.pop(k) for k in ("on_match_played", "on_matchday_closed") if k in kw}
        return Trainer(self.cfg(**kw), db or self.db(), Display(False, self.out), stop, **hooks)

    def q(self, sql, params=()):
        self.admin.execute(f'SET search_path TO "{self.schema}"')
        return self.admin.execute(sql, params).fetchall()

    def close(self):
        self.admin.execute(f'DROP SCHEMA "{self.schema}" CASCADE')
        self.admin.close()

    def signature(self) -> dict:
        """Resumen comparable entre ejecuciones (todo en colores, sin ids de BD)."""
        matches = self.q("""SELECT m.competition, m.matchday, m.seq, xi.color x, oi.color o, m.result, m.moves
                            FROM matches m JOIN individuals xi ON xi.id=m.x_individual_id
                            JOIN individuals oi ON oi.id=m.o_individual_id
                            ORDER BY m.competition, m.matchday, m.round_no, m.seq, xi.color, oi.color""")
        parts = self.q("""SELECT i.color, p.league_position, p.league_score, p.cup_score, p.recopa_score,
                          p.total_score, p.final_position FROM season_participants p
                          JOIN individuals i ON i.id=p.individual_id ORDER BY i.color""")
        plan = self.q("""SELECT e.position, i.color, e.mutation_level, e.mutation_seed, i.model_sha256
                         FROM evolution_plan e JOIN individuals i ON i.id=e.individual_id ORDER BY e.position""")
        h = hashlib.sha256(repr((matches, parts, plan)).encode()).hexdigest()
        return {"hash": h, "matches": matches, "parts": parts, "plan": plan}
