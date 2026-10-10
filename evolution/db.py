"""Conexión PostgreSQL: bloqueo de instancia única, transacciones y reconexión."""
from __future__ import annotations

import logging
import time
from importlib import resources
from typing import Callable, TypeVar

import psycopg
from psycopg.rows import dict_row

log = logging.getLogger("evolution.db")
T = TypeVar("T")

# Clave del bloqueo consultivo (advisory lock) que impide dos entrenamientos a la vez.
ADVISORY_LOCK_KEY = 0x7A7E7E1  # arbitrario pero fijo
SCHEMA_VERSION = "1"


class AlreadyRunningError(RuntimeError):
    """Otro proceso ya está entrenando sobre esta base de datos."""


class Database:
    """Una conexión con transacciones atómicas y reintentos ante caídas.

    - ``run(fn)`` ejecuta ``fn(conn)`` dentro de UNA transacción. Si la conexión se
      cae, reconecta (con espera exponencial), recupera el bloqueo y repite ``fn``:
      todas las operaciones de la aplicación son idempotentes, así que repetir una
      transacción cuyo COMMIT quedó en duda nunca contabiliza nada dos veces.
    - El bloqueo es de sesión (``pg_try_advisory_lock``): PostgreSQL lo libera solo
      si el proceso muere o se corta la conexión.
    """

    def __init__(self, url: str, retries: int = 8, base_delay: float = 0.5, schema: str | None = None):
        self.url, self.retries, self.base_delay, self.schema = url, retries, base_delay, schema
        self.conn: psycopg.Connection | None = None
        self.reconnections = 0

    # -- conexión ----------------------------------------------------------
    def connect(self) -> None:
        self.close()
        conn = psycopg.connect(self.url, autocommit=True, row_factory=dict_row)
        if self.schema:
            conn.execute(f'SET search_path TO "{self.schema}"')
        ok = conn.execute("SELECT pg_try_advisory_lock(%s) AS ok", (ADVISORY_LOCK_KEY,)).fetchone()["ok"]
        if not ok:
            conn.close()
            raise AlreadyRunningError("Otro entrenamiento está activo sobre esta base de datos")
        self.conn = conn

    def connect_with_retries(self) -> None:
        """Arranque: espera a que la base de datos esté disponible (orden de inicio en Docker)."""
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                self.connect()
                return
            except (psycopg.OperationalError, psycopg.InterfaceError) as e:
                last = e
                delay = min(self.base_delay * 2 ** attempt, 15)
                log.warning("Base de datos no disponible (%s); reintento en %.1fs", str(e).strip()[:120], delay)
                time.sleep(delay)
        raise ConnectionError(f"No se pudo conectar a la base de datos: {last}")

    def close(self) -> None:
        if self.conn is not None:
            try:
                self.conn.close()
            except Exception:  # noqa: BLE001
                pass
            self.conn = None

    # -- transacciones -----------------------------------------------------
    def run(self, fn: Callable[[psycopg.Connection], T]) -> T:
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                if self.conn is None or self.conn.closed:
                    self.connect()
                    self.reconnections += 1
                with self.conn.transaction():
                    return fn(self.conn)
            except (psycopg.OperationalError, psycopg.InterfaceError) as e:
                last = e
                delay = min(self.base_delay * 2 ** attempt, 15)
                log.warning("Conexión perdida (%s); reconectando en %.1fs", str(e).strip()[:120], delay)
                self.close()
                time.sleep(delay)
        raise ConnectionError(f"Base de datos inaccesible tras {self.retries + 1} intentos: {last}")

    # -- esquema -----------------------------------------------------------
    def init_schema(self) -> None:
        sql = resources.files("evolution").joinpath("schema.sql").read_text(encoding="utf-8")

        def _init(conn):
            conn.execute(sql)
            conn.execute(
                "INSERT INTO meta(key, value) VALUES ('schema_version', %s) ON CONFLICT (key) DO NOTHING",
                (SCHEMA_VERSION,),
            )
        self.run(_init)
