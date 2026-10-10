"""Punto de entrada: ``python -m evolution [run|status|season N|champions|menu]``."""
from __future__ import annotations

import logging
import signal
import sys

import psycopg
from psycopg.rows import dict_row

from . import reports
from .config import load_config
from .db import AlreadyRunningError, Database
from .display import Display
from .runner import StopFlag, Trainer

USAGE = """Uso: python -m evolution <comando>
  run            entrena (crea o reanuda la temporada)            [por defecto]
  status         lista las temporadas, con sus campeones
  season N       analiza la temporada N (Liga, Copa, Recopa, tabla anual)
  champions      campeones de cada temporada + ranking de campeones
  menu           menú interactivo con lo anterior
Los comandos de consulta son de solo lectura y se pueden usar mientras se entrena."""


def _read_only(cfg, fn) -> int:
    out = Display(cfg.color)
    try:
        with psycopg.connect(cfg.database_url, row_factory=dict_row) as conn:
            fn(conn, out)
    except psycopg.errors.UndefinedTable:
        print("La base de datos todavía no tiene temporadas (ejecutá primero `run`).", file=sys.stderr)
        return 1
    except psycopg.OperationalError as e:
        print(f"No se pudo conectar a la base de datos: {e}", file=sys.stderr)
        return 1
    return 0


def cmd_menu(cfg) -> int:
    def loop(conn, out):
        while True:
            print("\n  1) Estado de las temporadas\n  2) Ver una temporada en detalle\n  3) Campeones\n  0) Salir")
            try:
                choice = input("Opción: ").strip()
                if choice == "1":
                    reports.render_status(conn, out)
                elif choice == "2":
                    nums = [r["number"] for r in conn.execute("SELECT number FROM seasons ORDER BY number")]
                    print("Temporadas disponibles:", ", ".join(map(str, nums)) or "ninguna")
                    n = input("Número de temporada: ").strip()
                    if n.isdigit():
                        reports.render_season(conn, out, int(n))
                    else:
                        print("Ingresá un número.")
                elif choice == "3":
                    reports.render_champions(conn, out)
                elif choice in ("0", "q", ""):
                    return
            except (EOFError, KeyboardInterrupt):
                return
    return _read_only(cfg, loop)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    cfg = load_config()
    cmd = argv[0] if argv else "run"
    if cmd == "status":
        return _read_only(cfg, reports.render_status)
    if cmd == "champions":
        return _read_only(cfg, reports.render_champions)
    if cmd == "season":
        if len(argv) != 2 or not argv[1].isdigit():
            print("Uso: python -m evolution season N", file=sys.stderr)
            return 2
        return _read_only(cfg, lambda conn, out: reports.render_season(conn, out, int(argv[1])))
    if cmd == "menu":
        return cmd_menu(cfg)
    if cmd != "run":
        print(USAGE, file=sys.stderr)
        return 2
    stop = StopFlag()
    signal.signal(signal.SIGTERM, stop.set)   # docker stop
    signal.signal(signal.SIGINT, stop.set)    # Ctrl+C: termina la partida en curso y sale con el estado guardado
    db = Database(cfg.database_url, cfg.db_retries, cfg.db_retry_base_delay)
    try:
        return Trainer(cfg, db, Display(cfg.color), stop).run()
    except AlreadyRunningError as e:
        print(f"ERROR: {e}. Detené el otro proceso antes de iniciar uno nuevo.", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
