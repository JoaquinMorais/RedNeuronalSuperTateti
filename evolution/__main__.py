"""Punto de entrada: ``python -m evolution [run|status]``."""
from __future__ import annotations

import logging
import signal
import sys

from .config import load_config
from .db import AlreadyRunningError, Database
from .display import Display
from .runner import StopFlag, Trainer


def cmd_status(cfg) -> int:
    import psycopg
    from psycopg.rows import dict_row
    with psycopg.connect(cfg.database_url, row_factory=dict_row) as conn:
        for r in conn.execute("SELECT number, status, stage, current_matchday FROM seasons ORDER BY number"):
            print(f"Temporada {r['number']}: {r['status']} · etapa {r['stage']} · fecha cerrada {r['current_matchday']}/38")
        best = conn.execute("""SELECT s.number, i.color, h.total_score, h.model_file FROM historical_best h
                               JOIN seasons s ON s.id=h.season_id JOIN individuals i ON i.id=h.individual_id
                               ORDER BY h.total_score DESC, s.number LIMIT 1""").fetchone()
        if best:
            print(f"Mejor histórico: {best['color']} (temporada {best['number']}, {best['total_score']} pts) -> {best['model_file']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    cfg = load_config()
    cmd = argv[0] if argv else "run"
    if cmd == "status":
        return cmd_status(cfg)
    if cmd != "run":
        print("Uso: python -m evolution [run|status]", file=sys.stderr)
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
