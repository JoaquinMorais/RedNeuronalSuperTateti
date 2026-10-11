"""Punto de entrada: ``python -m evolution <comando>`` (ver USAGE)."""
from __future__ import annotations

import logging
import signal
import sys

import psycopg
from psycopg.rows import dict_row

from . import play, prune, reports
from .config import load_config
from .db import AlreadyRunningError, Database
from .display import Display
from .runner import StopFlag, Trainer

USAGE = """Uso: python -m evolution <comando>
  run                 entrena (crea o reanuda la temporada)            [por defecto]
  status              lista las temporadas, con sus campeones
  season N            analiza la temporada N (Liga, Copa, Recopa, tabla anual)
  champions [N]       campeones por temporada + ranking (los N mejores; por defecto 30)
  play [N]            juega contra el mejor de la tabla anual de la temporada N (por defecto, la última terminada)
  prune [--dry-run]   borra los .pth antiguos (ver MODEL_RETENTION_SEASONS); --dry-run solo muestra qué borraría
  menu                menú interactivo (admite "3 100", "2 21", "4 12", ...)
Los comandos de consulta son de solo lectura y se pueden usar mientras se entrena."""

MENU = """
  1) Estado de las temporadas
  2) Ver una temporada en detalle        (ej.: "2 21")
  3) Campeones                           (ej.: "3 100" muestra 100; por defecto 30)
  4) Jugar contra la IA                  (ej.: "4 12" = mejor de la temporada 12; por defecto, la última terminada)
  0) Salir"""


def parse_menu_line(line: str) -> tuple[str, list[str]]:
    """'3 100' -> ('3', ['100']); '' -> ('', [])."""
    parts = line.split()
    return (parts[0], parts[1:]) if parts else ("", [])


def _positive_int(args: list[str]) -> int | None:
    return int(args[0]) if args and args[0].isdigit() and int(args[0]) > 0 else None


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


def cmd_play(cfg, season: int | None) -> int:
    found: dict = {}

    def find(conn, out):
        try:
            found["opp"] = play.find_opponent(conn, cfg.models_dir, season)
        except play.PlayError as e:
            print(f"No se puede jugar: {e}", file=sys.stderr)

    rc = _read_only(cfg, find)
    if "opp" not in found:
        return rc or 1
    session = play.PlaySession(play.load_opponent_net(found["opp"]), found["opp"])
    play.serve(session, cfg.play_host, cfg.play_port)
    return 0


def cmd_prune(cfg, dry_run: bool) -> int:
    db = Database(cfg.database_url, cfg.db_retries, cfg.db_retry_base_delay)
    out = Display(cfg.color)
    try:
        db.connect_with_retries()
        db.init_schema()
        last = db.run(lambda c: c.execute("SELECT max(number) AS n FROM seasons WHERE status='completed'").fetchone()["n"])
        if last is None:
            print("Todavía no terminó ninguna temporada.")
            return 0
        if cfg.retention_seasons <= 0:
            print("MODEL_RETENTION_SEASONS=0: la limpieza está desactivada.")
            return 0
        rows = prune.prune_models(db, cfg, last, dry_run=dry_run)
        for r in rows:
            out.emit(f"  {'(simulado) ' if dry_run else ''}{r['file']}  IA {out.ia(r['color'], badge=False)}  "
                     f"expulsada en T{r['last_season']}  {r['bytes'] / 1e6:.2f} MB")
        verb = "se borrarían" if dry_run else "borrados"
        print(f"{len(rows)} archivo(s) .pth {verb} ({sum(r['bytes'] for r in rows) / 1e6:.1f} MB). "
              f"Criterio: expulsadas hace {cfg.retention_seasons}+ temporadas y sin títulos ni 1.º de tabla anual.")
        return 0
    except AlreadyRunningError as e:
        print(f"ERROR: {e}. Detené el entrenamiento o esperá a que termine la temporada.", file=sys.stderr)
        return 3
    finally:
        db.close()


def cmd_menu(cfg) -> int:
    def loop(conn, out):
        while True:
            print(MENU)
            try:
                choice, args = parse_menu_line(input("Opción: "))
                n = _positive_int(args)
                if args and n is None:
                    print("El argumento debe ser un número entero positivo (ej.: 3 100).")
                    continue
                if choice == "1":
                    reports.render_status(conn, out)
                elif choice == "2":
                    if n is None:
                        nums = [r["number"] for r in conn.execute("SELECT number FROM seasons ORDER BY number")]
                        print("Temporadas disponibles:", ", ".join(map(str, nums)) or "ninguna")
                        raw = input("Número de temporada: ").strip()
                        n = int(raw) if raw.isdigit() else None
                    if n is None:
                        print("Ingresá un número.")
                    else:
                        reports.render_season(conn, out, n)
                elif choice == "3":
                    reports.render_champions(conn, out, n or cfg.champions_top)
                elif choice == "4":
                    cmd_play(cfg, n)
                elif choice in ("0", "q"):
                    return
                elif choice:
                    print("Opción no válida.")
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
        n = _positive_int(argv[1:2])
        if len(argv) > 1 and n is None:
            print("Uso: python -m evolution champions [N]", file=sys.stderr)
            return 2
        return _read_only(cfg, lambda conn, out: reports.render_champions(conn, out, n or cfg.champions_top))
    if cmd == "season":
        if len(argv) != 2 or not argv[1].isdigit():
            print("Uso: python -m evolution season N", file=sys.stderr)
            return 2
        return _read_only(cfg, lambda conn, out: reports.render_season(conn, out, int(argv[1])))
    if cmd == "play":
        n = _positive_int(argv[1:2])
        if len(argv) > 1 and n is None:
            print("Uso: python -m evolution play [N]", file=sys.stderr)
            return 2
        return cmd_play(cfg, n)
    if cmd == "prune":
        return cmd_prune(cfg, "--dry-run" in argv[1:])
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