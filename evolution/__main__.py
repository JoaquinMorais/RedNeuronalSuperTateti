"""Punto de entrada: ``python -m evolution <comando>`` (ver USAGE)."""
from __future__ import annotations

import logging
import shlex
import signal
import sys

import psycopg
from psycopg.rows import dict_row

from . import additions, benchmark, disk, ladder, modelstore, play, progress, prune, reports, sandbox, statics
from .config import load_config
from .db import AlreadyRunningError, Database
from .display import Display
from .runner import StopFlag, Trainer

USAGE = """Uso: python -m evolution <comando>
  run                 entrena (crea o reanuda la temporada)            [por defecto]
  status              lista las temporadas, con sus campeones
  season N            analiza la temporada N (Liga, Copa, Recopa, tabla anual)
  champions [N]       campeones por temporada + ranking (los N mejores; por defecto 30)
  play [N|nombre]     juega contra la mejor RED de la temporada N (por defecto, la última terminada) o contra un estático por su nombre
  fight A B [C..]     pelea de ida y vuelta (2 IA) o mini liga (3 a 8). IA: nº de temporada = mejor RED de esa tabla anual, color #rrggbb,
                      o NOMBRE de un bot estático (los estáticos solo se invocan por su nombre)
  testleague A [B..]  Liga de prueba (no se guarda): tus IA + aleatorias hasta 20; opcional seed=N
  addai x [y..]       añade a la PRÓXIMA temporada: archivos .pth (redes) y/o NOMBRES de bots estáticos; --list · --cancel color|nombre
  statics             bots estáticos disponibles en models_statics/ y su estado
  progress [N]        curva de aprendizaje contra bots fijos (últimas N mediciones), ritmo y eficacia de mutaciones
                      progress --backfill [cada] [partidas]: mide temporadas ya jugadas para ver la curva sin esperar
  bench A [partidas]  mide una IA contra los bots fijos (aleatorio y táctico); por defecto 200 partidas por rival
  ladder [paso] [max] torneo maestro entre campeones de distintas temporadas (1 cada `paso`, hasta `max`)
  disk                qué ocupa espacio: modelos por categoría, archivos huérfanos y base de datos
  prune --orphans [--yes]  lista (y con --yes borra) archivos de modelos que la base de datos no usa
  prune [--dry-run]   borra los .pth antiguos (ver MODEL_RETENTION_SEASONS); --dry-run solo muestra qué borraría
  menu                menú interactivo (admite "3 100", "2 21", "4 12", ...)
Los comandos de consulta son de solo lectura y se pueden usar mientras se entrena."""

MENU = """
  1) Estado de las temporadas
  2) Ver una temporada en detalle        (ej.: "2 21")
  3) Campeones                           (ej.: "3 100" muestra 100; por defecto 30)
  4) Jugar contra la IA                  (ej.: "4 12" = mejor RED de la temporada 12; "4 BocaJrs" = un estático)
  5) Pelear entre IA                     (ej.: "5 1 311" · "5 311 BocaJrs" · "5 1 311 BocaJrs" = mini liga de 3)
  6) Liga de prueba con IA random        (ej.: "6 1 311 BocaJrs": esas IA + random hasta 20; opcional seed=N)
  7) Añadir IA a la próxima temporada    (ej.: "7 models/mi_ia.pth" · "7 BocaJrs" (estático) · "7 --list" · "7 --cancel BocaJrs")
  8) Progreso: curva de aprendizaje      (ej.: "8 50" = últimas 50 mediciones; "8 --backfill 10" mide temporadas ya jugadas)
  9) Medir una IA contra bots fijos      (ej.: "9 12" o "9 #ffffff 500" = 500 partidas por rival)
 10) Torneo maestro entre campeones      (ej.: "10 5" = un campeón cada 5 temporadas; "10 5 40" = hasta 40)
  11) Espacio en disco (modelos, huérfanos, base de datos)
  12) Bots estáticos (disponibles en models_statics/ y su estado)
  0) Salir"""


def parse_menu_line(line: str) -> tuple[str, list[str]]:
    """'3 100' -> ('3', ['100']); '' -> ('', []). Admite comillas para rutas con espacios."""
    try:
        parts = shlex.split(line)
    except ValueError:
        parts = line.split()
    return (parts[0], parts[1:]) if parts else ("", [])


def _positive_int(args: list[str]) -> int | None:
    return int(args[0]) if args and args[0].isdigit() and int(args[0]) > 0 else None


def _read_only(cfg, fn) -> int:
    out = Display(cfg.color)
    try:
        with psycopg.connect(cfg.database_url, row_factory=dict_row) as conn:
            out.statics = reports.load_statics(conn)
            fn(conn, out)
    except psycopg.errors.UndefinedTable:
        print("La base de datos todavía no tiene temporadas (ejecutá primero `run`).", file=sys.stderr)
        return 1
    except psycopg.OperationalError as e:
        print(f"No se pudo conectar a la base de datos: {e}", file=sys.stderr)
        return 1
    return 0


def cmd_play(cfg, season: int | None, name: str | None = None) -> int:
    """Juega contra la mejor RED de una temporada (por defecto la última terminada) o, con ``name``, contra un estático."""
    found: dict = {}

    def find(conn, out):
        try:
            if name:
                ia = sandbox.resolve_ia(conn, cfg.models_dir, name, cfg.statics_dir)
                if ia.get("type") != "estatico":
                    raise play.PlayError(f"'{name}' no es un bot estático (un número de temporada elige la mejor red).")
                found["opp"] = {"season": None, "color": ia["color"], "tag": ia["tag"], "static": True, "name": ia["name"],
                                "color2": ia["color2"], "total": None, "league": None, "cup": None, "recopa": None}
                found["player"] = lambda: sandbox.make_player(ia, 0)
            else:
                opp = play.find_opponent(conn, cfg.models_dir, season)
                found["opp"] = opp
                found["player"] = lambda: play.load_opponent_net(opp)
        except play.PlayError as e:
            print(f"No se puede jugar: {e}", file=sys.stderr)

    rc = _read_only(cfg, find)
    if "opp" not in found:
        return rc or 1
    session = play.PlaySession(found["player"](), found["opp"])
    play.serve(session, cfg.play_host, cfg.play_port)
    return 0


def _resolve(cfg, tokens: list[str]) -> list[dict] | None:
    found: dict = {}

    def fn(conn, out):
        try:
            found["ias"] = sandbox.resolve_many(conn, cfg.models_dir, tokens, cfg.statics_dir)
        except play.PlayError as e:
            print(f"No se puede: {e}", file=sys.stderr)

    _read_only(cfg, fn)
    return found.get("ias")


def cmd_fight(cfg, tokens: list[str]) -> int:
    import secrets
    seed, refs = None, []
    for t in tokens:
        if t.lower().startswith("seed=") and t[5:].isdigit():
            seed = int(t[5:])
        else:
            refs.append(t)
    if not 2 <= len(refs) <= 8:
        print("Uso: fight A B [C ...] [seed=N]  (2 IA = pelea de ida y vuelta; 3 a 8 = mini liga. Cada IA: nº de temporada = mejor "
              "RED de esa tabla anual, color #rrggbb o NOMBRE de un bot estático; ej.: fight 1 311 BocaJrs)", file=sys.stderr)
        return 2
    ias = _resolve(cfg, refs)
    if ias is None:
        return 1
    seed = secrets.randbits(31) if seed is None else seed
    out = Display(cfg.color)
    try:
        if len(ias) == 2:
            res = sandbox.fight(ias[0], ias[1], seed)
        else:
            res = sandbox.fight_round_robin(ias, seed)
    except Exception as e:  # noqa: BLE001  (modelo corrupto, hash que no coincide, ...)
        print(f"No se pudo pelear: {e}", file=sys.stderr)
        return 1
    if len(ias) == 2:
        sandbox.render_fight(out, ias[0], ias[1], res)
        if any(i.get("type") == "estatico" for i in ias):
            out.emit(out.dim(f"  Hay un bot estático: su azar sale de la semilla {seed} (repetí con seed={seed})."))
    else:
        sandbox.render_fight_table(out, ias, res, seed)
    return 0


def cmd_testleague(cfg, tokens: list[str]) -> int:
    import secrets
    seed, refs = None, []
    for t in tokens:
        if t.lower().startswith("seed="):
            if not t[5:].isdigit():
                print("seed debe ser un número entero (ej.: seed=5).", file=sys.stderr)
                return 2
            seed = int(t[5:])
        else:
            refs.append(t)
    if not refs:
        print("Uso: testleague A [B ...] [seed=N]  (cada IA: nº de temporada o color #rrggbb)", file=sys.stderr)
        return 2
    ias = _resolve(cfg, refs)
    if ias is None:
        return 1
    seed = secrets.randbits(31) if seed is None else seed
    out = Display(cfg.color)
    sandbox.show(out, ias)
    tested = {i["color"] for i in ias}

    def progress(md, games):
        for g in games:
            if g["x"] in tested or g["o"] in tested:
                res = {"x": f"gana {out.ia(g['x'])}", "o": f"gana {out.ia(g['o'])}", "draw": "empate"}[g["result"]]
                out.emit(f"  F{md:<2} {out.ia(g['x'])} (comienza) vs {out.ia(g['o'])} -> {res}")
        if md % 10 == 0:
            out.emit(out.dim(f"  ... fecha {md}/38"))

    out.emit(f"Liga de prueba con {len(ias)} IA + {20 - len(ias)} aleatorias (semilla {seed})...")
    try:
        res = sandbox.run_random_league(ias, seed, progress)
    except (play.PlayError, Exception) as e:  # noqa: BLE001
        print(f"No se pudo ejecutar la liga de prueba: {e}", file=sys.stderr)
        return 1
    sandbox.render_random_league(out, res, ias)
    return 0


def cmd_addai(cfg, args: list[str]) -> int:
    db = Database(cfg.database_url, cfg.db_retries, cfg.db_retry_base_delay, lock=False)   # se puede usar entrenando
    out = Display(cfg.color)
    try:
        db.connect_with_retries()
        if not db.run(lambda c: c.execute("SELECT to_regclass('season_additions') AS t").fetchone()["t"]):
            db.init_schema()     # base creada con una versión anterior: migrar
        out.statics = db.run(reports.load_statics)
        if args[:1] == ["--list"]:
            rows = db.run(lambda c: additions.list_pending(c))
            if not rows:
                print("No hay IA añadidas pendientes.")
            for r in rows:
                if r["ia_type"] == "estatico":
                    out.statics[r["color"]] = {"name": r["static_name"], "color2": r["static_color2"], "slug": ""}
                print(f"  {out.ia(r['color'], badge=False)} ({'estático' if r['ia_type'] == 'estatico' else 'red'}) -> "
                      f"temporada {r['season_number']}  (desde {r['source']})")
            return 0
        if args[:1] == ["--cancel"]:
            if len(args) != 2:
                print("Uso: addai --cancel color|nombre", file=sys.stderr)
                return 2
            r = additions.cancel_addition(db, cfg, args[1])
            print(f"Quitada {r['name'] or r['color']} de la temporada {r['season']}.")
            return 0
        res = additions.register_additions(db, cfg, args)
        f = res["format"]
        n_static = 0
        for a in res["added"]:
            if a["type"] == "estatico":
                n_static += 1
                out.statics[a["color"]] = {"name": a["name"], "color2": a["color2"], "slug": ""}
                print(f"  Añadido el estático {out.ia(a['color'], badge=False)}  (desde {a['source']})")
            else:
                print(f"  Añadida {out.ia(a['color'], badge=False)}  (desde {a['source']})")
        gs = "/".join(map(str, f["group_sizes"]))
        print(f"La temporada {res['target']} tendrá {f['teams']} equipos ({res['total_extra']} añadida(s)): "
              f"Liga de {f['league_matchdays']} fechas ({f['games_per_team']} partidas por IA"
              f"{', una IA descansa en cada fecha' if f['teams'] % 2 else ''}), grupos de Copa de {gs} "
              f"(quedan fuera {'/'.join(map(str, f['group_eliminated']))}).")
        if n_static:
            print("Los estáticos no se reproducen: siguen en la temporada siguiente mientras terminen dentro de los primeros "
                  "7 + (nº de estáticos) puestos de la tabla anual; si no, salen y la composición vuelve a 20.")
        else:
            print(f"La temporada {res['target'] + 1} vuelve a tener 20.")
        return 0
    except additions.AdditionError as e:
        print(f"No se pudo añadir: {e}", file=sys.stderr)
        return 1
    finally:
        db.close()


def cmd_progress(cfg, args: list[str]) -> int:
    """``progress [N]`` muestra la curva; ``progress --backfill [cada] [partidas]`` mide temporadas ya jugadas."""
    if args[:1] == ["--backfill"]:
        nums = args[1:]
        if len(nums) > 2 or not all(a.isdigit() and int(a) > 0 for a in nums):
            print("Uso: progress --backfill [cada_N_temporadas] [partidas_por_rival]  (ej.: progress --backfill 10 200)", file=sys.stderr)
            return 2
        step = int(nums[0]) if nums else 10
        games = int(nums[1]) if len(nums) > 1 else 120
        db = Database(cfg.database_url, cfg.db_retries, cfg.db_retry_base_delay, lock=False)   # se puede usar entrenando
        try:
            db.connect_with_retries()
            if not db.run(lambda c: c.execute("SELECT to_regclass('season_benchmarks') AS t").fetchone()["t"]):
                db.init_schema()
            print(f"Midiendo a los campeones de las temporadas sin medición (1 de cada {step}, {games} partidas por rival)...")
            n = benchmark.backfill(db, cfg, step, games)
            print(f"{n} temporada(s) medidas. Ahora: `progress`.")
            return 0
        finally:
            db.close()
    n = int(args[0]) if args and args[0].isdigit() and int(args[0]) > 0 else None
    if args and n is None:
        print("Uso: python -m evolution progress [N] | progress --backfill [cada] [partidas]", file=sys.stderr)
        return 2
    return _read_only(cfg, lambda conn, out: progress.render_progress(conn, out, n or 25))


def cmd_bench(cfg, tokens: list[str]) -> int:
    import secrets
    seed, rest = None, []
    for t in tokens:
        if t.lower().startswith("seed=") and t[5:].isdigit():
            seed = int(t[5:])
        else:
            rest.append(t)
    games = 200
    if len(rest) == 2 and rest[1].isdigit():
        games = int(rest[1])
    elif len(rest) != 1:
        print("Uso: bench A [partidas] [seed=N]  (A: nº de temporada = mejor RED, color #rrggbb o NOMBRE de un estático)", file=sys.stderr)
        return 2
    if not 2 <= games <= 5000:
        print("Las partidas por rival deben estar entre 2 y 5000.", file=sys.stderr)
        return 2
    ias = _resolve(cfg, rest[:1])
    if ias is None:
        return 1
    seed = secrets.randbits(31) if seed is None else seed
    out = Display(cfg.color)
    sandbox.show(out, ias)
    out.emit(f"Midiendo {out.ia(ias[0]['color'])} contra los bots fijos ({games} partidas por rival)...")
    try:
        res = benchmark.bench_ia(ias[0], games, seed, lambda ia: sandbox.make_player(ia, seed))
    except Exception as e:  # noqa: BLE001
        print(f"No se pudo medir: {e}", file=sys.stderr)
        return 1
    benchmark.render_bench(out, ias[0], res, games, seed)
    return 0


def cmd_ladder(cfg, args: list[str]) -> int:
    nums = [a for a in args if a.isdigit() and int(a) > 0]
    names = [a for a in args if a not in nums]
    if len(nums) > 2 or any(a.isdigit() for a in names):
        print("Uso: ladder [paso] [max] [nombre_estatico ...]  (ej.: ladder 5 40 BocaJrs)", file=sys.stderr)
        return 2
    step = int(nums[0]) if nums else None
    max_n = int(nums[1]) if len(nums) > 1 else 20
    if max_n > 60:
        print("Máximo 60 campeones (son n·(n-1) partidas).", file=sys.stderr)
        return 2
    found: dict = {}

    def fn(conn, out):
        try:
            champs = ladder.pick_champions(conn, cfg.models_dir, step, max_n)
            extra = sandbox.resolve_many(conn, cfg.models_dir, names, cfg.statics_dir) if names else []
            if any(e.get("type") != "estatico" for e in extra):
                raise play.PlayError("Los nombres extra del torneo deben ser bots estáticos (los campeones de cada temporada se eligen solos).")
            found["champs"] = champs + [{"season": None, "color": e["color"], "cls": e["cls"], "slug": e["slug"],
                                         "path": None, "sha256": None} for e in extra]
        except play.PlayError as e:
            print(f"No se puede: {e}", file=sys.stderr)

    rc = _read_only(cfg, fn)
    if "champs" not in found:
        return rc or 1
    out = Display(cfg.color)
    champs = found["champs"]
    if len(champs) < 3:
        out.emit("Hacen falta al menos 3 campeones distintos con su .pth guardado (más temporadas completadas) o estáticos.")
        return 1
    out.emit(f"Torneo maestro con {len(champs)} participantes ({len(champs) * (len(champs) - 1)} partidas)...")
    try:
        res = ladder.run_ladder(champs)
    except Exception as e:  # noqa: BLE001
        print(f"No se pudo ejecutar el torneo: {e}", file=sys.stderr)
        return 1
    ladder.render_ladder(out, res)
    return 0


def cmd_statics(cfg) -> int:
    return _read_only(cfg, lambda conn, out: reports.render_statics(conn, out, cfg))


def cmd_disk(cfg) -> int:
    return _read_only(cfg, lambda conn, out: disk.render(out, disk.collect(conn, cfg), cfg))


def cmd_prune_orphans(cfg, confirm: bool) -> int:
    db = Database(cfg.database_url, cfg.db_retries, cfg.db_retry_base_delay)
    try:
        db.connect_with_retries()
        orphans = db.run(lambda c: disk.find_orphans(c, cfg.models_dir))
        total = sum(p.stat().st_size for p in orphans)
        for p in orphans[:20]:
            print(f"  {p.name}  {p.stat().st_size / 1e6:.2f} MB")
        if len(orphans) > 20:
            print(f"  ... y {len(orphans) - 20} más")
        if not orphans:
            print("No hay archivos huérfanos.")
        elif not confirm:
            print(f"{len(orphans)} archivo(s) huérfanos ({total / 1e6:.1f} MB). Para borrarlos: prune --orphans --yes")
        else:
            freed = disk.delete_orphans(orphans, cfg.models_dir)
            print(f"{len(orphans)} archivo(s) huérfanos borrados ({freed / 1e6:.1f} MB).")
        return 0
    except AlreadyRunningError as e:
        print(f"ERROR: {e}. Detené el entrenamiento primero.", file=sys.stderr)
        return 3
    finally:
        db.close()


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
                if choice in ("2", "3") and args and n is None:
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
                    cmd_play(cfg, n, None if n or not args else args[0])
                elif choice == "5":
                    cmd_fight(cfg, args or input("IA 1 y IA 2 (temporada o #color): ").split())
                elif choice == "6":
                    cmd_testleague(cfg, args or input("IA a probar (temporada o #color, separadas por espacio): ").split())
                elif choice == "7":
                    if not args:
                        cmd_addai(cfg, ["--list"])
                        args = shlex.split(input("Archivo .pth (ruta o nombre dentro de models/) o NOMBRE de un bot estático: "))
                    if args:
                        cmd_addai(cfg, args)
                elif choice == "8":
                    cmd_progress(cfg, args)
                elif choice == "9":
                    cmd_bench(cfg, args or input("IA a medir (temporada, #color o nombre de un estático) y partidas opcionales: ").split())
                elif choice == "10":
                    cmd_ladder(cfg, args)
                elif choice == "11":
                    cmd_disk(cfg)
                elif choice == "12":
                    cmd_statics(cfg)
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
        arg = argv[1] if len(argv) > 1 else None
        if arg is not None and not (arg.isdigit() and int(arg) > 0):
            return cmd_play(cfg, None, arg)               # nombre de un bot estático
        if len(argv) > 2:
            print("Uso: python -m evolution play [N|nombre]", file=sys.stderr)
            return 2
        return cmd_play(cfg, int(arg) if arg else None)
    if cmd == "fight":
        return cmd_fight(cfg, argv[1:])
    if cmd == "testleague":
        return cmd_testleague(cfg, argv[1:])
    if cmd == "addai":
        if len(argv) < 2:
            print("Uso: python -m evolution addai archivo.pth [otro.pth ...] | --list | --cancel #color", file=sys.stderr)
            return 2
        return cmd_addai(cfg, argv[1:])
    if cmd == "progress":
        return cmd_progress(cfg, argv[1:])
    if cmd == "bench":
        return cmd_bench(cfg, argv[1:])
    if cmd == "ladder":
        return cmd_ladder(cfg, argv[1:])
    if cmd == "statics":
        return cmd_statics(cfg)
    if cmd == "disk":
        return cmd_disk(cfg)
    if cmd == "prune":
        if "--orphans" in argv[1:]:
            return cmd_prune_orphans(cfg, "--yes" in argv[1:])
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
        return Trainer(cfg, db, Display(cfg.color, level=cfg.output), stop).run()
    except AlreadyRunningError as e:
        print(f"ERROR: {e}. Detené el otro proceso antes de iniciar uno nuevo.", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())