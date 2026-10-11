"""Salida por terminal. Funciona con y sin colores (sin TTY, NO_COLOR, Docker sin -t)."""
from __future__ import annotations

import sys

from . import colors as colormod
from . import titles as titlemod

RESET = "\x1b[0m"
BOLD = "\x1b[1m"
DIM = "\x1b[2m"

COMP_LABEL = {"league": "LIGA ", "cup_group": "COPA·G", "cup_ko": "COPA·E", "recopa": "RECOPA"}
ROUND_NAME = {"R16": "Octavos", "QF": "Cuartos", "SF": "Semifinales", "F": "Final"}
RESOLUTION = {
    "two_wins": "gana las dos partidas", "win_and_draw": "una victoria y un empate",
    "third_game": "gana la 3.ª partida", "third_game_draw_second_starter": "3.ª partida empatada: pasa quien comenzó segundo",
}


def fmt_time(ms: float) -> str:
    s = ms / 1000.0
    if s < 1:
        return f"{ms:.0f}ms"
    if s < 60:
        return f"{s:.2f}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{int(m)}m{s:04.1f}s"
    h, m = divmod(m, 60)
    return f"{int(h)}h{int(m):02d}m"


LEVELS = {"temporada": 0, "fecha": 1, "full": 2}


def bench_text(b: dict | None) -> str:
    """'1.º vs aleatorio 63% · vs táctico 12% | población 52% · 6%' (vacío si esta temporada no se midió)."""
    if not b:
        return ""
    def g(subject, opp):
        st = b.get((subject, opp))
        return f"{st['score']:.0%}" if st and st["n"] else "—"
    return (f"1.º vs aleatorio {g('top1', 'random')} · vs táctico {g('top1', 'tactical')} | "
            f"población vs aleatorio {g('population', 'random')} · vs táctico {g('population', 'tactical')}")


class Display:
    """``level``: 'full' (todo) · 'fecha' (una línea por fecha + eventos de Copa + informe de temporada) ·
    'temporada' (una línea por temporada). Sirve para no llenar el disco con logs en entrenamientos largos."""

    def __init__(self, color: bool = False, stream=None, level: str = "full"):
        self.level = level
        self.color = color
        self.stream = stream or sys.stdout
        # {color: {"L": [...], "C": [...], "RC": [...]}}: IA que ya ganaron algo (ver titles.py)
        self.titles: dict = {}

    def shown(self, needed: str) -> bool:
        """¿Se imprime lo que corresponde al nivel ``needed``?"""
        return LEVELS[self.level] >= LEVELS[needed]

    def log(self, needed: str, text: str) -> None:
        if self.shown(needed):
            self.emit(text)

    def emit(self, text: str = "") -> None:
        try:
            print(text, file=self.stream, flush=True)
        except UnicodeEncodeError:
            print(text.encode("ascii", "replace").decode(), file=self.stream, flush=True)

    # -- estilo ----------------------------------------------------------
    def tag(self, hex_color: str, badge: bool = True) -> str:
        """Texto visible de una IA: '#7d48e6' o '#7d48e6:01L:00C:00RC' si ya ganó algo."""
        return hex_color + (titlemod.badge(self.titles.get(hex_color)) if badge else "")

    def paint(self, text: str, hex_color: str) -> str:
        if not self.color:
            return text
        r, g, b = colormod.to_rgb(hex_color)
        return f"\x1b[38;2;{r};{g};{b}m{text}{RESET}"

    def ia(self, hex_color: str, suffix: str = "", badge: bool = True) -> str:
        t = self.tag(hex_color, badge)
        if not self.color:
            return f"{t}{suffix}"
        return self.paint(f"■ {t}{suffix}", hex_color)

    def id_width(self) -> int:
        """Ancho de la columna 'IA' en las tablas (con insignia si alguien tiene títulos)."""
        longest = max((len(titlemod.badge(t)) for t in self.titles.values()), default=0)
        return 7 + longest

    def cell(self, hex_color: str, width: int | None = None, badge: bool = True) -> str:
        """``ia`` rellenada hasta ``width`` columnas VISIBLES (los códigos ANSI no cuentan)."""
        width = self.id_width() if width is None else width
        visible = len(self.tag(hex_color, badge)) + (2 if self.color else 0)
        return self.ia(hex_color, badge=badge) + " " * max(0, width - visible)

    def bold(self, t: str) -> str:
        return f"{BOLD}{t}{RESET}" if self.color else t

    def dim(self, t: str) -> str:
        return f"{DIM}{t}{RESET}" if self.color else t

    def banner(self, text: str) -> None:
        line = "=" * 78
        self.emit(self.bold(f"\n{line}\n {text}\n{line}"))

    def section(self, text: str) -> None:
        self.emit(self.bold(f"\n--- {text} ---"))

    # -- partidas --------------------------------------------------------
    def match_line(self, m: dict, color_of: dict, outcome, prog: dict, info: str = "") -> str:
        x, o = color_of[m["x_individual_id"]], color_of[m["o_individual_id"]]
        res = {"x": f"gana {self.ia(x)}", "o": f"gana {self.ia(o)}", "draw": "empate"}[outcome.result]
        tag = COMP_LABEL[m["competition"]]
        counter = self.dim("(%d/%d)" % (prog["done"], prog["total"]))
        return (f"  {tag:<7}{info:<12} {self.ia(x)} (comienza) vs {self.ia(o)}  -> {res}"
                f"  [{outcome.num_moves} jug., {fmt_time(outcome.duration_ms)}]  {counter}")

    # -- tablas ----------------------------------------------------------
    def league_table(self, rows: list[dict], md: int) -> None:
        if not self.shown("full"):
            return
        self.section(f"Tabla de Liga tras la fecha {md}")
        w = self.id_width() + (2 if self.color else 0)
        self.emit(f"  {'Pos':>3}  {'IA':<{w}} {'PJ':>3} {'G':>3} {'E':>3} {'P':>3} {'Pts':>4}")
        for r in rows:
            self.emit(f"  {r['position']:>3}  {self.cell(r['color'], w)} {r['played']:>3} {r['wins']:>3} "
                      f"{r['draws']:>3} {r['losses']:>3} {r['points']:>4}")

    def event(self, ev: dict) -> None:
        if not self.shown("fecha"):
            return
        t = ev["type"]
        if t == "groups_resolved":
            self.section("COPA · fase de grupos terminada")
            for g, rows in ev["tables"].items():
                self.emit(f"  Grupo {g}:")
                for pos, (p, r) in enumerate(rows, start=1):
                    mark = "clasifica" if pos <= 4 else "ELIMINADA"
                    self.emit(f"    {pos}. {self.ia(p)}  {r.points} pts ({r.wins}G {r.draws}E {r.losses}P)  {mark}")
            if ev["draws"]:
                self.emit(self.dim(f"  Desempates por sorteo reproducible: {', '.join(ev['draws'])}"))
            self.emit("  Octavos de final programados (fecha 18):")
            for sa, ca, sb, cb in ev["r16"]:
                self.emit(f"    {sa} {self.ia(ca)}  vs  {sb} {self.ia(cb)}")
        elif t == "tie_third_game":
            self.emit(self.bold(f"  COPA · {ROUND_NAME[ev['round']]} #{ev['tie_no']}: eliminatoria empatada -> 3.ª partida "
                                f"(comienza {self.ia(ev['x'])}, sorteado)"))
        elif t == "tie_finished":
            self.emit(self.bold(f"  COPA · {ROUND_NAME[ev['round']]} #{ev['tie_no']}: pasa {self.ia(ev['winner'])} "
                                f"sobre {self.ia(ev['loser'])} ({RESOLUTION[ev['resolution']]})"))
        elif t == "tie_scheduled":
            self.emit(f"  COPA · {ROUND_NAME[ev['round']]} #{ev['tie_no']} programado (fecha {ev['matchday']}): "
                      f"{self.ia(ev['a'])} vs {self.ia(ev['b'])}")
        elif t == "cup_finished":
            self.banner(f"COPA: campeón {self.ia(ev['champion'])} · subcampeón {self.ia(ev['runner_up'])}")
        elif t == "recopa_same":
            self.banner(f"RECOPA: {self.ia(ev['champion'])} ganó Liga y Copa -> campeón sin jugar")
        elif t == "recopa_scheduled":
            self.banner(f"RECOPA: campeón de Copa {self.ia(ev['cup'])} vs campeón de Liga {self.ia(ev['league'])} (10 partidas)")
        elif t == "recopa_finished":
            how = "sorteo reproducible" if ev["resolution"] == "random_draw" else "más victorias"
            self.emit(self.bold(f"  RECOPA: Copa {ev['wins_cup']} V · Liga {ev['wins_league']} V · {ev['draws']} empates "
                                f"-> campeón {self.ia(ev['champion'])} ({how})"))

    # -- resúmenes -------------------------------------------------------
    def header(self, *, season: dict, master_seed: int, cfg, db_status: str, progress: dict, resumed: bool) -> None:
        if not self.shown("fecha"):
            return
        self.banner(f"TEMPORADA {season['number']} {'(REANUDADA)' if resumed else '(NUEVA)'}")
        n, md_total = season.get("n_participants") or 20, season.get("league_matchdays") or 38
        extra = f"  ({n - 20} IA añadida{'s' if n - 20 != 1 else ''} a la generación normal)" if n != 20 else ""
        self.emit(f"  IA en la temporada      : {n}{extra}")
        self.emit(f"  Etapa                   : {season['stage']}  (última fecha cerrada: {season['current_matchday']}/{md_total})")
        sizes = (season.get("draw_meta") or {}).get("format", {}).get("group_sizes", [5, 5, 5, 5])
        self.emit(f"  Competiciones           : Liga ({md_total} fechas, {2 * (n - 1)} partidas por IA) · Copa (grupos de "
                  f"{'/'.join(map(str, sizes))}, octavos f18, cuartos f23, semis f30, final f37) · Recopa (tras f{md_total})")
        self.emit(f"  Directorio de modelos   : {cfg.models_dir}")
        self.emit(f"  Base de datos           : {db_status}")
        self.emit(f"  Semilla maestra/temporada: {master_seed} / {season['seed']}")
        self.emit(f"  Partidas                : {progress['done']} jugadas · {progress['pending']} pendientes (programadas)")

    def matchday_header(self, md: int, pending: int, total: int = 38) -> None:
        if not self.shown("full"):
            return
        self.banner(f"FECHA {md} / {total}")
        self.emit(f"  Partidas por jugar en esta fecha: {pending}")

    def matchday_footer(self, md: int, wall_ms: float, game_ms: float, prog: dict, season_ms: float) -> None:
        if not self.shown("fecha"):
            return
        self.emit(self.dim(f"  Fecha {md}: {fmt_time(wall_ms)} de reloj ({fmt_time(game_ms)} en partidas)"
                           f" · acumulado de partidas en la temporada: {fmt_time(season_ms)}"
                           f" · completadas {prog['done']} / faltan {prog['pending']} programadas"))

    def final_report(self, d: dict) -> None:
        if not self.shown("fecha"):
            top = d["global"][0]
            self.emit(f"Temporada {d['number']}: {len(d['global'])} equipos · Liga {self.ia(d['league'][0]['color'])} · "
                      f"Copa {self.ia(d['cup_champion'])} · Recopa {self.ia(d['recopa_champion'])} · "
                      f"1.º de la tabla anual {self.ia(top['color'])} ({top['total_score']} pts) · "
                      f"{fmt_time(d['sim_ms'])} en partidas" + (f" · MEDIDA: {bench_text(d.get('bench'))}" if d.get("bench") else ""))
            return
        self.banner(f"FIN DE LA TEMPORADA {d['number']}")
        self.section("Tabla final de Liga")
        w = self.id_width() + (2 if self.color else 0)
        self.emit(f"  {'Pos':>3}  {'IA':<{w}} {'Pts partido':>11}  {'Puntos evol.':>12}")
        for r in d["league"]:
            self.emit(f"  {r['league_position']:>3}  {self.cell(r['color'], w)} {r['league_points']:>11}  {r['league_score']:>12}")
        self.section("Copa")
        self.emit(f"  Campeón: {self.ia(d['cup_champion'])}   Subcampeón: {self.ia(d['cup_runner_up'])}")
        self.section("Recopa")
        self.emit(f"  Campeón: {self.ia(d['recopa_champion'])}")
        self.section("Clasificación global (Copa + Liga + Recopa)")
        self.emit(f"  {'Pos':>3}  {'IA':<{w}} {'Copa':>5} {'Liga':>5} {'Recopa':>7} {'TOTAL':>6}")
        for r in d["global"]:
            self.emit(f"  {r['final_position']:>3}  {self.cell(r['color'], w)} {r['cup_score']:>5} {r['league_score']:>5} "
                      f"{r['recopa_score']:>7} {r['total_score']:>6}")
        self.section("Siguiente generación")
        self.emit("  Se conservan intactas (7): " + ", ".join(self.ia(c) for c in d["survivors"]))
        self.emit("  Descendientes nuevos (13):")
        for ch in d["children"]:
            self.emit(f"    {self.ia(ch['color'])} <- hijo de {self.ia(ch['parent'])} (mutación {ch['level']})")
        if d.get("historical"):
            self.emit(f"  Nuevo mejor histórico: {self.ia(d['historical']['color'])} con {d['historical']['total']} pts -> {d['historical']['file']}")
        if d.get("bench"):
            self.emit(f"  Medición contra bots fijos ({'puntaje: victoria 1, empate 0,5'}): {bench_text(d['bench'])}")
        self.emit(self.bold(f"  Tiempo total de la temporada: {fmt_time(d['sim_ms'])} en partidas · {fmt_time(d['wall_ms'])} de reloj en esta sesión"))