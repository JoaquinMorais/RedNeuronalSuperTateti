"""Jugar contra una IA desde el navegador (servidor local mínimo, solo biblioteca estándar).

La IA juega exactamente como en los torneos: elige el movimiento legal de mayor
probabilidad según su red (``engine.select_move``), sin MCTS.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from game import SuperTateti

from . import modelstore, reports
from . import titles as titlemod
from .engine import select_move

UI_FILE = Path(__file__).with_name("play_ui.html")
MAX_BODY = 4096


class PlayError(Exception):
    """Error que se le muestra a la persona (rival no disponible, jugada ilegal, ...)."""


# ---------------------------------------------------------------------------
# Rival: mejor de la tabla anual de una temporada
# ---------------------------------------------------------------------------
def find_opponent(conn, models_dir: Path, season: int | None = None) -> dict:
    """1.ª IA de la tabla anual de la temporada ``season`` (por defecto, la última terminada)."""
    if season is None:
        season = conn.execute("SELECT max(number) AS n FROM seasons WHERE status='completed'").fetchone()["n"]
        if season is None:
            raise PlayError("Todavía no terminó ninguna temporada: no hay tabla anual.")
    s = conn.execute("SELECT * FROM seasons WHERE number=%s", (season,)).fetchone()
    if s is None:
        raise PlayError(f"No existe la temporada {season}.")
    if s["status"] != "completed":
        raise PlayError(f"La temporada {season} todavía no terminó: su tabla anual no está disponible.")
    r = conn.execute(
        """SELECT i.color, i.model_file, i.model_sha256, p.total_score, p.league_score, p.cup_score, p.recopa_score
           FROM season_participants p JOIN individuals i ON i.id = p.individual_id
           WHERE p.season_id=%s AND p.final_position=1""", (s["id"],)).fetchone()
    path = models_dir / r["model_file"]
    if not path.exists():
        raise PlayError(f"El modelo {r['model_file']} no está en {models_dir}.")
    badge = titlemod.badge(reports.titles_through(conn, season).get(r["color"]))
    return {"season": season, "color": r["color"], "tag": r["color"] + badge, "total": r["total_score"],
            "league": r["league_score"], "cup": r["cup_score"], "recopa": r["recopa_score"],
            "path": path, "sha256": r["model_sha256"]}


def load_opponent_net(opp: dict):
    return modelstore.load_net(opp["path"], opp["sha256"])


# ---------------------------------------------------------------------------
# Partida
# ---------------------------------------------------------------------------
class PlaySession:
    """Una persona contra una red. ``human``: 1 = X (comienza), -1 = O."""

    def __init__(self, net, opponent: dict):
        self.net, self.opponent = net, opponent
        self.lock = threading.Lock()
        self.game: SuperTateti | None = None
        self.human = 1
        self.last_human = self.last_ai = None
        self.wins = {"human": 0, "ai": 0, "draw": 0}
        self._counted = False

    # -- acciones --------------------------------------------------------
    def new_game(self, human: int) -> dict:
        if human not in (1, -1):
            raise PlayError("Elegí X (1) u O (-1).")
        with self.lock:
            self.game, self.human = SuperTateti(), human
            self.last_human = self.last_ai = None
            self._counted = False
            self._ai_turns()
            return self._state()

    def human_move(self, board: int, cell: int) -> dict:
        with self.lock:
            g = self.game
            if g is None:
                raise PlayError("Primero empezá una partida.")
            if g.winner is not None:
                raise PlayError("La partida ya terminó.")
            if g.current_player != self.human:
                raise PlayError("No es tu turno.")
            if not (isinstance(board, int) and isinstance(cell, int) and 0 <= board < 9 and 0 <= cell < 9) \
                    or g.get_valid_moves()[board * 9 + cell] != 1:
                raise PlayError("Jugada ilegal.")
            g.make_move(board, cell)
            self.last_human = [board, cell]
            self._ai_turns()
            return self._state()

    def state(self) -> dict:
        with self.lock:
            return self._state()

    # -- internos --------------------------------------------------------
    def _ai_turns(self) -> None:
        g = self.game
        while g.winner is None and g.current_player != self.human:
            mv = select_move(self.net, g)
            g.make_move(mv // 9, mv % 9)
            self.last_ai = [mv // 9, mv % 9]

    def _state(self) -> dict:
        opp = {k: self.opponent[k] for k in ("season", "color", "tag", "total", "league", "cup", "recopa")}
        if self.game is None:
            return {"started": False, "opponent": opp, "score": self.wins}
        g = self.game
        winner = None
        if g.winner is not None:
            winner = "draw" if g.winner == 0 else ("human" if g.winner == self.human else "ai")
            if not self._counted:
                self.wins[winner] += 1
                self._counted = True
        small = []
        for v in g.small_boards:
            small.append("p" if v == 0.5 else "d" if v == 0 else "x" if v == 1 else "o")
        return {
            "started": True, "opponent": opp, "score": self.wins,
            "cells": [[int(v) for v in row] for row in g.board], "small": small,
            "valid": [int(v) for v in g.get_valid_moves()], "active_board": g.active_board,
            "human": self.human, "current": g.current_player, "turn": g.turn, "winner": winner,
            "last_human": self.last_human, "last_ai": self.last_ai,
        }


# ---------------------------------------------------------------------------
# Servidor HTTP
# ---------------------------------------------------------------------------
def make_server(session: PlaySession, host: str, port: int) -> ThreadingHTTPServer:
    page = UI_FILE.read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # silencio: la terminal es del entrenador
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj: dict) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json; charset=utf-8")

        def do_GET(self):  # noqa: N802
            if self.path in ("/", "/index.html"):
                self._send(200, page, "text/html; charset=utf-8")
            elif self.path == "/api/state":
                self._json(200, session.state())
            else:
                self._json(404, {"error": "no encontrado"})

        def do_POST(self):  # noqa: N802
            try:
                n = int(self.headers.get("Content-Length") or 0)
                if n > MAX_BODY:
                    raise PlayError("Pedido demasiado grande.")
                data = json.loads(self.rfile.read(n) or b"{}")
                if self.path == "/api/new":
                    self._json(200, session.new_game({"x": 1, "o": -1}.get(data.get("human"), 0)))
                elif self.path == "/api/move":
                    self._json(200, session.human_move(data.get("board"), data.get("cell")))
                else:
                    self._json(404, {"error": "no encontrado"})
            except PlayError as e:
                self._json(400, {"error": str(e)})
            except (ValueError, TypeError, AttributeError):
                self._json(400, {"error": "Pedido inválido."})

    return ThreadingHTTPServer((host, port), Handler)


def serve(session: PlaySession, host: str, port: int, emit=print) -> None:
    """Sirve hasta Ctrl+C (y vuelve al llamador)."""
    server = make_server(session, host, port)
    o = session.opponent
    emit(f"\n  Rival: {o['tag']} · mejor de la tabla anual de la temporada {o['season']} ({o['total']} pts)")
    emit(f"  Abrí en el navegador:  http://localhost:{server.server_address[1]}")
    emit("  (Ctrl+C para terminar la partida y volver)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        emit("  Servidor de juego detenido.")