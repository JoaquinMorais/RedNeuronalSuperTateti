#!/usr/bin/env bash
# ==========================================================================
# retrain_loop_windows.sh -- Reintento automático de "mate en N" con MÁS
# DATOS, pensado para correr DESDE WINDOWS con Docker Desktop + WSL:
#
#     wsl bash trainPuzzles/retrain_loop_windows.sh
#
# Hace lo mismo que trainPuzzles/retrain_loop.sh, pero se ocupa de los
# detalles de Windows/WSL:
#   * Se ubica solo en la raíz del repo (no importa desde dónde lo lances).
#   * Verifica que Docker responde y elige python3/python para leer el reporte.
#
# IMPORTANTE (Windows): este archivo debe tener finales de línea LF, no CRLF.
# El .gitattributes del repo fuerza `*.sh text eol=lf`, así que un `git clone`
# en Windows lo deja bien. Si al correrlo ves errores raros tipo
# "$'\r': command not found", convertí los finales de línea con:
#   sed -i 's/\r$//' trainPuzzles/retrain_loop_windows.sh
#
# Ronda 1: arranca desde el modelo canónico (CANON_INIT) con PPL puzzles por
# nivel. Si el nivel objetivo no queda 'mastered', reintenta con más puzzles
# (+PPL_STEP), semilla de train nueva (=> datos nuevos) y --eval-seed FIJO
# (=> el mismo examen, comparación justa), reanudando desde el canon
# (puzzle_canon.pth), que cada ronda actualiza con el modelo que terminó.
#
# Cada ronda se archiva en checkpoints/runs/NNNN_retryN/ y actualiza el canon;
# el chequeo de 'mastered' lee siempre checkpoints/puzzle_canon_report.json.
#
# Parámetros (variables de entorno):
#   TARGET_LEVEL=2  ROUNDS=5  PPL=25000  PPL_STEP=25000
#   MAX_EPOCHS=3000   (tope DURO de epochs por nivel y ronda)
#   PLATEAU=1500      (corte anticipado si el mejor acierto no mejora)
#   SEED=0  EVAL_SEED=0  DROPOUT=0.2  WEIGHT_DECAY=1e-4  WARM_START=1
#   THRESHOLD=0.90  MAX_LEVEL=3
#   START_LEVEL=2     (nivel donde arranca el currículo; >1 salta el reentrenamiento
#                      de mate-1 y calienta directo en mate-2 desde CANON_INIT/canon)
#   CANON_INIT=trainPuzzles/checkpoints/puzzle_canon.pth
#   OUTDIR=trainPuzzles/checkpoints
#
# Ejemplos desde PowerShell / CMD (parado en la raíz del repo):
#   wsl bash trainPuzzles/retrain_loop_windows.sh
#   wsl bash -c "ROUNDS=2 bash trainPuzzles/retrain_loop_windows.sh"
#   wsl bash -c "ROUNDS=2 bash trainPuzzles/retrain_loop_windows.sh" > retrain.log 2>&1
# ==========================================================================

# --- 0. Robustez en Windows (ver nota del encabezado sobre LF/CRLF) --------
set -euo pipefail

# --- 1. Ubicarse en la raíz del repo ---------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
echo "== Repo: $REPO_ROOT"

# --- 2. Comprobar Docker y Python ------------------------------------------
if ! command -v docker >/dev/null 2>&1; then
  echo "!! No se encontró 'docker' en WSL."
  echo "   Instalá/abrí Docker Desktop y habilitá la integración con esta distro de WSL."
  exit 1
fi
if ! docker info >/dev/null 2>&1; then
  echo "!! Docker no responde. ¿Está corriendo Docker Desktop?"
  exit 1
fi

PY=""
for cand in python3 python; do
  if command -v "$cand" >/dev/null 2>&1; then PY="$cand"; break; fi
done
if [ -z "$PY" ]; then
  echo "!! Se necesita python3 (o python) en WSL para leer el reporte JSON."
  exit 1
fi

# --- 3. Parámetros ----------------------------------------------------------
TARGET_LEVEL="${TARGET_LEVEL:-2}"
ROUNDS="${ROUNDS:-5}"
PPL="${PPL:-25000}"
PPL_STEP="${PPL_STEP:-25000}"
MAX_EPOCHS="${MAX_EPOCHS:-3000}"
PLATEAU="${PLATEAU:-1500}"
SEED="${SEED:-0}"
EVAL_SEED="${EVAL_SEED:-0}"
DROPOUT="${DROPOUT:-0.2}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
WARM_START="${WARM_START:-1}"
THRESHOLD="${THRESHOLD:-0.90}"
MAX_LEVEL="${MAX_LEVEL:-3}"
START_LEVEL="${START_LEVEL:-1}"
CANON_INIT="${CANON_INIT:-trainPuzzles/checkpoints/puzzle_canon.pth}"
OUTDIR="${OUTDIR:-trainPuzzles/checkpoints}"

canon_pth="${OUTDIR}/puzzle_canon.pth"
report_host="${OUTDIR}/puzzle_canon_report.json"

ppl="$PPL"
seed="$SEED"
init="$CANON_INIT"
round=1

# --- 4. Bucle de rondas -----------------------------------------------------
while [ "$round" -le "$ROUNDS" ]; do
  run_name="retry${round}"
  maxg=$(( ppl * 5 ))

  echo "==== Ronda ${round}/${ROUNDS} | puzzles/nivel=${ppl} | max-games=${maxg} | seed=${seed} | eval-seed=${EVAL_SEED} | start-level=${START_LEVEL} | init=${init} | run=${run_name} ===="

  LR_SCHEDULE="${LR_SCHEDULE:-none}"
  LR_FINAL_FRAC="${LR_FINAL_FRAC:-0.03}"
  TTA="${TTA:-0}"
  EVAL_EVERY="${EVAL_EVERY:-1}"
  HARD_MINING="${HARD_MINING:-0}"
  MINE_EVERY="${MINE_EVERY:-10}"
  MINE_BOOST="${MINE_BOOST:-3.0}"
  MINE_WARMUP="${MINE_WARMUP:-25}"
  EMA="${EMA:-0}"
  EMA_DECAY="${EMA_DECAY:-0.999}"
  RANK_COEF="${RANK_COEF:-0.0}"
  RANK_MARGIN="${RANK_MARGIN:-1.0}"

  tta_flag=""
  if [ "$TTA" = "1" ] || [ "$TTA" = "true" ]; then
    tta_flag="--tta"
  fi
  hard_flag=""
  if [ "$HARD_MINING" = "1" ] || [ "$HARD_MINING" = "true" ]; then
    hard_flag="--hard-mining"
  fi
  ema_flag=""
  if [ "$EMA" = "1" ] || [ "$EMA" = "true" ]; then
    ema_flag="--ema"
  fi

  docker compose run --rm --no-deps --user "$(id -u):$(id -g)" \
    -v ./trainPuzzles:/app/trainPuzzles trainer \
    python -m trainPuzzles \
      --levels 1 --max-level "${MAX_LEVEL}" --threshold "${THRESHOLD}" \
      --start-level "${START_LEVEL}" --stop-at-level "${TARGET_LEVEL}" \
      --games-budget 0 --max-games "${maxg}" \
      --puzzles-per-level "${ppl}" \
      --dropout "${DROPOUT}" --weight-decay "${WEIGHT_DECAY}" \
      --lr-schedule "${LR_SCHEDULE}" --lr-final-frac "${LR_FINAL_FRAC}" \
      ${tta_flag} --eval-every "${EVAL_EVERY}" \
      ${hard_flag} --mine-every "${MINE_EVERY}" --mine-boost "${MINE_BOOST}" --mine-warmup "${MINE_WARMUP}" \
      ${ema_flag} --ema-decay "${EMA_DECAY}" \
      --rank-coef "${RANK_COEF}" --rank-margin "${RANK_MARGIN}" \
      --max-epochs "${MAX_EPOCHS}" --plateau-stop "${PLATEAU}" \
      --seed "${seed}" --eval-seed "${EVAL_SEED}" \
      --init "${init}" --out "${canon_pth}" --run-name "${run_name}"

  if [ ! -f "$report_host" ]; then
    echo "!! No se encontró el reporte ${report_host}; abortando."
    exit 1
  fi

  mastered=$("$PY" - "$report_host" "$TARGET_LEVEL" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
target = int(sys.argv[2])
for lvl in report.get("levels", []):
    if lvl.get("level") == target:
        print("1" if lvl.get("mastered") else "0")
        break
else:
    print("0")
PY
)

  echo "---- Ronda ${round}: mate ${TARGET_LEVEL} mastered=${mastered}"

  if [ "$mastered" = "1" ]; then
    echo "**** mate ${TARGET_LEVEL} DOMINADO en la ronda ${round}. Modelo canon: ${canon_pth}"
    exit 0
  fi

  ppl=$(( ppl + PPL_STEP ))
  seed=$(( seed + 1 ))
  if [ "$WARM_START" = "1" ]; then
    init="$canon_pth"
  fi
  round=$(( round + 1 ))
done

echo "**** Se agotaron las ${ROUNDS} rondas sin dominar mate ${TARGET_LEVEL}."
exit 2
