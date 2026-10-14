#!/usr/bin/env bash
# Reintento automático de "mate en N" con MÁS DATOS cuando el entrenamiento se estanca.
#
# Ronda 1: arranca desde el modelo canónico (CANON_INIT) con PPL puzzles por nivel.
# Si el nivel objetivo no queda 'mastered' en el reporte, reintenta con más puzzles
# (+PPL_STEP), una semilla de train nueva (=> datos nuevos) y --eval-seed FIJO (=> el
# mismo examen, comparación justa), reanudando desde el mejor modelo de la ronda previa.
#
# Uso:
#   setsid bash trainPuzzles/retrain_loop.sh > /tmp/opencode/retrain.log 2>&1 < /dev/null &
#
# Parámetros (variables de entorno):
#   TARGET_LEVEL=2  ROUNDS=5  PPL=25000  PPL_STEP=25000
#   MAX_EPOCHS=3000   (tope DURO de epochs por nivel y ronda: garantiza que la ronda
#                      termine para poder pasar a la siguiente con más datos)
#   PLATEAU=1500      (corte anticipado si el mejor acierto no mejora; opcional)
#   SEED=0  EVAL_SEED=0  DROPOUT=0.2  WEIGHT_DECAY=1e-4  WARM_START=1
#   THRESHOLD=0.90  MAX_LEVEL=3
#   CANON_INIT=trainPuzzles/checkpoints/puzzle_canon.pth
#   OUTDIR=trainPuzzles/checkpoints
set -euo pipefail

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
CANON_INIT="${CANON_INIT:-trainPuzzles/checkpoints/puzzle_canon.pth}"
OUTDIR="${OUTDIR:-trainPuzzles/checkpoints}"

REL="${OUTDIR#trainPuzzles/}"

ppl="$PPL"
seed="$SEED"
init="$CANON_INIT"
round=1

while [ "$round" -le "$ROUNDS" ]; do
  out_cont="trainPuzzles/${REL}/puzzle_retry${round}.pth"
  out_host="${OUTDIR}/puzzle_retry${round}.pth"
  report_host="${OUTDIR}/puzzle_retry${round}_report.json"
  maxg=$(( ppl * 5 ))

  echo "==== Ronda ${round}/${ROUNDS} | puzzles/nivel=${ppl} | max-games=${maxg} | seed=${seed} | eval-seed=${EVAL_SEED} | init=${init} ===="

  docker compose run --rm --no-deps --user "$(id -u):$(id -g)" \
    -v ./trainPuzzles:/app/trainPuzzles trainer \
    python -m trainPuzzles \
      --levels 1 --max-level "${MAX_LEVEL}" --threshold "${THRESHOLD}" \
      --games-budget 0 --max-games "${maxg}" \
      --puzzles-per-level "${ppl}" \
      --dropout "${DROPOUT}" --weight-decay "${WEIGHT_DECAY}" \
      --max-epochs "${MAX_EPOCHS}" --plateau-stop "${PLATEAU}" \
      --seed "${seed}" --eval-seed "${EVAL_SEED}" \
      --init "${init}" --out "${out_cont}"

  if [ ! -f "$report_host" ]; then
    echo "!! No se encontró el reporte ${report_host}; abortando."
    exit 1
  fi

  mastered=$(python3 - "$report_host" "$TARGET_LEVEL" <<'PY'
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
    echo "**** mate ${TARGET_LEVEL} DOMINADO en la ronda ${round}. Modelo: ${out_host}"
    exit 0
  fi

  ppl=$(( ppl + PPL_STEP ))
  seed=$(( seed + 1 ))
  if [ "$WARM_START" = "1" ]; then
    init="$out_cont"
  fi
  round=$(( round + 1 ))
done

echo "**** Se agotaron las ${ROUNDS} rondas sin dominar mate ${TARGET_LEVEL}."
exit 2
