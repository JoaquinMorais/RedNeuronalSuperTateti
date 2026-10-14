# Entrenamiento de puzzles (mate en N) — cómo leer el log y cómo ejecutarlo

Este archivo explica **qué significa cada campo** de las líneas que imprime el entrenador
(`python -m trainPuzzles`) y **cómo correrlo**. La referencia completa de flags está en
`trainPuzzles/README.md`.

---

## 1. Línea de entrenamiento `[TRAIN]`

Ejemplo real:

```
[20:41:16] [TRAIN]   N2 ep  60 | loss 0.883 (P 0.883 V 1.000) | acierto  67.5% base  66.6% d  +0.9 | train  73.9% brecha  +6.4 | mejor  68.6% | ret  80.3% | [########--] 3.7s | meta 90%
```

Campo por campo:

| Campo | Significado |
|---|---|
| `[20:41:16]` | Hora del registro (log). |
| `[TRAIN]` | Fase del programa. Las fases son `[GEN]` (generación), `[TRAIN]` (entrenamiento), `[REV]` (revisión anti-olvido) e `[INFO]` (información). |
| `N2` | Nivel 2, es decir "mate en 2". |
| `ep 60` | **Epoch 60**: la vuelta número 60 sobre *todos* los puzzles de entrenamiento. |
| `loss 0.883` | **Pérdida total** que minimiza el optimizador (combina política y valor). |
| `(P 0.883 V 1.000)` | Desglose de la pérdida: **P** = pérdida de **política** (qué jugada elegir), **V** = pérdida de **valor** (evaluar la posición). Con `--value-coef 0` (por defecto) la de valor no afecta y queda fija en `1.000`. |
| `acierto 67.5%` | **Acierto en el set de EVALUACIÓN** de mate-2: puzzles NUEVOS que la red no vio en entrenamiento. **Es la métrica que importa** (mide generalizar, no memorizar). |
| `base 66.6%` | **Baseline**: el acierto de la red inicial (antes de entrenar este nivel) sobre ese mismo set de evaluación. |
| `d +0.9` | **Delta** = `acierto − base` (67.5 − 66.6). Cuánto mejoró respecto de la base. |
| `train 73.9%` | Acierto sobre una **muestra fija de los puzzles de ENTRENAMIENTO** (el "probe"). Mide cuánto *memorizó*. |
| `brecha +6.4` | **Gap** = `train − acierto` (73.9 − 67.5). Diferencia entre memorizar y generalizar. **Cuanto más grande, más sobreajuste.** |
| `mejor 68.6%` | **Mejor acierto de evaluación** alcanzado en este nivel hasta ahora. |
| `ret 80.3%` | **Retención**: acierto sobre la unión de los sets de evaluación de los niveles vistos hasta el actual (1..N). Sirve para detectar que no se "olvidan" los niveles viejos. |
| `[########--]` | **Barra de progreso** hacia el umbral: `acierto / umbral` (67.5 / 90 ≈ 75%). 8 de 10. |
| `3.7s` | **Duración del epoch** (segundos). |
| `meta 90%` | **Estado**: falta llegar al umbral (`--threshold`, 90%). Cuando lo logra, dice `MASTERED`. |

### Cómo interpretarlo
- Subí `acierto` y se mantiene/disminuye `brecha` → la red **generaliza** (lo que buscamos).
- `train` cerca del 100% con `brecha` grande (ej. +30 o más) → está **memorizando**. El
  `--dropout` y el `--weight-decay` sirven justamente para bajarlo.
- `ret` cayendo mucho por debajo del umbral → se están **olvidando** los niveles previos.

---

## 2. Línea de generación `[GEN]`

Ejemplo real:

```
[20:37:27] [GEN]   mate 2: 24419/25000 (97.7%) [##########] 31.2 puz/s | partidas 42345 | ETA 0:18
```

Campo por campo:

| Campo | Significado |
|---|---|
| `[20:37:27]` | Hora del registro. |
| `[GEN]` | Fase de **generación** de puzzles (juega partidas al azar y filtra con el solver). |
| `mate 2:` | Nivel que se está generando (puzzles de victoria forzada en 2 jugadas propias). |
| `24419/25000` | Puzzles **conseguidos hasta ahora** / **objetivo** (`--puzzles-per-level`). |
| `(97.7%)` | Porcentaje del objetivo ya juntado. |
| `[##########]` | Barra de progreso (10 casillas). |
| `31.2 puz/s` | **Velocidad**: puzzles nuevos por segundo. |
| `partidas 42345` | **Partidas aleatorias** jugadas hasta ahora (el "combustible" de la generación). |
| `ETA 0:18` | **ETÁ** (*Estimated Time of Arrival*): **tiempo estimado que falta** para terminar, formato `m:ss` o `h:mm:ss` (aquí 18 segundos). Es solo una estimación que se recalcula según la velocidad real. |

### Notas sobre `--games-budget` y `--max-games`
El tope se mide en **partidas** (`partidas`), no en puzzles. Con `--games-budget 0` (sin tope)
el generador sigue jugando **hasta juntar el objetivo**, usando `--max-games` como tope de
seguridad. Si el objetivo no se alcanza, el log avisa: `aviso: no se alcanzó el target ...`.

---

## 3. Otros mensajes útiles

| Mensaje | Significado |
|---|---|
| `===== Nivel 2 (mate en 2) =====` | Empieza el entrenamiento de ese nivel; muestra `train=`, `eval=`, `umbral%`, `retención%`, `min-epochs`, `patience`. |
| `nivel 2 DOMINADO: ...` | El nivel superó el umbral durante `patience` evaluaciones → sube al siguiente. |
| `nivel 2: N epochs sin mejorar ... se corta por --plateau-stop N` | Se estancó y `--plateau-stop` cortó la corrida (para reintentar con más datos). |
| `Baseline (red inicial) por nivel: mate 1: 93.1% ...` | Acierto de la red de partida en cada set de evaluación. |
| `RESUMEN DEL APRENDIZAJE` (al final) | Tabla con `base`, `final`, `cambio`, `epochs`, `dominio` y una clasificación por nivel. |
| `Reporte guardado en ...` | Se escribió el JSON con todo el resumen (junto al modelo `<out>_report.json`). |

---

## 4. Comandos

> Todos se ejecutan desde la **raíz del proyecto**. La imagen de Docker ya tiene que estar
> construida (`docker compose build trainer`) si cambiaste `model.py`.

### 4.1 Construir la imagen (solo si cambió `model.py`)

```bash
docker compose build trainer
```

### 4.2 Corrida básica (Docker)

```bash
docker compose run --rm --no-deps --user "$(id -u):$(id -g)" \
  -v ./trainPuzzles:/app/trainPuzzles trainer \
  python -m trainPuzzles \
    --levels 1 --max-level 3 --threshold 0.90 \
    --games-budget 0 --max-games 200000 \
    --dropout 0.2 --weight-decay 1e-4 \
    --init trainPuzzles/checkpoints/puzzle_canon.pth \
    --out  trainPuzzles/checkpoints/puzzle_canon_m3.pth
```

### 4.3 Corrida local (sin Docker)

Si tenés `torch` instalado en el host (los tests lo usan):

```bash
python3 -m trainPuzzles \
  --levels 1 --max-level 3 --threshold 0.90 \
  --games-budget 0 --max-games 200000 \
  --dropout 0.2 --weight-decay 1e-4 \
  --init trainPuzzles/checkpoints/puzzle_canon.pth \
  --out  trainPuzzles/checkpoints/puzzle_canon_m3.pth
```

### 4.4 Experimento A/B de regularización (mismo examen, con y sin regularización)

`--eval-seed` fijo para que el set de evaluación NO cambie entre corridas.

```bash
# Control (sin regularización)
docker compose run --rm --no-deps --user "$(id -u):$(id -g)" \
  -v ./trainPuzzles:/app/trainPuzzles trainer \
  python -m trainPuzzles --levels 1 --max-level 2 --threshold 0.90 \
    --games-budget 0 --max-games 50000 --puzzles-per-level 10000 \
    --plateau-stop 1500 --seed 0 --eval-seed 0 \
    --init trainPuzzles/checkpoints/puzzle_canon.pth \
    --out  trainPuzzles/checkpoints/ab_control.pth

# Tratamiento (dropout + weight decay)
docker compose run --rm --no-deps --user "$(id -u):$(id -g)" \
  -v ./trainPuzzles:/app/trainPuzzles trainer \
  python -m trainPuzzles --levels 1 --max-level 2 --threshold 0.90 \
    --games-budget 0 --max-games 50000 --puzzles-per-level 10000 \
    --plateau-stop 1500 --seed 0 --eval-seed 0 \
    --dropout 0.2 --weight-decay 1e-4 \
    --init trainPuzzles/checkpoints/puzzle_canon.pth \
    --out  trainPuzzles/checkpoints/ab_dropout.pth
```

### 4.5 Reintento automático con más datos (`retrain_loop.sh`)

Repite el entrenamiento subiendo los puzzles cuando se estanca (`--max-epochs 3000` por ronda
como tope duro, `--plateau-stop 1500` como corte anticipado), reanudando desde el mejor modelo
de la ronda anterior.

**En primer plano (ves todo en vivo):**

```bash
cd /home/joaquin/Escritorio/Python/RedNeuronalSuperTateti
ROUNDS=2 bash trainPuzzles/retrain_loop.sh
```

**En segundo plano (escribe a un log):**

```bash
ROUNDS=2 setsid bash trainPuzzles/retrain_loop.sh > /tmp/opencode/retrain.log 2>&1 < /dev/null &
```

Parámetros por variable de entorno: `TARGET_LEVEL` (2), `ROUNDS` (5), `PPL` (25000),
`PPL_STEP` (25000), `MAX_EPOCHS` (3000), `PLATEAU` (1500), `SEED` (0), `EVAL_SEED` (0),
`DROPOUT` (0.2), `WEIGHT_DECAY` (1e-4), `WARM_START` (1), `THRESHOLD` (0.90), `MAX_LEVEL` (3).

### 4.6 Ver el log y frenar

```bash
# Seguir una corrida en segundo plano (salir con Ctrl-C NO la detiene)
tail -f /tmp/opencode/retrain.log

# Frenar una corrida en primer plano
Ctrl-C

# Frenar el contenedor del trainer (cierra prolijo: guarda mejor estado + reporte)
docker stop $(docker ps -q --filter ancestor=redneuronalsupertateti-trainer)

# Ver si quedó algún contenedor del trainer
docker ps --filter ancestor=redneuronalsupertateti-trainer
```

### 4.7 Leer el resultado de una corrida

```bash
python3 -c "import json; r=json.load(open('trainPuzzles/checkpoints/puzzle_retry1_report.json')); \
print([(l['level'], l['baseline'], l['final'], l['best'], l['mastered']) for l in r['levels']])"
```

---

## 5. Todos los flags de la CLI

### Rutas y salida

| Flag | Default | Qué hace |
|---|---|---|
| `--init` | `model.pth` | `.pth` de partida. Si no existe, arranca de una red aleatoria. |
| `--out` | `trainPuzzles/checkpoints/puzzle_net.pth` | `.pth` de salida (el mejor estado del último nivel). |
| `--report` | `<out>_report.json` | Reporte JSON del aprendizaje. |

### Currículo y dominio de nivel

| Flag | Default | Qué hace |
|---|---|---|
| `--levels` | `4` | Niveles iniciales pre-generados (`1..N`). |
| `--max-level` | `0` | Nivel máximo a alcanzar. `0` = sin tope. |
| `--min-level-puzzles` | `200` | Mínimo de puzzles para considerar entrenable un nivel nuevo. |
| `--threshold` | `0.95` | Acierto mínimo en evaluación para dominar un nivel. |
| `--retention-threshold` | `= --threshold` | Acierto mínimo en la unión de niveles previos (anti-olvido). |
| `--patience` | `2` | Evaluaciones seguidas sobre el umbral para dominar. |
| `--min-epochs` | `15` | Epochs mínimos en un nivel antes de poder avanzar. |
| `--max-epochs` | `0` | Tope de seguridad de epochs por nivel; `0` = sin tope. Si se alcanza sin dominar, corta el currículo. |
| `--plateau-patience` | `10` | Avisa si pasan N epochs sin mejorar el mejor acierto. |
| `--plateau-stop` | `0` | Corta el nivel y la corrida si pasan N epochs sin mejorar (`0` = off). |
| `--log-every` | `10` | Imprime la línea `[TRAIN]` 1 de cada N epochs (`<=0` = todas). |
| `--revision-epochs` | `6` | Epochs de revisión anti-olvido tras dominar cada nivel. |
| `--revision-puzzles` | `500` | Puzzles por fase de revisión. |
| `--big` | `0.20` | Cambio (en acierto) para clasificar "mejoró/empeoró mucho". |
| `--small` | `0.05` | Cambio (en acierto) para clasificar "mejoró/empeoró poco". |

### Datos y generación de puzzles

| Flag | Default | Qué hace |
|---|---|---|
| `--puzzles-per-level` | `25000` | Puzzles de entrenamiento por nivel (con pocos, la red los memoriza). |
| `--eval-size` | `1000` | Puzzles del set de evaluación por nivel. |
| `--games-budget` | `0` | Tope duro de partidas aleatorias por generación. `0` = sin tope: juega hasta juntar el target (usa `--max-games` como seguridad). |
| `--max-games` | `200000` | Tope de seguridad de partidas cuando `--games-budget=0`. |
| `--min-turn` | `16` | Solo se resuelven posiciones con turno ≥ este valor. |
| `--solver-max-nodes` | `120000` | Presupuesto de nodos del solver por posición (`0` = sin límite). |

### Entrenamiento y regularización

| Flag | Default | Qué hace |
|---|---|---|
| `--no-augment` | off | Desactiva las simetrías D4 (por defecto, una al azar por muestra y epoch). |
| `--batch-size` | `64` | Tamaño de lote. |
| `--lr` | `3e-4` | Learning rate (Adam). |
| `--dropout` | `0.0` | Dropout **funcional** en las capas compartidas (solo entrenando; en eval está apagado). No agrega módulos, el `state_dict` queda idéntico. `0` = off. |
| `--weight-decay` | `0.0` | Penalización L2 de Adam (weight decay). `0` = off. |
| `--value-coef` | `0.0` | Peso de la pérdida de la cabeza de valor. `0` = off. |
| `--replay-frac` | `0.30` | Fracción de puzzles de niveles previos en cada nivel (anti-olvido). |

### Reproducibilidad

| Flag | Default | Qué hace |
|---|---|---|
| `--seed` | `0` | Semilla de generación de train y mezcla (reproducible). |
| `--eval-seed` | `= --seed` | Semilla del set de **evaluación**. Fijala para comparar entre corridas con datos de train distintos (p. ej. reintentos). |

---

## 6. Resumen mental en una frase
- **`[GEN]`**: junta puzzles (mirá `puz/s` y `ETA`).
- **`[TRAIN]`**: entrena (mirá `acierto` = generalización, `brecha` = sobreajuste, `ret` = olvido).
- **`[REV]`**: repasa niveles viejos para no olvidarlos.
- Al final: tabla **RESUMEN DEL APRENDIZAJE** por nivel.
