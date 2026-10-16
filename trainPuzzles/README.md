# trainPuzzles — entrenamiento táctico "mate en N"

Método **independiente** para entrenar `SuperTatetiNet` (MLP) o `SuperTatetiCNN` con
puzzles tácticos. Todo vive en esta carpeta: **no modifica ni depende de** `mcts.py`,
`trainer.py`, `evolution/`, el `Dockerfile` ni `docker-compose.yml`. El modelo entrenado
se guarda dentro de `trainPuzzles/` (por defecto `checkpoints_mts/` para `--arch mlp` y
`checkpoint_cnn/` para `--arch cnn`).

## Idea

1. Se generan posiciones a las que les toca mover y que tienen **victoria forzada
   en exactamente N jugadas propias** (contra cualquier defensa). Se llaman
   "mate en 1", "mate en 2", ...
2. La red se entrena con descenso de gradiente para elegir una de las jugadas
   óptimas (cross-entropy de política). La entrada es **canónica** (ver abajo) y cada
   muestra se rota/refleja al azar en cada epoch. La cabeza de valor está apagada por
   defecto (`--value-coef 0`): todos los puzzles son ganados y aprendería una constante.
3. Un nivel se considera **dominado** cuando el acierto en un set de
   **evaluación fijo** supera un umbral (por defecto 95 %) durante varias
   evaluaciones seguidas y tras un mínimo de epochs, sin olvidar niveles previos
   (retención). Recién entonces sube a `N+1`; si no lo domina, **sigue entrenando**
   ese nivel (por defecto sin tope) en lugar de avanzar a las apuradas.
4. Al dominar los niveles pedidos, **sigue subiendo de nivel solo** (genera los
   puzzles del siguiente nivel on-demand) hasta `--max-level` (`0` = sin tope).
   Entre niveles corre una **fase de revisión** sobre niveles viejos para no
   olvidar. Si un nivel nuevo no se puede generar, entra en revisión permanente
   hasta que lo pares.

Definiciones y detalles del oráculo están en `puzzles.py`; el bucle de
entrenamiento y la CLI, en `train.py`.

## Requisitos

- La imagen de Docker del repositorio ya construida (trae PyTorch CPU + `game.py`
  y `model.py`). Si no existe:
  ```bash
  docker compose build
  ```
- El código de esta carpeta se inyecta con un **bind-mount** (el `Dockerfile` no
  lo copia a propósito, para no tocar nada existente).

## Entrenar con Docker (método oficial)

```bash
docker compose run --rm --no-deps --user "$(id -u):$(id -g)" \
  -v ./trainPuzzles:/app/trainPuzzles \
  -v ./model.pth:/app/model.pth:ro \
  trainer python -m trainPuzzles --levels 4 --threshold 0.95 \
          --init model.pth
```

- `--no-deps` evita levantar PostgreSQL (este entrenamiento no usa base de datos).
- `--user "$(id -u):$(id -g)"` hace que los `.pth` queden a tu nombre (sin esto los
  escribe el usuario root del contenedor).
- `-v ./trainPuzzles:/app/trainPuzzles` hace visible el código y deja la salida en
  el host.
- `-v ./model.pth:/app/model.pth:ro` provee la red de partida (ajustá la ruta a tu
  `.pth`; si no existe, arranca de una red aleatoria).
- **Ordenamiento de checkpoints** (ver `layout.py`): cada corrida archiva sus
  artefactos en `<dir>/runs/NNNN_<run-name>/` (modelo final + reporte +
  `levels/levelN_best.pth`). Además, el modelo final **siempre** se copia a
  `<dir>/puzzle_canon.pth` (con su `puzzle_canon_report.json`), que es el punto de
  partida por defecto (`--init`) del próximo entrenamiento. `<dir>` depende de
  `--arch`: `trainPuzzles/checkpoints_mts` (`mlp`) o `trainPuzzles/checkpoint_cnn`
  (`cnn`), de modo que los dos handoffs no se pisan.
- Si querés omitir la copia al canon (p. ej. en un experimento A/B que no debe
  pisar la cadena), pasá `--no-canon`.

Ejemplo rápido (para probar que corre, pocos puzzles y un tope pequeño de epochs):

```bash
docker compose run --rm --no-deps --user "$(id -u):$(id -g)" \
  -v ./trainPuzzles:/app/trainPuzzles \
  -v ./model.pth:/app/model.pth:ro \
  trainer python -m trainPuzzles --levels 2 --puzzles-per-level 25 --eval-size 8 \
          --max-epochs 2 --games-budget 400
```

> Sin `--max-epochs`, un nivel se entrena hasta dominarlo (puede tardar). El tope
> es solo una red de seguridad: si se alcanza sin dominar, **se detiene** el
> currículo (no avanza). `Ctrl-C` guarda el mejor estado parcial y el resumen.

## Ejecutar sin Docker

```bash
python -m trainPuzzles --levels 4 --threshold 0.9 --init model.pth
```

(el directorio de trabajo debe ser la raíz del repo, donde están `game.py` y `model.py`).

## Opciones

| Opción | Default | Qué hace |
|---|---|---|
| `--arch` | `mlp` | Arquitectura: `mlp` (`SuperTatetiNet`, carpeta `checkpoints_mts/`) o `cnn` (`SuperTatetiCNN`, carpeta `checkpoint_cnn/`). Define el directorio por defecto de `--init`/`--out`. |
| `--init` | canon del arch (`checkpoints_mts/puzzle_canon.pth` o `checkpoint_cnn/puzzle_canon.pth`) | `.pth` de partida (el canon: el último modelo terminado); si no existe, red aleatoria. |
| `--out` | canon del arch | `.pth` final (por defecto el canon). Además se archiva en `<dir>/runs/NNNN_<run-name>/`. |
| `--run-name` | `puzzle` | Etiqueta de la corrida archivada en `<dir>/runs/`. |
| `--no-canon` | off | No copiar el modelo final a `puzzle_canon.pth` (para experimentos que no deben pisar la cadena). |
| `--report` | `<out>_report.json` | Reporte JSON del aprendizaje (por defecto `puzzle_canon_report.json`). |
| `--levels` | `4` | Niveles iniciales pre-generados (`1..N`). La **evaluación** se genera para `1..N` (barata) y el **entrenamiento** solo desde `--start-level..N`. |
| `--start-level` | `1` | Nivel donde arranca el currículo. Con `>1` (p. ej. `2`) **salta el reentrenamiento de mate-1** y calienta directo en mate-2 desde `--init`, evitando que el paso por mate-1 degrade el modelo. |
| `--max-level` | `0` | Nivel máximo a alcanzar. `0` = sin tope: sube de nivel indefinidamente. |
| `--stop-at-level` | `0` | Termina el currículo tras procesar este nivel (dominado o no) y guarda el reporte. `0` = off. Con `retrain_loop.sh` se usa con el nivel objetivo para que el chequeo de dominio se ejecute sin seguir a niveles superiores. |
| `--min-level-puzzles` | `200` | Mínimo de puzzles para considerar entrenable un nivel nuevo. |
| `--threshold` | `0.95` | Acierto mínimo en evaluación para dominar un nivel. |
| `--retention-threshold` | `= --threshold` | Acierto mínimo en la unión de niveles previos (anti-olvido). |
| `--patience` | `2` | Evaluaciones seguidas sobre el umbral para dominar. |
| `--min-epochs` | `15` | Epochs mínimos en un nivel antes de poder avanzar. |
| `--max-epochs` | `0` | Tope de seguridad por nivel; `0` = sin tope (entrena hasta dominar). Si se alcanza sin dominar, se detiene el currículo. |
| `--plateau-patience` | `10` | Avisa si pasan N epochs sin mejorar el mejor acierto. |
| `--plateau-stop` | `0` | Si pasan N epochs sin mejorar el mejor acierto, **corta el nivel y la corrida** (guarda el mejor estado). `0` = off (no corta nunca). Útil para el ciclo de reintento con más datos. |
| `--log-every` | `10` | La línea por epoch de `[TRAIN]` se imprime 1 de cada N epochs (`<=0` = todas). Los hitos (nuevo mejor, dominado, plateau, parada) se imprimen siempre. |
| `--eval-every` | `1` | Evalúa/selecciona el mejor modelo cada N epochs (abarata corridas largas). |
| `--revision-epochs` | `6` | Epochs de revisión de niveles viejos (anti-olvido) tras dominar cada nivel. |
| `--revision-puzzles` | `500` | Puzzles por fase de revisión (repartidos entre los niveles viejos). |
| `--big` | `0.20` | Cambio (en acierto) para clasificar "mejoró/empeoró mucho". |
| `--small` | `0.05` | Cambio (en acierto) para clasificar "mejoró/empeoró poco". |
| `--puzzles-per-level` | `25000` | Puzzles de entrenamiento por nivel (con pocos, la red los memoriza). |
| `--eval-size` | `1000` | Puzzles del set de evaluación fijo por nivel. |
| `--no-augment` | off | Desactiva las simetrías D4 (por defecto, una al azar por muestra y epoch). |
| `--batch-size` | `64` | Tamaño de lote. |
| `--lr` | `3e-4` | Learning rate (Adam). |
| `--lr-schedule` | `none` | Programa de LR: `none` o `cosine` (decae de `--lr` a `--lr × --lr-final-frac`). |
| `--lr-final-frac` | `0.03` | Fracción final del LR para el cosine (`eta_min = lr × frac`). |
| `--lr-decay-epochs` | `0` | `T_max` del cosine; `0` = usar `--max-epochs` (o `3000` si no hay tope). |
| `--tta` | off | Promedia la política sobre las 8 simetrías D4 al evaluar (mejora el acierto, más lento). |
| `--dropout` | `0.0` | Dropout **funcional** en las capas compartidas (solo entrenando; en eval/inferencia está apagado). No agrega módulos: el `state_dict` queda idéntico. `0` = off. |
| `--weight-decay` | `0.0` | Penalización L2 de Adam (weight decay). `0` = off. |
| `--value-coef` | `0.0` | Peso de la pérdida de la cabeza de valor (0 = off). |
| `--hard-mining` | off | Hard-example mining: sobre-muestrea los puzzles donde el top-1 falla. |
| `--mine-every` | `10` | Recalcula los pesos del minado cada N epochs. |
| `--mine-boost` | `3.0` | Peso de los puzzles fallados en el minado. |
| `--mine-warmup` | `25` | Epochs iniciales sin minado. |
| `--ema` | off | Promedio exponencial de pesos (EMA); se evalúa y se guarda el mejor (red o EMA). |
| `--ema-decay` | `0.999` | Factor de decaimiento del EMA. |
| `--rank-coef` | `0.0` | Coeficiente de la pérdida de ranking (margen) sumada a la cross-entropy. |
| `--rank-margin` | `1.0` | Margen de la pérdida de ranking (empuja la mejor jugada por encima del mejor señuelo). |
| `--replay-frac` | `0.30` | Fracción de puzzles de niveles previos en cada nivel (anti-olvido). |
| `--min-turn` | `16` | Solo se resuelven posiciones con turno ≥ este valor (ahorra tiempo). |
| `--solver-max-nodes` | `120000` | Presupuesto de nodos del solver por posición (`0` = sin límite). |
| `--games-budget` | `0` | Tope duro de partidas aleatorias por generación. **`0` = sin tope**: juega hasta juntar `--puzzles-per-level` (con `--max-games` como seguridad). Un valor `>0` corta a esa cantidad de partidas. |
| `--max-games` | `200000` | Tope de seguridad de partidas **solo** cuando `--games-budget=0`. |
| `--seed` | `0` | Semilla (generación de train y mezcla reproducibles). |
| `--eval-seed` | `= --seed` | Semilla del set de **evaluación**. Fijala para comparar entre corridas con datos de train distintos (p. ej. reintentos). |

## Entrada canónica y simetrías (importante)

`game.get_state()` codifica el tablero en términos absolutos (X = 1, O = -1). Con eso la
red tiene que aprender cada patrón dos veces y, con pocos puzzles, **memoriza**: en
una prueba controlada con 6.000 puzzles de mate en 1 llegaba a ~99 % en entrenamiento y
solo ~62 % en puzzles nuevos. Por eso `canon.py` hace dos cosas:

1. **Perspectiva de quien mueve**: el tablero y los tableros chicos ganados se
   multiplican por +1 (mueve X) o -1 (mueve O); el 0.5 de "en juego" no se toca.
2. **Simetrías D4**: las 8 rotaciones/reflexiones (misma transformación en el tablero
   grande y dentro de cada casilla; la regla de envío se conserva) se aplican al azar,
   una por muestra y epoch.

En la prueba controlada, con 6.000 puzzles: 62 % -> 72 % (perspectiva) -> 84 %
(+ simetrías); con 24.000 puzzles y simetrías, ~93 % en 10 epochs.

> **Compatibilidad:** la perspectiva canónica la aplica ahora `SuperTatetiNet.forward`
> (`model.py`), así que al jugar o entrenar se pasa el estado CRUDO (`game.get_state()`)
> y la red lo transforma sola. Por eso **todos** los que juegan (liga `evolution/engine.py`,
> `play.py`, `mcts.py`) funcionan sin cambios. Los `.pth` siguen siendo `state_dict` planos
> de `SuperTatetiNet`, pero un modelo entrenado con `trainPuzzles` espera esa transformación:
> usarlo con un `forward` viejo (sin el canónico) lo haría jugar mal. Los reportes JSON llevan
> `"input_encoding": "canonical"`.

## Cómo saber si está aprendiendo

Mientras corre, la salida muestra en vivo, con una **etiqueta de fase** por línea:

- `[GEN]` — **generación de puzzles** (recolección de datasets). Ocurre antes de
  entrenar: por eso primero "pasa por mate 1, 2, 3...". Muestra
  `encontrados/objetivo`, barra, puz/s y ETA.
- `[TRAIN]` — **entrenamiento**. Encabezado de nivel y una línea por epoch:
  ```
  [TRAIN]   N1 ep  12 | loss 0.830 (P 0.610 V 0.000) | acierto  78.0% base  41.0% d +37.0 | train  80.0% brecha  +2.0 | mejor  78.0% | ret  76.0% | [########--] 2.3s | meta 95%
  ```
  `train` es el acierto sobre una muestra fija de puzzles de entrenamiento y `brecha` es
  `train - acierto`: si la brecha crece mucho, la red está memorizando en vez de aprender.
  `MASTERED` aparece cuando el nivel queda dominado, y luego
  `nivel N DOMINADO ...` + `nivel N finalizado; se continúa con el nivel N+1`.
- `[REV]` — **revisión anti-olvido** de niveles viejos (reporta la retención por nivel).
- `[INFO]` — configuración, baseline, resumen parcial y avisos.

Así nunca se confunde *generar* (los "mate 1, 2, 3, ..." de `[GEN]`) con *avanzar de
nivel* (los `[TRAIN]`). Un nivel nuevo se **genera recién** cuando dominás el anterior.

Avisa si hay **meseta** (muchos epochs sin mejorar el mejor acierto).

Al final imprime una tabla por nivel (`base`, `final`, `cambio`, `epochs`,
`dominio`) y un **veredicto global** que clasifica el aprendizaje en: *mejoró
mucho*, *mejoró poco*, *no muestra un aprendizaje claro*, *empeoró poco* o
*empeoró mucho* (comparando el acierto final contra el modelo inicial). El mismo
detalle queda en el reporte JSON.

## Corrida larga / nocturna

El entrenamiento está pensado para dejarlo corriendo horas. Con los defaults
(`--max-epochs 0`, `--max-level 0`) va subiendo de nivel a medida que domina cada
uno y nunca corta solo: lo pará cuando quieras.

```bash
docker compose run --rm --no-deps --user "$(id -u):$(id -g)" \
  -v ./trainPuzzles:/app/trainPuzzles \
  trainer python -m trainPuzzles --levels 4 --max-level 0 --threshold 0.95 \
          --init trainPuzzles/checkpoints_mts/puzzle_canon.pth
```

Si no pasás `--init`, arranca directamente del canon (`puzzle_canon.pth`); cada
nivel completado lo actualiza, así una corrida nueva reanuda desde el último modelo
sin hacer nada.

**Cómo pararlo y obtener el resumen:**

- En la misma terminal, **Ctrl-C** (SIGINT).
- O desde otra terminal: `docker compose stop trainer` (SIGTERM).

En ambos casos finaliza el epoch actual, guarda el mejor estado por nivel y
escribe el resumen en pantalla + `puzzle_canon_report.json` (y en el reporte de la
corrida archivada). El reporte también se **actualiza después de cada nivel**, así
que aunque el proceso muera de golpe siempre queda un JSON legible con el progreso.

- Si domina los 4 niveles iniciales, sigue con mate en 5, 6, ... (`--max-level 0`).
  Si un nivel es demasiado difícil de generar, entra en **modo revisión**: sigue
  entrenando niveles viejos (anti-olvido) hasta que lo pares.

## Entrenamiento evolutivo (alternativa memética)

El entrenamiento de un solo linaje tiende a **memorizar** el set de entrenamiento y
generalizar poco. La alternativa de `evolve_puzzles.py` mantiene una **población**
de redes que combina **gradiente** (búsqueda local de cada individuo) con
**mutación + selección** (los mejores sobreviven; el resto se recrea mutando). Así
se conserva diversidad y se frena el sobreajuste.

- Configuración **propia** en `evo_config.py` (sin colores, sin base de datos, sin
  torneos y **sin importar nada de `evolution/`**).
- Población por defecto **30** individuos, sobreviven los **10** mejores
  (`children = 20`).
- **Validación rotativa**: la selección y el early-stop usan subconjuntos
  **sorteados de nuevo en cada generación** desde un pool grande
  (`--eval-pool-size`), disjuntos entre sí. Así la población no puede memorizar
  un set fijo (el fallo del maratón anterior: subía la validación pero el test
  fresco quedaba plano).
- Cada individuo entrena `--epochs-per-gen` epochs sobre su propia **bootstrap**
  del pool y guarda su mejor estado según el set de early-stop; compite por el
  set de selección.
- **Diversidad**: al menos `--min-large-frac` de los hijos usan mutación `large`
  (inmigración) para frenar el colapso de linajes.
- El **mejor histórico por validación** se guarda en `<out>` (por defecto el canon
  `puzzle_canon.pth` de la arquitectura); el **mejor por test honesto** (si se define
  `--out-test`) en otro archivo, y hay un reporte JSON incremental con diversidad y
  ambos mejores.
- Cada corrida archiva su modelo/reporte en `<dir>/runs/NNNN_<run-name>/`
  (`best_test.pth` si hay `--out-test`). Con `--no-canon` no pisa `puzzle_canon.pth`.
- `--patience` corta la corrida si el test honesto no mejora en N generaciones.

```bash
python -m trainPuzzles.evolve_puzzles \
  --seed-model trainPuzzles/my_model.pth \
  --run-name evo
```

Con Docker:

```bash
docker compose run --rm --no-deps --user "$(id -u):$(id -g)" \
  -v ./trainPuzzles:/app/trainPuzzles \
  trainer python -m trainPuzzles.evolve_puzzles \
          --seed-model trainPuzzles/my_model.pth \
          --run-name evo
```

| Opción | Default | Qué hace |
|---|---|---|
| `--config` | — | JSON de configuración (los demás flags lo sobrescriben). |
| `--arch` | `mlp` | Arquitectura: `mlp` (carpeta `checkpoints_mts/`) o `cnn` (`checkpoint_cnn/`). El `--seed-model` debe ser del mismo arch. |
| `--seed-model` | `trainPuzzles/my_model.pth` | `.pth` semilla de toda la población. |
| `--out` | canon del arch | Mejor histórico por validación (el canon; se actualiza al mejorar). |
| `--out-test` | — (off) | Checkpoint del mejor **test honesto** (diagnóstico, no usado para seleccionar). Se archiva en `runs/.../best_test.pth`. |
| `--run-name` | `evo` | Etiqueta de la corrida archivada en `<dir>/runs/`. |
| `--no-canon` | off | No copiar el mejor a `puzzle_canon.pth`. |
| `--report` | `<out>_report.json` | Reporte JSON (generaciones, top, diversidad, linaje, test). |
| `--population` | `30` | Tamaño de la población. |
| `--survivors` | `10` | Cuántos pasan intactos (el resto se recrea mutando). |
| `--generations` | `50` | Generaciones a correr. |
| `--epochs-per-gen` | `10` | Epochs de gradiente por individuo y generación. |
| `--bootstrap-frac` | `1.0` | Fracción del pool muestreada con reemplazo por individuo. |
| `--mutation-probs` | `small:0.5,medium:0.35,large:0.15` | Probabilidad de cada nivel de mutación. |
| `--mutation-norm-threshold` | `4.0` | Reescala linajes cuyos pesos explotan (`0` = off). |
| `--epochs-per-gen`, `--lr`, `--batch-size`, `--value-coef` | `10`,`3e-4`,`64`,`0.0` | Búsqueda local. |
| `--levels` | `1` | Niveles de mate que definen el fitness. |
| `--puzzles-per-level` | `25000` | Puzzles de entrenamiento por nivel. |
| `--eval-pool-size` | `4000` | Pool grande de validación del que se sortea cada generación. |
| `--eval-size` / `--early-stop-size` | `1000` / `1000` | Subconjuntos por generación: selección y early-stop (disjuntos). |
| `--test-size` | `2000` | Puzzles del test fresco (honesto, solo reporte). |
| `--test-every` | `5` | Cada cuántas generaciones se mide en el test fresco. |
| `--min-large-frac` | `0.25` | Fracción mínima de hijos con mutación `large` (inmigración). |
| `--patience` | `12` | Corta si el test honesto no mejora en N generaciones (`0` = off). |
| `--seed` | `0` | Semilla (población, mutación y datos reproducibles). |

> **Cuándo usarlo:** primero conviene que el entrenamiento de un solo linaje
> (`python -m trainPuzzles`) supere el 95 % en mate en 1 con la entrada canónica. Con la
> entrada absoluta y pocos puzzles, la población quedaba clavada en 60-68 % durante 200
> generaciones: la mutación no arregla un problema de memorización. Cada ``test_every``
> generaciones se muestran dos tests: el del mejor **histórico** (es siempre el mismo
> modelo hasta que aparezca uno mejor) y el del **campeón de la generación**.

Salida en vivo con etiquetas `[GEN]`/`[TRAIN]`/`[EVO]`/`[INFO]`: una línea por
individuo (id, padre, nivel de mutación, acierto) y el resumen por generación
(mejor, histórico, media, diversidad). `Ctrl-C`/`docker stop` cortan tras la
generación actual y dejan el mejor modelo y el reporte.

## Salida y compatibilidad

- Los `.pth` son `state_dict` planos de `SuperTatetiNet`: se cargan con
  `play.py` y con `evolution.modelstore.load_net`.
- El layout de archivos vive en `layout.py`. El directorio raíz `<dir>` depende de
  `--arch`: `checkpoints_mts/` (`mlp`) o `checkpoint_cnn/` (`cnn`).
  - `<dir>/puzzle_canon.pth` + `puzzle_canon_report.json`: **el último modelo
    terminado** (handoff; `--init`/`--out` por defecto).
  - `<dir>/runs/NNNN_<run-name>/`: modelo final, reporte y
    `levels/levelN_best.pth` (mejor checkpoint por nivel).
  - `<dir>/legacy/`: corridas viejas con el naming plano anterior.

## A/B de regularización

Para medir si `--dropout`/`--weight-decay` ayudan a generalizar (y no empeoran), corré dos
veces con el **mismo `--eval-seed`, `--seed` y dataset**, cambiando solo la regularización.
Usá `--no-canon` (y `--run-name`) para no pisar la cadena de `puzzle_canon.pth`:

```bash
# Control
docker compose run --rm --no-deps --user "$(id -u):$(id -g)" \
  -v ./trainPuzzles:/app/trainPuzzles trainer \
  python -m trainPuzzles --levels 1 --max-level 2 --threshold 0.90 \
    --games-budget 0 --max-games 50000 --puzzles-per-level 10000 \
    --plateau-stop 1500 --seed 0 --eval-seed 0 --no-canon --run-name ab_control \
    --init trainPuzzles/checkpoints_mts/puzzle_canon.pth \
    --out  trainPuzzles/checkpoints_mts/ab_control.pth

# Tratamiento (dropout + weight decay)
docker compose run --rm --no-deps --user "$(id -u):$(id -g)" \
  -v ./trainPuzzles:/app/trainPuzzles trainer \
  python -m trainPuzzles --levels 1 --max-level 2 --threshold 0.90 \
    --games-budget 0 --max-games 50000 --puzzles-per-level 10000 \
    --plateau-stop 1500 --seed 0 --eval-seed 0 --no-canon --run-name ab_dropout \
    --dropout 0.2 --weight-decay 1e-4 \
    --init trainPuzzles/checkpoints_mts/puzzle_canon.pth \
    --out  trainPuzzles/checkpoints_mts/ab_dropout.pth
```

Comparar en mate-2 el `best`/`final` del `ab_control_report.json` vs `ab_dropout_report.json`
dentro de `<dir>/runs/` (generaliza mejor el que tenga mayor acierto de evaluación con
menor brecha train/eval).

## Reintento automático con más datos

`retrain_loop.sh` repite el entrenamiento de mate-2 subiendo los datos cuando se estanca
(`--plateau-stop 15000`), reanudando desde el canon que deja la ronda anterior:

```bash
setsid bash trainPuzzles/retrain_loop.sh > /tmp/opencode/retrain.log 2>&1 < /dev/null &
```

Parámetros por variable de entorno: `TARGET_LEVEL` (2), `ROUNDS` (5), `PPL` (25000),
`PPL_STEP` (25000), `PLATEAU` (1500), `SEED`, `EVAL_SEED`, `DROPOUT` (0.2),
`WEIGHT_DECAY` (1e-4), `WARM_START` (1), `START_LEVEL` (1), `ARCH` (mlp). Cada ronda corre con
`--stop-at-level $TARGET_LEVEL` y se archiva en `<OUTDIR>/runs/NNNN_retryN/` y
actualiza `<OUTDIR>/puzzle_canon.pth` (+ `puzzle_canon_report.json`); el loop lee ese reporte y
corta cuando `TARGET_LEVEL` queda `mastered`. `OUTDIR`/`CANON_INIT` se derivan del `ARCH` por
defecto (`checkpoints_mts` para mlp, `checkpoint_cnn` para cnn). Con `START_LEVEL=2` (mate-1 ya
dominado) la ronda **arranca directo en mate-2** calentando desde el canon, sin reentrenar mate-1.

### Técnicas de refuerzo (para exprimir el top-1)

Cuando mate-2 queda cerca pero no llega al umbral, la jugada óptima suele estar en el
**top-2/top-3** pero no en el **#1**: el problema es de calibración, no de táctica. Estas
opciones atacan ese desempate (todas por defecto **off**, se activan desde `retrain_loop.sh`
por variable de entorno):

| Variable | Flag | Qué hace |
|---|---|---|
| `LR_SCHEDULE=cosine` | `--lr-schedule cosine` | LR decae suave (`--lr` → `--lr × --lr-final-frac`), evita oscilar. |
| `TTA=1` | `--tta` | Promedia las 8 simetrías D4 al evaluar (+acierto). |
| `EVAL_EVERY=N` | `--eval-every N` | Evalúa 1 de cada N epochs (abarata la corrida con TTA/EMA). |
| `HARD_MINING=1` | `--hard-mining` | Sobre-muestrea los puzzles donde el top-1 falla (`MINE_BOOST`, `MINE_EVERY`, `MINE_WARMUP`). |
| `EMA=1` | `--ema` | Promedio exponencial de pesos; guarda el mejor entre red y EMA. |
| `RANK_COEF=0.5` | `--rank-coef` | Suma una pérdida de margen que empuja la mejor jugada por encima del mejor señuelo (`RANK_MARGIN`). |

Corrida nocturna combinando todo:

```bash
TARGET_LEVEL=2 ROUNDS=3 PPL=25000 PPL_STEP=25000 MAX_EPOCHS=3000 PLATEAU=0 \
SEED=0 EVAL_SEED=0 DROPOUT=0.2 WEIGHT_DECAY=1e-4 WARM_START=1 THRESHOLD=0.90 MAX_LEVEL=3 \
LR_SCHEDULE=cosine LR_FINAL_FRAC=0.03 TTA=1 EVAL_EVERY=2 \
HARD_MINING=1 MINE_EVERY=10 MINE_BOOST=3.0 MINE_WARMUP=25 \
EMA=1 EMA_DECAY=0.999 RANK_COEF=0.5 RANK_MARGIN=1.0 \
setsid bash trainPuzzles/retrain_loop.sh > /tmp/opencode/night.log 2>&1 < /dev/null &
```

El log `[TRAIN]` muestra siempre **T1/T2/T3** (top-1/2/3), que es lo que decide si conviene
seguir exprimiendo el entrenamiento o cambiar de arquitectura.

## Pruebas

```bash
python -m pytest trainPuzzles/tests -q
```

No requieren base de datos ni variables de entorno.

## Archivos

```
trainPuzzles/
├── __init__.py
├── __main__.py        # python -m trainPuzzles
├── puzzles.py         # oráculo min_mate/optimal_moves + generador
├── canon.py           # entrada canónica (perspectiva de quien mueve) + simetrías D4
├── layout.py          # ordenamiento de checkpoints (canon + runs/NNNN_<tag>)
├── train.py           # entrenador de gradiente + CLI (--arch mlp|cnn)
├── model_cnn.py       # SuperTatetiCNN (--arch cnn)
├── evo_config.py      # configuración del evolutivo (propia, sin evolution/)
├── evolve_puzzles.py  # entrenador evolutivo memético (python -m trainPuzzles.evolve_puzzles)
├── checkpoints_mts/   # canon + runs/NNNN_<tag>/ + legacy/ del MLP (los .pth ignorados por git)
├── checkpoint_cnn/    # canon + runs/NNNN_<tag>/ + legacy/ del CNN (los .pth ignorados por git)
├── tests/
│   ├── test_puzzles.py
│   ├── test_canon.py
│   ├── test_train.py
│   ├── test_evolve.py
│   └── test_model_cnn.py
└── README.md
```

## Notas

- La generación es por muestreo aleatorio de partidas + filtro del solver. Para
  niveles altos (`N` ≥ 4) puede tardar o encontrar pocos puzzles; subí
  `--games-budget`, bajá `--levels`/`--max-level` o subí `--solver-max-nodes`.
- Un nivel se entrena **hasta dominarlo** (umbral + paciencia + min-epochs). Si
  querés cortar antes, poné `--max-epochs`; al alcanzarlo sin dominar se detiene el
  currículo. `Ctrl-C`/`docker stop` guarda el mejor estado parcial y el resumen.
- Los `.pth` de las subcarpetas `runs/` y `legacy/` de `checkpoints_mts/` y
  `checkpoint_cnn/` están en el `.gitignore` de la raíz; el canon `.pth` sí se versiona.