# trainPuzzles — entrenamiento táctico "mate en N"

Método **independiente** para entrenar `SuperTatetiNet` con puzzles tácticos.
Todo vive en esta carpeta: **no modifica ni depende de** `mcts.py`, `trainer.py`,
`evolution/`, el `Dockerfile` ni `docker-compose.yml`. El modelo entrenado se
guarda dentro de `trainPuzzles/` (por defecto en `checkpoints/`).

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
          --init model.pth --out trainPuzzles/checkpoints/puzzle_net.pth
```

- `--no-deps` evita levantar PostgreSQL (este entrenamiento no usa base de datos).
- `--user "$(id -u):$(id -g)"` hace que los `.pth` queden a tu nombre (sin esto los
  escribe el usuario root del contenedor).
- `-v ./trainPuzzles:/app/trainPuzzles` hace visible el código y deja la salida en
  el host.
- `-v ./model.pth:/app/model.pth:ro` provee la red de partida (ajustá la ruta a tu
  `.pth`; si no existe, arranca de una red aleatoria).
- El `.pth` final queda en `./trainPuzzles/checkpoints/` y, por cada nivel, se
  guarda el **mejor** estado (`puzzle_net_level1_best.pth`, ...). También se escribe
  un reporte del aprendizaje en `puzzle_net_report.json`.

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
| `--init` | `model.pth` | `.pth` de partida; si no existe, red aleatoria. |
| `--out` | `trainPuzzles/checkpoints/puzzle_net.pth` | `.pth` final (además, mejor estado por nivel). |
| `--report` | `<out>_report.json` | Reporte JSON del aprendizaje. |
| `--levels` | `4` | Niveles iniciales pre-generados (`1..N`). |
| `--max-level` | `0` | Nivel máximo a alcanzar. `0` = sin tope: sube de nivel indefinidamente. |
| `--min-level-puzzles` | `200` | Mínimo de puzzles para considerar entrenable un nivel nuevo. |
| `--threshold` | `0.95` | Acierto mínimo en evaluación para dominar un nivel. |
| `--retention-threshold` | `= --threshold` | Acierto mínimo en la unión de niveles previos (anti-olvido). |
| `--patience` | `2` | Evaluaciones seguidas sobre el umbral para dominar. |
| `--min-epochs` | `15` | Epochs mínimos en un nivel antes de poder avanzar. |
| `--max-epochs` | `0` | Tope de seguridad por nivel; `0` = sin tope (entrena hasta dominar). Si se alcanza sin dominar, se detiene el currículo. |
| `--plateau-patience` | `10` | Avisa si pasan N epochs sin mejorar el mejor acierto. |
| `--revision-epochs` | `6` | Epochs de revisión de niveles viejos (anti-olvido) tras dominar cada nivel. |
| `--revision-puzzles` | `500` | Puzzles por fase de revisión (repartidos entre los niveles viejos). |
| `--big` | `0.20` | Cambio (en acierto) para clasificar "mejoró/empeoró mucho". |
| `--small` | `0.05` | Cambio (en acierto) para clasificar "mejoró/empeoró poco". |
| `--puzzles-per-level` | `25000` | Puzzles de entrenamiento por nivel (con pocos, la red los memoriza). |
| `--eval-size` | `1000` | Puzzles del set de evaluación fijo por nivel. |
| `--no-augment` | off | Desactiva las simetrías D4 (por defecto, una al azar por muestra y epoch). |
| `--batch-size` | `64` | Tamaño de lote. |
| `--lr` | `3e-4` | Learning rate (Adam). |
| `--value-coef` | `0.0` | Peso de la pérdida de la cabeza de valor (0 = off). |
| `--replay-frac` | `0.30` | Fracción de puzzles de niveles previos en cada nivel (anti-olvido). |
| `--min-turn` | `16` | Solo se resuelven posiciones con turno ≥ este valor (ahorra tiempo). |
| `--solver-max-nodes` | `120000` | Presupuesto de nodos del solver por posición (`0` = sin límite). |
| `--games-budget` | `8000` | Tope de partidas aleatorias por generación. |
| `--seed` | `0` | Semilla (generación y mezcla reproducibles). |

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
          --init trainPuzzles/my_model.pth \
          --out  trainPuzzles/checkpoints/puzzle_net.pth
```

**Cómo pararlo y obtener el resumen:**

- En la misma terminal, **Ctrl-C** (SIGINT).
- O desde otra terminal: `docker compose stop trainer` (SIGTERM).

En ambos casos finaliza el epoch actual, guarda el mejor estado por nivel y
escribe el resumen en pantalla + `puzzle_net_report.json`. El reporte también se
**actualiza después de cada nivel**, así que aunque el proceso muera de golpe
siempre queda un JSON legible con el progreso.

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
- El **mejor histórico por validación** se guarda en `<out>`; el **mejor por test
  honesto** (si se define `--out-test`) en otro archivo, y hay un reporte JSON
  incremental con diversidad y ambos mejores.
- `--patience` corta la corrida si el test honesto no mejora en N generaciones.

```bash
python -m trainPuzzles.evolve_puzzles \
  --seed-model trainPuzzles/my_model.pth \
  --out trainPuzzles/checkpoints/evo/best.pth \
  --out-test trainPuzzles/checkpoints/evo/best_test.pth
```

Con Docker:

```bash
docker compose run --rm --no-deps --user "$(id -u):$(id -g)" \
  -v ./trainPuzzles:/app/trainPuzzles \
  trainer python -m trainPuzzles.evolve_puzzles \
          --seed-model trainPuzzles/my_model.pth \
          --out trainPuzzles/checkpoints/evo/best.pth \
          --out-test trainPuzzles/checkpoints/evo/best_test.pth
```

| Opción | Default | Qué hace |
|---|---|---|
| `--config` | — | JSON de configuración (los demás flags lo sobrescriben). |
| `--seed-model` | `trainPuzzles/my_model.pth` | `.pth` semilla de toda la población. |
| `--out` | `trainPuzzles/checkpoints/evo/best.pth` | Mejor histórico por validación (se actualiza al mejorar). |
| `--out-test` | — (off) | Checkpoint del mejor **test honesto** (diagnóstico, no usado para seleccionar). |
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
- Además del `.pth` final se guarda el **mejor checkpoint por nivel**
  (`<out>_levelN_best.pth`) y un **reporte** `<out>_report.json`.

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
├── train.py           # entrenador de gradiente + CLI
├── evo_config.py      # configuración del evolutivo (propia, sin evolution/)
├── evolve_puzzles.py  # entrenador evolutivo memético (python -m trainPuzzles.evolve_puzzles)
├── checkpoints/       # aquí quedan los .pth entrenados (ignorados por git)
├── tests/
│   ├── test_puzzles.py
│   ├── test_canon.py
│   ├── test_train.py
│   └── test_evolve.py
└── README.md
```

## Notas

- La generación es por muestreo aleatorio de partidas + filtro del solver. Para
  niveles altos (`N` ≥ 4) puede tardar o encontrar pocos puzzles; subí
  `--games-budget`, bajá `--levels`/`--max-level` o subí `--solver-max-nodes`.
- Un nivel se entrena **hasta dominarlo** (umbral + paciencia + min-epochs). Si
  querés cortar antes, poné `--max-epochs`; al alcanzarlo sin dominar se detiene el
  currículo. `Ctrl-C`/`docker stop` guarda el mejor estado parcial y el resumen.
- Los `.pth` de `checkpoints/` están en el `.gitignore` local de esta carpeta.