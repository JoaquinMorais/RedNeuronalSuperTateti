# Entrenamiento evolutivo de Super Tateti

Sistema de 20 IA por temporada que compiten en **Liga + Copa + Recopa**; al final se
conservan las 7 mejores intactas y se crean 13 descendientes mutados. Sin MCTS.

## Arquitectura existente (inspeccionada)
- `model.py`: `SuperTatetiNet` (100 entradas → 3 capas ocultas → política de 81 + valor). Los pesos son un `state_dict` plano `.pth`.
- `game.py`: `SuperTateti` (tablero 9x9, `get_state()` de 100 floats, `get_valid_moves()` máscara de 81, `make_move`). **X (jugador 1) siempre comienza.**
- `mcts.py` / `trainer.py`: entrenamiento anterior (MCTS + autojuego + retropropagación). Siguen en el repo sin cambios, pero **el sistema nuevo no los importa** (hay una prueba que lo verifica). `play.py` y `test_ias.py` siguen usando MCTS para jugar contra humanos / comparar.
- No existían base de datos ni Docker.

## Cómo juega ahora una IA (sin MCTS)
`evolution/engine.py`: la red devuelve la política ya enmascarada con los movimientos legales; se juega el de mayor probabilidad (`argmax`, en empate exacto el de menor índice). Es determinista, por lo que una partida repetida da exactamente el mismo resultado. Se verifica además que la jugada sea legal.

Medición (`python scripts/benchmark_move_selection.py`, CPU, `model.pth`, partidas de 57-70 jugadas; muestras pequeñas para MCTS):

| Método | ms/partida | ms/jugada |
|---|---|---|
| MCTS 200 simulaciones (`trainer.py`) | ~2500 | ~48 |
| MCTS 50 simulaciones (`play.py`) | ~595 | ~11 |
| Selección directa (nuevo) | ~8 | ~0,14 |

Una temporada completa (≈500 partidas) tarda unos 6-10 s de simulación.

## Ejecución con Docker
```bash
docker compose build                       # construir
docker compose up -d                       # iniciar (PostgreSQL + entrenador)
docker compose logs -f trainer             # ver las fechas en la terminal
docker compose stop trainer                # detener de forma segura (SIGTERM: termina la partida en curso)
docker compose up -d trainer               # reanudar exactamente donde quedó
docker compose run --rm trainer python -m evolution status   # estado de las temporadas
docker compose down                        # apaga; los datos persisten (volumen pgdata y ./models)
docker compose down -v                     # ¡borra la base de datos!
```
Variables útiles (en `.env` o `docker compose run -e ...`): `SEED`, `MAX_SEASONS` (0 = sin límite), `INITIAL_POPULATION` (`auto|seed|random`), `MUTATION_PROBS=small:0.5,medium:0.35,large:0.15`, `EVO_CONFIG_FILE` (ver `evo_config.example.json`), `NO_COLOR`/`FORCE_COLOR`.
Una sola temporada: `MAX_SEASONS=1 docker compose up trainer`. Con `docker compose run -it` o `tty: true` se ven colores reales; sin terminal sale texto plano legible.

Sin Docker: `pip install torch -r requirements.txt`, levantar un PostgreSQL y `DATABASE_URL=postgresql://... python -m evolution run`.

## Analizar temporadas y campeones (solo lectura; sirve mientras se entrena)
```bash
docker compose run --rm trainer python -m evolution status      # temporadas + campeones de cada una
docker compose run --rm trainer python -m evolution season 1    # Liga, grupos + gráfico de la Copa, Recopa y tabla anual
docker compose run --rm trainer python -m evolution champions   # campeones por temporada + ranking de campeones
docker compose run --rm trainer python -m evolution menu        # lo mismo, con menú interactivo
```
- **Insignias**: una IA que ya ganó algo se muestra como `#7d48e6:01L:00C:00RC` (Ligas, Copas, Recopas). Si nunca ganó nada, no lleva nada. Aparece en tablas, partidas y resúmenes de la terminal; los hijos nacen sin títulos. En la vista de la temporada N se cuentan los títulos hasta el final de esa temporada.
- **Tabla anual**: puntos de Liga + Copa + Recopa por IA, y su destino: `se conserva`, `se conserva + N hijos`, `N hijos` o `sale`.
- **Ranking de campeones**: orden de prioridad Liga, Copa, Recopa; a la derecha, en qué temporadas ganó cada título (`Ligas: T1, T3 · Recopas: T2`).

## Menú, ranking y jugar contra la IA
```bash
docker compose run --rm -p 127.0.0.1:8765:8765 trainer python -m evolution menu
```
El `-p` publica el puerto para la opción 4 (jugar). Las opciones admiten un argumento:

| Escribís | Hace |
|---|---|
| `1` | estado de las temporadas con sus campeones |
| `2` / `2 21` | detalle de una temporada (sin número, lo pregunta) / de la temporada 21 |
| `3` / `3 100` | ranking de campeones: los 30 mejores (`CHAMPIONS_TOP`) / los 100 mejores |
| `4` / `4 12` | jugar contra el mejor de la tabla anual de la última temporada terminada / de la temporada 12 |
| `0` | salir |

Equivalentes directos: `python -m evolution champions 100`, `python -m evolution play 12`.

- **Ranking de campeones**: ahora tiene la columna *Estado*: `compitiendo`, o `expulsada T7` (la temporada en la que no pasó a la siguiente generación).
- **Jugar**: la opción 4 abre una página en `http://localhost:8765` (tablero 9x9, tablero activo resaltado, tus jugadas y las de la IA marcadas, marcador de la sesión; podés empezar vos o la IA). La IA juega igual que en los torneos (mejor movimiento legal de su red, sin MCTS). Ctrl+C vuelve al menú. Si el puerto está ocupado, `PLAY_PORT=9000`.

## Pelear entre IA, liga de prueba y añadir IA (menú 5, 6 y 7)
Las IA se indican con un **número de temporada** (= el 1.º de su tabla anual) o con su **color** (`#ffffff`; también vale sin `#`, porque la shell toma `#...` como comentario). Un color solo sirve si el `.pth` de esa IA sigue guardado (ver limpieza de modelos).

| Menú / comando | Qué hace |
|---|---|
| `5 1 311` · `fight 1 #ffffff` | **Pelea**: dos IA juegan una partida de ida y vuelta (comienza una, luego la otra). Gana quien suma más puntos (3/1/0). |
| `6 1 311 #ffffff` · `testleague 1 311 seed=5` | **Liga de prueba** (20 equipos, ida y vuelta, **no se guarda**): tus IA + redes aleatorias hasta llegar a 20 (con 2 IA, 18 aleatorias). Muestra la tabla y un resumen por IA (puesto, puntos, récord contra aleatorias y entre las probadas). `seed=N` repite exactamente lo mismo. |
| `7 models/mi_ia.pth` · `addai a.pth b.pth` | **Añadir IA a la próxima temporada**. Ruta o nombre (se busca también dentro de `models/`; con espacios, entre comillas). `7 --list` muestra las pendientes y `7 --cancel #abc123` quita una (solo antes de que se cree esa temporada). Se puede usar mientras se entrena. |

**Qué cambia en una temporada con IA añadidas** (solo esa; la siguiente vuelve a 20 y la composición sigue siendo 7 supervivientes + 13 descendientes, vengan o no de las añadidas):
- Si se está jugando la temporada 312, la IA entra en la 313 (siempre en la temporada siguiente a la última creada).
- **Liga**: igual, con más partidas: con N equipos cada IA juega 2·(N-1) (21 equipos → 40 partidas por IA). Como una IA juega una partida por fecha, las fechas son 2·(N-1) si N es par y 2·N si es impar (21 y 22 equipos → 42 fechas; con N impar descansa una IA en cada fecha). La Recopa se juega tras la última fecha.
- **Copa**: solo cambia la fase de grupos. Las IA extra se reparten de a una por grupo (A, B, C, D, A...): 1 extra → A de 6 · 3 extra → A, B y C de 6 y D de 5 · 6 extra → A y B de 7, C y D de 6. Siempre clasifican 4 por grupo (los demás quedan fuera) y el cuadro es el mismo de 16. Los grupos de 5 y 6 se juegan en las fechas 1-12; los de 7 y 8 necesitan 14 rondas y usan hasta la fecha 15 (siempre antes de los octavos).
- **Recompensa de Liga** para posiciones más allá de la 20.ª: -3.
- Máximo 12 IA añadidas por temporada.
- Cada IA añadida recibe un color nuevo y único y se guarda como `models/<color>.pth` + `.json` (`kind: imported`). Si queda entre las 7 primeras, se conserva y puede tener hijos como cualquier otra.

## Limpieza de modelos (espacio en disco)
Después de cada temporada se borra el `.pth` de las IA que cumplan **todo** esto:
1. fueron expulsadas (no pasaron a la siguiente generación) hace `MODEL_RETENTION_SEASONS` temporadas o más (por defecto **10**; `0` desactiva la limpieza);
2. nunca fueron 1.ª de la tabla anual en ninguna temporada;
3. nunca ganaron una Liga, una Copa ni una Recopa.

Se conservan siempre: las IA que siguen compitiendo, las 1.ª de tabla anual, las campeonas y `models/historical/`. El `.json` de cada modelo borrado se queda (con `weights_deleted: true`) y su fila en la base también (linaje, puntajes). Para ver qué se borraría sin borrar: `python -m evolution prune --dry-run`; para forzarla a mano: `python -m evolution prune` (necesita que no haya otro entrenamiento corriendo).

## Reiniciar desde cero (solo mientras se prueba)
`docker compose down -v` borra la base de datos. Si después querés empezar limpio, borrá también los modelos **generados** (no `models/model.pth` ni `models/model_easy.pth`, que son tuyos):
```bash
# Linux/macOS/Git Bash
rm -f models/??????.pth models/??????.json && rm -rf models/historical
# PowerShell
Remove-Item models\??????.pth, models\??????.json; Remove-Item -Recurse -Force models\historical
```
Lo que **no** conviene es lo contrario: borrar los modelos y conservar la base (la base referencia archivos que ya no existen y el entrenamiento se detiene con error). Una vez que entrenes en serio, ni la base ni `models/` se tocan: son una sola cosa.

## ¿Está aprendiendo? Medir el progreso (menú 8, 9 y 10)
Los puntos de la Liga/Copa son **relativos**: la población solo se compara consigo misma, así que puede mejorar (o no) sin que los puntos lo muestren. Por eso hay dos medidas independientes, que se leen juntas:

| | Qué mide | Cómo se ve | Dónde |
|---|---|---|---|
| **Propuesta 1: nivel absoluto** | El 1.º de cada tabla anual y la población entera juegan contra **rivales fijos** que no evolucionan: un *bot aleatorio* (el piso) y un *bot táctico* (gana tableros, bloquea, no regala la partida; le gana ≈ 96 % al aleatorio). | Curva por temporada con intervalo de confianza, tendencia estadística (¿sube o es ruido?) y nivel actual en palabras. | Se guarda sola cada `BENCH_EVERY` temporadas (5). Ver con `8`; medir una IA ahora con `9`; temporadas ya jugadas con `8 --backfill 10`. |
| **Propuesta 2: torneo maestro** | Los campeones (1.º de la tabla anual) de temporadas repartidas se enfrentan todos contra todos. ¿Los nuevos le ganan a los viejos? | Elo por campeón, % de puntos que se lleva la generación más nueva (50 % = nada), correlación temporada↔Elo y ciclos (piedra-papel-tijera). | `10` (`10 5` = uno cada 5 temporadas, `10 5 40` = hasta 40). |

**Cómo leerlas juntas**
- P1 sube **y** P2 sube → aprende de verdad.
- P1 plana (≈ 50 % vs aleatorio, ≈ 5 % vs táctico) **y** P2 sube → mejora *relativa*: los campeones se especializan en ganarles a sus ancestros (explotan manías de redes deterministas) sin jugar mejor. Es el caso típico de la coevolución cerrada.
- Ambas planas → la mutación no encuentra mejoras (ver la tabla *Eficacia de las mutaciones* del menú `8`: si los hijos entran al top 7 ≈ 35 % o menos y superan a su padre < 50 %, las mutaciones son ruido o empeoran).
- Referencias medidas: una red **sin entrenar** hace ≈ 52 % contra el aleatorio y ≈ 5 % contra el táctico.

Variables: `BENCH_EVERY` (0 = no medir), `BENCH_GAMES` (partidas del 1.º por rival, 120), `BENCH_POP_GAMES` (de cada IA de la población por rival, 8; 0 = solo el 1.º). Cada medición cuesta unos segundos; el menú `9` mide una IA puntual con la cantidad de partidas que quieras (`9 12 500`).

## Espacio en disco, logs y git
A ~390 temporadas/hora una noche son miles de temporadas. Qué crece y cómo se controla:

| Qué | Crecimiento sin control | Control |
|---|---|---|
| **Logs de Docker** | miles de líneas por temporada | `OUTPUT=temporada` (1 línea por temporada; `fecha` y `full` dan más detalle) y rotación en `docker-compose.yml` (3 × 10 MB). |
| **Modelos `.pth`** | ~0,6 MB por temporada de campeones conservados | `MODEL_RETENTION_SEASONS` (10) y `MODEL_PROTECT_EVERY=K`: de los campeones/1.º solo se conservan los de temporadas múltiplo de K (1 = todos). |
| **Base de datos** | ~0,5 MB por temporada | `DB_COMPACT_AFTER=N`: en temporadas con más de N de antigüedad se borran las jugadas de cada partida y el historial fecha a fecha (resultados, tablas finales y linajes quedan). |

`docker-compose.yml` ya trae valores para entrenar de noche (`OUTPUT=temporada`, `MODEL_PROTECT_EVERY=10`, `DB_COMPACT_AFTER=50`, `BENCH_EVERY=5`); se cambian en `.env` o por línea de comandos.
- `python -m evolution disk` (menú `11`): modelos por categoría, **archivos huérfanos** (sobras de `down -v`) y tamaño de la base. `prune --orphans` los lista y `prune --orphans --yes` los borra.
- **Git**: los modelos generados no deben versionarse (`.gitignore` ya ignora `models/*` salvo `model.pth` y `model_easy.pth`). Si ya estaban versionados, `git add .` tarda por hashearlos: `git rm -r -q --cached models && git add .gitignore models/model.pth models/model_easy.pth`. Si además `.git` pesa GB (`du -sh .git`), hay que reescribir la historia (p. ej. con `git filter-repo`; haz una copia antes).
- Diagnóstico rápido: `du -sh .git models`, `docker system df`, `sudo du -sh /var/lib/docker/containers/*/*-json.log`.

## Pruebas
```bash
TEST_DATABASE_URL=postgresql://usuario:clave@host:5432/base_de_pruebas python -m pytest tests -q
```
Las pruebas de integración usan un esquema aislado por prueba en esa base (sin `TEST_DATABASE_URL` se omiten explícitamente). Cubren: temporada completa e invariantes, reproducibilidad por semilla, interrupción entre fechas, durante una partida, corte de conexión (`pg_terminate_backend`), `SIGTERM` sobre un proceso real + reanudación, bloqueo de instancia única y creación de la temporada siguiente desde la base.

## Decisiones técnicas
- **Base de datos**: PostgreSQL 16 (fuente de verdad). Esquema en `evolution/schema.sql`. Cada resultado se registra en una transacción con `UPDATE ... WHERE status<>'finished'`: imposible contar dos veces. Si la conexión cae, se reconecta con espera exponencial y se repite la transacción (todas son idempotentes).
- **Instancia única**: bloqueo consultivo de PostgreSQL (`pg_try_advisory_lock`) de sesión; se libera solo si el proceso muere.
- **Partida interrumpida**: no se guarda estado a mitad de partida. Como la jugada es determinista y la partida dura ~10 ms, se reinicia desde cero y da el mismo resultado; solo cuenta al quedar `finished`.
- **Modelos** (`/app/models`, `./models` en el host): `<color>.pth` (state_dict plano, cargable por `play.py`) + `<color>.json` (arquitectura, versión de formato, hash SHA-256 de los pesos, linaje). Nunca se sobrescriben; los supervivientes siguen usando el mismo archivo. Mejor histórico: copia en `models/historical/seasonNNNN_<color>.pth` cuando un campeón supera en puntos totales al mejor registrado (tabla `historical_best`).
- **Calendarios**: Liga 38 fechas por método del círculo (cada IA comienza 19 veces; la vuelta invierte quién comienza). Grupos: 10 rondas repartidas en las fechas 1-5 y 7-11 (6 y 12 quedan de margen). Todo se guarda al crear la temporada. Las partidas de eliminatoria se crean cuando se conocen los participantes (los cruces y sus fechas 18/23/30/37 ya están guardados desde el inicio).
- **Aleatoriedad**: `semilla maestra → semilla de temporada → sub-semillas por etiqueta` (SHA-256). Los sorteos parten de un orden canónico (por color), así no dependen del orden de inserción. Cada sorteo queda en `tiebreak_draws`.
- **Mutación**: cada peso se perturba con probabilidad `rate` con ruido gaussiano de desvío `sigma × (desvío del tensor)`. Pequeña 10 %/0,05 · Media 30 %/0,20 · Grande 60 %/0,60; reparto entre hijos 50/35/15 %. Todo configurable. Solo mutan `shared.*` y `policy_head.*` (la cabeza de valor no influye en la jugada; configurable con `mutate_value_head`). El nivel de la mutación del color es el mismo que el de los pesos.
- **Colores**: variaciones sobre RGB (pequeña ±1-3, media ±4-24, grande ±25-80 por canal, 1-3 canales). Colisión → reintento; agotados los reintentos → escala de magnitud; si aún no hay hueco → búsqueda determinista del color libre más cercano (cubos RGB concéntricos). Unicidad global (`UNIQUE` en BD). La generación inicial recibe colores repartidos por todo el círculo cromático (matices separados, brillo alto) para distinguirse a simple vista; desde ahí, cada hijo se parece a su padre.
- **Padres (criterio por tabla anual)**: los 7 primeros de la tabla anual (Copa + Liga + Recopa) se conservan intactos y tienen hijos según su puesto: 1.º 5, 2.º 3, 3.º 2, 4.º 2, 5.º 1, 6.º y 7.º ninguno (13 en total). Es determinista y configurable con `parent_policy.children_per_rank` en `EVO_CONFIG_FILE`; debe sumar exactamente 13.
- **Recopa**: se comparan solo victorias; el empate de partida vale 0 para ambos. Si hay igualdad de victorias, sorteo reproducible (sin 11.ª partida).
- **Clasificación global**: puntos totales; desempate: mejor posición de Liga, luego más puntos de Copa, luego sorteo (inalcanzable en la práctica porque la posición de Liga es única).

## Supuestos que conviene que revises
1. **Cuadro de Copa** (cuartos en adelante): la especificación solo fija los octavos. Usé el orden del cuadro: ganadores de octavos 1v2, 3v4, 5v6, 7v8; luego 1v2, 3v4 de cuartos, etc. (p. ej. 1A/4B-ganador puede cruzarse con 2A/3B-ganador en cuartos).
2. **Quién comienza en la 3.ª partida de desempate** (no especificado): se sortea de forma reproducible y se guarda; si empata, pasa quien comenzó segundo.
3. **Población inicial** (`INITIAL_POPULATION=auto`): si existe `model.pth`, la IA 1 es una copia exacta y las otras 19 son sus descendientes mutados (padre registrado); si no, 20 redes aleatorias. Cambialo con `random` si preferís empezar de cero.
4. **Orden dentro de una fecha**: primero Liga, luego grupos/eliminatorias; ambas en la misma fecha.
5. Con la jugada determinista (`argmax`) muchas partidas entre redes parecidas terminan en empate y las eliminatorias recurren a la 3.ª partida con frecuencia (observado en la temporada de prueba). Si querés más variedad habría que introducir selección estocástica por temperatura; no lo hice por no alterar la regla "mejor movimiento legal".
6. **Reproducibilidad entre máquinas**: es exacta con la misma versión de torch/CPU; entre arquitecturas distintas, diferencias de punto flotante podrían cambiar una jugada en empates numéricos casi exactos.