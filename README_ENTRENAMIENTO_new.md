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

## Reiniciar desde cero (solo mientras se prueba)
`docker compose down -v` borra la base de datos. Si después querés empezar limpio, borrá también los modelos **generados** (no `models/model.pth` ni `models/model_easy.pth`, que son tuyos):
```bash
# Linux/macOS/Git Bash
rm -f models/??????.pth models/??????.json && rm -rf models/historical
# PowerShell
Remove-Item models\??????.pth, models\??????.json; Remove-Item -Recurse -Force models\historical
```
Lo que **no** conviene es lo contrario: borrar los modelos y conservar la base (la base referencia archivos que ya no existen y el entrenamiento se detiene con error). Una vez que entrenes en serio, ni la base ni `models/` se tocan: son una sola cosa.

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
- **Colores**: variaciones sobre RGB (pequeña ±1-3, media ±4-24, grande ±25-80 por canal, 1-3 canales). Colisión → reintento; agotados los reintentos → escala de magnitud; si aún no hay hueco → búsqueda determinista del color libre más cercano (cubos RGB concéntricos). Unicidad global (`UNIQUE` en BD).
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