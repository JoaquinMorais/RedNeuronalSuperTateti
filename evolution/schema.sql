-- Fuente de verdad del estado de las temporadas. Idempotente (CREATE ... IF NOT EXISTS).

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Un individuo = una red con identidad (color) única en toda la historia.
CREATE TABLE IF NOT EXISTS individuals (
    id                 SERIAL PRIMARY KEY,
    color              CHAR(7) NOT NULL UNIQUE CHECK (color ~ '^#[0-9a-f]{6}$'),
    origin_season      INT  NOT NULL,                 -- temporada en la que nació
    parent_id          INT  REFERENCES individuals(id), -- linaje explícito
    kind               TEXT NOT NULL CHECK (kind IN ('founder_seed','founder_random','mutant')),
    model_file         TEXT NOT NULL,                 -- ruta relativa a MODELS_DIR
    model_sha256       TEXT,                          -- NULL hasta que el archivo está escrito y verificado
    architecture       TEXT NOT NULL,
    architecture_sig   TEXT NOT NULL,
    model_version      INT  NOT NULL DEFAULT 1,
    mutation_level     TEXT CHECK (mutation_level IN ('small','medium','large')),
    mutation_seed      BIGINT,
    mutation_meta      JSONB NOT NULL DEFAULT '{}',   -- parámetros usados (reproducibilidad)
    color_mutation_meta JSONB NOT NULL DEFAULT '{}',
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS seasons (
    id               SERIAL PRIMARY KEY,
    number           INT  NOT NULL UNIQUE,
    seed             BIGINT NOT NULL,
    status           TEXT NOT NULL DEFAULT 'in_progress' CHECK (status IN ('in_progress','completed')),
    stage            TEXT NOT NULL DEFAULT 'league_cup'
                     CHECK (stage IN ('league_cup','recopa','scoring','evolution','completed')),
    current_matchday INT  NOT NULL DEFAULT 0,         -- última fecha CERRADA
    cup_groups_resolved BOOLEAN NOT NULL DEFAULT FALSE,
    cup_champion_id  INT REFERENCES individuals(id),
    cup_runner_up_id INT REFERENCES individuals(id),
    league_champion_id INT REFERENCES individuals(id),
    recopa_champion_id INT REFERENCES individuals(id),
    draw_meta        JSONB NOT NULL DEFAULT '{}',     -- sorteo de slots y grupos
    config           JSONB NOT NULL DEFAULT '{}',
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at     TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS season_participants (
    season_id       INT NOT NULL REFERENCES seasons(id),
    individual_id   INT NOT NULL REFERENCES individuals(id),
    slot            INT NOT NULL CHECK (slot BETWEEN 1 AND 20),
    cup_group       CHAR(1) NOT NULL CHECK (cup_group IN ('A','B','C','D')),
    -- Liga: puntos de PARTIDO (tabla)
    league_points   INT NOT NULL DEFAULT 0,
    league_played   INT NOT NULL DEFAULT 0,
    league_position INT,
    -- Puntuaciones EVOLUTIVAS
    league_score    INT NOT NULL DEFAULT 0,
    cup_stage       TEXT CHECK (cup_stage IN ('group','R16','QF','SF','runner_up','champion')),
    cup_score       INT NOT NULL DEFAULT 0,
    recopa_score    INT NOT NULL DEFAULT 0,
    total_score     INT,
    final_position  INT,
    PRIMARY KEY (season_id, individual_id),
    UNIQUE (season_id, slot)
);

CREATE TABLE IF NOT EXISTS cup_ties (
    id              SERIAL PRIMARY KEY,
    season_id       INT NOT NULL REFERENCES seasons(id),
    round           TEXT NOT NULL CHECK (round IN ('R16','QF','SF','F')),
    tie_no          INT NOT NULL,
    matchday        INT NOT NULL,
    slot_a          TEXT NOT NULL,                    -- p.ej. '1A' o 'W:R16-1'
    slot_b          TEXT NOT NULL,
    individual_a_id INT REFERENCES individuals(id),
    individual_b_id INT REFERENCES individuals(id),
    status          TEXT NOT NULL DEFAULT 'waiting' CHECK (status IN ('waiting','pending','finished')),
    winner_id       INT REFERENCES individuals(id),
    resolution      TEXT,
    UNIQUE (season_id, round, tie_no)
);

CREATE TABLE IF NOT EXISTS matches (
    id              BIGSERIAL PRIMARY KEY,
    season_id       INT NOT NULL REFERENCES seasons(id),
    competition     TEXT NOT NULL CHECK (competition IN ('league','cup_group','cup_ko','recopa')),
    matchday        INT NOT NULL,                     -- fecha de Liga (39 = Recopa)
    round_no        INT,                              -- ronda dentro de la competición
    group_name      CHAR(1),
    tie_id          INT REFERENCES cup_ties(id),
    seq             INT NOT NULL DEFAULT 1,           -- ida/vuelta/desempate o nº de partida de Recopa
    x_individual_id INT NOT NULL REFERENCES individuals(id),  -- X = quien COMIENZA
    o_individual_id INT NOT NULL REFERENCES individuals(id),
    status          TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','running','finished')),
    result          TEXT CHECK (result IN ('x','o','draw')),
    winner_id       INT REFERENCES individuals(id),
    moves           TEXT,
    num_moves       INT,
    duration_ms     DOUBLE PRECISION,
    started_at      TIMESTAMPTZ,
    finished_at     TIMESTAMPTZ,
    UNIQUE (season_id, competition, matchday, x_individual_id, o_individual_id, seq),
    CHECK (x_individual_id <> o_individual_id),
    CHECK ((status = 'finished') = (result IS NOT NULL))
);
CREATE INDEX IF NOT EXISTS matches_pending_idx ON matches (season_id, status, matchday);

CREATE TABLE IF NOT EXISTS league_history (            -- evolución de la tabla tras cada fecha
    season_id     INT NOT NULL REFERENCES seasons(id),
    matchday      INT NOT NULL,
    individual_id INT NOT NULL REFERENCES individuals(id),
    played        INT NOT NULL,
    wins          INT NOT NULL,
    draws         INT NOT NULL,
    losses        INT NOT NULL,
    points        INT NOT NULL,
    position      INT NOT NULL,                        -- posición provisional (sin sorteo)
    PRIMARY KEY (season_id, matchday, individual_id)
);

CREATE TABLE IF NOT EXISTS cup_group_table (
    season_id     INT NOT NULL REFERENCES seasons(id),
    group_name    CHAR(1) NOT NULL,
    individual_id INT NOT NULL REFERENCES individuals(id),
    position      INT NOT NULL,
    played        INT NOT NULL,
    wins          INT NOT NULL,
    draws         INT NOT NULL,
    losses        INT NOT NULL,
    points        INT NOT NULL,
    qualified     BOOLEAN NOT NULL,
    PRIMARY KEY (season_id, group_name, individual_id)
);

CREATE TABLE IF NOT EXISTS tiebreak_draws (            -- todo sorteo de desempate, reproducible
    id            SERIAL PRIMARY KEY,
    season_id     INT NOT NULL REFERENCES seasons(id),
    context       TEXT NOT NULL,
    seed          BIGINT NOT NULL,
    members       JSONB NOT NULL,                      -- colores en orden canónico
    result_order  JSONB NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (season_id, context)
);

CREATE TABLE IF NOT EXISTS recopa_results (
    season_id          INT PRIMARY KEY REFERENCES seasons(id),
    cup_champion_id    INT NOT NULL REFERENCES individuals(id),
    league_champion_id INT NOT NULL REFERENCES individuals(id),
    same_champion      BOOLEAN NOT NULL,
    cup_champ_wins     INT NOT NULL DEFAULT 0,
    league_champ_wins  INT NOT NULL DEFAULT 0,
    draws              INT NOT NULL DEFAULT 0,
    champion_id        INT REFERENCES individuals(id),
    resolution         TEXT CHECK (resolution IN ('same_champion','wins','random_draw'))
);

-- Composición de la SIGUIENTE generación (7 supervivientes + 13 descendientes)
CREATE TABLE IF NOT EXISTS evolution_plan (
    season_id      INT NOT NULL REFERENCES seasons(id),
    position       INT NOT NULL CHECK (position BETWEEN 1 AND 20),
    role           TEXT NOT NULL CHECK (role IN ('survivor','child')),
    individual_id  INT NOT NULL REFERENCES individuals(id),
    parent_id      INT REFERENCES individuals(id),
    mutation_level TEXT,
    mutation_seed  BIGINT,
    PRIMARY KEY (season_id, position),
    UNIQUE (season_id, individual_id)
);

CREATE TABLE IF NOT EXISTS historical_best (
    id            SERIAL PRIMARY KEY,
    season_id     INT NOT NULL UNIQUE REFERENCES seasons(id),
    individual_id INT NOT NULL REFERENCES individuals(id),
    total_score   INT NOT NULL,
    model_file    TEXT NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Migración idempotente: fecha en que se borró el .pth de un modelo antiguo (el .json y la fila se conservan)
ALTER TABLE individuals ADD COLUMN IF NOT EXISTS model_pruned_at TIMESTAMPTZ;