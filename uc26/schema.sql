-- UC26 point-in-time store (SQLite).
--
-- Rule: EVERY row carries `available_at` = the UTC moment the information became publicly
-- known (not when we scraped it, if we scraped it later; never earlier than the event itself).
-- The feature pipeline may only read rows with available_at < bet_time (features/pit.py).
--
-- Timestamps are TEXT in the fixed format 'YYYY-MM-DD HH:MM:SS' (UTC), so lexicographic
-- comparison == chronological comparison.

PRAGMA foreign_keys = ON;

-- Fixture: known as soon as the schedule is published.
CREATE TABLE IF NOT EXISTS games (
    game_id      TEXT PRIMARY KEY,
    league       TEXT NOT NULL,
    season       TEXT NOT NULL,
    start_time   TEXT NOT NULL,
    home_team    TEXT NOT NULL,
    away_team    TEXT NOT NULL,
    -- scheduled | final | postponed | cancelled  (non-final games are excluded from backtests,
    -- not silently dropped: survivorship trap, plan section 9)
    status       TEXT NOT NULL DEFAULT 'scheduled',
    source       TEXT,
    available_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_games_home ON games(home_team, available_at);
CREATE INDEX IF NOT EXISTS ix_games_away ON games(away_team, available_at);
CREATE INDEX IF NOT EXISTS ix_games_start ON games(start_time);

-- Result: known only after the final whistle. Split from `games` so that one row = one
-- `available_at` (the fixture is known weeks before the result).
-- Regulation result (60 min) drives the 1X2 market; per-period goals for later features.
CREATE TABLE IF NOT EXISTS results (
    game_id      TEXT PRIMARY KEY REFERENCES games(game_id),
    home_goals   INTEGER NOT NULL,          -- regulation
    away_goals   INTEGER NOT NULL,          -- regulation
    home_goals_p1 INTEGER, away_goals_p1 INTEGER,
    home_goals_p2 INTEGER, away_goals_p2 INTEGER,
    home_goals_p3 INTEGER, away_goals_p3 INTEGER,
    home_shots   INTEGER, away_shots INTEGER,
    home_pp      INTEGER, away_pp INTEGER,  -- power-play opportunities
    decided_in   TEXT,                      -- REG | OT | SO
    source       TEXT,
    available_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_results_avail ON results(available_at);

-- Confirmed lineup of the day (who actually dresses / starts in goal). In marginal leagues
-- this is confirmed shortly before puck drop -> realistic bet_time is ~T-1h.
CREATE TABLE IF NOT EXISTS lineups (
    game_id      TEXT NOT NULL REFERENCES games(game_id),
    team         TEXT NOT NULL,
    player_id    TEXT NOT NULL,
    position     TEXT NOT NULL,             -- F | D | G
    is_starting_goalie INTEGER NOT NULL DEFAULT 0,
    source       TEXT,
    available_at TEXT NOT NULL,
    PRIMARY KEY (game_id, team, player_id)
);
CREATE INDEX IF NOT EXISTS ix_lineups_game ON lineups(game_id, available_at);

-- Per-player box score, known after the game.
CREATE TABLE IF NOT EXISTS player_games (
    game_id      TEXT NOT NULL REFERENCES games(game_id),
    team         TEXT NOT NULL,
    player_id    TEXT NOT NULL,
    position     TEXT NOT NULL,
    toi_sec      INTEGER,
    goals        INTEGER,
    assists      INTEGER,
    shots_against INTEGER,                  -- goalies only
    saves        INTEGER,                   -- goalies only
    source       TEXT,
    available_at TEXT NOT NULL,
    PRIMARY KEY (game_id, team, player_id)
);
CREATE INDEX IF NOT EXISTS ix_pg_team ON player_games(team, available_at);

-- Every odds observation we ever saw, with the time we saw it (ts) and available_at
-- (normally == ts). Never use a price that did not exist at bet time.
-- market: 1X2 | HCP | TOTAL ; selection: H | D | A | OVER | UNDER ; line 0 for 1X2.
CREATE TABLE IF NOT EXISTS odds_snapshots (
    game_id      TEXT NOT NULL REFERENCES games(game_id),
    book         TEXT NOT NULL,
    market       TEXT NOT NULL,
    selection    TEXT NOT NULL,
    line         REAL NOT NULL DEFAULT 0,  -- 0 for 1X2
    ts           TEXT NOT NULL,
    odds         REAL NOT NULL CHECK (odds > 1.0),
    source       TEXT,
    available_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_odds_game ON odds_snapshots(game_id, book, market, available_at);

-- Closing line (last price before start). Used ONLY for CLV and Baseline A, never for profit.
CREATE TABLE IF NOT EXISTS closing_odds (
    game_id      TEXT NOT NULL REFERENCES games(game_id),
    book         TEXT NOT NULL,
    market       TEXT NOT NULL,
    selection    TEXT NOT NULL,
    line         REAL NOT NULL DEFAULT 0,  -- 0 for 1X2
    odds         REAL NOT NULL CHECK (odds > 1.0),
    source       TEXT,
    available_at TEXT NOT NULL,
    PRIMARY KEY (game_id, book, market, selection, line)
);
