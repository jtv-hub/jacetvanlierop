-- Enable WAL mode (for concurrency during reads/writes)
PRAGMA journal_mode=WAL;

-- ========================
-- 1. TRADE AUDIT & CAPITAL TRACKING
-- ========================
CREATE TABLE IF NOT EXISTS trades (
    trade_id TEXT PRIMARY KEY,
    timestamp DATETIME NOT NULL,
    pair TEXT NOT NULL,
    side TEXT CHECK(side IN ('buy', 'sell')) NOT NULL,
    entry_price REAL NOT NULL,
    exit_price REAL,
    size_usd REAL NOT NULL,
    roi REAL,
    pnl_usd REAL,
    confidence REAL CHECK(confidence BETWEEN 0.0 AND 1.0),
    strategy_id TEXT NOT NULL,
    status TEXT CHECK(status IN ('open', 'closed')) NOT NULL,
    exit_reason TEXT,
    rsi REAL,
    slippage_bps REAL
);

CREATE INDEX IF NOT EXISTS idx_trades_pnl ON trades(pnl_usd DESC, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_trades_strategy_month ON trades(strategy_id, strftime('%Y-%m', timestamp));

-- ========================
-- 2. LIVE POSITIONS
-- ========================
CREATE TABLE IF NOT EXISTS positions (
    trade_id TEXT PRIMARY KEY,
    pair TEXT NOT NULL,
    side TEXT NOT NULL,
    entry_price REAL NOT NULL,
    size_usd REAL NOT NULL,
    timestamp DATETIME NOT NULL,
    strategy_id TEXT NOT NULL,
    target_roi REAL,
    FOREIGN KEY (trade_id) REFERENCES trades(trade_id) ON DELETE CASCADE
);

-- ========================
-- 3. PPO REINFORCEMENT LEARNING
-- ========================
CREATE TABLE IF NOT EXISTS rl_episodes (
    episode_id INTEGER PRIMARY KEY AUTOINCREMENT,
    start_time DATETIME NOT NULL,
    end_time DATETIME,
    total_pnl_usd REAL,
    win_rate REAL,
    sharpe REAL,
    max_drawdown_pct REAL,
    strategy_id TEXT NOT NULL,
    monthly_goal_progress REAL
);

CREATE TABLE IF NOT EXISTS rl_actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    trade_id TEXT,
    state_vector BLOB NOT NULL,
    action TEXT NOT NULL,
    confidence REAL,
    strategy_id TEXT NOT NULL,
    reward REAL,
    episode_id INTEGER NOT NULL,
    FOREIGN KEY (trade_id) REFERENCES trades(trade_id),
    FOREIGN KEY (episode_id) REFERENCES rl_episodes(episode_id)
);

CREATE INDEX IF NOT EXISTS idx_rl_episode ON rl_actions(episode_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_rl_reward ON rl_actions(reward DESC);

-- ========================
-- 4. NSGA-3 GENETIC EVOLUTION
-- ========================
CREATE TABLE IF NOT EXISTS strategies (
    strategy_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    config JSON NOT NULL,
    parent_id TEXT,
    mutation_type TEXT,
    generation INTEGER NOT NULL,
    created_at DATETIME NOT NULL,
    is_live BOOLEAN DEFAULT 0,
    monthly_pnl_usd REAL DEFAULT 0,
    FOREIGN KEY (parent_id) REFERENCES strategies(strategy_id)
);

CREATE TABLE IF NOT EXISTS nsga_generations (
    gen INTEGER NOT NULL,
    strategy_id TEXT NOT NULL,
    monthly_roi_projected REAL,
    sharpe REAL,
    win_rate REAL,
    max_drawdown_pct REAL,
    calmar REAL,
    sortino REAL,
    capital_efficiency REAL,
    rank INTEGER,
    crowding_distance REAL,
    PRIMARY KEY (gen, strategy_id),
    FOREIGN KEY (strategy_id) REFERENCES strategies(strategy_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_nsga_projection ON nsga_generations(monthly_roi_projected DESC);
CREATE INDEX IF NOT EXISTS idx_nsga_gen_live ON nsga_generations(gen DESC) WHERE rank = 1;

CREATE TABLE IF NOT EXISTS pareto_fronts (
    gen INTEGER NOT NULL,
    strategy_id TEXT NOT NULL,
    front_rank INTEGER NOT NULL,
    projected_monthly_usd REAL,
    PRIMARY KEY (gen, strategy_id),
    FOREIGN KEY (strategy_id) REFERENCES strategies(strategy_id)
);

-- ========================
-- 5. SHADOW TESTING RESULTS
-- ========================
CREATE TABLE IF NOT EXISTS shadow_results (
    run_id TEXT PRIMARY KEY,
    strategy_id TEXT NOT NULL,
    timestamp DATETIME NOT NULL,
    win_rate REAL,
    avg_pnl_usd REAL,
    monthly_pnl_projection REAL,
    sharpe REAL,
    max_drawdown_pct REAL,
    total_trades INTEGER,
    config_snapshot JSON,
    passes_10k_test BOOLEAN,
    FOREIGN KEY (strategy_id) REFERENCES strategies(strategy_id)
);

-- ========================
-- 6. REAL-TIME EQUITY + RISK TRACKING
-- ========================
CREATE TABLE IF NOT EXISTS equity_curve (
    timestamp DATETIME PRIMARY KEY,
    total_equity_usd REAL NOT NULL,
    available_capital_usd REAL NOT NULL,
    daily_pnl_usd REAL NOT NULL,
    monthly_pnl_usd REAL NOT NULL,
    monthly_pnl_target REAL DEFAULT 10000,
    progress_pct REAL,
    drawdown_pct REAL NOT NULL,
    buffer_active BOOLEAN,
    risk_per_trade_pct REAL
);

CREATE INDEX IF NOT EXISTS idx_equity_month ON equity_curve(strftime('%Y-%m', timestamp));

CREATE TABLE IF NOT EXISTS risk_flags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    flag_type TEXT NOT NULL,
    reason TEXT,
    active BOOLEAN DEFAULT 1,
    cleared_at DATETIME,
    impact_on_10k_goal TEXT
);
