CREATE TABLE IF NOT EXISTS scheduled_events (
    id TEXT PRIMARY KEY,
    object TEXT NOT NULL,
    scheduled_time TIMESTAMP NOT NULL
); 