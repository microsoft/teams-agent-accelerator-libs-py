-- Convert memories.id to TEXT
CREATE TABLE memories_new (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL,
    user_id TEXT,
    memory_type TEXT NOT NULL DEFAULT 'semantic'
);

-- Copy data from old table to new table, converting id to TEXT
INSERT INTO memories_new SELECT CAST(id AS TEXT), content, created_at, user_id, memory_type FROM memories;

-- Drop the old table
DROP TABLE memories;

-- Rename the new table to the original name
ALTER TABLE memories_new RENAME TO memories;