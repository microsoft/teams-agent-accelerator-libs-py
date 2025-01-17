-- Create memories table
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP,
    user_id TEXT,
    memory_type TEXT NOT NULL DEFAULT 'semantic',
    topics TEXT
);

-- Create memory_attributions table
CREATE TABLE IF NOT EXISTS memory_attributions (
    memory_id TEXT NOT NULL,
    message_id TEXT NOT NULL,
    PRIMARY KEY (memory_id, message_id),
    FOREIGN KEY (memory_id) REFERENCES memories(id)
);

-- Create messages table
CREATE TABLE IF NOT EXISTS messages (
    id TEXT NOT NULL,
    content TEXT NOT NULL,
    author_id TEXT NOT NULL,
    conversation_ref TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL,
    deep_link TEXT,
    type TEXT NOT NULL,
    PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation_ref 
ON messages(conversation_ref);

-- Create buffered_messages table
CREATE TABLE IF NOT EXISTS buffered_messages (
    message_id TEXT NOT NULL,
    conversation_ref TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL,
    PRIMARY KEY (message_id)
);

CREATE INDEX IF NOT EXISTS idx_buffered_messages_conversation_ref 
ON buffered_messages(conversation_ref);

-- Create scheduled_events table
CREATE TABLE IF NOT EXISTS scheduled_events (
    id TEXT PRIMARY KEY,
    object TEXT NOT NULL,
    scheduled_time TIMESTAMP NOT NULL
);

-- Create embeddings table
CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id INTEGER NOT NULL,
    embedding float[1536],
    text TEXT
); 