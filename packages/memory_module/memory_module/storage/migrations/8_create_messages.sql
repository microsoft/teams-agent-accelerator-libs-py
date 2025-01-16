CREATE TABLE IF NOT EXISTS messages (
    id TEXT NOT NULL,
    content TEXT NOT NULL,
    author_id TEXT NOT NULL,
    conversation_ref TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL,
    is_assistant_message BOOLEAN NOT NULL DEFAULT FALSE,
    PRIMARY KEY (id)
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation_ref 
ON messages(conversation_ref);