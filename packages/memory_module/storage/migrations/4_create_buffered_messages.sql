CREATE TABLE IF NOT EXISTS buffered_messages (
    message_id TEXT NOT NULL,
    content TEXT NOT NULL,
    author_id TEXT NOT NULL,
    conversation_ref TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL,
    PRIMARY KEY (message_id)
);

CREATE INDEX IF NOT EXISTS idx_buffered_messages_conversation_ref 
ON buffered_messages(conversation_ref); 