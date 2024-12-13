-- Drop existing table
DROP TABLE IF EXISTS buffered_messages;

-- Create new buffered_messages table that only references messages
CREATE TABLE buffered_messages (
    message_id TEXT NOT NULL,
    conversation_ref TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL,
    PRIMARY KEY (message_id)
);

CREATE INDEX IF NOT EXISTS idx_buffered_messages_conversation_ref 
ON buffered_messages(conversation_ref); 