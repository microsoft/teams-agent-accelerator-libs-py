-- Create new table with desired schema
CREATE TABLE messages_new (
    id TEXT NOT NULL,
    content TEXT NOT NULL,
    author_id TEXT NOT NULL,
    conversation_ref TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL,
    deep_link TEXT,
    type TEXT NOT NULL
);

-- Copy data from old table to new table
INSERT INTO messages_new (id, content, author_id, conversation_ref, created_at, deep_link, type)
SELECT id, content, author_id, conversation_ref, created_at, deep_link,
    CASE 
        WHEN is_assistant_message = 1 THEN 'assistant'
        ELSE 'user'
    END
FROM messages;

-- Drop old table
DROP TABLE messages;

-- Rename new table to original name
ALTER TABLE messages_new RENAME TO messages;