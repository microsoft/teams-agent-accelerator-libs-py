CREATE TABLE IF NOT EXISTS memory_attributions (
    memory_id TEXT NOT NULL,
    message_id TEXT NOT NULL,
    PRIMARY KEY (memory_id, message_id),
    FOREIGN KEY (memory_id) REFERENCES memories(id)
)