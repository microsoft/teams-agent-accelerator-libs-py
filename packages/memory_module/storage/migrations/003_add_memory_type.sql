-- Add memory_type column
ALTER TABLE memories ADD COLUMN memory_type TEXT NOT NULL DEFAULT 'semantic'; 