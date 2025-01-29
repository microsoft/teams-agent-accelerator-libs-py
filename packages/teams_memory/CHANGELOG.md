# Release Notes

## Latest Changes

### Features

### Bug Fixes

## 0.1.1-alpha

### Features

- [Minor] Refactors to remove unused codepaths and improve readability
  - Remove get_memories_by_user_id and consolidate to get_memories
  - Change get_messages to accept message_ids and not memory_ids
  - Remove references to short_term_memory - that's just messages
  - Consolidate remove_memories to just 1 method (it was remove_memories, and clear_memories). In storage, call it "delete_memories"
  - Add remove_messages for completion's sake
- Updated memory_module to teams_memory
- Added documentation for public functions in public abstract base classes
- [Breaking] Updated search_memories to accept strings for topic instead of Topic objects
- [Breaking] Stop extracting memories automatically - now you must call listen/shutdown to begin automatic extraction
- Added `process_messages` to allow for manual extraction of memories
- Improved integration documentation

### Bug Fixes

## 0.1.0-alpha

### Features

- Added memory module with short-term and semantic memory extraction.
- Implemented decision-making logic and memory retrieval/search functionality.
- Added scoped memory module for targeted memory operations.
- Added SQLite storage support.
- Added In-Memory storage.
- Added Tech Support Assistant sample and memory confirmation with citations.
- Improved deduplication and similarity calculations (cosine distance).
- Added evals for memory extraction and retrieval.
