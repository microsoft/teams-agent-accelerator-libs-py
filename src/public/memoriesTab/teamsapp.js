// Initialize the Teams SDK with fallback for browser testing
function initializeTeamsOrBrowser() {
    try {
        return microsoftTeams.app.initialize();
    } catch (err) {
        console.log('Teams SDK not loaded - running in browser mode:', err);
        return Promise.resolve();
    }
}

// Replace the existing initialization with this one
initializeTeamsOrBrowser().then(() => {
    loadMemories();
    
    // Handle form submission
    document.getElementById('memoryForm').addEventListener('submit', (e) => {
        e.preventDefault();
        
        const memory = {
            id: Date.now(),
            title: document.getElementById('title').value,
            date: document.getElementById('date').value,
            description: document.getElementById('description').value
        };
        
        saveMemory(memory);
        document.getElementById('memoryForm').reset();
    });
});

/**
 * Saves a memory to localStorage
 * @param {Object} memory - The memory object to save
 */
function saveMemory(memory) {
    let memories = JSON.parse(localStorage.getItem('memories') || '[]');
    memories.push(memory);
    localStorage.setItem('memories', JSON.stringify(memories));
    loadMemories();
}

/**
 * Loads and displays memories from the server
 */
async function loadMemories(type = 'semantic') {
    const memoriesContainer = document.getElementById('memoriesContainer');
    try {
        const response = await fetch('/api/memories');
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const memories = await response.json();
        
        // Filter memories by type
        const filteredMemories = memories.filter(memory => 
            (memory.type || 'SEMANTIC').toUpperCase() === type.toUpperCase()
        );
        
        if (filteredMemories.length === 0) {
            memoriesContainer.innerHTML = '<div class="no-memories">No memories found</div>';
            return;
        }

        // Create header row
        const headerRow = `
            <div class="memory-row">
                <div class="memory-header">ID</div>
                <div class="memory-header">Content</div>
                <div class="memory-header">Date</div>
            </div>
        `;

        // Create memory rows
        const memoryRows = filteredMemories.map(memory => `
            <div class="memory-row">
                <div class="memory-cell memory-id">#${memory.id}</div>
                <div class="memory-cell memory-content">
                    ${memory.content}
                    <div class="memory-metadata">
                        <span class="metadata-item">
                            <span class="metadata-label">Type:</span>
                            ${memory.type === 'SEMANTIC' ? 'Semantic (Fact/Preference)' : 'Episodic (Event/Experience)'}
                        </span>
                    </div>
                </div>
                <div class="memory-cell memory-date">${new Date(memory.created_at).toLocaleString()}</div>
            </div>
        `).join('');

        memoriesContainer.innerHTML = headerRow + memoryRows;
    } catch (error) {
        console.error('Error loading memories:', error);
        memoriesContainer.innerHTML = `
            <div class="error">
                Error loading memories. Please try again later.
            </div>
        `;
    }
}

// Add event listeners for the toggle buttons
document.getElementById('semantic-toggle').addEventListener('click', () => toggleMemoryType('semantic'));
document.getElementById('episodic-toggle').addEventListener('click', () => toggleMemoryType('episodic'));

// Initialize with semantic memories
toggleMemoryType('semantic');

// Add these functions at the end of the file
function toggleMemoryType(type) {
    document.getElementById('semantic-toggle').classList.toggle('active', type === 'semantic');
    document.getElementById('episodic-toggle').classList.toggle('active', type === 'episodic');
    loadMemories(type);
}
  