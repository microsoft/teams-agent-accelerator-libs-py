// Initialize the Teams SDK
microsoftTeams.app.initialize().then(() => {
    // Load existing memories from localStorage
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
 * Loads and displays memories from localStorage
 */
function loadMemories() {
    const memoriesContainer = document.getElementById('memoriesContainer');
    const memories = JSON.parse(localStorage.getItem('memories') || '[]');
    
    memoriesContainer.innerHTML = memories.map(memory => `
        <div class="memory-card">
            <h3>${memory.title}</h3>
            <div class="date">${new Date(memory.date).toLocaleDateString()}</div>
            <div class="description">${memory.description}</div>
        </div>
    `).join('');
}
  