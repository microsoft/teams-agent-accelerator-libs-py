export function extractMemories(content: string): string[] {
  // Simple example: Extract sentences that contain key phrases
  const keyPhrases = ['remember', 'recall', 'memory', 'fact', 'learned'];
  
  return content
    .split('.')
    .filter(sentence => 
      keyPhrases.some(phrase => 
        sentence.toLowerCase().includes(phrase)
      )
    )
    .map(sentence => sentence.trim())
    .filter(Boolean);
}

export function formatTimestamp(date: Date): string {
  return new Intl.DateTimeFormat('en-US', {
    hour: 'numeric',
    minute: 'numeric',
    hour12: true
  }).format(date);
}