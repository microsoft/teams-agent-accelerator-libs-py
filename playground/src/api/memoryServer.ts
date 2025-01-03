const BASE_URL = 'http://127.0.0.1:8000';

export const addMessage = async (content: string, type: 'assistant' | 'user') => {
  try {
    const response = await fetch(`${BASE_URL}/message?type=${type}&content=${content}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      throw new Error('Failed to add message');
    }

    return await response.json();
  } catch (error) {
    console.error('Error adding message:', error);
    throw error;
  }
};

export const retrieveMemories = async () => {
  try {
    const response = await fetch(`${BASE_URL}/memories`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      throw new Error('Failed to retrieve memories');
    }

    return await response.json();
  } catch (error) {
    console.error('Error retrieving memories:', error);
    throw error;
  }
};
