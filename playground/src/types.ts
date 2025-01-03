export type Message = {
  role: 'user' | 'assistant';
  content: string;
};

export type Memory = {
  content: string;
  timestamp: string;
};

export type Conversation = {
  id: string;
  title: string;
  preview: string;
  timestamp: string;
  messages: Message[];
};