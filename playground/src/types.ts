export interface Memory {
  id: string;
  content: string;
  created_at: string;
  updated_at: string;
  memory_type: 'semantic' | 'episodic';
  user_id: string;
  message_attributions: string[];
}