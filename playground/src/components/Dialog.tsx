import React from 'react';
import { Memory } from '../types';

interface DialogProps {
  memory: Memory;
  onClose: () => void;
}

const Dialog: React.FC<DialogProps> = ({ memory, onClose }) => {
  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center" onClick={onClose}>
      <div className="bg-white p-8 rounded-lg shadow-lg w-1/3" onClick={(e) => e.stopPropagation()}>
        <h2 className="text-2xl font-bold mb-6 text-black">Memory Details</h2>
        <p className="mb-4 text-black"><strong>ID:</strong> {memory.id}</p>
        <p className="mb-4 text-black"><strong>Content:</strong> {memory.content}</p>
        <p className="mb-4 text-black"><strong>Created At:</strong> {memory.created_at}</p>
        <p className="mb-4 text-black"><strong>User Id:</strong> {memory.user_id}</p>
        <p className="mb-4 text-black"><strong>Memory Type:</strong> {memory.memory_type}</p>
        <p className="mb-4 text-black"><strong>Updated At:</strong> {memory.updated_at ?? "-"}</p>
        <div className="flex justify-end">
          <button className="mt-6 px-4 py-2 bg-teams text-white rounded" onClick={onClose}>Close</button>
        </div>
      </div>
    </div>
  );
};

export default Dialog;
