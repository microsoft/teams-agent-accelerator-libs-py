import os
import sys
from fastapi import APIRouter
from memory_service import MemoryService
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from memory_module.interfaces.types import Memory

router = APIRouter()
openai_api_key = os.environ.get("OPENAI_API_KEY")
memory_service = MemoryService(openai_api_key=openai_api_key)

@router.get("/memories", response_model=list[Memory])
async def get_memories(user_id: str):
    return await memory_service.get_all_memories(user_id=user_id)

class SearchQueryRequest(BaseModel):
    query: str
    user_id: str
    limit: int

@router.post("/search")
async def search_memories(request: SearchQueryRequest):
    if not request.user_id:
        raise ValueError("User ID is required to search memories")
    
    print(request)
    if len(request.query) > 0:
        return await memory_service.retrieve_memories(request.query, request.user_id, request.limit)
    else:
        # Return all memories if no query is provided
        return await memory_service.get_all_memories(user_id=request.user_id)

class AddMessageRequest(BaseModel):
    type: str
    content: str

@router.post("/message")
async def add_message(request: AddMessageRequest):
    await memory_service.add_message(request.type, request.content)
    return {"message": "Message added successfully"}