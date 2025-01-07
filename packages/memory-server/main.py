from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import memory

app = FastAPI(title="Chat API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(memory.router, tags=["memory"])

