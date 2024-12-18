"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""

import json
import os
from http import HTTPStatus
from pathlib import Path

from aiohttp import web
from botbuilder.core.integration import aiohttp_error_middleware
from memory_module import LLMConfig, MemoryModule, MemoryModuleConfig

from bot import bot_app, memory_llm_config
from config import Config

routes = web.RouteTableDef()

memory_module = MemoryModule(
    config=MemoryModuleConfig(
        llm=LLMConfig(**memory_llm_config),
        db_path=Path(os.path.join(os.path.dirname(__file__), "data", "memory.db")),
        timeout_seconds=60,
    )
)


@routes.post("/api/messages")
async def on_messages(req: web.Request) -> web.Response:
    res = await bot_app.process(req)
    if res is not None:
        return res
    return web.Response(status=HTTPStatus.OK)


@routes.get("/api/memories")
async def get_memories(request: web.Request) -> web.Response:
    # Get all memories with an empty query
    memories = await memory_module.retrieve_memories("", None, None)
    return web.Response(
        text=json.dumps(
            [
                {
                    "id": memory.id,
                    "content": memory.content,
                    "created_at": memory.created_at.isoformat() if memory.created_at else None,
                }
                for memory in memories
            ]
        ),
        content_type="application/json",
    )


app = web.Application(middlewares=[aiohttp_error_middleware])
app.add_routes(routes)

app.router.add_static("/public", os.path.join(os.path.dirname(__file__), "public"))

if __name__ == "__main__":
    web.run_app(app, host="localhost", port=Config.PORT)
