"""
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
"""

import os
from http import HTTPStatus

from aiohttp import web
from botbuilder.core.integration import aiohttp_error_middleware

from bot import bot_app, memory_middleware
from config import Config

routes = web.RouteTableDef()


@routes.post("/api/messages")
async def on_messages(req: web.Request) -> web.Response:
    res = await bot_app.process(req)
    if res is not None:
        return res
    return web.Response(status=HTTPStatus.OK)


@routes.get("/api/memories")
async def get_memories(request: web.Request) -> web.Response:
    # TODO: Auth
    user_id = request.query.get("userId")
    if not user_id:
        return web.Response(status=HTTPStatus.BAD_REQUEST, text="Missing userId parameter")
    print("Get_memories for user", user_id)
    memories = await memory_middleware.memory_module.get_user_memories(user_id)
    return web.json_response([memory.model_dump(mode="json", by_alias=True) for memory in memories])


app = web.Application(middlewares=[aiohttp_error_middleware])
app.add_routes(routes)

app.router.add_static("/memoriesTab", os.path.join(os.path.dirname(__file__), "public/memoriesTab"))

if __name__ == "__main__":
    web.run_app(app, host="localhost", port=Config.PORT)
