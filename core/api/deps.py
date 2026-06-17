from fastapi import Request


async def get_redis(request: Request):
    return request.app.state.redis


async def get_detector(request: Request):
    return getattr(request.app.state, "detector", None)


async def get_rl_agent(request: Request):
    return getattr(request.app.state, "rl_agent", None)
