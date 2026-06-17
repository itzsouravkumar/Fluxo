from fastapi import APIRouter
from fastapi.websockets import WebSocket

router = APIRouter(tags=["streams"])


@router.websocket("/ws/live")
async def live_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(data)
    except Exception:
        pass


@router.websocket("/ws/violations")
async def violation_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(data)
    except Exception:
        pass
