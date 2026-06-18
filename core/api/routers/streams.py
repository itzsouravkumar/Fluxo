from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import json
import asyncio

from ..state import event_bus

router = APIRouter(tags=["streams"])


class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, list[WebSocket]] = {
            "live": [],
            "violations": [],
            "junctions": [],
        }

    async def connect(self, websocket: WebSocket, channel: str):
        await websocket.accept()
        if channel not in self.active_connections:
            self.active_connections[channel] = []
        self.active_connections[channel].append(websocket)

    def disconnect(self, websocket: WebSocket, channel: str):
        if channel in self.active_connections:
            self.active_connections[channel] = [
                ws for ws in self.active_connections[channel] if ws != websocket
            ]

    async def broadcast(self, channel: str, data: dict):
        dead = []
        for ws in self.active_connections.get(channel, []):
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.active_connections[channel].remove(ws)


manager = ConnectionManager()


@router.websocket("/ws/live")
async def live_stream(websocket: WebSocket):
    await manager.connect(websocket, "live")
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            topic = msg.get("topic", "junctions")
            latest = event_bus.get_latest(topic)
            if latest:
                await websocket.send_json(latest)
    except WebSocketDisconnect:
        manager.disconnect(websocket, "live")
    except Exception:
        manager.disconnect(websocket, "live")


@router.websocket("/ws/violations")
async def violation_stream(websocket: WebSocket):
    await manager.connect(websocket, "violations")
    try:
        while True:
            await asyncio.sleep(1)
            latest = event_bus.get_latest("violation")
            if latest:
                await websocket.send_json(latest)
    except WebSocketDisconnect:
        manager.disconnect(websocket, "violations")
    except Exception:
        manager.disconnect(websocket, "violations")


@router.websocket("/ws/junctions")
async def junction_stream(websocket: WebSocket):
    await manager.connect(websocket, "junctions")
    try:
        while True:
            await asyncio.sleep(1)
            latest = event_bus.get_latest("junctions")
            if latest:
                await websocket.send_json(latest)
    except WebSocketDisconnect:
        manager.disconnect(websocket, "junctions")
    except Exception:
        manager.disconnect(websocket, "junctions")


@router.get("/api/v1/stream/status")
async def stream_status():
    return {
        "channels": {
            name: len(conns) for name, conns in manager.active_connections.items()
        },
        "total_connections": sum(len(conns) for conns in manager.active_connections.values()),
    }
