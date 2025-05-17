import asyncio, json
from typing import Dict

class EventBus:
    """
    Give each device a UUID.  All POST endpoints push
    events into that tab's asyncio.Queue; the SSE endpoint drains it.
    """
    _queues: Dict[str, asyncio.Queue] = {}

    @classmethod
    def queue(cls, client_id: str) -> asyncio.Queue:
        return cls._queues.setdefault(client_id, asyncio.Queue())

    @classmethod
    async def push(cls, client_id: str, event: str, data: dict | str):
        payload = {
            "event": event,
            "data": json.dumps(data) if isinstance(data, (dict, list)) else str(data),
        }
        await cls.queue(client_id).put(payload)