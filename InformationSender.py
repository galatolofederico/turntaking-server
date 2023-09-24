import asyncio
import websockets


async def send(uri, queue):
    async with websockets.connect(uri) as websocket:
        while True:
            information = queue.get()
            await websocket.send(information)


def send_information(uri, queue):
    asyncio.run(send(uri, queue))
