import asyncio
from websockets.server import serve
import wave
import numpy as np
import torch
import torchaudio


def int2float(sound):
    abs_max = np.abs(sound).max()
    sound = sound.astype('float32')
    if abs_max > 0:
        sound *= 1 / 32768
    sound = sound.squeeze()  # depends on the use case
    return sound


async def echo(websocket):
    async for message in websocket:
        print(message)


async def main():
    async with serve(echo, "localhost", 8765):
        await asyncio.Future()  # run forever


torch.set_num_threads(1)
torchaudio.set_audio_backend("soundfile")
vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad',
                                  force_reload=False,
                                  trust_repo=True)
asyncio.run(main())
