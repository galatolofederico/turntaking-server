import wave
import sys

import pyaudio

CHUNK = 1024

# Instantiate PyAudio and initialize PortAudio system resources (1)
p = pyaudio.PyAudio()

# Open stream (2)
stream = p.open(format=p.get_format_from_width(2),
                channels=2,
                rate=16000,
                output=True)

# Close stream (4)
stream.close()

# Release PortAudio system resources (5)
p.terminate()


def play(frame):
    global stream
    stream.write(frame)
