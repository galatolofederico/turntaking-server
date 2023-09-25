import sys
import numpy as np
import pyaudio
import torch
import torchaudio
from enum import Enum


class State(Enum):
    NOT_STARTED = 0
    STARTED = 1
    POTENTIAL_TURN_CHANGE = 2
    CONVERSATION_NOT_STARTED = 3


# PyAudio parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1 if sys.platform == 'darwin' else 2
RATE = 16000  # Hz
FRAME_DURATION = 0.05  # s
CHUNK = int(RATE * FRAME_DURATION)

# Detection parameters
SILENCE_THRESHOLD = 0.5  # s
CONFIRMED_SILENCE_THRESHOLD = 2  # s
CONFIDENCE_THRESHOLD = 0.5  #
START_CONVERSATION_THRESHOLD = 10  # s
CONVERSATION_NOT_STARTED_THRESHOLD = 5  # s


def int2float(sound):
    abs_max = np.abs(sound).max()
    sound = sound.astype('float32')
    if abs_max > 0:
        sound *= 1 / 32768
    sound = sound.squeeze()  # depends on the use case
    return sound


def detect_turn_change(stream, vad_model, queue):
    print("Started Recording")
    print(" ")
    state = State.NOT_STARTED
    cumulative_silence = 0  # s
    while True:
        # gets the audio_chunk
        audio_chunk = int2float(np.frombuffer(stream.read(CHUNK), np.int16))

        # get the confidences
        new_confidence = vad_model(torch.from_numpy(audio_chunk), 16000).item()

        # check if the new frame has voice
        if new_confidence <= CONFIDENCE_THRESHOLD:
            cumulative_silence += FRAME_DURATION
            if state == State.STARTED and cumulative_silence >= SILENCE_THRESHOLD:
                state = State.POTENTIAL_TURN_CHANGE
                print("Potential turn change")
                queue.put("Potential turn change")

            elif state == State.POTENTIAL_TURN_CHANGE and cumulative_silence >= CONFIRMED_SILENCE_THRESHOLD:
                state = State.NOT_STARTED  # we need to go back to the NOT_STARTED state to initiate a new turn
                print("Turn change confirmed")
                print(" ")
                queue.put("Turn change confirmed")

            # if for more than CONVERSATION_NOT_STARTED_THRESHOLD there is silence
            # the user may have not understood the response and we should repeat it
            elif state == State.NOT_STARTED and cumulative_silence >= CONVERSATION_NOT_STARTED_THRESHOLD:
                state = State.CONVERSATION_NOT_STARTED
                print("Conversation not started")
                print(" ")
                queue.put("Conversation not started")

            # if the user stay silent for more than START_CONVERSATION_THRESHOLD
            # the robot may want to start the conversation
            elif state == State.CONVERSATION_NOT_STARTED and cumulative_silence >= START_CONVERSATION_THRESHOLD:
                cumulative_silence = 0
                state = state.NOT_STARTED
                print("Start conversation")
                print(" ")
                queue.put("Start conversation")

        else:
            cumulative_silence = 0
            if state == State.NOT_STARTED or state == State.CONVERSATION_NOT_STARTED:
                state = State.STARTED
                print("Conversation started")
                queue.put("Conversation started")

            elif state == State.POTENTIAL_TURN_CHANGE:
                # if it was detected a potential turn change we need to send a message to
                # interrupt the pipeline
                state = State.STARTED
                print("Potential turn change aborted")
                queue.put("Potential turn change aborted")


# plot the confidences for the speech
# plt.figure(figsize=(20, 6))
# plt.plot(voiced_confidences)
# plt.show()  # activate to show the plot of the frames and relative confidence vad


def analyze(queue):
    torch.set_num_threads(1)
    torchaudio.set_audio_backend("soundfile")
    vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                      model='silero_vad',
                                      force_reload=False)

    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

    detect_turn_change(stream, vad_model, queue)

    stream.close()
    audio.terminate()
