import sys
import numpy as np
import pyaudio
import torch
import torchaudio

# PyAudio parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1 if sys.platform == 'darwin' else 2
RATE = 16000  # Hz
FRAME_DURATION = 0.05  # s
CHUNK = int(RATE * FRAME_DURATION)

# Detection parameters
SILENCE_THRESHOLD = 0.7  # s
CONFIDENCE_THRESHOLD = 0.5  #
START_CONVERSATION_THRESHOLD = 5  # s


def int2float(sound):
    abs_max = np.abs(sound).max()
    sound = sound.astype('float32')
    if abs_max > 0:
        sound *= 1 / 32768
    sound = sound.squeeze()  # depends on the use case
    return sound


def detect_turn_change(stream, vad_model):
    print("Started Recording")
    cumulative_silence = 0  # s
    conversation_started = False
    while True:
        audio_chunk = stream.read(CHUNK)

        audio_int16 = np.frombuffer(audio_chunk, np.int16)

        audio_float32 = int2float(audio_int16)

        # get the confidences
        new_confidence = vad_model(torch.from_numpy(audio_float32), 16000).item()

        # check if the new frame has voice
        if new_confidence <= CONFIDENCE_THRESHOLD:
            cumulative_silence += FRAME_DURATION
            if cumulative_silence >= SILENCE_THRESHOLD and conversation_started:
                print("Turn change")
                print(" ")
                return "Turn change"
            if cumulative_silence >= START_CONVERSATION_THRESHOLD and not conversation_started:
                print("Conversation not started")
                print(" ")
                return "Conversation not started"
        else:
            conversation_started = True
            cumulative_silence = 0


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

    while True:
        detection = detect_turn_change(stream, vad_model)
        queue.put(detection)

    stream.close()
    audio.terminate()
