import torch
import torchaudio
import sys
import wave
import matplotlib.pylab as plt
import pyaudio
import numpy as np
import whisper

whisper_model = whisper.load_model("base")
torch.set_num_threads(1)
torchaudio.set_audio_backend("soundfile")

vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad',
                                  force_reload=True)

(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils

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

audio = pyaudio.PyAudio()

stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
data = []
voiced_confidences = []


def int2float(sound):
    abs_max = np.abs(sound).max()
    sound = sound.astype('float32')
    if abs_max > 0:
        sound *= 1 / 32768
    sound = sound.squeeze()  # depends on the use case
    return sound


with wave.open('output.wav', 'wb') as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)

    print("Started Recording")
    cumulative_silence = 0  # s
    conversation_started = False
    while True:
        audio_chunk = stream.read(CHUNK)

        # in case you want to save the audio later
        data.append(audio_chunk)
        wf.writeframes(audio_chunk)

        audio_int16 = np.frombuffer(audio_chunk, np.int16)

        audio_float32 = int2float(audio_int16)

        # get the confidences and add them to the list to plot them later
        new_confidence = vad_model(torch.from_numpy(audio_float32), 16000).item()
        voiced_confidences.append(new_confidence)

        # check if the new frame has voice
        if new_confidence <= CONFIDENCE_THRESHOLD:
            cumulative_silence += FRAME_DURATION
            if cumulative_silence >= SILENCE_THRESHOLD and conversation_started:
                print("Turn change")
                break
            if cumulative_silence >= START_CONVERSATION_THRESHOLD and not conversation_started:
                print("Conversation not started")
                break
        else:
            conversation_started = True
            cumulative_silence = 0

    print("Stopped the recording")
    stream.close()
    audio.terminate()

result = whisper_model.transcribe("output.wav", language="it")
print(result["text"])
# plot the confidences for the speech
plt.figure(figsize=(20, 6))
plt.plot(voiced_confidences)
# plt.show()  # activate to show the plot of the frames and relative confidence vad
