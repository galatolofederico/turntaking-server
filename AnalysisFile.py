SAMPLING_RATE = 16000

import torch

torch.set_num_threads(1)

model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=False,
                              trust_repo=True)

(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils

wav = read_audio('provaSeria.wav', sampling_rate=SAMPLING_RATE)
speech_probs = []
window_size_samples = 800
for i in range(0, len(wav), window_size_samples):
    chunk = wav[i: i + window_size_samples]
    if len(chunk) < window_size_samples:
        break
    speech_prob = model(chunk, SAMPLING_RATE).item()
    speech_probs.append(speech_prob)
print(speech_probs)
