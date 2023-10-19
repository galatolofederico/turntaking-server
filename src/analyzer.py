import torch
import numpy as np
from av import AudioResampler, AudioFifo
from enum import Enum

# Detection parameters
SILENCE_THRESHOLD = 0.5  # s
CONFIRMED_SILENCE_THRESHOLD = 2.0  # s
CONFIDENCE_THRESHOLD = 0.5  #
START_CONVERSATION_THRESHOLD = 10.0  # s
CONVERSATION_NOT_STARTED_THRESHOLD = 5.0  # s

RATE = 16000
FRAME_DURATION = 0.05  # s

class State(Enum):
    NOT_STARTED = 0
    STARTED = 1
    POTENTIAL_TURN_CHANGE = 2
    CONVERSATION_NOT_STARTED = 3

class Analyzer:
    
    def __init__(self, *, mqtt_client=None, mqtt_topic=None):
        self.state = State.NOT_STARTED
        self.cumulative_silence = 0.0
        self.resampler = AudioResampler(format='s16', layout='mono', rate=RATE)
        self.audio_fifo = AudioFifo()
        torch.set_num_threads(1)
        self.vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                      model='silero_vad',
                                      force_reload=False,
                                      trust_repo=True)
        self.mqtt_client = mqtt_client
        self.mqtt_topic = mqtt_topic

    def int2float(self, sound):
        abs_max = np.abs(sound).max()
        sound = sound.astype('float32')
        if abs_max > 0:
            sound *= 1 / 32768
            sound = sound.squeeze()  # depends on the use case
        return sound
    
    def analyze(self, frame):
        frame = self.resampler.resample(frame)[0]
        self.audio_fifo.write(frame)
        frame = self.audio_fifo.read(800, False)
        if frame is not None:
            # get the confidences
            tensor = torch.from_numpy(self.int2float(frame.to_ndarray()))
            new_confidence = self.vad_model(tensor, RATE).item()
            self.set_state(new_confidence)
    
    def signal(self, message):
        if self.mqtt_client is not None:
            print(f"Sending message: {message} on topic {self.mqtt_topic}")
            self.mqtt_client.publish(self.mqtt_topic, message)

    def set_state(self, speaking_probability):
        # check if the new frame has voice
            if speaking_probability <= CONFIDENCE_THRESHOLD:
                self.cumulative_silence += FRAME_DURATION
                if self.state == State.STARTED and self.cumulative_silence >= SILENCE_THRESHOLD:
                    self.state = State.POTENTIAL_TURN_CHANGE
                    print("Potential turn change")

                    self.signal("potential_stop_speaking")

                elif self.state == State.POTENTIAL_TURN_CHANGE and self.cumulative_silence >= CONFIRMED_SILENCE_THRESHOLD:
                    self.state = State.NOT_STARTED  # we need to go back to the NOT_STARTED state to initiate a new turn
                    print("Turn change confirmed")

                    self.signal("stop_speaking")

                # if for more than CONVERSATION_NOT_STARTED_THRESHOLD there is silence
                # the user may have not understood the response and we should repeat it
                elif self.state == State.NOT_STARTED and (self.cumulative_silence >= CONVERSATION_NOT_STARTED_THRESHOLD):
                    self.state = State.CONVERSATION_NOT_STARTED
                    print("Conversation not started")

                    self.signal("conversation_not_started")

                # if the user stay silent for more than START_CONVERSATION_THRESHOLD
                # the robot may want to start the conversation
                elif self.state == State.CONVERSATION_NOT_STARTED and self.cumulative_silence >= START_CONVERSATION_THRESHOLD:
                    self.cumulative_silence = 0
                    self.state = State.NOT_STARTED
                    print("Start conversation")

                    self.signal("start_conversation")

            else:
                self.cumulative_silence = 0
                if self.state == State.NOT_STARTED or self.state == State.CONVERSATION_NOT_STARTED:
                    self.state = State.STARTED
                    print("Conversation started")

                    self.signal("start_speaking")

                elif self.state == State.POTENTIAL_TURN_CHANGE:
                    # if it was detected a potential turn change we need to send a message to
                    # interrupt the pipeline
                    self.state = State.STARTED
                    print("Potential turn change aborted")

                    self.signal("potential_stop_speaking_aborted")


    


