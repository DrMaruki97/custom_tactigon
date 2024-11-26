import time
from .utils import most_common
from multiprocessing import Pipe
from tactigon_gear import Ble, TSkinConfig, Hand, OneFingerGesture, TwoFingerGesture
from typing import Optional
from .middleware import ITSGesture

class CustomTskin(Ble):
    middleware: ITSGesture

    def __init__(self, address: str, hand: Hand):
        Ble.__init__(self, address, hand)

        sensor_rx, self._sensor_tx = Pipe(duplex=False)
        audio_rx, self._audio_tx = Pipe(duplex=False)
        self._action_pipe, action_pipe = Pipe(duplex=False)

        self.middleware = ITSGesture(sensor_rx, audio_rx,action_pipe)
        self.middleware.can_run.set()

    def start(self):
        self.middleware.start()
        Ble.start(self)

    def join(self, timeout: Optional[float] = None):
        Ble.join(self, timeout)
        self.middleware.can_run.clear()
        self.middleware.join(timeout)

    def evaluate_move(self):
        action = []
        while self._action_pipe.poll(timeout=0.02):
            action.append(self._action_pipe.recv())
        return most_common(action)
        