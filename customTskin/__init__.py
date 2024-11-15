import time

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

        self.middleware = ITSGesture(sensor_rx, audio_rx)
        self.middleware.can_run.set()

    def start(self):
        self.middleware.start()
        Ble.start(self)

    def join(self, timeout: Optional[float] = None):
        Ble.join(self, timeout)
        self.middleware.can_run.clear()
        self.middleware.join(timeout)