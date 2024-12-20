import time
from joblib import load
from multiprocessing import Pipe
from tactigon_gear import Ble, TSkinConfig, Hand, OneFingerGesture, TwoFingerGesture
from typing import Optional
from .middleware import ITSGesture
import whisper
import warnings

warnings.filterwarnings("ignore")

class CustomTskin(Ble):
    middleware: ITSGesture

    def __init__(self, address: str, hand: Hand, modelType):
        Ble.__init__(self, address, hand)

        sensor_rx, self._sensor_tx = Pipe(duplex=False)
        audio_rx, self._audio_tx = Pipe(duplex=False)
        self._action_pipe, action_pipe = Pipe(duplex=False)

        model = self.get_model(modelType)

        self.middleware = ITSGesture(sensor_rx, audio_rx,action_pipe,model)
        self.middleware.can_run.set()

    def get_model(self,modelType):
    
        model = load(f'tactigon_pw\custom_tactigon\customTskin\middleware\models\{modelType}_model.joblib')
        return model

    def start(self):
        self.middleware.start()
        Ble.start(self)

    def join(self, timeout: Optional[float] = None):
        Ble.join(self, timeout)
        self.middleware.can_run.clear()
        self.middleware.join(timeout)

    def evaluate_move(self):
        while True:
            action = self._action_pipe.recv()
            if action:
                self.middleware.predicted.clear()
                return action[0]
    
    def wernicke(self,path,model):
        modello = whisper.load_model(model)
        risultato = modello.transcribe(path,language='italian')
        testo = risultato['text']
        return testo

        