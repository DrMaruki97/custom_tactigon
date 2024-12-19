import pandas as pd
import numpy as np
import wave
from joblib import load
from email.mime import audio
from .models.randomForestModel import MovementClassifier
from math import sqrt
from multiprocessing import Process, Event
from multiprocessing.connection import _ConnectionBase
import warnings
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action='ignore', category=DataConversionWarning)


class Tester():
    def __init__(self) -> None:
        self.x = []
        self.y = []
        self.z = []
        self.gx = []
        self.gy = []
        self.gz = []
        self.label = []
        self.movimento = 'up'

    def new_data(self,accX,accY,accZ,gyroX,gyroY,gyroZ,movimento):
        self.x.append(accX)
        self.y.append(accY)
        self.z.append(accZ)
        self.gx.append(gyroX)
        self.gy.append(gyroY)
        self.gz.append(gyroZ)
        self.label.append(movimento)

    def scrivi(self):
        df = pd.DataFrame({'ax':self.x,'ay':self.y,'az':self.z,'gx':self.gx,'gy':self.gy,'gz':self.gz,'label':self.label})
        df.to_csv(r'C:\Users\LucaGiovagnoli\OneDrive - ITS Angelo Rizzoli\Desktop\Materiali\UFS15\Esercizi\tactigon_pw\custom_tactigon\movement_data\raw_data.csv')
        

        



class ITSGesture(Process):
    
    sensor_rx: _ConnectionBase
    audio_rx: _ConnectionBase
    action_pipe: _ConnectionBase
    model : MovementClassifier

    def __init__(self, sensor_rx, audio_rx, action_pipe,model):
        Process.__init__(self)

        self.sensor_rx = sensor_rx
        self.audio_rx = audio_rx
        self.model = model
        self.can_run = Event()
        self.registratore = Tester()
        self.registro = Event()
        self.scrivo = Event()
        self.down = Event()
        self.right = Event()
        self.left = Event()
        self.forward = Event()
        self.backward = Event()
        self.predicted = Event()
        self.action_pipe = action_pipe
        self.buffer = []
        self.predict_data = []
        self.can_predict = Event()
    
    def acc_vector(self, accX, accY, accZ):
        vector = accX**2 + accY**2 + accZ**2
        module =  sqrt(vector)
        return module
    
    def is_rotating(self,gyroX,gyroY,gyroZ):
        gyrox_base = 0.1
        gyroy_base = 0.0
        gyroz_base = -0.0

        if gyroX != gyrox_base or gyroY != gyroy_base or gyroZ != gyroz_base:
            return 'gira'
        else:
            return 'non gira'

    def guess_movement(self):
        data = pd.DataFrame(self.buffer)
        data = data.transpose()
        movement_type = self.model.predict(data)
        return movement_type
    
    def get_gesture(self,accX,accY,accZ,gyroX,gyroY,gyroZ):
        if len(self.buffer) <= 594:

            self.buffer.extend([accX,accY,accZ,gyroX,gyroY,gyroZ])

        else:
            self.can_predict.set()

    def run(self): 

        accX = []
        accY = []
        accZ = []
        gyroX = []
        gyroY = [] 
        gyroZ = []
        
        while self.can_run.is_set():

            if self.sensor_rx.poll(timeout=0.02):

                accX, accY, accZ, gyroX, gyroY, gyroZ = self.sensor_rx.recv()

                if self.down.is_set():
                    self.registratore.movimento = 'down'

                if self.left.is_set():
                    self.registratore.movimento = 'left'

                if self.right.is_set():
                    self.registratore.movimento = 'right'

                if self.forward.is_set():
                    self.registratore.movimento = 'for'

                if self.backward.is_set():
                    self.registratore.movimento = 'back'


                if self.registro.is_set():                    

                    #self.registratore.new_data(accX,accY,accZ,gyroX,gyroY,gyroZ,self.registratore.movimento)
                    
                    self.get_gesture(accX,accY,accZ,gyroX,gyroY,gyroZ)

                    if self.can_predict.is_set():

                        action = self.guess_movement()
                        self.predicted.set()
                        
                        while self.predicted.is_set():
                            self.action_pipe.send(action)
                        self.can_predict.clear()
                        self.predict_data.clear()

                else:
                    if self.buffer:
                        self.buffer.clear()

                '''elif self.scrivo.is_set():
                    self.registratore.scrivi()
                    self.scrivo.clear()'''
                
                
            if self.audio_rx.poll():
                with wave.open(r'C:\Users\LucaGiovagnoli\OneDrive - ITS Angelo Rizzoli\Desktop\Materiali\UFS15\Esercizi\tactigon_pw\custom_tactigon\audio_data\test_data.wav', "wb") as audio_file:
                    audio_file.setnchannels(1)
                    audio_file.setframerate(16000)
                    audio_file.setsampwidth(2)

                    while self.audio_rx.poll(1):
                        audio_bytes = self.audio_rx.recv_bytes()
                        audio_file.writeframes(audio_bytes)

        #self.registratore.scrivi()