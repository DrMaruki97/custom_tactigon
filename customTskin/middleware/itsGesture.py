from email.mime import audio
from math import sqrt
import wave
from multiprocessing import Process, Event
from multiprocessing.connection import _ConnectionBase
import pandas as pd

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
        df.to_csv(f'C:\\Users\\LucaGiovagnoli\\OneDrive - ITS Angelo Rizzoli\\Desktop\\Materiali\\UFS15\\Esercizi\\tactigon_pw\\testing.csv')
        

        



class ITSGesture(Process):
    sensor_rx: _ConnectionBase
    audio_rx: _ConnectionBase

    def __init__(self, sensor_rx, audio_rx):
        Process.__init__(self)

        self.sensor_rx = sensor_rx
        self.audio_rx = audio_rx

        self.can_run = Event()
        self.registratore = Tester()
        self.registro = Event()
        self.down = Event()
        self.right = Event()
        self.left = Event()
        self.forward = Event()
        self.backward = Event()
        
        
    
    
        
    def is_moving(self, accX, accY, accZ):
        vector = accX**2 + accY**2 + accZ**2
        l =  sqrt(vector)
        return l
    
    def is_rotating(self,gyroX,gyroY,gyroZ):
        gyrox_base = 0.1
        gyroy_base = 0.0
        gyroz_base = -0.0



        if gyroX != gyrox_base or gyroY != gyroy_base or gyroZ != gyroz_base:
            return 'gira'
        else:
            return 'non gira'

    def run(self):        
        
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

                    self.registratore.new_data(accX,accY,accZ,gyroX,gyroY,gyroZ,self.registratore.movimento)          
                
                
            if self.audio_rx.poll():
                with wave.open("test.wav", "wb") as audio_file:
                    audio_file.setnchannels(1)
                    audio_file.setframerate(16000)
                    audio_file.setsampwidth(2)

                    while self.audio_rx.poll(1):
                        audio_bytes = self.audio_rx.recv_bytes()
                        audio_file.writeframes(audio_bytes)

        self.registratore.scrivi()



