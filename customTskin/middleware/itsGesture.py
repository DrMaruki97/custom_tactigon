from email.mime import audio
from math import sqrt
import wave
from multiprocessing import Process, Event
from multiprocessing.connection import _ConnectionBase
import csv

class Tester():
    def __init__(self) -> None:
        self.data = {'ax':[],
                     'ay':[],
                     'az':[],
                     'reg':[]}
        self.regis = 1

    def new_data(self,accX,accY,accZ,reg):
        self.data['ax'].append(accX)
        self.data['ay'].append(accY)
        self.data['az'].append(accZ)
        self.data['reg'].append(reg)

    def scrivi(self):
        with open(r'C:\Users\LucaGiovagnoli\OneDrive - ITS Angelo Rizzoli\Desktop\Materiali\UFS15\Esercizi\tactigon_pw\testing.csv','a') as file:
            writer = csv.DictWriter(file,fieldnames=self.data.keys())
            writer.writeheader()
            for row in zip(*self.data.values()):
                writer.writerow(dict(zip(self.data.keys(),row)))
        



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
        
    def fine_reg(self):
        self.registratore.regis += 1


    def reg(self):
        if self.registro:
            self.registro = False
        else:
            self.registro = True


    def run(self):
        l0 = None
        velx,vely,velz=0,0,0
        
        #angolox0, angoloy0, angoloz0 = None
        while self.can_run.is_set():
            if self.sensor_rx.poll(timeout=0.02):
                accX, accY, accZ, gyroX, gyroY, gyroZ = self.sensor_rx.recv()
                if self.registro.is_set():
                    self.registratore.new_data(accX,accY,accZ,self.registratore.regis)
                    self.registratore.scrivi()
                l1 = self.is_moving(accX,accY,accZ)
                '''if l0:
                    moving = abs(l1 - l0)
                    if moving > 0.5:
                        movement = 'si muove'
                    else:
                        movement = 'è fermo'
                    l0 = l1
                    rotating = self.is_rotating(gyroX,gyroY,gyroZ)
                    print(f'tactigon {movement} e {rotating}')
                else:
                    l0 = l1               
                deltaT=0.02
                if accX>0.1 or accX<0.1:
                    velx=velx+accX*deltaT
                if accY>0.1 or accY<0.1:
                    vely=vely+accY*deltaT
                if accZ>0.01 or accZ<-0.01:
                    velz=velz+accZ*deltaT
                print(velx,vely,velz) '''                  
                       
                    
                
                
                
            if self.audio_rx.poll():
                with wave.open("test.wav", "wb") as audio_file:
                    audio_file.setnchannels(1)
                    audio_file.setframerate(16000)
                    audio_file.setsampwidth(2)

                    while self.audio_rx.poll(1):
                        audio_bytes = self.audio_rx.recv_bytes()
                        audio_file.writeframes(audio_bytes)
            
            '''if angolox0 & angoloy0 & angoloz0:
                deltax = angolox0 - gyroX
                if deltax < 1:
                    print('non gira')
                    else:
                        print()'''



