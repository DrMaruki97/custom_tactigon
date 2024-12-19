import time
import pandas as pd
from customTskin.middleware.models.randomForestModel import MovementClassifier
from customTskin import CustomTskin, Hand, OneFingerGesture, TwoFingerGesture
import whisper


if __name__ == "__main__":
    # tskin = CustomTskin(....)
    with CustomTskin("C0:83:43:39:21:57", Hand.RIGHT,'rf') as tskin:
        dati = tskin.middleware.registratore
        tocchi = 0
        movimento = 1
        while True:
            if not tskin.connected:
                print("Connecting...")
                time.sleep(0.1)
                continue

            touch = tskin.touch

            if touch and touch.one_finger == OneFingerGesture.TAP_AND_HOLD:
                print('Spegnimento...')
                tskin.join()
                break

            '''if movimento == 2:
                tskin.middleware.down.set()

            elif movimento == 3 and tskin.middleware.down.is_set():
                tskin.middleware.down.clear()
                tskin.middleware.left.set()

            elif movimento == 4 and tskin.middleware.left.is_set():
                tskin.middleware.left.clear()
                tskin.middleware.right.set()

            elif movimento == 5 and tskin.middleware.right.is_set():
                tskin.middleware.right.clear()
                tskin.middleware.forward.set()

            elif movimento == 6 and tskin.middleware.forward.is_set():
                tskin.middleware.forward.clear()
                tskin.middleware.backward.set()

            if movimento == 7:
                tskin.join()
                break'''
 
            

            if touch and touch.one_finger == OneFingerGesture.SINGLE_TAP:
                print("ascolto.....")
                tskin.select_audio()
                time.sleep(2)
                tskin.select_sensors()
                print('Fine registrazione, ora capisco...')
                action = tskin.broca(path=r'C:\Users\LucaGiovagnoli\OneDrive - ITS Angelo Rizzoli\Desktop\Materiali\UFS15\Esercizi\tactigon_pw\custom_tactigon\audio_data\test_data.wav',model = 'base')
                print(action)

                
            if touch and touch.two_finger == TwoFingerGesture.TWO_FINGER_TAP:
                print("registro.....")
                tskin.middleware.registro.set()
                time.sleep(0.5)
                tskin.middleware.registro.clear()
                print(tskin.evaluate_move())
                
                print("ho finito")

            time.sleep(tskin.TICK)