import time
import pandas as pd
from customTskin.middleware.models.randomForestModel import MovementClassifier
from customTskin import CustomTskin, Hand, OneFingerGesture, TwoFingerGesture
from tactigon_ironboy import IronBoy, IronBoyConfig, IronBoyCommand
import whisper


if __name__ == "__main__":

    i_config = IronBoyConfig(
        "Ironboy",
        "FE:E2:2B:47:5D:80"
    )

    # tskin = CustomTskin(....)
    with CustomTskin("C0:83:43:39:21:57", Hand.RIGHT,'rf') as tskin, IronBoy(i_config) as ironboy:
        
        while True:

            if not tskin.connected:
                print("Connecting...")
                time.sleep(0.1)
                continue

            if not ironboy.connected:
                print("Connecting ironboy")
                time.sleep(0.5)
                continue

            touch = tskin.touch            

            #if touch and touch.one_finger == OneFingerGesture.SINGLE_TAP:

             #   print("ascolto.....")
              #  tskin.select_audio()
               # time.sleep(1)
               # tskin.select_sensors()
               # print('Fine registrazione, ora capisco...')
               # action = tskin.wernicke(path=r'C:\Users\LucaGiovagnoli\OneDrive - ITS Angelo Rizzoli\Desktop\Materiali\UFS15\Esercizi\tactigon_pw\custom_tactigon\audio_data\test_data.wav',model = 'base')
               # print(action)
               # if action == 'Muoviti':
               #     ironboy.send_command(IronBoyCommand.KPOP)

                
            if touch and touch.two_finger == TwoFingerGesture.TWO_FINGER_TAP:
                print("registro.....")
                tskin.middleware.registro.set()
                while not tskin.middleware.can_predict.is_set():
                    continue
                tskin.middleware.registro.clear()
                azione = tskin.evaluate_move()
                print(azione)
                if azione == 'right':
                    ironboy.send_command(IronBoyCommand.KPOP)

            if touch and touch.one_finger == OneFingerGesture.TAP_AND_HOLD:
                break