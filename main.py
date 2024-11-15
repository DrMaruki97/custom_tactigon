import time
import pandas as pd
from customTskin import CustomTskin, Hand, OneFingerGesture


if __name__ == "__main__":
    # tskin = CustomTskin(....)
    with CustomTskin("C0:83:35:39:21:57", Hand.RIGHT) as tskin:
        dati = tskin.middleware.registratore
        tocchi = 0
        while True:
            if not tskin.connected:
                print("Connecting...")
                time.sleep(0.1)
                continue

            touch = tskin.touch
 
            if touch and touch.one_finger == OneFingerGesture.SINGLE_TAP:
                print("registro.....")
                tskin.middleware.reg()
                time.sleep(3)
                tskin.middleware.reg()
                tskin.middleware.fine_reg()
                print("ho finito")
                tocchi += 1

            time.sleep(tskin.TICK)

            if tocchi == 5:
                tskin.join()
                break

        dati.scrivi()


