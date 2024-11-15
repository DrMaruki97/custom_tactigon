import time
import pandas as pd
from customTskin import CustomTskin, Hand, OneFingerGesture, TwoFingerGesture


if __name__ == "__main__":
    # tskin = CustomTskin(....)
    with CustomTskin("C0:83:3E:39:21:57", Hand.RIGHT) as tskin:
        dati = tskin.middleware.registratore
        tocchi = 0
        movimento = 1
        while True:
            if not tskin.connected:
                print("Connecting...")
                time.sleep(0.1)
                continue

            touch = tskin.touch

            if touch and touch.two_finger == TwoFingerGesture.TWO_FINGER_TAP:
                movimento += 1
                print('gesture modificata')

            if movimento == 2:
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
                break
 
            if touch and touch.one_finger == OneFingerGesture.SINGLE_TAP:
                print("registro.....")
                tskin.middleware.registro.set()
                time.sleep(2)
                tskin.middleware.registro.clear()
                print("ho finito")

            time.sleep(tskin.TICK)


