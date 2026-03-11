from collections import deque

class PID():
    def __init__(self, KP, KI, KD):
        self.KP = KP
        self.KI = KI
        self.KD = KD
        self.sum_err = 0
        self.pre_err = 0

    def output_cal(self, err):
        self.sum_err += err
        delta_err = err - self.pre_err
        out_put = self.KP * err + self.sum_err * self.KI + self.KD * delta_err
        return out_put

    def clear_err(self):
        self.sum_err = 0
        self.pre_err = 0



