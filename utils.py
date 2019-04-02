import numpy as np
import matplotlib.pyplot as plt
import PIL
import datetime



def make_image(sequence):
    seq = np.cumsum(sequence.transpose(), 1)
    seq = seq.transpose()
    seq[:,2:] = sequence[:,2:]
    split_loc = np.where(seq[:,3]>0)[0]+1
    strokes = np.split(seq, split_loc)
    fig = plt.figure()
    for s in strokes:
        plt.plot(s[:,0],-s[:,1])
    return fig

def make_image2(sequence):
    """plot drawing with separated strokes"""
    sequence = np.split(sequence, np.where(sequence[:,3]>0)[0]+1)[0]
    strokes = np.split(sequence, np.where(sequence[:,2]>0)[0]+1)
    fig = plt.figure()
    for s in strokes:
        plt.plot(s[:,0],-s[:,1])
    return fig

def make_image3(sequence):
    """plot drawing with separated strokes"""
    strokes = np.split(sequence, np.where(sequence[:,2]>0)[0]+1)
    fig = plt.figure()
    for s in strokes:
        plt.plot(s[:,0],-s[:,1])
    return fig


def TimestampMillisec64():
    return float((datetime.datetime.utcnow() - datetime.datetime(1970, 1, 1)).total_seconds() * 1000)

def SetSeed(seed=771986):
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    import random
    random.seed(seed)
    import numpy
    numpy.random.seed(seed)

class StopWatch():
    
    def __init__(self, name, perMin=False):
        self.name = name
        self.perMin = perMin
        self.reset()

    def lap(self):
        stop = TimestampMillisec64()
        dif = (stop - self.lap_start)
        rate = 1 / (-1 if dif == 0 else dif) * 1000 * (60 if self.perMin else 1)
        unit = ("per/min" if self.perMin else "per/sec")
        print("{0:>20}: {1:6.2f} {2}".format(self.name, rate,unit))
        self.lap_start = stop

    def stop(self):
        stop = TimestampMillisec64()
        dif = (stop - self.start)
        rate = 1 / (-1 if dif == 0 else dif) * 1000 * (60 if self.perMin else 1)
        unit = ("per/min" if self.perMin else "per/sec")
        print("{0:>20}: {1:6.2f} {2}".format(self.name, rate,unit))

    def reset(self):
        self.start = self.lap_start = TimestampMillisec64()