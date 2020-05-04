import numpy as np

def decision(p=0.5):
    val = np.random.choice(2, 1, p=[1-p, p])[0]
    return val == 1

