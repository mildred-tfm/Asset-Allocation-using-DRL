import numpy as np

class EqWeights:
    def __init__(self):
        self.a_dim=0

    def predict(self,s,a):
        weights=np.ones(len(a[0]))/len(a[0])
        weights=weights[None,:]
        return weights