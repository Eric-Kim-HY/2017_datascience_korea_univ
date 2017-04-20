import numpy as np

class util :
    def __init__(self):
        pass

    # cost functinon
    def nce(self, sth):
        return sth

    def backprop(self, sth):
        return sth

    def neg_sample(self, nsample):
        return
    
    ## update weight matrix
    
    alpha = 0.5
    hiddenSize = 32
    
    # compute sigmoid nonlinearity
    def sigmoid(x):
        output = 1/(1+np.exp(-x))
        return output
    
    # convert output of sigmoid function to its derivative
    def sig2deriv(output):
        return output*(1-output)
    
    
    
    
    
    
    
    
    
    
    
    
    
    