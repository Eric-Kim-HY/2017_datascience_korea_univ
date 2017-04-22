import numpy as np

class util :
    def __init__(self):
        pass


    # compute sigmoid nonlinearity
    def sigmoid(x):
        output = 1/(1+np.exp(-x))
        return output
    
    # convert output of sigmoid function to its derivative
    def sig2deriv(output):
        return output*(1-output)

    
    # X = np.array(input vectors)             # vector들로 이루어진 input 정의 
    # y = np.array(output vector)             # 기대되는 결과
    
    np.random.seed(1)                         # 매 번 랜덤화 시키기
    
    # 평균값을 0으로 weight matrix 초기화



    
    
    
    
    
    
    
    
    
    
    