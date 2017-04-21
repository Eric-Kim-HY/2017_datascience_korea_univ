import numpy as np

class util :
    
    ## update weight matrix
    
    alpha = 0.5                # 적당한 값을 정한다
    hiddenSize = 32
    
    '''
    # compute sigmoid nonlinearity
    def sigmoid(x):
        output = 1/(1+np.exp(-x))
        return output
    
    # convert output of sigmoid function to its derivative
    def sig2deriv(output):
        return output*(1-output)
    '''
    
    # compute reLU
    def relu(x):
        return max(0,x)
    
    # X = np.array(input vectors)             # vector들로 이루어진 input 정의 
    # y = np.array(output vector)             # 기대되는 결과
    
    np.random.seed(1)                         # 매 번 랜덤화 시키기
    
    # 평균값을 0으로 weight matrix 초기화
    # col_len_X: input데이터의 길이
    W_0 = 2*np.random.random((col_len_X, hiddenSize)) - 1            # 첫번째 weight matrix의 크기 == (X의 열) X (hiddenSize)
    W_1 = 2*np.random.random((hiddenSize,1)) - 1                     # 두번째 weight matrix의 크기 == (hiddenSize) X (1)
    
    train = 60000           # 몇 번 train 할지 정한다
    for j in range(train):
        
        # forward propagation
        layer_0 = X                                    # 0번째 layer == input matrix
        layer_1 = relu(np.dot(layer_0,W_0))            # 1번째 layer: layer0에 W_0 행렬곱
        layer_2 = relu(np.dot(later_1,W_1))            # 2번째 layer: layer1에 W_1 행렬곱
        
        # calculate error
        layer_2_error = layer_2 - y                    # error는 원래값(y)과의 차이
        
        if (j%10000) == 0:
            print(str(j)+"번 학습 후 error:"+str(np.mean(np.abs(layer_2_error))))     # 10000번 학습할때마다 layer2 error 가 얼만큼 변하는지 보기
        
#       layer_2_delta = layer_2_error*sig2deriv(layer_2)         # reLU를 사용하면 오른쪽에 곱해지는 값이 항상 1
        
#       layer_1_error = layer_2_delta.dot(W_1.T)
        layer_1_error = layer_2_error.dot(W_1.T)                 # reLU의 미분값은 양의 값에서 항상 1 이므로 'delta = error'
        
#       layer_1_delta = layer_1_error*sig2deriv(layer_1)
        
        # update weight matrix
        W_1 -= alpha*(layer_1.T.dot(layer_2_error))
        W_0 -= alpha*(layer_0.T.dot(layer_1_error))
    
    
    
    
    
    
    
    
    
    
    