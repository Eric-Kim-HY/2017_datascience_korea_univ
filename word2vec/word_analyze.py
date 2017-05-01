import numpy as np
import math

def cosineSimilarity(v1, v2):       # returns the cosine similarity value of two input vectors v1 and v2
    v1 = np.array(v1)
    v2 = np.array(v2)
    multi = (v1.dot(v2)).sum()
    x = math.sqrt((v1 * v1).sum())
    y = math.sqrt((v2 * v2).sum())
    result = multi / (x * y)
    return result

class word_analysis():
    def __init__(self):
        pass

    # <1> 유사한 관계의 단어 추출
    # word analogy  vector('woman') + [ vector('king') - vector('man') ] +   => vector('queen')
    def analogy(self):
        pass


    # <2> 가장 유사한 N개의 단어 추출
    def sim_word(self):
        pass


    # <3> 두 단어의 Similarity
    def nearest(self):
        pass
