import numpy as np
import math
from agent import W

def getVector(word):
    return W[word]

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
        w = input("단어를 입력하세요: ")
        
        pass


    # <3> 두 단어의 Similarity
    def nearest(self):
        w1 = input("첫 번째 단어를 입력하세요: ")
        w2 = input("두 번째 단어를 입력하세요: ")
        v1 = getVector(w1)
        v2 = getVector(w2)
        result = cosineSimilarity(v1, v2)
        print("{0}과 {1} 의 cosine similarity 값은 {2}입니다.", w1, w2, result)


while(1):
    print("<1> 유사한 관계의 단어 찾기\n"
          "<2> 입력한 단어와 가장 가까운 N개의 단어 찾기\n"
          "<3> 두 단어의 유사도 확인하기(-1과 1 사이의 값을 리턴)\n")
    menu = int(input("작업을 원하는 메뉴의 번호를 입력하세요: "))

    if menu == 1:
        word_analysis.analogy()
    elif menu == 2:
        word_analysis.sim_word()
    elif menu == 3:
        word_analysis.nearest()
    else:
        print("잘못된 값을 입력하였습니다. 유효한 메뉴 번호를 입력해주세요.")