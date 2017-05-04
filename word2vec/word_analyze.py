import numpy as np
import math
import pandas as pd




class word_analysis():
    def __init__(self):
        self.k = 10          # 상위 몇 개의 단어를 받을지 정수 입력
        self.W = pd.read_csv('./wordvector.csv', header= 0, index_col = 0)


    def getVector(self, word):
        # word에 해당하는 벡터를 행렬 'W' 에서 가져와 array로 리턴한다.
        if word in self.W.ix:
            return self.W.ix[word]
        else:
            print("해당 단어는 우리의 딕셔너리에 없습니다. 다른 단어를 입력해주세요.")

    def cosineSimilarity(self, v1, v2):  # returns the cosine similarity value of two input vectors v1 and v2
        v1 = np.array(v1)
        v2 = np.array(v2)
        multi = (v1.dot(v2)).sum()
        x = math.sqrt((v1 * v1).sum())
        y = math.sqrt((v2 * v2).sum())
        result = multi / (x * y)
        return result

    def nearestWord(self, v, k):
        # 기준 단어의 벡터(v)와 가장 가까운 상위 k개의 단어를 리턴한다.
        W = self.W
        W.applymap
        # TODO
        pass




    # <1> 유사한 관계의 단어 추출
    # word analogy  vector('woman') + [ vector('king') - vector('man') ] +   => vector('queen')
    def analogy(self):
        print("A : B의 관계는 C : D의 관계와 같다")
        vec_A = self.getVector(input("A의 자리에 넣을 단어를 입력하세요"))
        vec_B = self.getVector(input("B의 자리에 넣을 단어를 입력하세요"))
        vec_C = self.getVector(input("C의 자리에 넣을 단어를 입력하세요"))
        vec_D = vec_C + vec_B - vec_A
        words_nearD = self.nearestWord(vec_D, self.k)
        result = words_nearD
        print("D의 자리에 가장 가까운 {0}개의 단어들: {1}", self.k, result)


    # <2> 가장 유사한 N개의 단어 추출
    def similar(self):
        vec = self.getVector(input("A의 자리에 넣을 단어를 입력하세요"))
        result = self.nearestWord(vec, self.k)
        print("가장 가까운 {0}개의 단어들: {1}", self.k, result)


    # <3> 두 단어의 Similarity
    def nearest(self):
        w1 = input("첫 번째 단어를 입력하세요: ")
        w2 = input("두 번째 단어를 입력하세요: ")
        v1 = self.getVector(w1)
        v2 = self.getVector(w2)
        result = self.cosineSimilarity(v1, v2)
        print("{0}과 {1} 의 cosine similarity 값은 {2}입니다.", w1, w2, result)


while(1):
    print("<1> 유사한 관계의 단어 찾기\n"
          "<2> 입력한 단어와 가장 가까운 N개의 단어 찾기\n"
          "<3> 두 단어의 유사도 확인하기(-1과 1 사이의 값을 리턴)\n")
    menu = int(input("작업을 원하는 메뉴의 번호를 입력하세요: "))

    if menu == 1:
        word_analysis.analogy()
    elif menu == 2:
        word_analysis.similar()
    elif menu == 3:
        word_analysis.nearest()
    else:
        print("잘못된 값을 입력하였습니다. 유효한 메뉴 번호를 입력해주세요.")