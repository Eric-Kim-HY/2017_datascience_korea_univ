import numpy as np
import math
from numba import jit
import pandas as pd
import sklearn.metrics
import sys


class word_analysis():
    @jit
    def __init__(self):
        self.k = 10          # 상위 몇 개의 단어를 받을지 정수 입력
        self.W = pd.read_csv('./wordvector.csv', header= 0, index_col = 0)  # weight parameter 불러오기
        self.W.dropna(how='any', inplace=True)                              # None 포함하는 행 제거
        self.COSIM = sklearn.metrics.pairwise.cosine_similarity             # Cosine similarity function

    @jit
    def getVector(self, word):
        # word에 해당하는 벡터를 행렬 'W' 에서 가져와 array로 리턴한다.
        if word in self.W.index.tolist():
            return self.W.ix[word].values
        else:
            print("{0} 단어는 우리의 딕셔너리에 없습니다. 다른 단어를 입력해주세요.".format(word))

    @jit
    def nearestWord(self, v, k):
        ## 기준 단어의 벡터(v)와 가장 가까운 상위 k개의 단어를 리턴한다. ##

        # cosine similarity 칼럼 생성을 위해 weight를 새로운 matrix로 복사
        W = self.W

        # similarity 연산을 위해 vector reshape
        v = v.reshape([1,W.shape[1]])

        # cosine similarity 칼럼 생성,
        W['sim'] = self.COSIM(W.as_matrix().reshape([W.shape[0],W.shape[1]]),v)

        # Similarity 높은 순으로 정렬
        W.sort_values(by='sim', axis=0, ascending=False, inplace=True, kind='quicksort', na_position='last')

        # 상위 k 개의 단어 list 형태로 저장
        ret = W.index[:k].tolist()

        return ret


    # <1> 유사한 관계의 단어 추출
    # word analogy  vector('woman') + [ vector('king') - vector('man') ] +   => vector('queen')
    @jit
    def analogy(self):

        print("A : B의 관계는 C : D의 관계와 같다")
        vec_A = self.getVector(input("A의 자리에 넣을 단어를 입력하세요 : "))
        vec_B = self.getVector(input("B의 자리에 넣을 단어를 입력하세요 : "))
        vec_C = self.getVector(input("C의 자리에 넣을 단어를 입력하세요 : "))
        vec_D = vec_B - vec_A + vec_C
        words_nearD = self.nearestWord(vec_D, self.k)

        print("D의 자리에 가장 가까운 {0}개의 단어들: {1}\n".format( self.k, words_nearD))


    # <2> 가장 유사한 N개의 단어 추출
    @jit
    def similar(self):
        word = input("탐색할 단어를 입력하세요 : ")
        vec = self.getVector(word)
        result = self.nearestWord(vec, self.k)
        print("가장 가까운 {0}개의 단어들: {1}\n".format( self.k, result))


    # <3> 두 단어의 Similarity
    @jit
    def nearest(self):
        w1 = input("첫 번째 단어를 입력하세요: ")
        w2 = input("두 번째 단어를 입력하세요: ")
        v1 = self.getVector(w1); v1 = v1.reshape([1,len(v1)])
        v2 = self.getVector(w2); v2 = v2.reshape([1,len(v2)])
        result = self.COSIM(v1, v2)
        print("{0}과 {1} 의 cosine similarity 값은 {2}입니다.\n".format( w1, w2, result))


    # <4> 프로그램 종료
    @jit
    def quit(self):
        sys.exit()

check = word_analysis()
while(1):
    print("<1> 유사한 관계의 단어 찾기\n"
          "<2> 입력한 단어와 가장 가까운 N개의 단어 찾기\n"
          "<3> 두 단어의 유사도 확인하기(-1과 1 사이의 값을 리턴)\n"
          "<4> 프로그램 종료\n")
    menu = int(input("작업을 원하는 메뉴의 번호를 입력하세요: "))

    if menu == 1:
        check.analogy()
    elif menu == 2:
        check.similar()
    elif menu == 3:
        check.nearest()
    elif menu == 4:
        check.quit()
    else:
        print("잘못된 값을 입력하였습니다. 유효한 메뉴 번호를 입력해주세요.")