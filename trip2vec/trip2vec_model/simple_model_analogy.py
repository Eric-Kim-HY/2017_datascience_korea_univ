import gensim
import pandas as pd
from sklearn import utils
import re

### Set Hyperparameter and settings ###
VECTOR_SIZE = 300
WINDOW = 10
MIN_COUNT = 5
PARALELL_SIZE = 8
LEARNING_RATE = 0.025
MIN_LEARNING_RATE = 0.025
ITERATIONS = 20
ATTRACTION_RE = re.compile('.*about.')
MODEL_NAME = 'TripAdvisor_doc2vec_model'


### Set Functions ###

# Preprocessing function
def cleanText(corpus) :
    punctuation = ".,?!:;(){}[]"
    for char in punctuation: corpus = corpus.replace(char,"")
    corpus = corpus.lower()
    corpus = corpus.split()
    return corpus

# Attraction preprocessing
def treat_attraction(s):
    m = ATTRACTION_RE.search(s)
    if m : s = re.sub(m.group(),'',s)
    return s

# label the dataframe and get unique words
def label_dataframe(df):
    LabeledSentence = gensim.models.doc2vec.LabeledSentence
    ret = []
    for idx, row in df.iterrows():
        temp_corpus = cleanText(row['review_text'])
        trip_id = row['attraction']
        ret1.append(LabeledSentence(temp_corpus, trip_id))

    return ret


### Proceed the doc2vec

# Load csv data
raw_data = pd.read_csv('./trip.csv', header = 0 )

# Paris data 만 가져오기
raw_data = raw_data[raw_data['city'] =='Paris']

# Attraction 앞 about 제거
raw_data['attraction'] = raw_data['attraction'].apply(treat_attraction)

# Load model
model = gensim.models.Doc2Vec.load(MODEL_NAME)

# Introduce attracion list to use trip site analogy
print(attraction_list)

# 유사한 여행지 구하기
model.docvecs.most_similar('Palais Royal')

# 여행지 간 유사도 구하기
model.docvecs.similarity('Palais Royal','St. Sulpice Fountain')
