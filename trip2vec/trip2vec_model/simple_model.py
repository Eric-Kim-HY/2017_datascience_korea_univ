import gensim
import pandas as pd
from sklearn import utils
import re
### Set Hyperparameter ###
VECTOR_SIZE = 300
WINDOW = 10
MIN_COUNT = 5
PARALELL_SIZE = 12
LEARNING_RATE = 0.025
MIN_LEARNING_RATE = 0.025

ITERATIONS = 10

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
    if s.startswith('about ') :
        s = s[6:]
    elif s.startswith('A about '):
        s = s[8:]
    return s

# label the dataframe and get unique words
def label_dataframe(df):
    LabeledSentence = gensim.models.doc2vec.LabeledSentence
    ret1 = []
    ret2 = []
    for idx, row in df.iterrows():
        temp_corpus = cleanText(row['review_text'])
        trip_id = row['attraction']
        ret1.append(LabeledSentence(temp_corpus, trip_id))
        ret2 = list(set(ret2 + temp_corpus))

    return (ret1, ret2)


### Proceed the doc2vec

# Load csv data
raw_data = pd.read_csv('./trip.csv', header = 0 )

# Paris data 만 가져오기
raw_data = raw_data[raw_data['city'] =='Paris']

# Attraction 앞 about 제거
raw_data['attraction'] = raw_data['attraction'].apply(treat_attraction)

# corpus building, prepare data for doc2vec
train_data, all_corpus = label_dataframe(raw_data)


# Define model
model = gensim.models.Doc2Vec(size = VECTOR_SIZE, window = WINDOW, min_count= MIN_COUNT,
                              workers=PARALELL_SIZE, alpha=LEARNING_RATE, min_alpha=MIN_LEARNING_RATE)

model.build_vocab(all_corpus)

for epoch in range(ITERATIONS):
    model.train(utils.shuffle(train_data))

model.save('Model_after_train')