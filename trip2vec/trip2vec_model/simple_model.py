import gensim
import pandas as pd
from sklearn import utils
import re
### Set Hyperparameter ###
VECTOR_SIZE = 300
WINDOW = 10
MIN_COUNT = 5
PARALELL_SIZE = 8
LEARNING_RATE = 0.025
MIN_LEARNING_RATE = 0.025
ITERATIONS = 20
LOAD_MODEL = False
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
        ret.append(LabeledSentence(temp_corpus, trip_id))

    return ret


### Proceed the doc2vec

# Load csv data
raw_data = pd.read_csv('./trip.csv', header = 0 )

# Paris data 만 가져오기
raw_data = raw_data[raw_data['city'] =='Paris']

# Attraction 앞 about 제거
raw_data['attraction'] = raw_data['attraction'].apply(treat_attraction)

# Process Train Data
train_data = label_dataframe(raw_data)

# Define or Load model
if LOAD_MODEL :
    model = gensim.models.Doc2Vec.load(MODEL_NAME)
else :
    model = gensim.models.Doc2Vec(size = VECTOR_SIZE, window = WINDOW, min_count= MIN_COUNT,
                              workers=PARALELL_SIZE, alpha=LEARNING_RATE, min_alpha=MIN_LEARNING_RATE)

model.build_vocab(train_data)

for epoch in range(ITERATIONS):
    model.train(utils.shuffle(train_data),
                total_examples = len(train_data) ,
                epochs = model.iter)

model.save(MODEL_NAME)
print("Model saved")

