import gensim
import pandas as pd






### Set Hyperparameter ###
VECTOR_SIZE = 300
WINDOW = 10
MIN_COUNT = 5
PARALELL_SIZE = 12
LEARNING_RATE = 0.025
MIN_LEARNING_RATE = 0.025



# Load csv data
raw_data = pd.read_csv('./trip.csv', header = 0 )

# Preprocess the data

#TODO erase about some attraction name




# Set data to fit Gensim doc2vec
LabeledSentence = gensim.models.doc2vec.LabeledSentence
gensim.models.doc2ec.TaggedDocumnet()

# Define model
model = gensim.models.Doc2Vec(size = VECTOR_SIZE, window = WINDOW, min_count= MIN_COUNT,
                              workers=PARALELL_SIZE, alpha=LEARNING_RATE, min_alpha=MIN_LEARNING_RATE)

model.build_vocab(data)