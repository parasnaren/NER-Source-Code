from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split as tts
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from collections import Counter
import re
import zipfile
import warnings
warnings.filterwarnings("ignore")

import time
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from xgboost import XGBClassifier
import textblob, string
import en_core_web_md
from keras.preprocessing import text, sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout, Embedding, LSTM, SpatialDropout1D, Flatten
from keras.layers import GRU, Bidirectional, Convolution1D, GlobalMaxPool1D, TimeDistributed
from keras.callbacks import EarlyStopping

zf = zipfile.ZipFile('../../data/temp-files/dummy.zip') 
df = pd.read_csv('../../data/temp-files/dummy.csv')
#df = pd.read_csv('../../data/temp-files/dummy.csv')


def clean_text(text):
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    if len(text.split()) < 4:
        return np.nan
    return text

def make_table(filename, c1, x_axis, c2, y_axis='Accuracy'):
    filename = './docs/' + filename + '.csv'
    table = pd.DataFrame({
            x_axis: c1,
            y_axis: c2
            })
    table.to_csv(filename, index=False)

df['headline'] = df['headline'].apply(lambda text: clean_text(text))
df = df.dropna()

X = df['headline']
y = df.iloc[:, 2:]

train_x, valid_x, train_y, valid_y = tts(X, y, test_size=0.2, random_state=0)

encoder = LabelEncoder()
train_y_onehot = train_y.copy().as_matrix()
train_y = encoder.fit_transform(train_y.idxmax(axis=1))
valid_y_onehot = valid_y.copy().as_matrix()
valid_y = encoder.fit_transform(valid_y.idxmax(axis=1))

"""
Analysis of words and n-grams for different categories
"""
def get_ngram_counts(df, X, y, ngrams=1):
    vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}',
                                 stop_words=stop_words, ngram_range=(ngrams, ngrams))
    categories = list(y.columns)    
    for category in categories:
        filename = ('category/'+category+'_{}_grams').format(ngrams)
        print(category + '...')
        cat_df = df.where(df[category] == 1).dropna()
        cat_headline = vect.fit_transform(cat_df['headline'])
        sum_words = cat_headline.sum(axis=0)
        words_freq = [[word, sum_words[0, idx]] for word, idx in vect.vocabulary_.items()]
        words_freq = pd.DataFrame(sorted(words_freq, key = lambda x: x[1], reverse=True))
        make_table(filename, words_freq[0], ('{}_grams').format(ngrams),
                   words_freq[1], 'Frequency')

get_ngram_counts(df, X, y, ngrams=3)

max_sent_len = 0
for i in range(len(X)):
    max_sent_len = max(len(X.at[i].split()), max_sent_len)

print("Length of longest sentence: ", max_sent_len)
# Longest sentence is 26 characters


"""
Count Vectorizer
"""

count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}',
                             stop_words=stop_words, ngram_range=(1,2))
count_vect.fit(X)
xtrain_count =  count_vect.transform(train_x)
xvalid_count =  count_vect.transform(valid_x)

"""
TF-IDF VEctorizer
"""

# Word level
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}',
                             ngram_range=(1,2))
tfidf_vect.fit(X)
xtrain_tfidf = tfidf_vect.transform(train_x)
xvalid_tfidf = tfidf_vect.transform(valid_x)

# n-grams
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', 
                                   ngram_range=(1,2), max_features=5000)
tfidf_vect_ngram.fit(df['headline'])
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)

# Character level
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}',
                                         ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram_chars.fit(df['headline'])
xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x) 
xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x)

"""
Using Embeddings

(currently using spaCy embeddings, will add custom embeddings later)
"""

nlp = en_core_web_md.load()
token = text.Tokenizer()
token.fit_on_texts(X)
word_index = token.word_index

train_seq_x = sequence.pad_sequences(token.texts_to_sequences(train_x), maxlen=70)
valid_seq_x = sequence.pad_sequences(token.texts_to_sequences(valid_x), maxlen=70)

embedding_matrix = np.zeros((len(word_index) + 1, 300))
for debug, word, i in enumerate(word_index.items()):
    embedding_matrix[i] = nlp(word).vector
    print(debug)

#np.save('spacy-embedding-matrix', embedding_matrix)
embedding_matrix = np.load('spacy-embedding-matrix.npy')

        

"""
LDA topic modelling (takes too long, try on cloud)
"""

lda_model = LatentDirichletAllocation(n_components=20, 
                                      learning_method='online', max_iter=20)
X_topics = lda_model.fit_transform(xtrain_count)
topic_word = lda_model.components_ 
vocab = count_vect.get_feature_names()

# view the topic models
n_top_words = 10
topic_summaries = []
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    topic_summaries.append(' '.join(topic_words))



"""
Evaluation of supervised models
"""

def train_model(classifier, train, label, valid, is_neural_net=False):
    start = time.time()
    
    if is_neural_net:
        classifier.fit(train, label, epochs=1, batch_size=256)
        predictions = classifier.predict(valid)
        print("Time taken: %.2f seconds", time.time()-start)
        return predictions, accuracy_score(predictions, valid_y_onehot)
    else:
        classifier.fit(train, label)
        predictions = classifier.predict(valid)    
        print("Time taken: %.2f seconds", time.time()-start)
        return predictions, accuracy_score(predictions, valid_y)


# Logistic Regression
lr_accuracy = []
types = ['count-vectors', 'tfidf-vectors', 'n-gram-tfidf', 'character-level-vectors']
pred, accuracy = train_model(LogisticRegression(), xtrain_count, train_y, xvalid_count)
print("LR, Count Vectors: ", accuracy)
lr_accuracy.append(accuracy)
pred, accuracy = train_model(LogisticRegression(), xtrain_tfidf, train_y, xvalid_tfidf)
print("LR, WordLevel TF-IDF: ", accuracy)
lr_accuracy.append(accuracy)
pred, accuracy = train_model(LogisticRegression(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print("LR, N-Gram Vectors: ", accuracy)
lr_accuracy.append(accuracy)
pred, accuracy = train_model(LogisticRegression(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
print("LR, CharLevel Vectors: ", accuracy)
lr_accuracy.append(accuracy)
make_table('logistic-regression-accuracy.csv', types, 'Feature-types', lr_accuracy)


# Naive Bayes
nb_accuracy = []
types = ['count-vectors', 'tfidf-vectors', 'n-gram-tfidf', 'character-level-vectors']
accuracy = train_model(MultinomialNB(), xtrain_count, train_y, xvalid_count)
print("Naive Bayes, Count Vectors: ", accuracy)
nb_accuracy.append(accuracy)
accuracy = train_model(MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf)
print("Naive Bayes, WordLevel TF-IDF: ", accuracy)
nb_accuracy.append(accuracy)
accuracy = train_model(MultinomialNB(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print("Naive Bayes, N-Gram Vectors: ", accuracy)
nb_accuracy.append(accuracy)
accuracy = train_model(MultinomialNB(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
print("Naive Bayes, CharLevel Vectors: ", accuracy)
nb_accuracy.append(accuracy)
make_table('naive-bayes-accuracy.csv', types, 'Feature-types', nb_accuracy)


# Random Forests
accuracy = train_model(RandomForestClassifier(), xtrain_count, train_y, xvalid_count)
print("RandomForest, Count Vectors: ", accuracy)
accuracy = train_model(RandomForestClassifier(), xtrain_tfidf, train_y, xvalid_tfidf)
print("RandomForest, WordLevel TF-IDF: ", accuracy)


# Xtreme Gradient Boosting
accuracy = train_model(XGBClassifier(), xtrain_count.tocsc(), train_y, xvalid_count.tocsc())
print("Xgb, Count Vectors: ", accuracy)
accuracy = train_model(XGBClassifier(), xtrain_tfidf.tocsc(), train_y, xvalid_tfidf.tocsc())
print("Xgb, WordLevel TF-IDF: ", accuracy)
accuracy = train_model(XGBClassifier(), xtrain_tfidf_ngram_chars.tocsc(), train_y, xvalid_tfidf_ngram_chars.tocsc())
print("Xgb, CharLevel Vectors: ", accuracy)


"""
Pipeline to verify accuracies on individual categories
"""
NB_pipeline = Pipeline([    
        ('classifier', OneVsRestClassifier(MultinomialNB(
                fit_prior=True, class_prior=None))),
        ])

categories = list(df.columns)[2:]
accuracies = []

# CountVectorizer is working slightly better than Tfidf
for category in categories:
    print('For category: ', category)
    NB_pipeline.fit(xtrain_count, train_y_onehot[category])    
    prediction = NB_pipeline.predict(xvalid_count)
    #NB_pipeline.fit(xtrain_tfidf, train_y_onehot[category])
    #prediction = NB_pipeline.predict(xvalid_tfidf)
    accuracies.append(accuracy_score(valid_y_onehot[category], prediction))
    print('Accuracy = %.2f', accuracy_score(valid_y_onehot[category], prediction))

make_table('OneVsRest-accuracy', categories, 'Category', accuracies)


"""
Neural network models
"""

# ANNs
def create_ann(input_size):
    model = Sequential()
    model.add(Dense(100, input_dim=input_size, activation='relu'))
    model.add(Dense(15, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

    """input_layer = Input((input_size, ), sparse=True)
    
    hidden_layer1 = Dense(100, activation="relu")(input_layer)
    hidden_layer2 = Dense(20, activation="relu")(hidden_layer1)
    hidden_layer3 = Dense(4, activation="relu")(hidden_layer2)
    
    output_layer = Dense(1, activation="sigmoid")(hidden_layer3)

    classifier = Model(inputs = input_layer, outputs = output_layer)
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier"""


classifier = create_ann(xtrain_count.shape[1])
ann_pred, accuracy = train_model(classifier, xtrain_count, train_y_onehot, xvalid_count, is_neural_net=True)
print("ANN, Count Vector", accuracy)


# CNNs
def create_cnn():
    input_layer = Input((70, ))

    embedding_layer = Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(input_layer)
    embedding_layer = SpatialDropout1D(0.3)(embedding_layer)
    
    conv_layer = Convolution1D(100, 3, activation="relu")(embedding_layer)

    pooling_layer = GlobalMaxPool1D()(conv_layer)
    
    output_layer1 = Dense(50, activation="relu")(pooling_layer)
    output_layer1 = Dropout(0.25)(output_layer1)
    output_layer2 = Dense(1, activation="sigmoid")(output_layer1)

    model = Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


# RNNs (LSTM and GRU)
def create_rnn_lstm(gru=False):
    input_layer = Input((70, ))

    embedding_layer = Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], trainable=False)(input_layer)
    embedding_layer = SpatialDropout1D(0.3)(embedding_layer)

    if gru:
        lstm_layer = GRU(100)(embedding_layer)
    else:
        lstm_layer = LSTM(100)(embedding_layer)

    output_layer1 = Dense(50, activation="relu")(lstm_layer)
    output_layer1 = Dropout(0.25)(output_layer1)
    output_layer2 = Dense(1, activation="sigmoid")(output_layer1)

    model = Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    
    return model


# Bi-directional RNNs
def create_bidirectional_rnn(train_seq_x, train_y_onehot):     
    input_layer = Input((70, ))

    embedding_layer = Embedding(len(word_index)+1, 300, weights=[embedding_matrix], trainable=False)(input_layer)
    embedding_layer = SpatialDropout1D(0.3)(embedding_layer)

    lstm_layer = Bidirectional(GRU(100))(embedding_layer)

    output_layer1 = Dense(50, activation="relu")(lstm_layer)
    output_layer1 = Dropout(0.25)(output_layer1)
    output_layer2 = Dense(1, activation="sigmoid")(output_layer1)

    model = Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    
    history = model.fit(train_seq_x, train_y_onehot, epochs=1, batch_size=64, validation_split=0.1)
    """, 
          callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
    plt.title('Accuracy')
    plt.plot(history.history['acc'], label='train')
    plt.plot(history.history['val_acc'], label='test')
    plt.legend()
    plt.show();"""

# LSTM
def run_lstm(train_seq_x, train_y_onehot):
    model = Sequential()
    model.add(Embedding(len(word_index)+1, 300, weights=[embedding_matrix],
                        trainable=False, input_length=train_seq_x.shape[1]))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(15, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(train_seq_x, train_y_onehot, epochs=1, batch_size=64, validation_split=0.1, 
          callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
    plt.title('Accuracy')
    plt.plot(history.history['acc'], label='train')
    plt.plot(history.history['val_acc'], label='test')
    plt.legend()
    plt.show();
    
# Bi-LSTM    
def run_bi_lstm(train_seq_x, train_y):
    model = Sequential()
    model.add(Embedding(len(word_index)+1, 300, weights=[embedding_matrix],
                        trainable=False, input_length=train_seq_x.shape[1]))
    model.add(SpatialDropout1D(0.2))
    model.add(Bidirectional(
            LSTM(100, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),merge_mode='concat'))
    #model.add(TimeDistributed(Dense(100, activation='relu')))
    #model.add(Flatten())
    #model.add(Dense(100, activation='relu'))
    model.add(Dense(15, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print(model.summary())
    history = model.fit(train_seq_x, train_y, epochs=1, batch_size=64, validation_split=0.1, 
          callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
    plt.title('Accuracy')
    plt.plot(history.history['acc'], label='train')
    plt.plot(history.history['val_acc'], label='test')
    plt.legend()
    plt.show();
    
    



classifier = create_ann(xtrain_count.shape[1])
ann_pred, accuracy = train_model(classifier, xtrain_count, train_y_onehot, xvalid_count, is_neural_net=True)
print("ANN, Count Vector", accuracy)

classifier = create_cnn()
cnn_pred, accuracy = train_model(classifier, train_seq_x, train_y, valid_seq_x, is_neural_net=True)
print("CNN, Word Embeddings", accuracy)

classifier = create_rnn_lstm()
rnn_lstm_pred, accuracy = train_model(classifier, train_seq_x, train_y, valid_seq_x, is_neural_net=True)
print("RNN-LSTM, Word Embeddings", accuracy)

classifier = create_rnn_lstm(gru=True)
rnn_gru_pred, accuracy = train_model(classifier, train_seq_x, train_y, valid_seq_x, is_neural_net=True)
print("RNN-GRU, Word Embeddings", accuracy)

classifier = create_bidirectional_rnn()
bilstm_pred, accuracy = train_model(classifier, train_seq_x, train_y, valid_seq_x, is_neural_net=True)
print("RNN-Bidirectional, Word Embeddings", accuracy)


run_lstm(train_seq_x, train_y_onehot)
run_bi_lstm(train_seq_x, train_y)


create_bidirectional_rnn(train_seq_x, train_y_onehot)









 