import tensorflow as tf
from tensorflow import keras
import numpy as np
import re

data = keras.datasets.imdb

(X_train, y_train), (X_test, y_test) = data.load_data(num_words=1000)

# CHECKING OUT THE FIRST REVIEW
'''
print(X_train[0])
'''

# WORD MAPPING
word_index = data.get_word_index()

word_index = {k:(v+3) for k,v in word_index.items()}
word_index['<PAD>'] = 0
word_index['<START>'] = 1
word_index['<UNK>'] = 2
word_index['<UNUSED>'] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# MAKE ALL REVIEWS THE SAME LENGTH
X_train = keras.preprocessing.sequence.pad_sequences(X_train, value = word_index['<PAD>'], padding = 'post', maxlen = 250)
X_test = keras.preprocessing.sequence.pad_sequences(X_test, value = word_index['<PAD>'], padding = 'post', maxlen = 250)

# DEFINE A FUNCTION TO DECODE WORDS FOR A REVIEW
def decode_review (text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])

#CHECKING OUT THE FIRST REVIEW'S ACTUAL TEXT
'''
print(decode_review(X_test[0]))
'''

# MODEL, EVALUATE, AND SAVE
'''
model = keras.Sequential()
model.add(keras.layers.Embedding(88000,16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation = 'sigmoid'))

model.summary()

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

x_val = X_train[:10000]
x_train = X_train[10000:]

y_val = y_train[:10000]
Y_train = y_train[10000:]

fitModel = model.fit(x_train, Y_train, epochs = 40, batch_size = 512, validation_data = (x_val, y_val), verbose = 1)

results = model.evaluate(X_test, y_test)

print(results)

#SAVE THE MODEL
model.save('textclassmodel.h5')
'''

def review_encode(s):
    encoded = [1]

    for word in s:
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)
    return encoded

#LOAD THE MODEL
model = keras.models.load_model('textclassmodel.h5')

with open('texttest.txt', encoding = 'utf-8') as f:
    for line in f.readlines():
        nline = line.replace(',', '').replace('.', '').replace('(', '').replace(')', '').replace('\"', '').replace(':', '').replace('!', '').strip().split(' ')
        encode = review_encode(nline)
        encode = keras.preprocessing.sequence.pad_sequences([encode], value = word_index['<PAD>'], padding = 'post', maxlen = 250)
        predict = model.predict(encode)
        print(nline)
        print(line)
        print(encode)
        print(predict[0])

# MAKE A PREDICTION FROM IMDB DATA, BUT FIRST PRINT THE DECODED REVIEW
'''
test_review = X_test[0]
predict = model.predict([test_review])
print("Review: ")
print(decode_review(test_review))
print("Prediction: " + str(predict[0]))
print("Actual: " + str(y_test[0]))
print(results)
'''
