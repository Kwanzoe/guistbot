import random
from tensorflow.keras.optimizers import SGD
from keras.layers import Dense, Dropout
from keras.models import Sequential
import numpy as np
import pickle
import json
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


words = []            # Bag of words
classes = []        # List of classes
documents = []      # list of elements which are tuples - words and their corresponding tag
ignore_words = ['?', '!']
data_file = open('botIntents.json').read()
intents = json.loads(data_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:

        # tokenize
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # This is the list of elements which are the words against a corresponding tag
        documents.append((w, intent['tag']))

        # classes list formation
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# lemmatize and keep unique values
words = [lemmatizer.lemmatize(w.lower())
         for w in words if w not in ignore_words]
words = list(set(words))
# sort classes
classes = list(set(classes))

print(len(documents), "documents")

print(len(classes), "classes", classes)

print(len(words), "unique lemmatized words", words)


pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Training data
training = []
# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(
        word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = [0] * len(classes)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])
# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)
# create train and test lists. X - patterns, Y - intents
train_x = list(training[:, 0])
train_y = list(training[:, 1])
print("Training data created")

# Model: 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd is output layer

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd, metrics=['accuracy'])

# fitting and saving the model
h = model.fit(np.array(train_x), np.array(train_y),
              epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', h)

print("model created")
