# Import necessary libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split


class LSTM:
    def __init__(self,x_train,y_train,x_val,y_val):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

    def __call__(self):
        # Tokenize the text
        tokenizer = Tokenizer(num_words=5000)
        tokenizer.fit_on_texts(self.x_train)

        # Convert text to sequences
        train_seq = tokenizer.texts_to_sequences(x_train)
        val_seq = tokenizer.texts_to_sequences(x_val)

        # Pad sequences to same length
        max_length = 100
        train_seq = pad_sequences(train_seq, maxlen=max_length)
        val_seq = pad_sequences(val_seq, maxlen=max_length)

        # Load the pre-trained GloVe embeddings
        embedding_dict = {}
        with open(gdrive_path+'/glove.6B.100d.txt', 'r') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embedding_dict[word] = coefs

        # Create embedding matrix
        embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, 100))
        for word, i in tokenizer.word_index.items():
            embedding_vector = embedding_dict.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        # Define the model architecture
        model = Sequential()
        model.add(Embedding(len(tokenizer.word_index) + 1, 100, weights=[embedding_matrix], input_length=max_length, trainable=False))
        model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
        # model.add(Dense(1, activation='sigmoid'))
        model.add(Dense(3, activation='softmax'))

        # Compile the model
        model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

        # Convert sentiment labels to one-hot encoding
        y_train = tf.keras.utils.to_categorical(y_train, num_classes=3)
        y_val = tf.keras.utils.to_categorical(y_val, num_classes=3)

        # Train model
        model.fit(train_seq, y_train, validation_data=(val_seq, y_val), epochs=20, batch_size=32)

        # Save model
        model.save(gdrive_path+'/sentiment_analysis_of_review.h5')

    def predict(self, x_test):
        # Tokenize the text
        tokenizer = Tokenizer(num_words=5000)
        tokenizer.fit_on_texts(x_text_test)

        # Convert text to sequences
        new_seq = tokenizer.texts_to_sequences(x_text_test)

        
        # Pad sequences to same length
        max_length = 100
        test_seq = pad_sequences(new_seq, maxlen=max_length)

        # Load the trained model
        bce_model = tf.keras.models.load_model(gdrive_path+'/sentiment_analysis_of_review.h5', compile=False)
        
        # Make predictions
        prediction = model.predict(test_seq)

        return prediction

    


