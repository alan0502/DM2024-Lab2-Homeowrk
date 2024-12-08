import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, ReLU, Softmax, BatchNormalization
import keras

train_df = pd.read_csv('train_data.csv')
test_df = pd.read_csv('test_data.csv')

nltk.download('stopwords')
train_df.drop_duplicates(subset=['text'], keep=False, inplace=True)

TFIDF = TfidfVectorizer(max_features=25000, stop_words='english')
TFIDF.fit(train_df['text'])

X_train = TFIDF.transform(train_df['text'])
y_train = train_df['emotion']
X_test = TFIDF.transform(test_df['text'])

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

label_encoder = LabelEncoder()
label_encoder.fit(y_train)

def label_encode(le, labels):
    enc = le.transform(labels)
    return keras.utils.to_categorical(enc)

def label_decode(le, one_hot_label):
    dec = np.argmax(one_hot_label, axis=1)
    return le.inverse_transform(dec)

y_train = label_encode(label_encoder, y_train)
y_val = label_encode(label_encoder, y_val)

print('X_train.shape: ', X_train.shape)
print('y_train.shape: ', y_train.shape)
print('X_test.shape: ', X_test.shape)

# Input layer
model_input = Input(shape=(25000, ))

# First hidden layer
X = Dense(units=1024)(model_input)
X = BatchNormalization()(X)  # 加速收斂
X = ReLU()(X)
X = Dropout(0.3)(X)

# Second hidden layer
X = Dense(units=512)(X)
X = BatchNormalization()(X)
X = ReLU()(X)
X = Dropout(0.3)(X)

# Third hidden layer
X = Dense(units=256)(X)
X = BatchNormalization()(X)
X = ReLU()(X)
X = Dropout(0.3)(X)

# Fourth hidden layer
X = Dense(units=128)(X)
X = BatchNormalization()(X)
X = ReLU()(X)

# Fifth hidden layer
X = Dense(units=64)(X)
X = BatchNormalization()(X)
X = ReLU()(X)

# Sixth hidden layer
X = Dense(units=32)(X)
X = BatchNormalization()(X)
X = ReLU()(X)

# Output layer
model_output = Dense(units=8, activation='softmax')(X)

# Build the model
model = Model(inputs=[model_input], outputs=[model_output])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Model summary
model.summary()

epochs = 30
batch_size = 256

from tensorflow.keras.callbacks import EarlyStopping

# Define EarlyStopping
early_stopping = EarlyStopping(
    monitor='val_accuracy',  
    patience=3,            
    mode='max',             
    restore_best_weights=True 
)

history = model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping]
)

y_pred = model.predict(X_test)
y_pred = label_decode(label_encoder, y_pred)

submit_df = pd.DataFrame({
    'id': test_df['tweet_id'],
    'emotion': y_pred
})
submit_df.to_csv('submission.csv', index=False)