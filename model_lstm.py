import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pickle
import os

print("="*50)
print("TRAINING MODEL LSTM")
print("="*50)

os.makedirs('models', exist_ok=True)

df = pd.read_csv('data/youtube_comments_labeled.csv')
print(f"Total data: {len(df)}")
print(df['sentimen'].value_counts())

X = df['komentar_clean'].values
y = df['label'].values

tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(X)
X_seq = tokenizer.texts_to_sequences(X)
X_pad = pad_sequences(X_seq, maxlen=100, padding='post', truncating='post')

y_cat = tf.keras.utils.to_categorical(y, num_classes=3)

X_train, X_test, y_train, y_test = train_test_split(X_pad, y_cat, test_size=0.2, random_state=42, stratify=y)

class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = dict(enumerate(class_weights))

model = Sequential([
    Embedding(10000, 128, input_length=100),
    Bidirectional(LSTM(128, dropout=0.2, return_sequences=True)),
    Bidirectional(LSTM(64, dropout=0.2)),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2,
                    class_weight=class_weight_dict, callbacks=[early_stop, reduce_lr], verbose=1)

loss, acc = model.evaluate(X_test, y_test)
print(f"\n✅ Test Accuracy: {acc:.4f}")

y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['positif', 'netral', 'negatif']))

model.save('models/lstm_model.h5')
with open('models/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

print("\n✅ Model dan tokenizer tersimpan di folder 'models/'")