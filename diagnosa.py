# diagnosa.py
import pandas as pd
import tensorflow as tf
import pickle
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

print("="*60)
print("🔍 DIAGNOSA MODEL SENTIMEN")
print("="*60)

# 1. Cek data training
print("\n1️⃣ Cek Data Training:")
try:
    df = pd.read_csv('data/youtube_comments_labeled.csv')
    print(f"   Total data: {len(df)}")
    print(f"   Distribusi sentimen:")
    print(df['sentimen'].value_counts())
    
    # Cek apakah semua sentimen ada
    if len(df['sentimen'].unique()) < 3:
        print(f"   ⚠️ PERINGATAN: Hanya ada {len(df['sentimen'].unique())} jenis sentimen!")
        print(f"   Sentimen yang ada: {df['sentimen'].unique()}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# 2. Cek model
print("\n2️⃣ Cek Model LSTM:")
try:
    model = tf.keras.models.load_model('models/lstm_model.h5')
    print("   ✅ Model berhasil dimuat")
    print(f"   Model summary:")
    model.summary()
except Exception as e:
    print(f"   ❌ Error: {e}")

# 3. Cek tokenizer
print("\n3️⃣ Cek Tokenizer:")
try:
    with open('models/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    print("   ✅ Tokenizer berhasil dimuat")
    print(f"   Vocabulary size: {len(tokenizer.word_index)} kata")
except Exception as e:
    print(f"   ❌ Error: {e}")

# 4. Test dengan berbagai komentar
print("\n4️⃣ Test Prediksi dengan Berbagai Komentar:")

factory = StemmerFactory()
stemmer = factory.create_stemmer()

stopwords = {'yang', 'dan', 'di', 'dari', 'ini', 'itu', 'adalah', 'untuk', 
             'dengan', 'pada', 'mereka', 'kami', 'kita', 'anda', 'aku', 
             'kamu', 'dia', 'ia', 'karena', 'jadi', 'sangat', 'bisa'}

def simple_preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stopwords]
    words = [stemmer.stem(w) for w in words]
    return ' '.join(words)

test_comments = [
    "Timnas Indonesia bagus banget",
    "Saya suka timnas",
    "Pemainnya hebat dan berkualitas",
    "Biasa aja sih",
    "Jelek banget timnasnya",
    "Saya tidak suka dengan permainannya",
    "Luar biasa pertandingannya!"
]

for comment in test_comments:
    cleaned = simple_preprocess(comment)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=100)
    pred = model.predict(padded, verbose=0)
    
    sentiment = ['positif', 'netral', 'negatif'][pred.argmax()]
    confidence = pred.max()
    
    print(f"\n   Teks: {comment}")
    print(f"   Cleaned: {cleaned}")
    print(f"   Prediksi: {sentiment.upper()} ({confidence:.2%})")
    print(f"   Detail: Positif={pred[0][0]:.3f}, Netral={pred[0][1]:.3f}, Negatif={pred[0][2]:.3f}")

print("\n" + "="*60)