import pandas as pd
import os
import re
import sys

# Cek library
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    SASTRAWI_AVAILABLE = True
except ImportError:
    SASTRAWI_AVAILABLE = False
    print("⚠️ Sastrawi tidak tersedia. Install: pip install Sastrawi")
    sys.exit(1)

# Buat folder
os.makedirs('data', exist_ok=True)

# Baca data
data_file = 'data/youtube_comments_raw.csv'
if not os.path.exists(data_file):
    print(f"❌ ERROR: File '{data_file}' tidak ditemukan!")
    sys.exit(1)

df = pd.read_csv(data_file)
print(f"✅ Jumlah komentar awal: {len(df)}")

# Clean data
df = df.dropna(subset=['komentar'])
df = df[df['komentar'].str.strip() != '']
print(f"✅ Setelah cleaning: {len(df)} komentar")

# Daftar negasi
NEGATION_WORDS = {'tidak', 'tak', 'bukan', 'jangan', 'belum', 'kurang', 'ga', 'gak', 'nggak', 'enggak'}

# Stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def clean_text(text):
    if not isinstance(text, str):
        return "", 1
    
    original_lower = text.lower()
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Deteksi negasi
    has_negation = any(neg in original_lower.split() for neg in NEGATION_WORDS)
    multiplier = -1 if has_negation else 1
    
    # Stemming
    words = text.split()
    words = [stemmer.stem(w) for w in words if len(w) > 2]
    
    if has_negation:
        words.insert(0, 'NEGASI')
    
    return ' '.join(words), multiplier

# Proses
print("Melakukan preprocessing...")
results = df['komentar'].apply(clean_text)
df['komentar_clean'] = [r[0] for r in results]
df['negasi'] = [r[1] for r in results]

df = df[df['komentar_clean'] != '']
df.to_csv('data/youtube_comments_preprocessed.csv', index=False)

print(f"✅ Selesai! {len(df)} komentar tersimpan")
print("\n📝 Contoh hasil:")
for i in range(min(5, len(df))):
    print(f"  {df['komentar'].iloc[i][:50]} → {df['komentar_clean'].iloc[i][:50]}")