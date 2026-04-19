import pandas as pd

print("="*50)
print("LABELING SENTIMEN DENGAN NEGASI")
print("="*50)

df = pd.read_csv('data/youtube_comments_preprocessed.csv')
print(f"Total data: {len(df)}")

# Kamus kata
POSITIF = {'bagus', 'baik', 'keren', 'mantap', 'suka', 'senang', 'hebat', 'luar',
           'best', 'good', 'great', 'awesome', 'perfect', 'juara', 'menang',
           'top', 'setuju', 'dukung', 'bangga', 'wow', 'gacor', 'fantastis',
           'berhasil', 'sukses', 'puas', 'istimewa', 'berkualitas'}

NEGATIF = {'jelek', 'buruk', 'benci', 'kesal', 'kecewa', 'goblok', 'tolol',
           'bodoh', 'bad', 'worst', 'suck', 'terrible', 'gagal', 'kalah',
           'korupsi', 'curang', 'sampah', 'hancur', 'rusak', 'parah', 'ampas',
           'memalukan', 'benci', 'marah', 'kesel', 'frustasi', 'payah'}

def label_sentiment(row):
    text = row['komentar_clean'].lower()
    has_negation = 'negasi' in text
    
    # Hapus token NEGASI untuk scoring
    clean_text = text.replace('negasi', '').strip()
    
    # Hitung skor
    pos_score = sum(1 for w in POSITIF if w in clean_text)
    neg_score = sum(1 for w in NEGATIF if w in clean_text)
    
    # Jika ada negasi, balik skor
    if has_negation:
        pos_score, neg_score = neg_score, pos_score
    
    # Tentukan sentimen
    if pos_score > neg_score and pos_score >= 1:
        return 'positif'
    elif neg_score > pos_score and neg_score >= 1:
        return 'negatif'
    else:
        return 'netral'

df['sentimen'] = df.apply(label_sentiment, axis=1)
sentimen_map = {'positif': 0, 'netral': 1, 'negatif': 2}
df['label'] = df['sentimen'].map(sentimen_map)

print("\nDistribusi sentimen:")
print(df['sentimen'].value_counts())
print("\nPersentase:")
print(df['sentimen'].value_counts(normalize=True) * 100)

df.to_csv('data/youtube_comments_labeled.csv', index=False)
print("\n✅ Data berhasil dilabel!")