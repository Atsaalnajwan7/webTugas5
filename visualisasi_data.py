import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os

# Buat folder untuk menyimpan visualisasi
if not os.path.exists('outputs'):
    os.makedirs('outputs')
if not os.path.exists('outputs/visualizations'):
    os.makedirs('outputs/visualizations')
if not os.path.exists('outputs/wordclouds'):
    os.makedirs('outputs/wordclouds')

# Baca data yang sudah dilabel
df = pd.read_csv('data/youtube_comments_labeled.csv')

print("="*50)
print("📊 MEMBUAT VISUALISASI DATA")
print("="*50)

# 1. Distribusi Sentimen (Bar Chart)
plt.figure(figsize=(10, 6))
colors = {'positif': 'green', 'netral': 'gray', 'negatif': 'red'}
sentimen_counts = df['sentimen'].value_counts()
bars = plt.bar(sentimen_counts.index, sentimen_counts.values, 
               color=[colors[s] for s in sentimen_counts.index])
plt.title('Distribusi Sentimen Komentar YouTube', fontsize=16)
plt.xlabel('Sentimen', fontsize=12)
plt.ylabel('Jumlah Komentar', fontsize=12)

# Tambahkan angka di atas bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)} ({height/len(df)*100:.1f}%)',
             ha='center', va='bottom')

plt.tight_layout()
plt.savefig('outputs/visualizations/distribusi_sentimen.png', dpi=300)
plt.show()

# 2. Pie Chart
plt.figure(figsize=(8, 8))
plt.pie(sentimen_counts.values, labels=sentimen_counts.index, 
        autopct='%1.1f%%', colors=[colors[s] for s in sentimen_counts.index],
        startangle=90)
plt.title('Persentase Sentimen', fontsize=16)
plt.savefig('outputs/visualizations/pie_chart_sentimen.png', dpi=300)
plt.show()

# 3. Wordcloud untuk setiap sentimen
for sentiment in ['positif', 'netral', 'negatif']:
    # Gabungkan semua teks untuk sentimen tertentu
    text = ' '.join(df[df['sentimen'] == sentiment]['komentar_clean'])
    
    if text.strip():  # Jika ada teks
        wordcloud = WordCloud(width=800, height=400, 
                             background_color='white',
                             colormap='viridis',
                             max_words=100).generate(text)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f'Wordcloud - Sentimen {sentiment.upper()}', fontsize=16)
        plt.axis('off')
        plt.savefig(f'outputs/wordclouds/wordcloud_{sentiment}.png', dpi=300)
        plt.show()
    else:
        print(f"Tidak ada teks untuk sentimen {sentiment}")

print("\n✅ Visualisasi selesai! File disimpan di folder 'outputs/'")