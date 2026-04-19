# app.py - Web App Analisis Sentimen Lengkap
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
import re
import base64
from io import BytesIO
import time

app = Flask(__name__)

# Buat folder untuk menyimpan gambar
os.makedirs('static/images', exist_ok=True)

# ============================================
# KAMUS KATA UNTUK SENTIMEN
# ============================================

# Kata POSITIF
POSITIVE_WORDS = {
    'bagus', 'baik', 'keren', 'mantap', 'suka', 'senang', 'hebat',
    'luar biasa', 'best', 'good', 'great', 'awesome', 'amazing',
    'perfect', 'nice', 'juara', 'menang', 'top', 'setuju', 'dukung',
    'bangga', 'wow', 'gacor', 'fantastis', 'berhasil', 'sukses',
    'puas', 'istimewa', 'berkualitas', 'profesional', 'berprestasi',
    'handal', 'cerdas', 'brilian', 'gemilang', 'membanggakan',
    'terbaik', 'keren abis', 'luar biasa', 'sangat bagus'
}

# Kata NEGATIF
NEGATIVE_WORDS = {
    'jelek', 'buruk', 'benci', 'kesal', 'kecewa', 'goblok', 'tolol',
    'bodoh', 'bad', 'worst', 'suck', 'terrible', 'horrible',
    'gagal', 'kalah', 'sampah', 'hancur', 'rusak', 'parah', 'ampas',
    'memalukan', 'marah', 'kesel', 'frustasi', 'payah', 'kecewa',
    'benci', 'tidak becus', 'mengecewakan', 'parah banget', 'buruk sekali'
}

# Kata NEGASI
NEGATION_WORDS = {'tidak', 'tak', 'bukan', 'jangan', 'belum', 'ga', 'gak', 'nggak', 'enggak'}

# ============================================
# FUNGSI PREDIKSI SENTIMEN
# ============================================

def predict_sentiment(text):
    """
    Prediksi sentimen menggunakan rule-based method
    Returns: sentiment (positif/netral/negatif), confidence (0-1)
    """
    if not isinstance(text, str) or text.strip() == '':
        return 'netral', 0.5
    
    text_lower = text.lower()
    
    # Deteksi negasi
    words = text_lower.split()
    has_negation = any(neg in words for neg in NEGATION_WORDS)
    
    # Hitung skor
    pos_score = 0
    neg_score = 0
    
    # Hitung kata positif
    for word in POSITIVE_WORDS:
        if word in text_lower:
            pos_score += 1
    
    # Hitung kata negatif
    for word in NEGATIVE_WORDS:
        if word in text_lower:
            neg_score += 1
    
    # Jika ada negasi, balik skor positif dan negatif
    if has_negation:
        pos_score, neg_score = neg_score, pos_score
    
    # Hitung confidence berdasarkan selisih skor
    total_score = pos_score + neg_score
    
    if pos_score > neg_score and pos_score > 0:
        sentiment = 'positif'
        confidence = min(0.6 + (pos_score * 0.1), 0.95)
    elif neg_score > pos_score and neg_score > 0:
        sentiment = 'negatif'
        confidence = min(0.6 + (neg_score * 0.1), 0.95)
    else:
        sentiment = 'netral'
        confidence = 0.55
    
    # Tambahan: deteksi tanda seru untuk memperkuat
    if '!' in text_lower and sentiment != 'netral':
        confidence = min(confidence + 0.05, 0.95)
    
    # Tambahan: kata "banget" atau "sekali" memperkuat
    if ('banget' in text_lower or 'sekali' in text_lower) and sentiment != 'netral':
        confidence = min(confidence + 0.05, 0.95)
    
    return sentiment, confidence

# ============================================
# FUNGSI PREPROCESSING UNTUK WORDCLOUD
# ============================================

def preprocess_for_wordcloud(text):
    """Preprocessing teks untuk wordcloud"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [w for w in words if len(w) > 2]
    return ' '.join(words)

# ============================================
# FUNGSI GENERATE VISUALISASI
# ============================================

def generate_visualizations(comments_data):
    """
    Generate visualisasi dan wordcloud dari data komentar
    Returns: dict dengan base64 encoded images
    """
    images = {}
    
    if not comments_data:
        return images
    
    # Buat dataframe
    df = pd.DataFrame(comments_data)
    
    # Hitung distribusi sentimen
    sentiment_counts = df['sentimen'].value_counts()
    
    # Pastikan semua sentimen ada
    for s in ['positif', 'netral', 'negatif']:
        if s not in sentiment_counts.index:
            sentiment_counts[s] = 0
    
    # ========================================
    # 1. BAR CHART - Distribusi Sentimen
    # ========================================
    plt.figure(figsize=(10, 6))
    colors = {'positif': '#28a745', 'netral': '#ffc107', 'negatif': '#dc3545'}
    bars = plt.bar(sentiment_counts.index, sentiment_counts.values, 
                   color=[colors.get(x, '#667eea') for x in sentiment_counts.index])
    plt.title('Distribusi Sentimen Komentar', fontsize=16, fontweight='bold')
    plt.xlabel('Sentimen', fontsize=12)
    plt.ylabel('Jumlah Komentar', fontsize=12)
    
    # Tambahkan angka di atas bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Simpan ke buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    images['bar_chart'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    
    # ========================================
    # 2. PIE CHART - Persentase Sentimen
    # ========================================
    plt.figure(figsize=(8, 8))
    plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%',
            colors=['#28a745', '#ffc107', '#dc3545'], startangle=90)
    plt.title('Persentase Sentimen', fontsize=16, fontweight='bold')
    
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    images['pie_chart'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    
    # ========================================
    # 3. WORDCLOUD untuk setiap sentimen
    # ========================================
    for sentiment in ['positif', 'netral', 'negatif']:
        sentiment_df = df[df['sentimen'] == sentiment]
        if len(sentiment_df) > 0:
            # Gabungkan semua teks
            text = ' '.join(sentiment_df['komentar_clean'].astype(str))
            if text.strip():
                # Generate wordcloud
                wordcloud = WordCloud(width=800, height=400, 
                                     background_color='white',
                                     colormap='viridis',
                                     max_words=100).generate(text)
                
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.title(f'Wordcloud - Sentimen {sentiment.upper()}', fontsize=14, fontweight='bold')
                plt.axis('off')
                
                buf = BytesIO()
                plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                buf.seek(0)
                images[f'wordcloud_{sentiment}'] = base64.b64encode(buf.getvalue()).decode('utf-8')
                plt.close()
    
    return images

# ============================================
# DATA STORAGE (untuk menyimpan komentar)
# ============================================
comments_history = []

# ============================================
# ROUTES
# ============================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Prediksi satu komentar"""
    text = request.form.get('text', '')
    if not text:
        return jsonify({'error': 'Masukkan komentar terlebih dahulu!'}), 400
    
    # Prediksi sentimen
    sentiment, confidence = predict_sentiment(text)
    
    # Simpan ke history
    comments_history.append({
        'komentar': text,
        'komentar_clean': preprocess_for_wordcloud(text),
        'sentimen': sentiment,
        'confidence': confidence,
        'timestamp': time.time()
    })
    
    # Batasi history hanya 100 komentar terakhir
    if len(comments_history) > 100:
        comments_history.pop(0)
    
    # Generate visualisasi dari semua history
    visualizations = generate_visualizations(comments_history)
    
    # Hitung statistik
    df_history = pd.DataFrame(comments_history)
    if len(df_history) > 0:
        sentiment_counts = df_history['sentimen'].value_counts().to_dict()
    else:
        sentiment_counts = {'positif': 0, 'netral': 0, 'negatif': 0}
    
    return jsonify({
        'sentiment': sentiment,
        'confidence': confidence,
        'text': text,
        'visualizations': visualizations,
        'stats': {
            'total': len(comments_history),
            'positif': sentiment_counts.get('positif', 0),
            'netral': sentiment_counts.get('netral', 0),
            'negatif': sentiment_counts.get('negatif', 0)
        }
    })

@app.route('/reset', methods=['POST'])
def reset():
    """Reset history komentar"""
    global comments_history
    comments_history = []
    return jsonify({'status': 'success', 'message': 'History telah direset!'})

@app.route('/history', methods=['GET'])
def get_history():
    """Get history komentar"""
    return jsonify({
        'total': len(comments_history),
        'comments': comments_history[-20:]  # 20 komentar terakhir
    })

@app.route('/visualizations', methods=['GET'])
def get_visualizations():
    """Get visualisasi dari semua history"""
    visualizations = generate_visualizations(comments_history)
    return jsonify({'visualizations': visualizations})

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 ANALISIS SENTIMEN - WEB APP")
    print("📱 Buka di: http://localhost:5000")
    print("="*50)
    app.run(debug=True, host='0.0.0.0', port=5000)