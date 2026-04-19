from youtube_comment_downloader import YoutubeCommentDownloader
import pandas as pd

downloader = YoutubeCommentDownloader()

# Ganti dengan link video viral pilihan Anda (setelah disetujui dosen)
url = "https://www.youtube.com/watch?v=VTlfbgDxiZk"

comments = []
count = 0
max_comments = 10000  # Target 10.000 komentar

print("Mulai scraping komentar...")

for comment in downloader.get_comments_from_url(url):
    comments.append({
        "platform": "YouTube",
        "komentar": comment['text'],
        "timestamp": comment['time'],  # atau comment['time_parsed']
        "sentimen": ""  # Kolom kosong, akan diisi nanti
    })
    
    count += 1
    if count % 1000 == 0:
        print(f"Sudah mengumpulkan {count} komentar...")
    
    if count >= max_comments:
        break

# Buat DataFrame
df = pd.DataFrame(comments)

# Simpan ke CSV
df.to_csv("data/youtube_comments_raw.csv", index=False)

print(f"\nSelesai! Total komentar: {len(df)}")
print(df.head())
print("\nInfo dataset:")
print(df.info())