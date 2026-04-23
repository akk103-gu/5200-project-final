import time, pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

bpd = pd.read_csv("data/bpd_fp_posts.csv")
loved = pd.read_csv("data/bpdlovedones_fp_posts.csv")
df = pd.concat([bpd, loved], ignore_index=True).drop_duplicates(subset="id")
df["text_combined"] = (df["title"].fillna("") + " " + df["text"].fillna("")).str.strip()
df = df[df["text_combined"].str.len() > 10].copy()
df = df.dropna(subset=["year"])
print(f"Posts: {len(df)}")

analyzer = SentimentIntensityAnalyzer()

t = time.time()
_ = [analyzer.polarity_scores(str(x)) for x in df["text_combined"].head(1000)]
elapsed = time.time() - t
print(f"1000 posts: {elapsed:.2f}s")
print(f"Estimated full run: {elapsed * len(df) / 1000:.0f}s")
