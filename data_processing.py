import pandas as pd
from utils import detect_lang, clean

df = pd.read_csv('data/lyrics.csv', sep = '\t')
df.dropna(subset=['lyrics'], inplace = True)
df.drop_duplicates(subset=['lyrics'],keep=False, inplace=True)
df['language'] = df['lyrics'].apply(detect_lang)
eng = df[df['language']=='en'].copy()
eng['word_count'] = eng['lyrics'].str.split().apply(len)
eng_removed = eng[eng['word_count']<=500]
#eng_removed = eng[eng['word_count']<=(eng['word_count'].quantile(0.75) + 1.5*(eng['word_count'].quantile(0.75) - eng['word_count'].quantile(0.25)))]
eng_cleaned = clean(eng_removed.copy())
corpus = '\n'.join(eng_cleaned['lyrics'].tolist())
with open('data/lyrics.txt', 'w+') as fh:
    fh.write(corpus)