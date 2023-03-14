from langdetect import detect
import re

def detect_lang(text):
    try:
        lang = detect(text)
        return lang
    except:
        lang = 'other'
    return lang

def clean(texts):
    texts['lyrics'] = texts['lyrics'].str.lower()
    texts['lyrics'] = texts['lyrics'].str.replace('\\\\n', '\n', regex=True)
    texts['lyrics'] = texts['lyrics'].str.replace('\\', '', regex=True)
    texts['lyrics'] = [re.sub(r'\[.*?\]','',text) for text in texts['lyrics']]
    texts['lyrics'] = [re.sub(r'[^A-Za-z0-9 \n.,?!\'":;]+','',text) for text in texts['lyrics']]
    return texts