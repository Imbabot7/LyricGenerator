import streamlit as st
import tensorflow as tf
from LSTM import StackedLSTM
from LyricsGenerator import LyricsGenerator

st.title('Lyrics Generator :musical_note:')

prompt = st.text_area('Enter a lyric prompt here')
temp = st.slider('Temperature for predictions (the higher the value the more random generated lyrics are)', 0.1,2.0,1.0,0.1)
lengths = st.select_slider('Select length of generated lyrics',['short','medium','long'],value='medium')

dct_length = {'short':100,'medium':250,'long':500}

generator = LyricsGenerator()
generator.load_weights('checkpoints/checkpoint_model3')

if st.button('Generate lyrics'):
    prompt = ' ' if len(prompt) == 0 else prompt
    generated = generator.generate(prompt=prompt,temperature=temp,length=dct_length[lengths])
    print(generator.generate())
    st.write(generated)



