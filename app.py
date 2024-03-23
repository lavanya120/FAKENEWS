import streamlit as st
import pandas as pd
import joblib
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
vector_form = joblib.load(open('vector.pkl', 'rb'))
load_model = joblib.load('FakeNews_detector.pkl')
def wordopt(text):
    text=text.lower()
    text=re.sub('\[,*?\]','',text)
    text=re.sub('\\W','',text)
    text=re.sub('https?://\S+|www\.\S+','',text)
    text=re.sub('<,*?>+','',text)
    text=re.sub('\n','',text)
    text=re.sub('\w*\d\w*','',text)
    return text
def testing(news):
    news=wordopt(news)
    input_data=[news]
    vector_form1=vector_form.transform(input_data)
    prediction = load_model.predict(vector_form1)
    return prediction[0]
  


if __name__ == '__main__':
    st.title('Fake News Classification app ')
    st.subheader("Input the News content below")
    sentence = st.text_area("Enter your news content here", "",height=200)
    predict_btt = st.button("predict")
    if predict_btt:
        prediction_class=testing(sentence)
        if prediction_class == 'REAL':
            st.success('The news is REAL!!')
        if prediction_class == 'FAKE':
            st.warning('The news is FAKE!!')