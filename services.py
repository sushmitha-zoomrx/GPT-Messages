import re
import subprocess
from subprocess import PIPE

import numpy as np
import openai
import pandas as pd
import scipy
import streamlit as st
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
from xgboost import XGBRegressor

from constants import GPT3_API_KEY


@st.cache_resource
def load_models():
    b_model = XGBRegressor({'nthread': 4})
    b_model.load_model("./models/0001.model")
    m_model = XGBRegressor({'nthread': 4})
    m_model.load_model("./models/0002.model")
    d_model = XGBRegressor({'nthread': 4})
    d_model.load_model("./models/0003.model")
    emb_model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    return b_model, m_model, d_model, emb_model


def GPT3_Generation(prompt, max_tokens, i):
    openai.api_key = GPT3_API_KEY
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            temperature=0.7,
            top_p=1,
            max_tokens=max_tokens,
            frequency_penalty=0,
            presence_penalty=0
        )
    except Exception as e:
        if i <= 5 and 'server had an error' in str(e):
            st.markdown(f'Server is down, Hence re-generating... {i}')
            i += 1
            return GPT3_Generation(prompt, max_tokens=2000, i=i)
        else:
            st.error('GPT Server is down. Please try again later')
            return None
    return response.choices[0].text


def generate_messages(prompt):
    try:
        print(prompt)
        st.markdown('Generating messages...')
        msgs = GPT3_Generation(prompt, max_tokens=2000, i=1)
        if not msgs:
            return
        st.markdown('Generated, Predicting the scores of the messages...')
        print('Generated!!!!!!!')
        print(msgs)
        generated_messages = []
        sentences = re.split('^\d+\.+ ', msgs, flags=re.M)
        for i, text in enumerate(sentences):
            msg = text.strip()
            if len(msg) > 5:
                generated_messages.append(msg)
        print('List of msgs : ')
        print(generated_messages)
        df = pd.DataFrame({
            'Message': [],
            'Believability \n(ZoomRx Industry Average is 63 %)': [],
            'Differentiation \n(ZoomRx Industry Average is 55 %)': [],
            'Motivation \n(ZoomRx Industry Average is 59 %)': [],
            'Overall Score \n(ZoomRx Industry Average is 59 %)': [],
            'Rank Percentile': []
        })
        for msg in generated_messages:
            df = df.append({'Message': msg, **predict_scores(msg)}, ignore_index=True)
        return df.sort_values('Overall Score \n(ZoomRx Industry Average is 59 %)', ascending=False)
    except:
        import traceback
        traceback.print_exc()


@st.cache_resource
def read_data():
    data_df = pd.read_csv('./input_files/data.csv')
    return data_df


def predict_scores(message):

    def round_off_score(val):
        val = "{:.2f} %".format(val)
        return val

    def calc_percentile(final_score):
        data = read_data()
        data["%Believable"] = data["%Believable"].astype(float)
        data["%Differentiation"] = data["%Differentiation"].astype(float)
        data["%Motivation"] = data["%Motivation"].astype(float)
        data["Final_Score"] = np.cbrt(data["%Believable"] * data["%Differentiation"] * data["%Motivation"])
        percentile = scipy.stats.percentileofscore(data["Final_Score"], final_score)
        return percentile

    b_model, m_model, d_model, emb_model = load_models()
    word_count = len(str(message).split(" "))
    char_count = sum(len(word) for word in str(message).split(" "))
    sentence_count = len(str(message).split("."))
    avg_word_length = char_count / word_count
    avg_sentence_lenght = word_count / sentence_count
    sentiment = TextBlob(message).sentiment.polarity
    SEmbeddings = emb_model.encode(message)
    X_test = [SEmbeddings.tolist() + [
        word_count, char_count, sentence_count,
        avg_word_length, avg_sentence_lenght, sentiment]]
    B_score, M_Score, D_Score = b_model.predict(X_test)[0], m_model.predict(X_test)[0], d_model.predict(X_test)[0]
    overall_score = np.cbrt(B_score * M_Score * D_Score)
    return {
        'Believability \n(ZoomRx Industry Average is 63 %)': round_off_score(B_score),
        'Differentiation \n(ZoomRx Industry Average is 55 %)': round_off_score(D_Score),
        'Motivation \n(ZoomRx Industry Average is 59 %)': round_off_score(M_Score),
        'Overall Score \n(ZoomRx Industry Average is 59 %)': round_off_score(overall_score),
        'Rank Percentile': int(calc_percentile(overall_score))
    }
