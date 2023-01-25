import re

import numpy as np
import pandas as pd
import scipy
import streamlit as st
from pyChatGPT import ChatGPT
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
from xgboost import XGBRegressor

from constants import SESSION_TOKEN


@st.experimental_singleton
def load_models():
    b_model = XGBRegressor({'nthread': 4})
    b_model.load_model("./models/0001.model")
    m_model = XGBRegressor({'nthread': 4})
    m_model.load_model("./models/0002.model")
    d_model = XGBRegressor({'nthread': 4})
    d_model.load_model("./models/0003.model")
    emb_model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    return b_model, m_model, d_model, emb_model


@st.experimental_singleton
def load_cgpt():
    cg_api = ChatGPT(session_token=SESSION_TOKEN)
    print('API created')
    return cg_api


def ChatGPT_Generation(prompt):
    api = load_cgpt()
    try:
        return api.send_message(prompt)['message']
    except:
        api.reset_conversation()
        api.clear_conversations()
        api.refresh_chat_page()
        return api.send_message(prompt)['message']


def generate_messages(prompt):
    try:
        msgs = ChatGPT_Generation(prompt)
        generated_messages = []
        sentences = re.split('^\d+\.+ ', msgs, flags=re.M)
        for i, text in enumerate(sentences):
            msg = text.strip()
            if len(msg) > 5:
                generated_messages.append(msg)
        print(generated_messages)
        df = pd.DataFrame()
        for msg in generated_messages:
            df = df.append({'Message': msg, **predict_scores(msg)}, ignore_index=True)
        return df
    except:
        import traceback
        traceback.print_exc()


@st.experimental_singleton
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
        'Believability': round_off_score(B_score),
        'Differentiation': round_off_score(D_Score),
        'Motivation': round_off_score(M_Score),
        'Overall_Score': round_off_score(overall_score),
        'Rank_Percentile': int(calc_percentile(overall_score))
    }
