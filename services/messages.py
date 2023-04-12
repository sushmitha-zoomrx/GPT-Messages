import json
import re

import numpy as np
import openai
import pandas as pd
import scipy
from icecream import ic
from textblob import TextBlob

from config.settings import constants
from utilities.gsheets import log_to_gsheets


class MessagesService():
    def GPT_Generation(self, prompt, max_tokens, i):
        ic()
        openai.api_key = constants.GPT_API_KEY
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                temperature=1,
                messages=[
                    {"role": "user", "content": constants.GPT_PROMPT.substitute(
                        {'question': prompt})},
                ],
                max_tokens=max_tokens
            )
            ic(response)
        except Exception as e:
            if i <= 5 and 'server had an error' in str(e):
                ic(e)
                return self.GPT_Generation(prompt, max_tokens=max_tokens, i=i)
            else:
                ic(e)
                return

        return response['choices'][0]['message']['content']

    def generate_messages(self, email, prompt):
        try:
            print(prompt)
            response = json.loads(self.GPT_Generation(prompt, max_tokens=1000, i=1))
            if not response:
                ic()
                return
            if response['category'] == 3:
                return {
                    'data': '\n'.join(response['messages'])
                }

            generated_messages = response['messages']
            print('List of msgs : ')
            print(generated_messages)
            df = pd.DataFrame({
                'Message': [],
                'Believability': [],
                'Differentiation': [],
                'Motivation': [],
                'Overall Score': [],
                'Rank Percentile': []
            })
            for msg in generated_messages:
                df = pd.concat(
                    [df, pd.DataFrame({'Message': [msg], **self.predict_scores(msg)})],
                    ignore_index=True
                )
            df = df.sort_values('Overall Score', ascending=False)
            log_to_gsheets(email, prompt, df)
            return {
                'data': self.format_df(df)
            }

        except:
            import traceback
            traceback.print_exc()

    def format_df(self, df):
        # styles = dict(selector="table, th, td", props=[("border", "1px solid black")])

        res = df.to_html(index=False, escape=False, justify='center', border=0)

        res = re.sub(r'<thead>[\s\S]*?</thead>', constants.HEADER_HTML, res)
        res = res.replace('<table class="dataframe">', '<table>')

        return res

    def predict_scores(self, message):

        def round_off_score(val):
            val = "{:.1f} %".format(val)
            return val

        def calc_percentile(final_score):
            data = constants.CSV_DATA.copy(deep=True)
            data["%Believable"] = data["%Believable"].astype(float)
            data["%Differentiation"] = data["%Differentiation"].astype(float)
            data["%Motivation"] = data["%Motivation"].astype(float)
            data["Final_Score"] = np.cbrt(
                data["%Believable"] * data["%Differentiation"] * data["%Motivation"])
            percentile = scipy.stats.percentileofscore(data["Final_Score"], final_score)
            return str(int(percentile))

        word_count = len(str(message).split(" "))
        char_count = sum(len(word) for word in str(message).split(" "))
        sentence_count = len(str(message).split("."))
        avg_word_length = char_count / word_count
        avg_sentence_lenght = word_count / sentence_count
        sentiment = TextBlob(message).sentiment.polarity
        SEmbeddings = constants.EMBEDDING_MODEL.encode(message)
        X_test = [SEmbeddings.tolist() + [
            word_count, char_count, sentence_count,
            avg_word_length, avg_sentence_lenght, sentiment]]
        B_score, M_Score, D_Score = (
            constants.BELIEVABILITY_SCORE_PREDICTOR.predict(X_test)[0],
            constants.MOTIVATION_SCORE_PREDICTOR.predict(X_test)[0],
            constants.DIFFERENTIATION_SCORE_PREDICTOR.predict(X_test)[0]
        )
        overall_score = np.cbrt(B_score * M_Score * D_Score)
        return {
            'Believability': [round_off_score(B_score)],
            'Differentiation': [round_off_score(D_Score)],
            'Motivation': [round_off_score(M_Score)],
            'Overall Score': [round_off_score(overall_score)],
            'Rank Percentile': [calc_percentile(overall_score)]
        }


messages_service = MessagesService()
