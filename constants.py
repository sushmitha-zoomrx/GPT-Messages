import pygsheets
import streamlit as st
from google.oauth2 import service_account
from PIL import Image

APP_URL = 'https://me-prediction-zoomrx.streamlit.app/'
BENCHMARKS = {
    "Believability": "63 %",
    "Differentiation": "55 %",
    "Motivation": "59 %",
    "Overall_Score": "59 %"
}
EMAIL_VALIDATOR = '^[a-z0-9]+[\._]?[ a-z0-9]+[@]\w+[. ]\w{2,3}$'
LOGO = Image.open('input_files/logo.png')


GPT3_API_KEY = "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"
GPT3_PROMPT = """
If the Text given below text asks for prediction of any score,  must give the "exact message" from the Text, for which score should be predicted. If Text is not asking for prediction, just answer to the Text. For creating or generating messages, if no count of required messages mentioned, generate 5 messages by default.

Text:
Predict the score of Fight NSCLC with OPDIVO + YERVOY and platinum-based chemotherapy and live longer.
Output:
Fight NSCLC with OPDIVO + YERVOY and platinum-based chemotherapy and live longer.

Text:
Create 2 messages for Opdivo in NSCLC using : In 14.1 months, half the people were alive on OPDIVO + YERVOY and platinum-based chemotherapy
Output:
1. Improve your outlook with OPDIVO + YERVOY and platinum-based chemotherapy. In 14.1 months, half the people were living longer and stronger.
2.  Give yourself the best chance of beating non-small cell lung cancer with OPDIVO + YERVOY and platinum-based chemotherapy. In 14.1 months, half the people saw long-term survival.

Text:
What is the prediction score of cabometyx + opdivo offers a balance of data
Output:
cabometyx + opdivo offers a balance of data

Text:
{input}
Output:
"""
PERCENTILE_TEXT = (
    "To improve its performance, use precise messages with data & qualifiers as relevant."
    "[Read more>>](https://learnmore.zoomrx.com/pet-registration)"
)
HIDE_TABLE_ROW_INDEX = """
    <style>
    thead tr th:first-child {display:none}
    tbody th {display:none}
    </style>
"""
FOOTER_TEXT = (
    "This is a Beta version of the product. Our goal is to help build AI systems that could aid in Pharma "
    "promotions. Please let us know your feedback at info@zoomrx.com")

# Access to gsheet
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"],
    scopes=[
        "https://www.googleapis.com/auth/spreadsheets",
    ],
)
gc = pygsheets.authorize(custom_credentials=credentials)
GSHEET = gc.open_by_url(st.secrets["private_gsheets_url"])[0]

DISPLAY_MESSAGES = {
    'create': (
        'Create 5 messages for Opdivo in NSCLC using : In 14.1 months, half the people were alive on '
        'OPDIVO + YERVOY and platinum-based chemotherapy'
    ),
    'predict': (
        'Predict the score of "Fight NSCLC with OPDIVO + YERVOY and platinum-based chemotherapy and live longer."'
    )
}
