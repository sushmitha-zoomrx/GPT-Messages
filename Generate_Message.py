import re

import pygsheets
import streamlit as st
from google.oauth2 import service_account

from constants import BENCHMARKS, IMAGE
from features import list_messages

credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"],
    scopes=[
        "https://www.googleapis.com/auth/spreadsheets",
    ],
)
gc = pygsheets.authorize(custom_credentials=credentials)
sh = gc.open_by_url(st.secrets["private_gsheets_url"])
wk = sh[0]
st.set_page_config(
    page_title="Message Effectiveness Prediction",
    page_icon=IMAGE,
    initial_sidebar_state="expanded",
    layout="wide"
)
user_email = st.text_input('Please enter your valid work email id to access the app')
regex = '^[a-z0-9]+[\._]?[ a-z0-9]+[@]\w+[. ]\w{2,3}$'
valid_email = re.findall(regex, user_email)
if len(valid_email) > 0:
    st.markdown(
        "<h2 style='text-align: center;'>Message Effectiveness Prediction Application</h2>",
        unsafe_allow_html=True)

    prompt = st.text_input('Type in the instruction for message generation')
    st.caption('Sample Instruction')
    st.code('"Create messages for Opdivo in NSCLC using : In 14.1 months, half the people were alive on '
            'OPDIVO + YERVOY and platinum-based chemotherapy"')
    if prompt:
        wk.append_table(values=[f'{valid_email[0]}', prompt])

        res = list_messages(prompt)
        if res is not None:
            if len(res) == 1:
                res = res.to_dict(orient='records')
                st.subheader('Generated Message')
                st.markdown(f"<p>{res[0]['Message']}<p>", unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    st.header('Predicted Scores')
                    for key in ['Believability', 'Differentiation', 'Motivation', 'Overall_Score']:
                        col1.metric(key, res[0][key])
                with col2:
                    st.header('Industry Averages')
                    for key in ['Believability', 'Differentiation', 'Motivation', 'Overall_Score']:
                        col2.metric(key, BENCHMARKS[key])

            else:
                # CSS to inject contained in a string
                hide_table_row_index = """
                            <style>
                            thead tr th:first-child {display:none}
                            tbody th {display:none}
                            </style>
                            """

                # Inject CSS with Markdown
                st.markdown(hide_table_row_index, unsafe_allow_html=True)

                # Display a static table
                st.table(res)
                # st.dataframe(data=res, use_container_width=False)

            percentile_text = (
                "To improve its performance, use precise messages with data & qualifiers as relevant."
                "[Read more>>](https://learnmore.zoomrx.com/pet-registration)"
            )
            st.markdown(percentile_text)
            if st.button('Share App Link'):
                st.markdown('https://me-prediction-zrx.streamlit.app/')

        else:
            st.markdown('Try with appropriate instruction to generate message!')

    st.caption(
        "This is a Beta version of the product. Our goal is to help build AI systems that could aid in Pharma promotions."
        " Please let us know your feedback at info@zoomrx.com")
