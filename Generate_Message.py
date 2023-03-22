import re
import webbrowser

import streamlit as st

import constants
from features import load_page_with_components

st.set_page_config(
    page_title="Message Effectiveness Prediction",
    page_icon=constants.LOGO,
    initial_sidebar_state="expanded",
    layout="wide"
)

email = st.experimental_get_query_params().get('email')
if not email:
    email = st.session_state.get('email')
if not email:
    email_text_container = st.empty()
    email = email_text_container.text_input(
        'Please enter your valid work Email ID to access the app')

    if len(re.findall(constants.EMAIL_VALIDATOR, email)) > 0:
        st.experimental_set_query_params(email=email)
        st.session_state.email = email
        email_text_container.empty()


def share_app():
    subject = 'Check out this cool Streamlit app!'
    body = f'Hi,\n\nI thought you might be interested in this Streamlit app: {constants.APP_URL} \n\nCheers!'
    webbrowser.open(f'mailto:?subject={subject}&body={body}')


if email:
    load_page_with_components(email)
    st.button(label='Share this App!', on_click=share_app, type='primary')
    st.caption(constants.FOOTER_TEXT)
