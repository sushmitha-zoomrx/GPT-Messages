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


def send_email(email):
    email_subject = "Check out this Expeimental AI Tool"
    email_body = f"Hi,\n\nPlease check out this experimental tool that utilizes AI to help generate branding messages and predict the message effectiveness scores - {constants.APP_URL}\n\nCheers!"
    gmail_url = f"https://mail.google.com/mail/u/0/?view=cm&fs=1&tf=1&to={email}&su={email_subject}&body={email_body}"
    html = f'<a href="{gmail_url}" target="_blank"><button type = "button" style = "background:#330933;color:white;height:30px;width:150px;">Share this App!</button></a>'
    return html


if email:
    load_page_with_components(email)
    st.components.v1.html(send_email(email))
    st.caption(constants.FOOTER_TEXT)
