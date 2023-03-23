import streamlit as st

import constants
from services import generate_messages


def list_messages(prompt: str):
    return generate_messages(prompt)


def load_page_with_components(email):

    with st.container():
        st.markdown(
            "<h2 style='text-align: center;'>Message Effectiveness Prediction Application</h2>",
            unsafe_allow_html=True)
        prompt_container = st.empty()
        prompt = prompt_container.text_input('You:')
        st.caption('Sample Instructions:')
        create_button = st.button(constants.DISPLAY_MESSAGES['create'])
        predict_button = st.button(constants.DISPLAY_MESSAGES['predict'])
        if create_button:
            prompt = constants.DISPLAY_MESSAGES['create']
            prompt_container.text_input('You:', constants.DISPLAY_MESSAGES['create'])
        elif predict_button:
            prompt = constants.DISPLAY_MESSAGES['predict']
            prompt_container.text_input('You:', constants.DISPLAY_MESSAGES['predict'])

        if prompt:

            res = list_messages(prompt)

            if res is not None:
                st.markdown(constants.HIDE_TABLE_ROW_INDEX, unsafe_allow_html=True)
                st.table(res)
                st.markdown(constants.PERCENTILE_TEXT)
                constants.GSHEET.append_table(values=[f'{email[0]}', prompt, res.to_string()])
            else:
                st.markdown('Try with appropriate instruction to generate message!')
                constants.GSHEET.append_table(values=[f'{email[0]}', prompt, 'No result'])
