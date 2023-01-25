import streamlit as st

from constants import BENCHMARKS, IMAGE
from features import list_messages

st.set_page_config(
    page_title="Message Effectiveness Prediction",
    page_icon=IMAGE,
    initial_sidebar_state="expanded",
    layout="wide"
)

st.markdown(
    "<h2 style='text-align: center;'>Message Effectiveness Prediction Application</h2>",
    unsafe_allow_html=True)

prompt = st.text_input('Type in the instruction for message generation')
st.caption('Sample Instruction')
st.code('"Create messages for Opdivo in NSCLC using : In 14.1 months, half the people were alive on '
        'OPDIVO + YERVOY and platinum-based chemotherapy"')
if prompt:
    print(prompt)
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
            st.dataframe(data=res, use_container_width=False)
            st.header('Industry Averages')
            for key in ['Believability', 'Differentiation', 'Motivation', 'Overall_Score']:
                st.metric(key, BENCHMARKS[key])
    else:
        st.markdown('Try someother instruction, or use the same sometime later!')
