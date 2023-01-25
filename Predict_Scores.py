import streamlit as st

from constants import IMAGE
from features import get_scores

st.set_page_config(
    page_title="Message Effectiveness Prediction",
    page_icon=IMAGE,
    initial_sidebar_state="expanded",
    layout="wide"
)

st.markdown(
    "<h2 style='text-align: center;'>Message Effectiveness Prediction Application</h2>",
    unsafe_allow_html=True)


message = st.text_input('Input the message to predict the Message Effectiveness scores')
st.markdown('Sample Messge:')
st.code('"LYBALVI offers the efficacy demonstrated by olanzapine in adequate and '
        'well-controlled studies in bipolar I disorder"')

if message:
    scores = get_scores(message)
    benchmarks = {
        "Believability": "63 %",
        "Differentiation": "55 %",
        "Motivation": "59 %",
        "Overall_Score": "59 %"
    }
    col1, col2 = st.columns(2)
    with col1:
        st.header('Predicted Scores')

        for key in ['Believability', 'Differentiation', 'Motivation', 'Overall_Score']:
            col1.metric(key, scores[key])
    with col2:
        st.header('Industry Averages')
        for key in ['Believability', 'Differentiation', 'Motivation', 'Overall_Score']:
            col2.metric(key, benchmarks[key])

    percentile_text = (
        f"Your message ranks in **:green[{scores['Rank_Percentile']}]** Percentile. To improve its performance, "
        "use precise messages with data & qualifiers as relevant."
        "[Read more>>](https://learnmore.zoomrx.com/pet-registration)"
    )
    st.markdown(percentile_text)

    if st.button('Share App Link'):
        st.markdown('https://me-prediction-zrx.streamlit.app/')

st.caption(
    "This is a Beta version of the product. Our goal is to help build AI systems that could aid in Pharma promotions."
    " Please let us know your feedback at info@zoomrx.com")
