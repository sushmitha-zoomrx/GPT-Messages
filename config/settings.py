from string import Template

import pandas as pd
import pygsheets
from google.oauth2 import service_account
from pydantic import BaseSettings
from sentence_transformers import SentenceTransformer
from xgboost import XGBRegressor


class Constants(BaseSettings):
    BELIEVABILITY_SCORE_PREDICTOR = XGBRegressor({'n_jobs': 4})
    BELIEVABILITY_SCORE_PREDICTOR.load_model("config/models/0001.model")
    MOTIVATION_SCORE_PREDICTOR = XGBRegressor({'n_jobs': 4})
    MOTIVATION_SCORE_PREDICTOR.load_model("config/models/0002.model")
    DIFFERENTIATION_SCORE_PREDICTOR = XGBRegressor({'n_jobs': 4})
    DIFFERENTIATION_SCORE_PREDICTOR.load_model("config/models/0003.model")
    EMBEDDING_MODEL = SentenceTransformer('distilbert-base-nli-mean-tokens')
    CSV_DATA = pd.read_csv('config/input_files/data.csv')
    GPT_API_KEY = "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"
    GPT_PROMPT = Template("""
    Please focus on the question intent closely. Based on he below questions, give me the appropriate output:
    1. If the Question is asking for messages generation, give me a python dictionary:
    {"messages":["messages that are generated for given question, give them as a list of strings in python format, without any numberings"],
    "category": 1 (To indicate that it is falling into this category)

    2. If the Question is asking for messages score prediction, give me a python dictionary:
    {"messages":["Extract actual messages(given inside "") that are to be predicted from the given question, must not generate additional text other than the messages given in the question"],
    "category": 2 (To indicate that it is falling into this category)

    3. If it doesnot fall into the above mentioned category, Just answer to the question and give me output in similar format, as mentioned below:
    {
    "messages": ["actual answer"],
    "category": 3 (To indicate that it is falling into this category)
    }

    If the ask is not clear, ask them the rephrase the question saying, "The scope of this tool is to create/predict scores for messages. Please rephrase your question accordingly to help you on your ask."
    
    Input Question:
    $question
    """)
    PERCENTILE_TEXT = (
        "To improve its performance, use precise messages with data & qualifiers as relevant."
        "[Read more>>](https://learnmore.zoomrx.com/pet-registration)"
    )
    FOOTER_TEXT = (
        "This is a Beta version of the product. Our goal is to help build AI systems that could aid in Pharma "
        "promotions. Please let us know your feedback at info@zoomrx.com")
    credentials = service_account.Credentials.from_service_account_info(
        {
            "type": "service_account",
            "project_id": "sushi-j",
            "private_key_id": "04c3fc5c8b9bcb9f8ef22ff66c9f018f912bef24",
            "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQDCaa3XZ3YWyr/v\nesCPDifvFQKyLzUfiVsRclfuMTojiytOE7li1au8DlYRFNAxnADDJglW78imRXq/\nDwjthj2WF+h9AsT3HnITJ0Kdp+xQ4IGBFAGkdndtJg3Ti+vtlboy6lAO8nKpKU72\nAqlEWXLhAHVQci0RcLvSJRfhZdfB6QnessRxO1mRcanCvXmAkKkxGmS1QJF4h1+d\nj6Awrnt3Tz+zOwruBDrPGxF5/xXpU0pkP+xx0Pn0Squ1aYur6r9jLyaVsHcHwrPe\nSescdSiQPO/gnrCI2yHK28cRq2ES7I+S12zl1MYKZjHoOKEUpcHyyvi6ZtWO2KPs\nICObPexnAgMBAAECggEAGZ5LZiMgEIjPGgOW9ELtSDgAjvJhkbJL6dSjeaPLAXwJ\nTNSUjU2Sv8kz1jRj6uWfxBdzC521VxO4xQx6JLKX0vt7i12eDuJYLeXyhUvnfBDZ\nf+TfAokJ27wz/jhl8nzUeHkf65hPO7NR0GExZOxUkwm4a81f2vh8B4kTyIPfFpIf\n0HMcV5Fo3gb5E9LkD0QEaVGyFn234BH7LMzCFKM0XZT17hzl7nQ2H5+EVTL12Z0F\nKPShEmW7l+8ddBB7LSWAu+HOb4mVBS1f1ZCCkTjas2eb2jIFhShZ+AoD0mX7M9/T\nwmrUhpMqdetcI8MfaTCBc/sznSFMyCMEauqdVWy7YQKBgQDzrsD09CT5RDh89pjH\nUQ14605IC3ZWJZtmzMMpXh5pviJ9Kkwbwb+vEvqqp4OMAJxk1f9O/KVeeq8rczRT\nHry/1A6s6/RN+9M9z3CpRsRf2z9qXt/ePf7M+Dm/YKKDRFG4q+g8Jo/09lbqPRCb\naWwpArWYhxegi/TSdtuRsxbZpQKBgQDMPWAPcFkev0uy8Ivbva4Azb0SGsvSVVtV\nsdpe1E2jbT+gpn4rbWtruZaAiAoXllbjezpFRJc2hQdcl6RYwVT0BDO7593v7lby\nHgQO6lboW8EwjgV/6IIBGS9BTlP+mM0gHSoIvwpyqWiBM+Sya8pqcU8iDuFZ9q7M\nqGbUiz+YGwKBgQCVlrZe6Kz109o1ZA/fczMpApHYiijHs2hVT+eSMnPLB+wWF+wG\nsgZgi+8S6ahIPmvDPtbufwtpFzkHHD6Hs/u8aonjvykG4ksHy5rmX0nXajjgrIMS\n483Rt6ODhufcWwkrq2Px4N5ISxyJyJi0PqAmAMLHck6fwKq2tD4Pj/e7/QKBgC/5\nUrENsMFaKcvUWOW6vj6OFRVFmg7D4fpVFngj4kC7DrELqqNExnC9XS6/xa8Yrzwr\n29odbG9v+/Sx4fa/ItdWjVhb9HPBRkcE6ese/F8D/nMLSRtsX+0mH0V1wqEQ/03F\ny/PV+/xG8rc2m0eVriwmhXH4kNJy8Ug9Xjoao0t1AoGBAOrkMXTAZK3aOqKILDi3\nq2380OPOnywipX3pfrNW2Dz0DEPizFPUoJJxt5046vkopP+NUNMQ45xEcgu7l7i5\n5SZfp6mofXaUZi5CeHP0ROkgtNNy/OBADKGhTyoMHqHRL1GRUFpwqOy04zj/EQxr\nj86fgencLAU73dYabSrypQfW\n-----END PRIVATE KEY-----\n",
            "client_email": "sushi-j-1@sushi-j.iam.gserviceaccount.com",
            "client_id": "103234480024712790951",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/sushi-j-1%40sushi-j.iam.gserviceaccount.com"
        },
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
        ],
    )
    GSHEET_URL = "https://docs.google.com/spreadsheets/d/1sQJm0z3yuTSTqpEg-vDRiV60vvl6SpCcyRb6iRMU3l0/edit"
    gc = pygsheets.authorize(custom_credentials=credentials)
    GSHEET = gc.open_by_url(GSHEET_URL)[0]

    HEADER_HTML = """
    <tr>
        <th style="text-align: center; min-width: 500px" rowspan="1">Message</th>\n
        <th rowspan="1">Believability</th>
        <th rowspan="1">Differentiation</th>
        <th rowspan="1">Motivation</th>
        <th rowspan="1">Overall Score</th>
        <th rowspan="2">Rank Percentile</th>
    </tr>
    <tr style="text-align: center;">\n
        <th>ZoomRx Industry Averages -></th>\n
        <th> 63 %</th>\n
        <th> 55 %</th>\n
        <th> 59 %</th>\n
        <th> 59 %</th>\n
    </tr>\n
    """


constants = Constants()
