from string import Template
from typing import List, ClassVar
import os
import pandas as pd
import pygsheets
from dotenv import load_dotenv
from google.oauth2 import service_account
from pydantic_settings import BaseSettings
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBRegressor
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import pickle


# Load environment variables
load_dotenv()


class Constants(BaseSettings):
    BELIEVABILITY_SCORE_PREDICTOR: ClassVar[XGBRegressor] = pickle.load(
        open("config/models/B.pkl", "rb"))
    MOTIVATION_SCORE_PREDICTOR: ClassVar[XGBRegressor] = pickle.load(
        open("config/models/M.pkl", "rb"))
    DIFFERENTIATION_SCORE_PREDICTOR: ClassVar[XGBRegressor] = pickle.load(
        open("config/models/D.pkl", "rb"))
    TFIDF_VECTORIZER_LESS_FEATURES: ClassVar[TfidfVectorizer] = TfidfVectorizer(
        max_features=10, stop_words="english")
    TFIDF_VECTORIZER_MORE_FEATURES: ClassVar[TfidfVectorizer] = TfidfVectorizer(
        max_features=200, stop_words="english")
    CSV_DATA: ClassVar[pd.DataFrame] = pd.read_csv('config/input_files/benchmark_data.csv')
    GPT_API_KEY: str = os.getenv('OPENAI_API_KEY')
    GPT_PROMPT: ClassVar[Template] = Template("""
    Please focus on the question intent closely. Based on the below questions, give me the appropriate output:
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
    CHAT_GPT_PROMPT: ClassVar[Template] = Template("""Guidelines:
            1] For any message given, your task is to classify the same under each below fields.
            2] Understand the description of each and answer the same.
            3] Leave fields empty when we have no data
            4] Shouldnt list any words that's out of respective message
            5] Output the same in json format.

            Description:
            0] message: Pharma specific message from the given input message
            1] word_count: Number of words in each messages. [number of words]
            2] char_count: Number of character in each messages. [number of chars]
            3] average_char: divide char_count/word_count [round upto two decimal values]
            4] comparator: Identifies words or phrases that explicitly indicate comparison between two or more entities. (e.g., 'earlier' ,'lowest', 'comparable', 'better', 'versus', 'vs').This also includes implicit comparisons, such as when multiple numerical values are presented for different groups (e.g., different dosages, treatment arms, or placebo comparisons).  [list them all]
            5] passive_voice: Flags usage of passive voice, though identifying this in short text might require natural language processing. [1 for passive and 0 for active]
            6] data_points: Counts the number of explicit data or quantifiable statistical datapoints (dosages, percentages, survival rates, or study results) (e.g., dosage amounts, '52 week safety data'). [datapoints and not metrics]  [count of it]
	    7] metric_count: Counts the number of any metrics which are metrics that HCPs care about. [count of it]
            8] brand_name: Identify mention of brand names alone. (e.g., 'Vivlodex', 'Zorvolex').  [list them all]
            9] specific_condition: References to medical conditions (e.g., 'osteoarthritis', 'acute pain'). [list them all]
            10] action_words: Messages with action verbs or calls to action (e.g., 'use', 'provides', 'delivers')  [list them all]
            11] action_words_count: Number of action_words in each messages. [Number]
            12] unique_selling_proposition:  Make sure to identify the differentiators that positively highlights the product’s benefits and favours selling point  (e.g., 'earliest median time to Tmax', 'lowest effective daily dose', 'novel', 'first-in-class'). This can include Clinical endorsements, guideline recommendations, distinctive treatment advantages,established patient management approaches, innovative mechanisms of action. Exclude risk warnings, adverse effects and anything negative. Be precise. [list them all]
            13] tone: Flags as Positive (benefits/improvements), Neutral (objective facts), Negative (risks/unfavorable outcomes) tone. [positive, neutral, negative]
            14] technical_terms: Flags presence of technical or medical terms.  [list them all]
            15] presence_of_datapoints: Identify where the explicit data or statistics datapoints on the message arrives: First, middle, Last [First, middle, last][only if we had valid data_points]
            16] emotion_elements:  Make sure to analyze messages and get the emotional elements . [list them all]
            17] word_complexities: Identifies words that may require additional cognitive processing or clarification, even for HCPs. This excludes common medical descriptors (e.g., fatal, severe), but includes multi-syllabic terms, abstract concepts, or words that might have nuanced interpretations in a clinical setting. No technical terms should be considered here. [list them all]
            18] readability_score (Flesch-Kincaid or Automated Readability Index):
Calculate the readability score of the message using either the Flesch-Kincaid Grade Level formula. Output the result as a grade level (e.g., 8.5, indicating an 8th-grade reading level).  Ensure the calculation accounts for the number of words, sentences, and syllables (for Flesch-Kincaid) or characters (for ARI) in the message text. Exclude any non-text elements (e.g., images, charts) from the calculation.
            19] call_to_action: Determine whether the message contains a clear call to action (e.g., a directive like 'prescribe this treatment', 'contact us for more information' or 'review the study'). Output the result as a binary value: 1 (Yes, a call to action is present) or 0 (No, a call to action is absent). Identify a call to action by detecting imperative verbs (e.g: 'prescribe,' 'learn,' 'visit') combined with a specific request or instruction relevant to the message’s purpose. Ignore vague or implied actions (e.g: 'this is important') unless explicitly tied to a directive.

 Input Message: $message

            OUTPUT FORMAT[JSON]:
            {
                [0]: branding message from input text
                [1]: word_count,
                [2]: char_count,
                [3]: average_char,
                [4]: comparator,
                [5]: passive_voice,
                [6]: data_points,
                [7]: metric_count,
                [8]: brand_name,
                [9]: specific_condition,
                [10] action_words,:
                [11]: action_words_count,
                [12]: unique_selling_proposition,
                [13]: tone,
                [14]: technical_terms,
                [15]: presence_of_datapoints,
                [16]: emotion_elements,
                [17]: word_complexities,
                [18]: readability_score,
                [19]:  call_to_action
            }""")
    EVALUATION_PROMPT: ClassVar[Template] = Template("""You are an expert pharmaceutical marketing analyst with extensive experience in evaluating message effectiveness for pharmaceutical brands. You will analyze marketing messages for pharmaceutical products and provide detailed evaluations.

    MESSAGE TO ANALYZE:
    $message

    EVALUATION INSTRUCTIONS:
    Please analyze this pharmaceutical message across three primary dimensions:

    1. MOTIVATION (Scale 0-100)
    - Does this message motivate the target audience to take action?
    - Does it address key needs or pain points?
    - Does it create a sense of urgency where appropriate?
    - Does it clearly communicate value for the target audience?

    2. BELIEVABILITY (Scale 0-100)
    - How credible and trustworthy are the claims?
    - Are claims appropriately substantiated or qualified?
    - Is the message consistent with the product's clinical profile?
    - Does it avoid overpromising or using hyperbole?

    3. DIFFERENTIATION (Scale 0-100)
    - How effectively does the message distinguish the product from competitors?
    - Does it highlight unique aspects or benefits?
    - Would the audience understand why to choose this product over alternatives?
    - Does it leverage the product's unique positioning?

    REGULATORY ANALYSIS:
    - Identify any potential regulatory compliance issues
    - Note any problematic claims or language
    - Flag any fair balance concerns
    - Highlight potential MLR review issues

    ADDITIONAL ANALYSIS:
    - Linguistic strengths and weaknesses
    - Emotional resonance with target audience
    - Structural effectiveness of the message
    - Clarity and simplicity of communication

    FORMAT YOUR RESPONSE AS FOLLOWS IN A PROPER JSON CONSUMABLE STRUCTURE IN SOFTWARE CODES:
    {
        "scores": {
            "motivation": 0,
            "believability": 0,
            "differentiation": 0,
            "overall_effectiveness": 0
        },
        "strengths": [
            "Strength `n`"
        ],
        "weaknesses": [
            "Weakness `n`"
        ],
        "regulatory_concerns": [
            "Regulatory concern `n`"
        ],
        "detailed_analysis": "Provide a max of 100 word analysis explaining your scores and reasoning",
        "improvement_recommendations": [
            "Recommendation n"
        ],
        "improvised_messages": [
            "Improvised message n"
        ]
    }
    *** Details on output ***
    SCORES:
    - Motivation: [0-100]
    - Believability: [0-100]
    - Differentiation: [0-100]
    - Overall Effectiveness: [0-100]

    STRENGTHS:
    - [List 1-3 major strengths of the message]

    WEAKNESSES:
    - [List 1-3 major weaknesses of the message]

    REGULATORY CONCERNS:
    - [List any potential regulatory issues. Yes/No Is enough. If yes, then provide details. ]

    DETAILED ANALYSIS:
    [Provide a max of 100 word analysis explaining your scores and reasoning]

    IMPROVEMENT RECOMMENDATIONS:
    - [List 1-3 specific, actionable recommendations to improve the message]

    IMPROVISED MESSAGES:
    - [List 1-3 improved versions of the message that address the weaknesses and enhance strengths, in not more than 25 words. Donot use call, contact, support kind of instructions.]

    Note:
    - There may be labels mentioned in the message like ([Product], [Trial Acronym]). Use them as it is with the associated phrases.

    *** End of Details ***
    """)
    DISCREPANCY_THRESHOLD: int = 20
    REQUIRED_FEATURES: dict = {
        "Motivation": ['brand_name', 'technical_terms', 'specific_condition', 'metric_count', 'tone', 'action_words', 'bert_33', 'bert_46', 'word_complexities', 'bert_91', 'bert_37', 'bert_20', 'bert_3', 'presence_of_datapoints', 'bert_4', 'bert_60', 'unique_selling_proposition', 'bert_0', 'readability_score', 'bert_1', 'word_count', 'bert_85', 'bert_40', 'bert_43', 'bert_22', 'bert_87', 'bert_95', 'sentiment', 'bert_76', 'tfidf_129', 'bert_62', 'bert_44', 'bert_82', 'bert_53', 'tfidf_199', 'tfidf_109', 'bert_17', 'tfidf_57', 'tfidf_144', 'tfidf_81', 'bert_52', 'tfidf_37', 'tfidf_45', 'tfidf_139', 'bert_58', 'tfidf_158', 'bert_5', 'tfidf_142', 'tfidf_23', 'tfidf_8', 'bert_99', 'tfidf_71', 'tfidf_15', 'tfidf_157', 'comparator', 'bert_75', 'bert_31', 'tfidf_193', 'bert_86', 'tfidf_66', 'bert_90', 'tfidf_153', 'tfidf_135', 'tfidf_113', 'bert_49', 'bert_80', 'tfidf_42', 'tfidf_128', 'tfidf_125', 'bert_66', 'bert_41', 'bert_50', 'bert_34', 'tfidf_70', 'tfidf_68', 'bert_6', 'bert_10', 'tfidf_75', 'tfidf_174', 'tfidf_106', 'tfidf_80', 'tfidf_103', 'tfidf_180', 'tfidf_92', 'bert_69', 'tfidf_10', 'bert_21', 'bert_14', 'tfidf_164', 'tfidf_94', 'tfidf_121', 'tfidf_175', 'tfidf_95', 'tfidf_78', 'tfidf_59', 'tfidf_177', 'tfidf_156', 'tfidf_195', 'tfidf_141', 'datapoint_count', 'tfidf_148', 'tfidf_12', 'tfidf_85', 'tfidf_131', 'tfidf_197', 'tfidf_47', 'tfidf_3', 'bert_92', 'tfidf_84', 'tfidf_143', 'tfidf_49', 'passive_voice', 'tfidf_155', 'tfidf_28', 'action_words_count', 'tfidf_123', 'tfidf_22', 'tfidf_16', 'tfidf_61', 'tfidf_105', 'bert_63', 'tfidf_89', 'tfidf_152', 'bert_59', 'tfidf_32', 'tfidf_118', 'tfidf_159', 'tfidf_130', 'bert_64', 'bert_83', 'tfidf_120', 'tfidf_146', 'bert_19', 'bert_65', 'tfidf_114', 'tfidf_18', 'tfidf_149', 'tfidf_51', 'bert_28', 'tfidf_117', 'tfidf_145', 'tfidf_184', 'char_count', 'tfidf_27', 'bert_8', 'tfidf_1', 'tfidf_36', 'tfidf_50', 'bert_18', 'tfidf_127', 'tfidf_65', 'bert_71', 'bert_68', 'tfidf_93', 'tfidf_52', 'tfidf_6', 'tfidf_77', 'tfidf_185', 'tfidf_79', 'bert_45', 'tfidf_44', 'tfidf_178', 'bert_12', 'tfidf_111', 'tfidf_176', 'tfidf_136', 'tfidf_62', 'call_to_action', 'tfidf_124', 'tfidf_140', 'tfidf_60', 'tfidf_189', 'tfidf_196', 'tfidf_104', 'tfidf_55', 'bert_89', 'bert_2', 'bert_7', 'tfidf_86', 'tfidf_91', 'tfidf_169', 'tfidf_101', 'tfidf_192', 'tfidf_90', 'tfidf_63', 'tfidf_138', 'tfidf_137', 'tfidf_25', 'tfidf_39', 'tfidf_24', 'tfidf_172', 'tfidf_98', 'tfidf_168', 'tfidf_166', 'tfidf_21'],
        "Believability": ['unique_selling_proposition', 'specific_condition_tfidf_9', 'tfidf_95', 'tfidf_83', 'technical_terms_tfidf_3', 'tfidf_134', 'tfidf_131', 'word_complexities_tfidf_2', 'brand_name_tfidf_3', 'technical_terms_tfidf_0', 'tfidf_171', 'tfidf_35', 'tfidf_168', 'tfidf_66', 'specific_condition_tfidf_7', 'tfidf_160', 'brand_name_tfidf_6', 'action_words_tfidf_9', 'tfidf_27', 'brand_name_tfidf_7', 'tfidf_29', 'tfidf_112', 'tfidf_47', 'tfidf_143', 'tfidf_136', 'tfidf_12', 'action_words_tfidf_6', 'tfidf_175', 'tfidf_22', 'tfidf_82', 'tfidf_65', 'tfidf_165', 'tfidf_69', 'tfidf_125', 'tfidf_1', 'tfidf_38', 'tfidf_7', 'comparator', 'tfidf_100', 'tfidf_183', 'tfidf_155', 'tfidf_193', 'tfidf_71', 'word_complexities_tfidf_3', 'tfidf_43', 'tfidf_110', 'tfidf_198', 'tfidf_26', 'tfidf_67', 'technical_terms_tfidf_8', 'tfidf_56', 'technical_terms_tfidf_5', 'tfidf_178', 'tfidf_166', 'tfidf_53', 'tfidf_149', 'tfidf_181', 'tfidf_126', 'tfidf_14', 'tfidf_197', 'tfidf_6', 'specific_condition_tfidf_2', 'tfidf_70', 'tfidf_23', 'tfidf_144', 'tfidf_158', 'tfidf_192', 'tfidf_142', 'tfidf_182', 'tfidf_186', 'tfidf_87', 'tfidf_91', 'tfidf_51', 'tfidf_92', 'specific_condition_tfidf_3', 'tfidf_116', 'tfidf_33', 'tfidf_173', 'technical_terms_tfidf_4', 'technical_terms_tfidf_2', 'emotion_elements', 'tfidf_161', 'tfidf_76', 'tfidf_49', 'tfidf_190', 'tfidf_89', 'tfidf_146', 'tfidf_17', 'tfidf_148', 'tfidf_130', 'word_complexities_tfidf_9', 'tfidf_109', 'tfidf_108', 'tfidf_25', 'tfidf_138', 'tfidf_9', 'action_words_tfidf_7', 'tfidf_151', 'tfidf_121', 'tfidf_169', 'tfidf_157', 'tfidf_195', 'tfidf_107', 'brand_name_tfidf_9', 'tfidf_59', 'tfidf_24', 'tfidf_196', 'tfidf_80', 'tfidf_73', 'tfidf_84', 'tfidf_104', 'readability_score', 'tfidf_124', 'specific_condition_tfidf_5', 'tfidf_156', 'tfidf_120', 'tfidf_5', 'tfidf_37', 'call_to_action', 'tfidf_52', 'tfidf_4', 'tfidf_135', 'tfidf_129', 'tfidf_163', 'tfidf_63', 'brand_name_tfidf_0', 'action_words_tfidf_0', 'tfidf_159', 'tfidf_128', 'tfidf_79', 'tfidf_93', 'action_words_tfidf_1', 'tfidf_140', 'specific_condition_tfidf_8', 'tfidf_180', 'word_complexities_tfidf_8', 'tfidf_97', 'tfidf_30', 'tfidf_41', 'tfidf_85', 'word_complexities_tfidf_0', 'tfidf_8', 'tfidf_2', 'tfidf_123', 'tfidf_10', 'action_words_tfidf_5', 'tfidf_3', 'tfidf_118', 'tfidf_153', 'tfidf_199', 'tfidf_50', 'tfidf_147', 'tfidf_11', 'char_count', 'tfidf_74', 'specific_condition_tfidf_0', 'average_char_count', 'brand_name_tfidf_2', 'tfidf_45', 'technical_terms_tfidf_9', 'tfidf_176', 'tfidf_177', 'tfidf_86', 'tfidf_60', 'tfidf_154', 'tfidf_48', 'tfidf_13', 'tfidf_55', 'tfidf_81', 'tfidf_170', 'tfidf_103', 'tfidf_162', 'tfidf_141', 'tfidf_139', 'tfidf_137', 'presence_of_datapoints', 'passive_voice', 'tfidf_132', 'tfidf_127', 'tfidf_122', 'tfidf_133', 'tfidf_145', 'tfidf_150', 'tfidf_119', 'tfidf_106', 'tfidf_105', 'tfidf_164', 'tfidf_152', 'tone', 'action_words_count', 'metric_count', 'datapoint_count', 'tfidf_18', 'tfidf_16', 'tfidf_19', 'tfidf_20', 'word_complexities_tfidf_7', 'tfidf_21', 'word_count', 'tfidf_15'],
        "Differentiation": ['comparator', 'tfidf_126', 'tfidf_134', 'tfidf_168', 'tfidf_113', 'unique_selling_proposition', 'tfidf_179', 'readability_score', 'tfidf_125', 'tfidf_1', 'tfidf_34', 'tfidf_151', 'tfidf_27', 'technical_terms_tfidf_5', 'tfidf_85', 'tfidf_80', 'passive_voice', 'tfidf_70', 'tfidf_131', 'tfidf_54', 'tfidf_19', 'tfidf_49', 'tfidf_6', 'technical_terms_tfidf_1', 'call_to_action', 'tfidf_29', 'tfidf_172', 'tfidf_10', 'tone', 'tfidf_68', 'tfidf_118', 'tfidf_165', 'tfidf_30', 'tfidf_177', 'tfidf_137', 'tfidf_74', 'action_words_tfidf_0', 'tfidf_152', 'brand_name_tfidf_2', 'tfidf_81', 'technical_terms_tfidf_4', 'tfidf_198', 'action_words_tfidf_5', 'char_count', 'technical_terms_tfidf_6', 'tfidf_148', 'tfidf_5', 'technical_terms_tfidf_9', 'tfidf_146', 'tfidf_140', 'tfidf_7', 'tfidf_167', 'datapoint_count', 'tfidf_42', 'tfidf_133', 'specific_condition_tfidf_7', 'tfidf_159', 'technical_terms_tfidf_7', 'tfidf_180', 'tfidf_121', 'specific_condition_tfidf_1', 'tfidf_47', 'technical_terms_tfidf_2', 'tfidf_139', 'tfidf_161', 'tfidf_3', 'tfidf_82', 'word_complexities_tfidf_1', 'tfidf_103', 'specific_condition_tfidf_4', 'word_complexities_tfidf_6', 'brand_name_tfidf_0', 'tfidf_53', 'tfidf_79', 'action_words_tfidf_7', 'metric_count', 'tfidf_58', 'tfidf_135', 'brand_name_tfidf_6', 'tfidf_50', 'specific_condition_tfidf_3', 'tfidf_190', 'brand_name_tfidf_7', 'tfidf_69', 'tfidf_108', 'tfidf_31', 'tfidf_55', 'tfidf_35', 'tfidf_105', 'brand_name_tfidf_3', 'tfidf_169', 'tfidf_183', 'word_count', 'action_words_tfidf_9', 'tfidf_63', 'tfidf_2', 'tfidf_122', 'tfidf_86', 'tfidf_138', 'tfidf_93', 'tfidf_199', 'tfidf_89', 'word_complexities_tfidf_4', 'tfidf_164', 'tfidf_20', 'tfidf_143', 'tfidf_41', 'presence_of_datapoints', 'word_complexities_tfidf_5', 'action_words_tfidf_2', 'tfidf_66', 'tfidf_99', 'tfidf_136', 'brand_name_tfidf_1', 'tfidf_141', 'tfidf_196', 'tfidf_186', 'tfidf_40', 'tfidf_154', 'tfidf_192', 'tfidf_189', 'tfidf_33', 'tfidf_4', 'tfidf_184', 'tfidf_17', 'tfidf_191', 'tfidf_104', 'brand_name_tfidf_8', 'tfidf_185', 'action_words_count', 'tfidf_22', 'tfidf_150', 'specific_condition_tfidf_5', 'tfidf_170', 'tfidf_13', 'tfidf_160', 'word_complexities_tfidf_7', 'tfidf_130', 'tfidf_36', 'tfidf_23', 'tfidf_72', 'tfidf_158', 'tfidf_128', 'action_words_tfidf_1', 'tfidf_46', 'tfidf_102', 'tfidf_87', 'tfidf_119', 'tfidf_37', 'tfidf_187', 'tfidf_95', 'word_complexities_tfidf_3', 'tfidf_106', 'tfidf_84', 'tfidf_91', 'tfidf_115', 'tfidf_76', 'tfidf_100', 'tfidf_117', 'tfidf_120', 'tfidf_111', 'tfidf_112', 'tfidf_97', 'tfidf_98', 'tfidf_96', 'tfidf_107', 'tfidf_109', 'tfidf_110', 'tfidf_60', 'tfidf_59', 'tfidf_73', 'tfidf_75', 'tfidf_12', 'tfidf_8', 'tfidf_9', 'tfidf_11', 'tfidf_18', 'tfidf_16', 'tfidf_14', 'word_complexities_tfidf_8', 'average_char_count', 'word_complexities_tfidf_9', 'tfidf_52', 'tfidf_56', 'tfidf_15', 'tfidf_101', 'tfidf_71', 'tfidf_67', 'tfidf_65', 'tfidf_61', 'tfidf_62', 'tfidf_64', 'tfidf_57', 'tfidf_83', 'tfidf_77', 'tfidf_78', 'tfidf_90', 'tfidf_88', 'tfidf_92', 'tfidf_94']
    }
    VECTORIZE_COLUMNS: list = [
        "brand_name",
        "specific_condition",
        "action_words",
        "technical_terms",
        "word_complexities",
    ]
    PERCENTILE_TEXT: str = (
        "To improve its performance, use precise messages with data & qualifiers as relevant."
        "[Read more>>](https://learnmore.zoomrx.com/pet-registration)"
    )
    FOOTER_TEXT: str = (
        "This is a Beta version of the product. Our goal is to help build AI systems that could aid in Pharma "
        "promotions. Please let us know your feedback at info@zoomrx.com")
    # credentials: ClassVar[service_account.Credentials] = service_account.Credentials.from_service_account_info(
    #     {
    #         "type": "service_account",
    #         "project_id": "sushi-j",
    #         "private_key_id": "04c3fc5c8b9bcb9f8ef22ff66c9f018f912bef24",
    #         "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQDCaa3XZ3YWyr/v\nesCPDifvFQKyLzUfiVsRclfuMTojiytOE7li1au8DlYRFNAxnADDJglW78imRXq/\nDwjthj2WF+h9AsT3HnITJ0Kdp+xQ4IGBFAGkdndtJg3Ti+vtlboy6lAO8nKpKU72\nAqlEWXLhAHVQci0RcLvSJRfhZdfB6QnessRxO1mRcanCvXmAkKkxGmS1QJF4h1+d\nj6Awrnt3Tz+zOwruBDrPGxF5/xXpU0pkP+xx0Pn0Squ1aYur6r9jLyaVsHcHwrPe\nSescdSiQPO/gnrCI2yHK28cRq2ES7I+S12zl1MYKZjHoOKEUpcHyyvi6ZtWO2KPs\nICObPexnAgMBAAECggEAGZ5LZiMgEIjPGgOW9ELtSDgAjvJhkbJL6dSjeaPLAXwJ\nTNSUjU2Sv8kz1jRj6uWfxBdzC521VxO4xQx6JLKX0vt7i12eDuJYLeXyhUvnfBDZ\nf+TfAokJ27wz/jhl8nzUeHkf65hPO7NR0GExZOxUkwm4a81f2vh8B4kTyIPfFpIf\n0HMcV5Fo3gb5E9LkD0QEaVGyFn234BH7LMzCFKM0XZT17hzl7nQ2H5+EVTL12Z0F\nKPShEmW7l+8ddBB7LSWAu+HOb4mVBS1f1ZCCkTjas2eb2jIFhShZ+AoD0mX7M9/T\nwmrUhpMqdetcI8MfaTCBc/sznSFMyCMEauqdVWy7YQKBgQDzrsD09CT5RDh89pjH\nUQ14605IC3ZWJZtmzMMpXh5pviJ9Kkwbwb+vEvqqp4OMAJxk1f9O/KVeeq8rczRT\nHry/1A6s6/RN+9M9z3CpRsRf2z9qXt/ePf7M+Dm/YKKDRFG4q+g8Jo/09lbqPRCb\naWwpArWYhxegi/TSdtuRsxbZpQKBgQDMPWAPcFkev0uy8Ivbva4Azb0SGsvSVVtV\nsdpe1E2jbT+gpn4rbWtruZaAiAoXllbjezpFRJc2hQdcl6RYwVT0BDO7593v7lby\nHgQO6lboW8EwjgV/6IIBGS9BTlP+mM0gHSoIvwpyqWiBM+Sya8pqcU8iDuFZ9q7M\nqGbUiz+YGwKBgQCVlrZe6Kz109o1ZA/fczMpApHYiijHs2hVT+eSMnPLB+wWF+wG\nsgZgi+8S6ahIPmvDPtbufwtpFzkHHD6Hs/u8aonjvykG4ksHy5rmX0nXajjgrIMS\n483Rt6ODhufcWwkrq2Px4N5ISxyJyJi0PqAmAMLHck6fwKq2tD4Pj/e7/QKBgC/5\nUrENsMFaKcvUWOW6vj6OFRVFmg7D4fpVFngj4kC7DrELqqNExnC9XS6/xa8Yrzwr\n29odbG9v+/Sx4fa/ItdWjVhb9HPBRkcE6ese/F8D/nMLSRtsX+0mH0V1wqEQ/03F\ny/PV+/xG8rc2m0eVriwmhXH4kNJy8Ug9Xjoao0t1AoGBAOrkMXTAZK3aOqKILDi3\nq2380OPOnywipX3pfrNW2Dz0DEPizFPUoJJxt5046vkopP+NUNMQ45xEcgu7l7i5\n5SZfp6mofXaUZi5CeHP0ROkgtNNy/OBADKGhTyoMHqHRL1GRUFpwqOy04zj/EQxr\nj86fgencLAU73dYabSrypQfW\n-----END PRIVATE KEY-----\n",
    #         "client_email": "sushi-j-1@sushi-j.iam.gserviceaccount.com",
    #         "client_id": "103234480024712790951",
    #         "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    #         "token_uri": "https://oauth2.googleapis.com/token",
    #         "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    #         "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/sushi-j-1%40sushi-j.iam.gserviceaccount.com"
    #     },
    #     scopes=[
    #         "https://www.googleapis.com/auth/spreadsheets",
    #     ],
    # )
    # GSHEET_URL: str = "https://docs.google.com/spreadsheets/d/1sQJm0z3yuTSTqpEg-vDRiV60vvl6SpCcyRb6iRMU3l0/edit"
    # gc: ClassVar[pygsheets.authorize] = pygsheets.authorize(custom_credentials=credentials)
    # GSHEET: ClassVar[pygsheets.Worksheet] = gc.open_by_url(GSHEET_URL)[0]
    # Column for message text - <th style="text-align: center; min-width: 500px" rowspan="1">Message</th>\n
    # 2nd row's column for message test - <th>ZoomRx Industry Averages -></th>\n
    HEADER_HTML: str = """
    <tr>
        <th rowspan="1">Believability</th>
        <th rowspan="1">Differentiation</th>
        <th rowspan="1">Motivation</th>
        <th rowspan="1">Overall ME Score</th>
        <th rowspan="2">Rank Percentile</th>
    </tr>
    <tr style="text-align: center;">\n
        <th> 63 %</th>\n
        <th> 55 %</th>\n
        <th> 59 %</th>\n
        <th> 59 %</th>\n
    </tr>\n
    """

    class QuestionIntent(BaseModel):
        messages: List[str] = Field(
            description=(
                '- messages that are generated for given question, without any numberings \n'
                '- actual messages(given inside "") that are to be predicted from the given question \n'
                '- any response to the user regarding the given question'
            )
        )
        category: int = Field(
            description=(
                '1 (If the ask is message generation) or 2 (If the ask is message score prediction) or '
                '3 (If the ask is not message generation and prediction)'
            )
        )

    CHAT_PROMPT_TEMPLATE: str = """
    You are a creative agency consultant helping Pharma brands with messaging to HCPs.

    Focus on the ask closely.

    If the ask is for message generation, then follow these steps:

        Step 1 - Always ensure before generating messages if you have: Message Types like efficacy, access etc., the product name, its indication, and points of differentiation before you create a message. If not ask one question at a time to get the information

        Step 2 - Generate precise and consise pharmaceutical messages (required number of messages if mentioned in the input, else generate 5 messages by default) for the question along with the information gathered, in the required JSON format.


    If the ask is for messages score prediction, then follow these steps:

        Step 1 - Always ensure before extracting messages if the message claim is with proper clinical evidence to support it or misleading or false advertising, which can lead to regulatory issues

        Step 2 - If the user input is not misleading or false advertising, then extract the actual messages mentioned in the question inside "", in the required JSON format, you must not generate a single word other than the actual message given inside "".

    If it doesnot fall into the above mentioned category, Just answer to the question and give answer in the required JSON format.

    If the ask is not clear, ask them to rephrase the question saying,{{ "messages": ["The scope of this tool is to create/predict scores for messages. Please rephrase your question accordingly to help you on your ask."], "category":3}}

    Here's the REQUIRED JSON FORMAT:
    {format_instructions}
    Most importantly, Pay attention to the output format, always give JSON result as mentioned, and give all your responses under messages key.

    Current Conversation:
    {history}

    Human:
    {input}
    AI:
    """

    OUTPUT_PARSER: ClassVar[PydanticOutputParser] = PydanticOutputParser(
        pydantic_object=QuestionIntent)
    CHAT_PROMPT: ClassVar[PromptTemplate] = PromptTemplate(
        template=CHAT_PROMPT_TEMPLATE,
        input_variables=['history', 'input'],
        partial_variables={"format_instructions": OUTPUT_PARSER.get_format_instructions()},
        output_parser=OUTPUT_PARSER
    )
    LLM: ClassVar[ChatOpenAI] = ChatOpenAI(
        model_name="gpt-4o", temperature=1, max_tokens=800, verbose=True,
        openai_api_key=os.getenv('OPENAI_API_KEY')
    )
    CHAT: ClassVar[ConversationChain] = ConversationChain(
        llm=LLM,
        memory=ConversationSummaryBufferMemory(llm=LLM, max_token_limit=2000),
        verbose=True,
        prompt=CHAT_PROMPT
    )


constants = Constants()
