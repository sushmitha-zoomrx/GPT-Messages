import re
import json
import os
import csv
import datetime
import numpy as np
import pandas as pd
import scipy
from icecream import ic
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

from config.settings import constants
from utilities.gsheets import log_to_gsheets

class MessagesService():

    def format_df(self, df):
        res = df.to_html(index=False, escape=False, justify='center', border=0)
        res = re.sub(r'<thead>[\s\S]*?</thead>', constants.HEADER_HTML, res)
        res = res.replace('<table class="dataframe">', '<table>')
        return res

    def extract_features_from_gpt(self, message):
        """Extract features from the message using GPT model"""
        try:
            llm = ChatOpenAI(
                model_name="gpt-4o",
                temperature=0,
                openai_api_key=constants.GPT_API_KEY
            )
            formatted_prompt = constants.CHAT_GPT_PROMPT.substitute(message=message)
            response = llm.invoke([HumanMessage(content=formatted_prompt)])
            try:
                features = json.loads(response.content)
            except json.JSONDecodeError:
                ic("JSON parsing failed, attempting to extract JSON from response")
                print(features)
                content = response.content
                # Find JSON content between curly braces
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = content[json_start:json_end]
                    features = json.loads(json_str)
                else:
                    raise ValueError("Could not extract valid JSON from response")

            validated_features = self.validate_and_convert_features(features)
            return validated_features
        except Exception as e:
            print(f"Error extracting features from GPT: {e}")
            return self.get_default_features()

    def validate_and_convert_features(self, features):
        """Validate and convert features to appropriate types"""
        validated = {}

        expected_types = {
            "word_count": int,
            "char_count": int,
            "average_char": float,
            "passive_voice": int,
            "data_points": int,
            "metric_count": int,
            "action_words_count": int,
            "readability_score": float,
            "call_to_action": int
        }

        if all(isinstance(k, int) or (isinstance(k, str) and k.isdigit()) for k in features.keys()):
            feature_mapping = {
                0: "message",
                1: "word_count",
                2: "char_count",
                3: "average_char",
                4: "comparator",
                5: "passive_voice",
                6: "data_points",
                7: "metric_count",
                8: "brand_name",
                9: "specific_condition",
                10:"action_words",
                11: "action_words_count",
                12: "unique_selling_proposition",
                13: "tone",
                14: "technical_terms",
                15: "presence_of_datapoints",
                16: "emotion_elements",
                17: "word_complexities",
                18: "readability_score",
                19: "call_to_action",
            }

            for key, value in features.items():
                index = int(key) if isinstance(key, str) else key
                if index in feature_mapping:
                    validated[feature_mapping[index]] = value
        else:
            validated = features

        # Convert types as needed
        for key, expected_type in expected_types.items():
            if key in validated:
                try:
                    validated[key] = expected_type(validated[key])
                except (ValueError, TypeError):
                    # Use default if conversion fails
                    if expected_type == int:
                        validated[key] = 0
                    elif expected_type == float:
                        validated[key] = 0.0

        return validated

    def get_default_features(self):
        """Return default features when extraction fails"""
        return {
            "word_count": 0,
            "char_count": 0,
            "average_char": 0.0,
            "comparator": [],
            "passive_voice": 0,
            "data_points": 0,
            "metric_count": 0,
            "brand_name": [],
            "specific_condition": [],
            "action_words": [],
            "action_words_count": 0,
            "unique_selling_proposition": [],
            "tone": "neutral",
            "technical_terms": [],
            "presence_of_datapoints": "",
            "emotion_elements": [],
            "word_complexities": [],
            "readability_score": 0.0,
            "call_to_action": 0
        }

    def get_tfidf_vectors(self, vectorizer, text, column_name=None):
        """Generate TF-IDF vectors for text"""
        if vectorizer is None:
            return {}
        vector = vectorizer.fit_transform([text])
        vectors_dict = {}
        for i, value in enumerate(vector.toarray()[0]):
            if value > 0:
                if column_name:
                    vector_name = f"{column_name}_tfidf_{i}"
                else:
                    vector_name = f"tfidf_{i}"
                vectors_dict[vector_name] = value

        return vectors_dict

    def form_test_data(self, features, message):
        """Form test data for prediction models"""
        test_data = {}

        for column in constants.VECTORIZE_COLUMNS:
            if column in features and features[column]:
                if isinstance(features[column], list):
                    column_text = " ".join(features[column])
                else:
                    column_text = str(features[column])

                column_vectors = self.get_tfidf_vectors(constants.TFIDF_VECTORIZER_LESS_FEATURES,column_text, column)
                test_data.update(column_vectors)

        message_vectors = self.get_tfidf_vectors(constants.TFIDF_VECTORIZER_MORE_FEATURES, message)
        test_data.update(message_vectors)

        # Add other features from GPT
        for key, value in features.items():
            if key not in constants.VECTORIZE_COLUMNS:
                if isinstance(value, list):
                    # Count number of items if it's a list
                    test_data[f"{key}_count"] = len(value)
                elif isinstance(value, (int, float, bool)):
                    # Use as is if it's a numeric or boolean value
                    test_data[key] = value

        # Create test data for each model
        X_test = {
            "Motivation": [],
            "Believability": [],
            "Differentiation": []
        }

        # For each model, select the required features
        for model_name in X_test.keys():
            # Skip if the required features key doesn't exist
            if model_name not in constants.REQUIRED_FEATURES:
                ic(f"Warning: {model_name} not found in constants.REQUIRED_FEATURES")
                X_test[model_name] = np.array([[0]])
                continue

            required_features = constants.REQUIRED_FEATURES[model_name]

            # Validate that we're using the exact list from constants
            if not isinstance(required_features, list):
                ic(f"Warning: required_features for {model_name} is not a list. Converting...")
                required_features = list(required_features)

            # Log the feature order for debugging
            ic(f"Creating feature vector for {model_name} with {len(required_features)} features")

            # Create a feature vector with the required features in the exact order specified in constants.REQUIRED_FEATURES
            feature_vector = []
            ordered_features_dict = {}  # For debugging purposes

            for i, feature in enumerate(required_features):  # This preserves the exact order from constants.REQUIRED_FEATURES
                # Use the feature value if available, otherwise use 0
                value = test_data.get(feature, 0)
                feature_vector.append(value)
                ordered_features_dict[f"{i}_{feature}"] = value  # Store with index for order verification

            # Save ordered features for debugging if needed
            if hasattr(constants, 'DEBUG_MODE') and constants.DEBUG_MODE:
                with open(f"debug_{model_name}_features.json", "w") as f:
                    json.dump(ordered_features_dict, f, indent=2)

            # Ensure our feature vector has the expected length
            if len(feature_vector) != len(required_features):
                ic(f"Warning: Feature vector length ({len(feature_vector)}) doesn't match required features ({len(required_features)})")

            # Convert to numpy array for the model
            X_test[model_name] = np.array([feature_vector])

        return X_test

    def analyze_with_llm(self, message, context):
        """Analyze the message using Langchain with GPT model."""
        try:
            # Create a ChatOpenAI instance with the appropriate model
            llm = ChatOpenAI(
                model_name="gpt-4o",
                temperature=0.2,
                openai_api_key=constants.GPT_API_KEY
            )
            formatted_prompt = context.substitute(message=message)
            response = llm.invoke([HumanMessage(content=formatted_prompt)])
            try:
                return json.loads(response.content)
            except json.JSONDecodeError:
                ic("JSON parsing failed, attempting to extract JSON from response")
                content = response.content
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = content[json_start:json_end]
                    return json.loads(json_str)
                else:
                    raise ValueError("Could not extract valid JSON from response")
        except Exception as e:
            ic(f"Error analyzing with LLM: {e}")
            return {}

    def weighted_average(self, llm_score, ml_score, llm_confidence=0, ml_confidence=0):
        """Calculate weighted average of LLM and ML scores based on confidence."""
        total_confidence = llm_confidence + ml_confidence
        if total_confidence == 0:
            return (llm_score + ml_score) / 2
        return (llm_score * llm_confidence + ml_score * ml_confidence) / total_confidence

    def resolve_discrepancy(self, metric, llm_score, ml_score, llm_results, ml_results):
        """Resolve discrepancies between LLM and ML scores using weighted average based on confidence."""

        llm_confidence = llm_results.get('confidence', 2)
        ml_confidence = ml_results.get('confidence', 1)
        return self.weighted_average(llm_score, ml_score, llm_confidence, ml_confidence)


    def integrate_analyses(self, llm_results, ml_results):
        def clamp_score(score):
            """Clamp the score between 0 and 100."""
            return max(0, min(100, score))

        integrated_results = {}

        # Handle each metric
        for metric in ['Motivation', 'Believability', 'Differentiation']:
            llm_score = clamp_score(llm_results[metric.lower()])
            ml_score = clamp_score(ml_results[metric])

            # Check for significant discrepancy
            if abs(llm_score - ml_score) > constants.DISCREPANCY_THRESHOLD:
                integrated_score = self.resolve_discrepancy(
                    metric, llm_score, ml_score,
                    llm_results, ml_results
                )
            else:
                integrated_score = self.weighted_average(
                    llm_score, ml_score,
                )

            integrated_results.update({
                metric: integrated_score
            })

        return integrated_results

    def log_predict_scores(self, user_input, gpt_output, ml_scores, llm_scores, final_results):
        """
        Log prediction data to a single CSV file with columns:
        [timestamp, user_input, GPT outputs, ml_scores, llm_scores, final_results]

        Args:
            user_input (str): The user's input message
            gpt_output (dict): The extracted features from GPT
            ml_scores (dict): The machine learning model scores
            llm_scores (dict): The LLM scores before integration
            final_results (dict): The final integrated results
        """
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)

        # Use a single static filename
        filename = "logs/predict_scores.csv"

        # Check if file exists to decide whether to write headers
        file_exists = os.path.isfile(filename)

        # Prepare row data with timestamp
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Format data as per requirement
        row_data = {
            'timestamp': timestamp,
            'user_input': user_input,
            'gpt_outputs': json.dumps(gpt_output, ensure_ascii=False),
            'ml_scores': json.dumps(ml_scores, ensure_ascii=False),
            'llm_scores': json.dumps(llm_scores, ensure_ascii=False),
            'final_results': json.dumps(final_results, ensure_ascii=False)
        }

        try:
            # Write to CSV
            with open(filename, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=row_data.keys())

                # Write header if file didn't exist
                if not file_exists:
                    writer.writeheader()

                writer.writerow(row_data)
            ic(f"Prediction data logged to {filename}")
        except Exception as e:
            ic(f"Error logging prediction data: {str(e)}")

    def predict_scores(self, message):
        def round_off_score(val):
            val = "{:.1f} %".format(val)
            return val

        def calc_percentile(final_score):
            data = constants.CSV_DATA.copy(deep=True)
            data["Believability"] = data["Believability"].astype(float)
            data["Differentiation"] = data["Differentiation"].astype(float)
            data["Motivation"] = data["Motivation"].astype(float)
            data["Final_Score"] = np.cbrt(
                data["Believability"] * data["Differentiation"] * data["Motivation"])
            percentile = scipy.stats.percentileofscore(data["Final_Score"], final_score)
            return str(int(percentile))

        features = self.extract_features_from_gpt(message)
        pharma_message = features.get('message', message)
        X_test = self.form_test_data(features, pharma_message)

        B_score, M_Score, D_Score = (
            constants.BELIEVABILITY_SCORE_PREDICTOR.predict(X_test["Believability"])[0],
            constants.MOTIVATION_SCORE_PREDICTOR.predict(X_test["Motivation"])[0],
            constants.DIFFERENTIATION_SCORE_PREDICTOR.predict(X_test["Differentiation"])[0]
        )
        ml_scores = {
            'Believability': float(B_score),
            'Differentiation': float(D_Score),
            'Motivation': float(M_Score),
        }
        ic(ml_scores)

         # Get LLM analysis
        llm_results = self.analyze_with_llm(pharma_message, constants.EVALUATION_PROMPT)
        ic(llm_results)
        llm_scores = llm_results['scores']
        if not llm_results:
            return "Something went wrong! Failed to produce the scores and analysis results. Please give us sometime "

        # Integrate results
        final_results = self.integrate_analyses(llm_scores , ml_scores)

        overall_score = np.cbrt(final_results['Believability'] * final_results['Differentiation'] * final_results['Motivation'])
        llm_results['scores'] =  {
            'Believability': [round_off_score(final_results['Believability'])],
            'Differentiation': [round_off_score(final_results['Differentiation'])],
            'Motivation': [round_off_score(final_results['Motivation'])],
            'Overall Effectiveness': [round_off_score(overall_score)],
            'Rank Percentile': [calc_percentile(overall_score)]
        }
        # Log the prediction data
        self.log_predict_scores(
            user_input=message,
            gpt_output=features,
            ml_scores=ml_scores,
            llm_scores=llm_scores,
            final_results=llm_results
        )

        return self.format_final_answer(llm_results)

    def format_final_answer(self, llm_results):
        """Format the final answer into an HTML string."""
        html_parts = []

        # Format scores
        if "scores" in llm_results and llm_results["scores"]:
            html_parts.append("<h3>Scores:</h3>")
            html_parts.append("<ul>")
            for key, value in llm_results["scores"].items():
                html_parts.append(f"<li>{key}: {value}</li>")
            html_parts.append("</ul>")

        # Format strengths
        if "strengths" in llm_results and llm_results["strengths"]:
            html_parts.append("<h3>Strengths:</h3>")
            html_parts.append("<ul>")
            for strength in llm_results["strengths"]:
                html_parts.append(f"<li>{strength}</li>")
            html_parts.append("</ul>")

        # Format weaknesses
        if "weaknesses" in llm_results and llm_results["weaknesses"]:
            html_parts.append("<h3>Weaknesses:</h3>")
            html_parts.append("<ul>")
            for weakness in llm_results["weaknesses"]:
                html_parts.append(f"<li>{weakness}</li>")
            html_parts.append("</ul>")

        # Format regulatory concerns
        if "regulatory_concerns" in llm_results and llm_results["regulatory_concerns"]:
            html_parts.append("<h3>Regulatory Concerns:</h3>")
            html_parts.append("<ul>")
            for concern in llm_results["regulatory_concerns"]:
                html_parts.append(f"<li>{concern}</li>")
            html_parts.append("</ul>")

        # Format detailed analysis
        if "detailed_analysis" in llm_results and llm_results["detailed_analysis"]:
            html_parts.append("<h3>Detailed Analysis:</h3>")
            html_parts.append(f"<p>{llm_results['detailed_analysis']}</p>")

        # Format improvement recommendations
        if "improvement_recommendations" in llm_results and llm_results["improvement_recommendations"]:
            html_parts.append("<h3>Improvement Recommendations:</h3>")
            html_parts.append("<ul>")
            for recommendation in llm_results["improvement_recommendations"]:
                html_parts.append(f"<li>{recommendation}</li>")
            html_parts.append("</ul>")

        # Format improvised messages
        if "improvised_messages" in llm_results and llm_results["improvised_messages"]:
            html_parts.append("<h3>Sample Improvised Messages:</h3>")
            html_parts.append("<ul>")
            for message in llm_results["improvised_messages"]:
                html_parts.append(f"<li>{message}</li>")
            html_parts.append("</ul>")

        return "\n".join(html_parts)

messages_service = MessagesService()
