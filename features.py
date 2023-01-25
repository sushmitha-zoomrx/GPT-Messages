from services import generate_messages, predict_scores


def list_messages(prompt: str):
    return generate_messages(prompt)


def get_scores(message: str):
    return predict_scores(message)
