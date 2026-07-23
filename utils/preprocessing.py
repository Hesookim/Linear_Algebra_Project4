import re

def preprocess_text(text):

    text = re.sub(r"[^\w\s]", "", text)

    text = " ".join(text.split())

    return text