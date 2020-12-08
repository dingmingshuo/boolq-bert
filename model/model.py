from transformers import AutoModelForSequenceClassification


def get_model(model_name):
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return model
