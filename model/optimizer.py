from transformers import AdamW


def get_AdamW(model, lr, weight_decay=0.0):
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    return optimizer
