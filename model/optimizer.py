from transformers import AdamW


def get_AdamW(model, lr, weight_decay):
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    return optimizer
