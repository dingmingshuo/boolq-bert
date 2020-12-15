from utils import yaml_load
from data import get_dev_data
from model.model import get_model

import os

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
from transformers import AutoTokenizer

from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score, roc_auc_score

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

config = yaml_load("./config.yaml")
model_name = config.get("model")
preprocess_cfg = config.get("preprocess", {})
data_cfg = config.get("data", {})
dev_cfg = config.get("dev", {})

tokenizer = AutoTokenizer.from_pretrained(model_name)
dev_data = get_dev_data(
    data_cfg['data_path'], data_cfg['dev_data'], tokenizer, preprocess_cfg['max_sent_len'])

model_path = os.path.join(dev_cfg['model_dir'], dev_cfg['model_filename'])
model = get_model(model_path).to(device)
model.eval()

dev_loader = DataLoader(
    dev_data,
    batch_size=dev_cfg["batch_size"],
    shuffle=True
)

# Evaluate
total_loss = 0
pred = np.array([])
answer = np.array([])
prob = np.array([])

with torch.no_grad():
    for input_ids, attention_mask, labels in dev_loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        outputs = model(input_ids, attention_mask=attention_mask,
                        labels=labels, return_dict=True)
        loss = outputs.loss
        total_loss += loss.cpu().item()
        results = softmax(outputs.logits, dim=1)
        now_pred = torch.argmax(results, dim=1)
        pred = np.concatenate((pred, now_pred.numpy()), axis=0)
        answer = np.concatenate((answer, labels.numpy()), axis=0)
        prob = np.concatenate((prob, results.numpy()[:,1]), axis=0)

# Calculate metric scores
accuracy = accuracy_score(answer, pred)
recall = recall_score(answer, pred)
precision = precision_score(answer, pred)
f1 = f1_score(answer, pred)
roc_auc = roc_auc_score(answer, prob)

print("accuracy: ", accuracy)
print("recall: ", recall)
print("precision: ", precision)
print("f1 score: ", f1)
print("roc_auc score: ", roc_auc)

# Save results
if not os.path.isdir(dev_cfg["result_dir"]):
    os.mkdir(dev_cfg["result_dir"])
output_path = os.path.join(
    dev_cfg["result_dir"], dev_cfg["result_filename"])
np.savetxt(output_path, pred, fmt = "%d", delimiter = "\n")