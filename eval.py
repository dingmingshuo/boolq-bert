from utils import yaml_load
from data import get_dev_data, get_train_data
from model.model import get_model
from model.optimizer import get_AdamW
from model.memory import EpisodicMemory
from data.dataset import BoolQDataset

import os
import copy

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
from transformers import AutoTokenizer

from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score, roc_auc_score

from tqdm import tqdm
import faulthandler
# 在import之后直接添加以下启用代码即可
faulthandler.enable()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

config = yaml_load("./config.yaml")
model_name = config.get("model")
preprocess_cfg = config.get("preprocess", {})
data_cfg = config.get("data", {})
dev_cfg = config.get("dev", {})
eval_cfg = config.get("eval", {})
train_cfg = config.get("train", {})

tokenizer = AutoTokenizer.from_pretrained(model_name)
train_data = get_train_data(
    data_cfg['data_path'], data_cfg['train_data'], tokenizer, preprocess_cfg['max_sent_len'])
eval_data = get_dev_data(
    data_cfg['data_path'], data_cfg['dev_data'], tokenizer, preprocess_cfg['max_sent_len'])

eval_loader = DataLoader(
    eval_data,
    batch_size=1,
    shuffle=False
)

e_memory = EpisodicMemory()
e_memory.build(train_data, rate = eval_cfg['sample_rate'])

model = get_model(model_name)
checkpoint = torch.load(eval_cfg['model_path'], map_location=torch.device("cpu"))
model.load_state_dict(checkpoint)
model.eval()

# Evaluate
pred = np.array([])
answer = np.array([])
prob = np.array([])

for input_ids, attention_mask, labels in tqdm(eval_loader):

    # only one piece of data in a batch
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
    attention_mask = torch.tensor([attention_mask], dtype=torch.long).to(device)
    labels = torch.tensor([labels], dtype=torch.long).to(device)

    # get current piece of data and its k nearest neighbours
    keys = e_memory.get_keys(input_ids.cpu().numpy())
    neighbour = e_memory.get_neighbours(keys, k = eval_cfg['k_neighbours'])[0]

    mem_data = BoolQDataset(neighbour[0], neighbour[1], neighbour[2])
    mem_loader = DataLoader(
         mem_data,
         batch_size=eval_cfg['batch_size'],
         shuffle = False
     )

    # copy a model, set optimizer
    infer_model = copy.deepcopy(model)
    infer_model = infer_model.to(device)
    infer_model.train()
    optimizer = get_AdamW(infer_model, eval_cfg["lr"])

    # finetune the model on the k nearest neighbours
    for mem_ids, mem_attnmasks, mem_labels in mem_loader:
        optimizer.zero_grad()
        mem_ids = mem_ids.to(device)
        mem_attnmasks = mem_attnmasks.to(device)
        mem_labels = mem_labels.to(device)
        train_outputs = infer_model(mem_ids, attention_mask=mem_attnmasks,
                        labels=mem_labels, return_dict=True)
        loss = train_outputs.loss
        loss.backward()
        optimizer.step()

    # do inference on current piece of data
    infer_model.eval()
    with torch.no_grad():
        eval_outputs = infer_model(input_ids, attention_mask=attention_mask,
                    labels=labels, return_dict=True)
        results = softmax(eval_outputs.logits, dim=1)
        now_pred = torch.argmax(results, dim=1)
        pred = np.concatenate((pred, now_pred.cpu().numpy()), axis=0)
        answer = np.concatenate((answer, labels.cpu().numpy()), axis=0)
        prob = np.concatenate((prob, results.cpu().numpy()[:,1]), axis=0)

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
if not os.path.isdir(eval_cfg["result_dir"]):
    os.mkdir(eval_cfg["result_dir"])
output_path = os.path.join(
    eval_cfg["result_dir"], eval_cfg["result_filename"])
np.savetxt(output_path, pred, fmt = "%d", delimiter = "\n")
