from utils import yaml_load
from data import get_train_data, get_dev_data
from model.model import get_model
from model.optimizer import get_AdamW

import os

import torch
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
from transformers import AutoTokenizer
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = yaml_load("./config.yaml")
model_name = config.get("model")
preprocess_cfg = config.get("preprocess", {})
data_cfg = config.get("data", {})
dev_cfg = config.get("dev", {})
train_cfg = config.get("train", {})

tokenizer = AutoTokenizer.from_pretrained(model_name)
train_data = get_train_data(
    data_cfg['data_path'], data_cfg['train_data'], tokenizer, preprocess_cfg['max_sent_len'])
dev_data = get_dev_data(
    data_cfg['data_path'], data_cfg['dev_data'], tokenizer, preprocess_cfg['max_sent_len'])

model = get_model(model_name).to(device)
model.train()
optimizer = get_AdamW(model, train_cfg["lr"], train_cfg["weight_decay"])


train_loader = DataLoader(
    train_data,
    batch_size=train_cfg["batch_size"],
    shuffle=True
)
dev_loader = DataLoader(
    dev_data,
    batch_size=dev_cfg["batch_size"],
    shuffle=True
)

logging_step = train_cfg["logging_step"]
b = train_cfg["b"]

for epoch in range(train_cfg["epochs"]):
    total_loss = 0
    step_now = 0
    dev_loss = 0
    dev_acc = 0
    with tqdm(train_loader) as tl:
        for input_ids, attention_mask, labels in tl:
            step_now += 1
            optimizer.zero_grad()
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            outputs = model(input_ids, attention_mask=attention_mask,
                            labels=labels, return_dict=True)
            loss = (outputs.loss - b).abs() + b
            loss.backward()
            optimizer.step()
            total_loss += loss.cpu().item()
            
            # Evaluate
            if step_now % logging_step == 0:
                dev_loss = 0
                dev_acc = 0
                model.eval()
                with torch.no_grad():
                    for input_ids, attention_mask, labels in dev_loader:
                        input_ids = input_ids.to(device)
                        attention_mask = attention_mask.to(device)
                        labels = labels.to(device)
                        outputs = model(input_ids, attention_mask=attention_mask,
                                        labels=labels, return_dict=True)
                        loss = outputs.loss
                        dev_loss += loss.cpu().item()
                        results = softmax(outputs.logits, dim=1)
                        pred = torch.argmax(results, dim=1)
                        dev_acc += torch.eq(pred, labels).sum().float().item()

                model.train()

            # Load loggings
            tl.set_postfix(loss=loss.cpu().item(),
                           avg_loss=total_loss/step_now,
                           dev_loss=dev_loss/len(dev_loader),
                           dev_acc=dev_acc/len(dev_data.input_ids))
           
    # Save model
    if not os.path.isdir(train_cfg["output_dir"]):
        os.mkdir(train_cfg["output_dir"])
    output_path = os.path.join(
        train_cfg["output_dir"], train_cfg["output_filename_prefix"])
    torch.save(model.state_dict(), (output_path+"_epoch=%d") % (epoch+1))