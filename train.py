from transformers import AutoTokenizer
from utils import yaml_load
from data import get_train_data, get_dev_data

config = yaml_load("./config.yaml")
model_name = config.get("model")
preprocess_cfg = config.get("preprocess", {})
data_cfg = config.get("data", {})

tokenizer = AutoTokenizer.from_pretrained(model_name)
train_data = get_train_data(
    data_cfg['data_path'], data_cfg['train_data'], tokenizer, preprocess_cfg['max_sent_len'])
dev_data = get_dev_data(
    data_cfg['data_path'], data_cfg['dev_data'], tokenizer, preprocess_cfg['max_sent_len'])