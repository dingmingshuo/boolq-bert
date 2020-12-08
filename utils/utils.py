import yaml

def yaml_load(config_path):
    with open(config_path) as f:
        param = yaml.safe_load(f)
    return param