# boolq-bert
Finetune a BERT on google's dataset BoolQ: https://github.com/google-research-datasets/boolean-questions

Some of the codes infers from https://medium.com/illuin/deep-learning-has-almost-all-the-answers-yes-no-question-answering-with-transformers-223bebb70189.

## Environment

```
conda create -n boolq python=3.7
conda activate boolq
pip install -r requirements.txt
```

If you change the environment, or use some packages which are not mentioned in `requirements.txt`, please run `pipreqs` as following:

```
pipreqs . --force
```