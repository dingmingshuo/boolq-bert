# boolq-bert
Finetune a BERT on google's dataset BoolQ: https://github.com/google-research-datasets/boolean-questions

Some of the codes infers from:

- https://medium.com/illuin/deep-learning-has-almost-all-the-answers-yes-no-question-answering-with-transformers-223bebb70189.
- https://github.com/h3lio5/episodic-lifelong-learning.

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

## Trained Params
- Of commit [66b11cb](https://github.com/dingmingshuo/boolq-bert/commit/66b11cb6339e82f282c97f1e94365d25060d143e): https://disk.pku.edu.cn:443/link/DAC6F75EC7D8010CDB1013EF601B7281