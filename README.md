# Chinese_Resume_NER
Using HMM,CRF,LSTM,LSTM+CRF to achieve the goal of name entity recognition.

Environment: python3 + tensorflow 1.12+

Refer to [this github](https://github.com/luopeixiang/named_entity_recognition)

Dataset is published by ACL 2018 [Chinese NER using Lattice LSTM](https://github.com/jiesutd/LatticeLSTM)

This project don't contain the model saver, and the evaluation is about precision, recall and F1 of token pair, not about word, so you can complete these parts.
The result of this project:
|      | HMM    | CRF    | BiLSTM | BiLSTM+CRF |
| ---- | ------ | ------ | ------ | ---------- |
| recall  | 90.93% | 94.78% | 93.96% | 95.39%     |
| precision  | 93.13% | 96.51% | 96.41% | 96.61%  |
| F1 score | 91.97% | 95.63% | 95.13% | 95.91%     |
