> [See in spanish/Ver en español](https://github.com/LuisMiSanVe/TransformersNERTrainer/blob/main/README.es.md)
# 🤗 Transformers NER Model Trainer
[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
[![image](https://img.shields.io/badge/Visual_Studio_Code-0078D4?style=for-the-badge&logo=visual%20studio%20code&logoColor=white)](https://code.visualstudio.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)

Train your own NER Model using HuggingFace's Transformers with this Python Scripts.

## 📝 Technology Explanation
A NER Model (Named Entity Recognition) is a AI tool capable of recognizing words and patterns and clasify them, depending of the training data.\
There's already trained models like [SpaCy](https://spacy.io/) but with this simple script you can train your own model with custom training datasets.

## 🛠️ Setup
You'll obviously need Python to install the dependencies and run the scripts.\
Open a CMD and install the necessary depndencies:
```
pip install transformers datasets seqeval scikit-learn torch transformers[torch] accelerate>=0.26.0
```
Or if it fails or you're using a newer version of Python:
```
py -m pip install transformers datasets seqeval scikit-learn torch transformers[torch] accelerate>=0.26.0
```
Check if Python is in Windows' PATH:
`C:\Users\USER_NAME\AppData\Local\Programs\Python\Python313\Scripts`

> [!NOTE]
> The folder `Python313\` represents that the installed version is the '3.13', if you have other version installed, change it.

## 🚀 Project Usage Explanation
In [trainmodel.py](https://github.com/LuisMiSanVe/TransformersNERTrainer/blob/main/trainmodel.py), change the default dataset with the data you want to use to train your NER Model (explained in comments).\
In the `line 77` there are the training arguments, you can change them to test the results.\
Run the training Script using:
```
python trainmodel.py
```
Or if it fails or you're using a newer version of Python:
```
py trainmodel.py
```
Now, in [inferencemodel.py](https://github.com/LuisMiSanVe/TransformersNERTrainer/blob/main/inferencemodel.py), change the label map to match the used in the training step.\
Run the inference Script:
```
python inferencemodel.py
```
Or if it fails or you're using a newer version of Python:
```
py inferencemodel.py
```

## 📂 Files
If the scripts ran succesfully, the model will be generated in the same folder the script is, this files are:
- **my_ner_model**: here is stored all the model's data and configuration.
- **ner_model**: here are the different Model's checkpoints.

## 💻 Technologies Used
- Programming Language: [Python](https://www.python.org/)
- Framework: [seqeval](https://github.com/chakki-works/seqeval) (1.2.2)
- Libraries:
  - [datasets](https://pypi.org/project/datasets/) (3.3.2)
  - [scikit-learn](https://pypi.org/project/scikit-learn/) (1.6.1)
  - [torch](https://pypi.org/project/torch/) (2.6.0)
  - [transformers (with PyTorch)](https://huggingface.co/docs/transformers/en/installation)
  - [accelerate](https://pypi.org/project/accelerate/) (0.26.0)
- Other:
  - Model Base: [xlm-roberta-base](https://huggingface.co/FacebookAI/xlm-roberta-base)
- Recommended IDE: [VS Code](https://code.visualstudio.com/)
