# Project Overview

In this project, we developed multiple machine learning models for the purpose of conducting sentiment analysis on Twitter posts. The objective was to classify the posts as either positive or negative. 
The models tested include BERT, RoBERTa, XLNet, TF-IDF, GloVe and Word2Vec. The performance of our best model was evaluated on the testing set, yielding an accuracy of 0.892.

# Getting Started

To run the code in this project, you will need to have the following software installed on your machine:

- Python 3.6 or higher
- The following Python packages:
  - numpy
  - pandas
  - scikit-learn
  - nltk
  - transformers (for BERT, RoBERTa, and XLNet)
  - gensim (for Word2Vec, and GloVe)
  - contractions
  - sentencepiece (for XLNet)
  - pytorch
  - pytorch lightning


To install theses packages, you can run the following command: 
```
conda install --file requirements.txt
```
# Data and pre-trained models

The dataset used in this project is a collection of tweets, with labels indicating whether each tweet is positive or negative. The dataset and our pre-trained models are not included in this repository due to their size, but they can be obtained from **[this URL](https://drive.google.com/drive/folders/1AToMGRPNWx2LGKRe_EbzKjDXm6nUqe1l?usp=sharing)**.

Please make sure that the structure of the project is the following: 

```
├── LICENSE
├── Pretrained_model
│   └── roBERTa
│       ├── config.json
│       ├── pytorch_model.bin
│       └── training_args.bin
├── README.md
├── data
│   ├── pred.csv
│   ├── sample_submission.csv
│   ├── test_data.txt
│   ├── train_neg.txt
│   ├── train_neg_full.txt
│   ├── train_pos.txt
│   └── train_pos_full.txt
├── notebooks
│   ├── GloVe.ipynb
│   ├── TFIDF.ipynb
│   ├── XLNet.ipynb
│   ├── bert.ipynb
│   ├── roBERTa.ipynb
│   └── word2Vec.ipynb
└── src
    ├── cooc.py
    ├── glove.py
    ├── helpers.py
    └── pickle_vocab.py
```

# Make predictions
Once the data and the pretrained models are in the correct folder, you can run our model on the testing set and make prediction using the following command at the top of the folder:

```
Python3 run.py
```

# Notebook Descriptions

- `bert.ipynb`: This notebook contains the code for training and evaluating a BERT model on the tweet classification task.
- `GloVE.ipynb`: This notebook contains the code for training and evaluating a model using the GlovE feature representation on the tweet classification task.
- `roBERTa.ipynb`: This notebook contains the code for training and evaluating a RoBERTa model on the tweet classification task.
- `XLNet.ipynb`: This notebook contains the code for training and evaluating an XLN model on the tweet classification task.
- `TFIDF.ipynb`: This notebook contains the code for training and evaluating a model using the TF-IDF feature representation on the tweet classification task.
- `word2Vec.ipynb`: This notebook contains the code for training and evaluating a model using the Word2Vec feature representation on the tweet classification task.

# Authors
- Daniel Tavares Agostinho
- Thomas Castiglione
- Jeremy Di Dio
