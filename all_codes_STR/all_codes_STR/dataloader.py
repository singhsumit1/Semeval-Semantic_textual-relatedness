from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
import os
import torch
um_labels=1
seed=7
batch_size=16
lr=2e-5
# set the number of epochs
n_epochs = 5 #3
l_lr=5e-2
emded_d = 1024
max_len=92
early_stoping_patience=3
import random

random.seed(seed)
import re

import zipfile
import pandas as pd
# from tqdm import tqdmn
import numpy as np
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from transformers import  XLMRobertaTokenizer, XLMRobertaModel, XLMRobertaForSequenceClassification
# -*- coding: utf-8 -*-


"""# Import the Libraries"""
import pandas as pd
from collections import Counter
import sys
import json
import json
# visualization libraries
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# pytorch libraries
import torch # the main pytorch library
import torch.nn as nn # the sub-library containing Softmax, Module and other useful functions
import torch.optim as optim # the sub-library containing the common optimizers (SGD, Adam, etc.)

# huggingface's transformers library
from transformers import RobertaForSequenceClassification, AutoTokenizer, XLMRobertaForSequenceClassification

# huggingface's datasets library
from datasets import load_dataset
from datasets import  load_from_disk

# the tqdm library used to show the iteration progress

# tqdmn = tqdm.notebook.tqdm#tqdm.tqdm# tqdm.notebook.tqdm

from datasets import load_dataset
import os
from transformers import get_scheduler
import torch
from datasets import Dataset
from datasets import load_from_disk
import sys
from sklearn.metrics import confusion_matrix
import copy

class DataProcessor:
    def __init__(self, dataset_root, languages, max_length, batch_size):
        self.dataset_root = dataset_root
        self.languages = languages
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")

    def process_data(self, lang):
        train_data_path = {}
        dev_data_path = {}
        test_data_path = {}

        train_dataset = {}
        dev_dataset = {}
        test_dataset = {}
        train_dataloader = {}
        dev_dataloader = {}
        test_dataloader = {}

        train_data_path[lang] = os.path.join(self.dataset_root, lang, f"{lang}_train.csv")
        dev_data_path[lang] = os.path.join(self.dataset_root, lang, f"{lang}_dev_with_labels.csv")
        test_data_path[lang] = os.path.join(self.dataset_root, lang, f"{lang}_test_with_labels.csv")

        train_dataset[lang] = self._process_dataset(train_data_path[lang], nrows=10)
        dev_dataset[lang] = self._process_dataset(dev_data_path[lang], nrows=10)
        test_dataset[lang] = self._process_dataset(test_data_path[lang], nrows=10)

        train_dataset[lang] = self.set_format(train_dataset[lang])
        dev_dataset[lang] = self.set_format(dev_dataset[lang])
        test_dataset[lang] = self.set_format(test_dataset[lang])

        train_dataloader[lang] = torch.utils.data.DataLoader(
            train_dataset[lang],
            batch_size=self.batch_size,
            shuffle=True
        )
        dev_dataloader[lang] = torch.utils.data.DataLoader(
            dev_dataset[lang],
            batch_size=self.batch_size
        )
        test_dataloader[lang] = torch.utils.data.DataLoader(
            test_dataset[lang],
            batch_size=self.batch_size
        )

        return train_dataloader[lang], dev_dataloader[lang], test_dataloader[lang]

    def _process_dataset(self, data_path, nrows=None):
        dataset = pd.read_csv(data_path, nrows=nrows)
        dataset['Split_Text'] = dataset['Text'].apply(lambda x: x.split("\n"))

        def add_encodings(example):
            encodings = self.tokenizer(
                example['Split_Text'][0] + "</s>" + example['Split_Text'][1],
                truncation=True,
                padding='max_length',
                is_split_into_words=False
            )
            score=example["Score"]
            # score = example.get('Score', 0)
            return {**encodings, 'labels': score}

        dataset = Dataset.from_pandas(dataset)
        dataset = dataset.rename_column("PairID", "sentence_id")

        dataset = dataset.map(add_encodings)

        return dataset

    def set_format(self, dataset):
        dataset.set_format(type="torch", columns=['input_ids', 'attention_mask', 'labels'])
        return dataset


# class DataProcessor:
#     def __init__(self, dataset_root, languages, max_length, batch_size):
#         self.dataset_root = dataset_root
#         self.languages = languages
#         self.max_length = max_length
#         self.batch_size = batch_size
#         self.tokenizer = AutoTokenizer.from_pretrained("roberta-large")

#     def process_data(self,lang):
#         train_data_path={}
#         dev_data_path={}
#         test_data_path={}
    	
#         train_dataset={}
#         dev_dataset={}
#         test_dataset={}
#         train_dataloader = {}
#         dev_dataloader = {}
#         test_dataloader = {}
	    
#         train_data_path[lang] = os.path.join(self.dataset_root, lang, f"{lang}_train.csv")
#         dev_data_path[lang] = os.path.join(self.dataset_root, lang, f"{lang}_dev.csv")
#         test_data_path[lang] = os.path.join(self.dataset_root, lang, f"{lang}_test.csv")

#         train_dataset[lang]  = self._process_dataset(train_data_path[lang])
#         dev_dataset[lang] = self._process_dataset(dev_data_path[lang])
#         test_dataset[lang] = self._process_dataset(test_data_path[lang])
#         # print("train_dataset:",train_dataset[lang]['Score'])
#         print(dev_dataset[lang])
#         train_dataset[lang] = self.set_format(train_dataset[lang])
#         dev_dataset[lang] = self.set_format(dev_dataset[lang])
#         test_dataset[lang] = self.set_format(test_dataset[lang])
#         # print("train_dataset:",train_dataset[lang])
#         train_dataloader[lang] = torch.utils.data.DataLoader(
#             train_dataset[lang],
#             batch_size=self.batch_size,
#             shuffle=True
#         )
#         dev_dataloader[lang] = torch.utils.data.DataLoader(
#             dev_dataset[lang],
#             batch_size=self.batch_size
#         )
#         test_dataloader[lang] = torch.utils.data.DataLoader(
#             test_dataset[lang],
#             batch_size=self.batch_size
#         )

#         return train_dataloader[lang], dev_dataloader[lang], test_dataloader[lang]

#     def _process_dataset(self, data_path):
#         dataset = pd.read_csv(data_path)
#         dataset['Split_Text'] = dataset['Text'].apply(lambda x: x.split("\n"))

#         def add_encodings(example):
#             encodings = self.tokenizer(
#                 example['Split_Text'][0] + "</s>" + example['Split_Text'][1],
#                 truncation=True,
#                 padding='max_length',
#                 is_split_into_words=False
#             )
#             score = example.get('Score', 0)
#             return {**encodings, 'labels': score}

#         dataset = Dataset.from_pandas(dataset)
#         dataset = dataset.rename_column("PairID", "sentence_id")

#         dataset = dataset.map(add_encodings)

#         return dataset

#     def set_format(self, dataset):
#         dataset.set_format(type="torch", columns=['input_ids', 'attention_mask','labels'])
#         return dataset



# # if __name__ == "__main__":
# #     # Your data loading code goes here
# #     # Make sure you have loaded train, dev, and test data for each language
# #     dataset_root = "/home/naive123/nlp/Sumit/Textural_Relateness/Semantic_Relatedness_SemEval2024/textuaL_relatedness_test_phase/Semantic_Relatedness_SemEval2024/Dataset/Track A"  # Specify the root path to your dataset directory
# #     languages = ["eng", "esp", "amh", "arq", "ary", "hau", "kin", "mar", "tel"]
# #     max_length = 92  # Your max_length
# #     batch_size = 16   # Your batch_size

# #     # Your data loading code goes here
# #     # # Make sure you have loaded train, dev, and test data for each language
# #     # # Assuming train_data, dev_data, and test_data are populated here

# #     # data_processor = DataProcessor(dataset_root, languages, max_length, batch_size)
# #     # train_dataloader, dev_dataloader, test_dataloader = data_processor.process_data()

#     # Now you have dataloaders for train, dev, and test datasets for each language

