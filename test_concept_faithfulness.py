import sys
from apetokenizer.src.apetokenizer.ape_tokenizer import APETokenizer
import pandas as pd
from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import torch
from backbone.backbone import ModelXtoCtoY_function
import numpy as np
from sklearn.metrics import roc_auc_score
from utils import set_seed, agent
import ast
import pickle as pkl
from utils import MyDataset
import yaml

import csv

def get_features(outfile):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open('args.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    set_seed(config['seed'])
    #data_type = config['data_type']
    data_type = "ames"
    num_epochs = config['num_epochs']
    num_concepts = config['num_concepts']
    loss_weight = config['loss_weight']

    tokenizer = APETokenizer()
    tokenizer.load_vocabulary('apetokenizer/tokenizer.json')
    model = AutoModelForSequenceClassification.from_pretrained('mikemayuare/SMILY-APE-BBBP').to(device)

    # load data
    DATA = {}
    DATA['train'] = pd.read_csv(f'data/train_{data_type}.csv')
    DATA['val'] = pd.read_csv(f'data/val_{data_type}.csv')
    DATA['test'] = pd.read_csv(f'data/test_{data_type}.csv')

    # choose num_concepts features with llm agent
    feature_collection = []
    for _ in range(10):
        features = agent(data_type, DATA['train'].drop(columns=['Drug', 'Y', 'Drug_ID']).columns.tolist(), num_concepts)
        features = ast.literal_eval(features)
        features = sorted(features)[:40]
        feature_collection.append(features)

    with open(outfile, "w", encoding = "utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([f"feat_{i}" for i in range(1,41)])
        writer.writerows(feature_collection)


def eval_concepts(infile, outfile):
    # get freqs for each concept
    df = pd.read_csv(infile, )
    counts = df.stack().value_counts()
    counts.to_csv(outfile)



if __name__ == "__main__":
    concept_file = "concept_faithfulness_ames.csv"
    count_file = "concept_counts_ames.csv"

    get_features(concept_file)

    eval_concepts(concept_file, count_file)