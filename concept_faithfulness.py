import pandas as pd 
from utils import set_seed, agent
import ast

# set the same hyperparamters as in original GlassMol code
set_seed(1)
num_concepts = 29

# iterate over all datasets from original GlassMol paper
for data_type in ['dili', 'ames', 'bbbp', 'lipo', 'avail', 'solubility', 'caco', 'hia_hou', 'pgp', 'ppbr', 'vdss', 'cyp2c9', 'cyp2d6', 'cyp3a4', 'cyp2c9_substrate', 'cyp2d6_substrate', 'cyp3a4_substrate', 'half_life', 'ld50', 'herg']:
    
    print(f"selecting concepts for {data_type}")

    # load data
    DATA = {}
    DATA['train'] = pd.read_csv(f'data/train_{data_type}.csv')
    DATA['val'] = pd.read_csv(f'data/val_{data_type}.csv')
    DATA['test'] = pd.read_csv(f'data/test_{data_type}.csv')

    # run feature selection 10x for current dataset
    run_to_features = {}
    for i in range(10):

        print(f"run {i+1}")

        # choose num_concepts features with llm agent
        features = agent(data_type, DATA['train'].drop(columns=['Drug', 'Y', 'Drug_ID']).columns.tolist(), num_concepts)
        features = ast.literal_eval(features)
        run_to_features[str(i+1)] = features[:29]

    # write selected features from all runs to csv file    
    outfile = f"concept_faithfulness_{data_type}.csv"
    df = pd.DataFrame.from_dict(run_to_features)
    df.to_csv(outfile)
