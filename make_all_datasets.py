from data.get_data import get_data

from rdkit import RDLogger                                                                                                                                                               
RDLogger.DisableLog('rdApp.*') 

# make train, val, test split + get descriptors for all datasets from GlassMol paper
for dataset in ['bbbp', 'lipo', 'avail', 'solubility', 'caco', 'hia_hou', 'pgp', 'ppbr', 'vdss', 'cyp2c9', 'cyp2d6', 'cyp3a4', 'cyp2c9_substrate', 'cyp2d6_substrate', 'cyp3a4_substrate', 'half_life', 'ld50', 'herg']:
    print(f"getting dataset {dataset}...")
    get_data(dataset)