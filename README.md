## GlassMol: Interpretable Molecular Property Prediction with Concept Bottleneck Models

This repository contains the implementation of the newly accepted PAKDD (full paper and oral presentation) paper "GlassMol: Interpretable Molecular Property Prediction with Concept Bottleneck Models". The goal of this work is to develop and demonstrate the efficacy of a concept bottleneck architecture for a variety of chemical tasks, using the embeddings from both LLM and GNN architectures.

### Setup
Install dependencies:
```python
pip install -r requirements.txt

```

To generate the augmented dataset as used in the paper:
```python
python data/get_data.py <dataset>

```
- --dataset is the desired dataset
  - these can be any of 'dili', 'lipo', 'bbbp', 'avail', 'solubility', 'caco', 'hia_hou', 'pgp', 'ppbr', 'vdss', 'cyp2c9', 'cyp2d6', 'cyp3a4', 'cyp2c9_substrate', 'cyp2d6_substrate', 'cyp3a4_substrate', 'half_life', 'ld50', 'herg', 'ames'

Also, before training, if the GPT selector is desired, set your API key:
```
export OPENAI_API_KEY=<your-api-key>
```

### Scripts

To train the GNN-based model as trained in the paper:

```python
python gnn_architecture.py

```

To train the LLM-based model as trained in the paper:

```python
python llm_architecture.py

```

After training the LLM model, generate explanations using the concepts selected by a state-of-the-art LLM:


```python
python -m generate_explanations.explainer_step1_llm <dataset>

```

After training the GNN model, generate explanations using the concepts selected by a state-of-the-art LLM:


```python
python -m generate_explanations.explainer_step1_gnn <dataset>

```
- In both cases, --dataset is the dataset used during training (set in the args.yaml).

Finally, to see textual output:
```python
python -m generate_explanations.explainer_step2 <model> <dataset> <mol_to_analyze>

```
- --model is whichever model has been trained (either 'gnn' or 'llm').
- --dataset is the dataset used during training (set in the args.yaml).
- --mol_to_analyze is the # of the molecule in the test set that you would like to analyze.

### Helpful Information

Some arguments used during the training of both models can be found and modified in args.yaml.
