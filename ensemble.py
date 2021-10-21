import pandas as pd
import numpy as np
import json
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

mmbt_grid = pd.read_csv("result/mmbt_grid.csv")
mmbt_region = pd.read_csv("result/mmbt_region.csv")
vilbert = pd.read_csv("result/vilbert.csv")
vilbert_cc = pd.read_csv("result/vilbert_cc.csv")
visualbert = pd.read_csv("result/visualbert.csv")
visualbert_coco = pd.read_csv("result/visualbert_coco.csv")

models = [mmbt_grid, mmbt_region, vilbert, vilbert_cc, visualbert, visualbert_coco]
modelnames = ["mmbt_grid", "mmbt_region", "vilbery", "vilbert_cc", "visualbert", "visualbert_coco"]

models_v2 = [vilbert, vilbert_cc, visualbert, visualbert_coco]
modelname_v2 = ["vilbery", "vilbert_cc", "visualbert", "visualbert_coco"]

def ensemble(models, modelnames):
    # print([len(df) for df in models])
    #   id     proba  label
    labels = pd.DataFrame();
    probs = pd.DataFrame();

    for i, model in enumerate(modelnames):
        labels[model] = models[i]['label']
        probs[model] = models[i]['proba']

    # print(labels.head())
    # print(probs.head())

    # majority vote
    labels["sum"] = labels.sum(axis=1)
    labels_vote = []
    for i in range(0, len(labels)):
        labels_vote.append(1 if labels.iloc[i]['sum'] >(len(modelnames)/2) else 0)
    labels["majority_vote"] = labels_vote
    # print(labels.head())

    # probability average
    probs["sum"]=probs.sum(axis=1)
    probs_value = []
    for i in range(0, len(probs)):
        probs_value.append(1 if (probs.iloc[i]['sum']/len(modelnames)) > 0.5 else 0)
    probs["probs_average"] = probs_value
    # print(probs.head())

    result = pd.DataFrame()
    result['id'] = mmbt_grid['id']
    result['majority_vote'] = labels["majority_vote"]
    result['probs_averag'] = probs["probs_average"]
    result.to_csv("result/ensemble_result.csv", index=False)

def process_test_set():
    with open("data/hateful_memes/test_unseen.jsonl", "r") as f:
        test_unseen = f.read().splitlines()

    test_labels = []
    test_ids = []
    for i, item in enumerate(test_unseen):
        test_labels.append(json.loads(item)["label"])
        test_ids.append(json.loads(item)['id'])

    test = pd.DataFrame()
    test["id"] = test_ids
    test['labels'] = test_labels
    test.to_csv("result/test.csv", index=False)

def evaluation_metrics():
    ensemble_result = pd.read_csv("result/ensemble_result.csv")
    test_set = pd.read_csv("result/test.csv")
    df = ensemble_result.merge(test_set, how="left", on="id")
    # correct_majority = []
    # correct_probs = []
    # for idx, row in df.iterrows():
    #     correct_majority.append(1 if row['majority_vote'] == row['labels'] else 0)
    #     correct_probs.append(1 if row['probs_averag']==row['labels'] else 0)

    roc_auc1 = roc_auc_score(df['majority_vote'].tolist(), df['labels'].tolist())
    roc_auc2 = roc_auc_score(df['probs_averag'].tolist(), df['labels'].tolist())

    accuracy1 = accuracy_score(df['majority_vote'].tolist(), df['labels'].tolist())
    accuracy2 = accuracy_score(df['probs_averag'].tolist(), df['labels'].tolist())

    f1_score1 = f1_score(df['majority_vote'].tolist(), df['labels'].tolist())
    f1_score2 = f1_score(df['probs_averag'].tolist(), df['labels'].tolist())


    print(f'Majority vote accuracy: {accuracy1}')
    print(f'Probability average accuracy: {accuracy2}')
    print(f'Majority vote roc_auc: {roc_auc1}')
    print(f'Probability average roc_auc: {roc_auc2}')
    print(f'Majority vote f1: {f1_score1}')
    print(f'Probability average f1: {f1_score2}')


# ensemble(models, modelnames)
evaluation_metrics()

