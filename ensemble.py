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
finetune_visual_bert_with_caption = pd.read_csv("result/fintune_visual_bert_with_caption.csv")
finetune_visual_bert_without_caption = pd.read_csv("result/fintune_visual_bert_without_caption.csv")
finetune_visual_bert_coco_with_caption = pd.read_csv("result/fintune_visual_bert_coco_with_caption.csv")
finetune_visual_bert_coco_without_caption = pd.read_csv("result/fintune_visual_bert_coco_without_caption.csv")
finetune_vilbert_with_caption = pd.read_csv("result/fintune_vilbert_with_caption.csv")
vilbert_default_with_caption = pd.read_csv("result/vilbert_default_with_caption.csv")
finetune_vilbert_without_caption = pd.read_csv("result/fintune_vilbert_without_caption.csv")
finetune_vilbert_cc_with_caption = pd.read_csv("result/fintune_vilbert_cc_with_caption.csv")
finetune_vilbert_cc_without_caption = pd.read_csv("result/fintune_vilbert_cc_without_caption.csv")
finetune_vilbert_dense = pd.read_csv("result/finetune_vilbert_dense.csv")
finetune_visual_bert_dense = pd.read_csv("result/finetune_visual_bert_dense.csv")
finetune_vilbert_cc_dense = pd.read_csv("result/finetune_vilbert_cc_dense.csv")
finetune_visual_bert_coco_dense = pd.read_csv("result/finetune_visual_bert_coco_dense.csv")

# unimodal pretraining
models = [mmbt_grid, mmbt_region, finetune_vilbert_without_caption, finetune_vilbert_with_caption, finetune_visual_bert_without_caption, finetune_visual_bert_with_caption, finetune_vilbert_dense, finetune_visual_bert_dense]
modelnames = ["mmbt_grid", "mmbt_region", "vilbert", "vilbert_caption", "visualbert", "visualbert_caption", "vilbert_dense", "visualbert_dense"]

# multimodal pretraining
models_v2 = [finetune_vilbert_cc_dense, finetune_vilbert_cc_without_caption, finetune_vilbert_cc_with_caption, finetune_visual_bert_coco_dense, finetune_visual_bert_coco_without_caption, finetune_visual_bert_coco_with_caption]
modelname_v2 = ["vilbert_cc_dense", "vilbert_cc", "vilbert_cc_caption", "visualbert_coco_dense", "visualbert_coco", "visualbert_coco_caption"]

# excluding MMBT
models_v3 = [finetune_vilbert_without_caption, finetune_vilbert_with_caption, finetune_visual_bert_without_caption, finetune_visual_bert_with_caption, finetune_vilbert_dense, finetune_visual_bert_dense, finetune_vilbert_cc_dense, finetune_vilbert_cc_without_caption, finetune_vilbert_cc_with_caption, finetune_visual_bert_coco_dense, finetune_visual_bert_coco_without_caption, finetune_visual_bert_coco_with_caption]
modelname_v3 = ["vilbert", "vilbert_caption", "visualbert", "visualbert_caption", "vilbert_dense", "visualbert_dense", "vilbert_cc_dense", "vilbert_cc", "vilbert_cc_caption", "visualbert_coco_dense", "visualbert_coco", "visualbert_coco_caption"]

# Caption + denseCap
models_v4 = [finetune_vilbert_cc_with_caption, finetune_vilbert_with_caption, finetune_vilbert_dense, finetune_vilbert_cc_dense,
finetune_visual_bert_coco_with_caption, finetune_visual_bert_with_caption, finetune_visual_bert_dense, finetune_visual_bert_coco_dense]
modelname_v4 = ["vilbert_cc_caption", "vilbert_caption", "vilbert_dense", "vilbert_cc_dense", "visual_bert_coco_caption", "visual_bert_caption", "visual_bert_dense", "visual_bert_coco_dense"]

# all
models_v5 = [mmbt_grid, mmbt_region, finetune_vilbert_without_caption, finetune_vilbert_with_caption, finetune_visual_bert_without_caption, finetune_visual_bert_with_caption, finetune_vilbert_dense, finetune_visual_bert_dense, finetune_vilbert_cc_dense, finetune_vilbert_cc_without_caption, finetune_vilbert_cc_with_caption, finetune_visual_bert_coco_dense, finetune_visual_bert_coco_without_caption, finetune_visual_bert_coco_with_caption]
modelname_v5 = ["mmbt_grid", "mmbt_region", "vilbert", "vilbert_caption", "visualbert", "visualbert_caption", "vilbert_dense", "visualbert_dense", "vilbert_cc_dense", "vilbert_cc", "vilbert_cc_caption", "visualbert_coco_dense", "visualbert_coco", "visualbert_coco_caption"]

models_v7 = [mmbt_grid, mmbt_region, finetune_vilbert_cc_with_caption, finetune_vilbert_with_caption,
finetune_visual_bert_coco_with_caption, finetune_visual_bert_with_caption]
modelname_v7 = ["mmbt_grid", "mmbt_region","vilbert_cc_caption", "vilbert_caption",  "visual_bert_coco_caption", "visual_bert_caption"]

models_v8 = [finetune_vilbert_cc_with_caption, finetune_vilbert_cc_without_caption, finetune_vilbert_with_caption,
finetune_visual_bert_coco_with_caption,finetune_visual_bert_coco_without_caption, finetune_visual_bert_with_caption]
modelname_v8 = ["vilbert_cc_caption", "vilbert_cc", "vilbert_caption", "visual_bert_coco_caption", "visual_bert_coco", "visual_bert_caption"]

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
    labels_p = []
    for i in range(0, len(labels)):
        labels_vote.append(1 if labels.iloc[i]['sum'] >(len(modelnames)/2) else 0)
        labels_p.append(labels.iloc[i]['sum']/len(modelnames))
    labels["majority_vote"] = labels_vote
    labels["majority_vote_p"] = labels_p
    # print(labels.head())

    # probability average
    probs["sum"]=probs.sum(axis=1)
    probs_value = []
    probs_p = []
    for i in range(0, len(probs)):
        probs_value.append(1 if (probs.iloc[i]['sum']/len(modelnames)) > 0.5 else 0)
        probs_p.append(probs.iloc[i]['sum']/len(modelnames))
    probs["probs_average"] = probs_value
    probs["probs_average_p"] = probs_p
    # print(probs.head())

    result = pd.DataFrame()
    result['id'] = mmbt_grid['id']
    result['majority_vote'] = labels["majority_vote"]
    result['majority_vote_p'] = labels["majority_vote_p"]
    result['probs_averag'] = probs["probs_average"]
    result['probs_averag_p'] = probs["probs_average_p"]
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

    # roc_auc1 = roc_auc_score(df['majority_vote'].tolist(), df['labels'].tolist())
    # roc_auc2 = roc_auc_score(df['probs_averag'].tolist(), df['labels'].tolist())
    roc_auc1 = roc_auc_score(df['labels'].tolist(), df['majority_vote_p'].tolist())
    roc_auc2 = roc_auc_score(df['labels'].tolist(), df['probs_averag_p'].tolist())

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


ensemble(models_v3, modelname_v3)
evaluation_metrics()

