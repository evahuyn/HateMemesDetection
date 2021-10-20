# Reproducing Baseline

## Using Pretrained Checkpoints

#### **Image-Grid**

```latex
mmf_run config=projects/hateful_memes/configs/unimodal/image.yaml \
model=unimodal_image dataset=hateful_memes run_type=val \
checkpoint.resume_zoo=unimodal_image.hateful_memes.images \
checkpoint.resume_pretrained=False

mmf_run config=projects/hateful_memes/configs/unimodal/image.yaml \
model=unimodal_image dataset=hateful_memes run_type=test \
checkpoint.resume_zoo=unimodal_image.hateful_memes.images \
checkpoint.resume_pretrained=False

mmf_predict config=projects/hateful_memes/configs/unimodal/image.yaml model=unimodal_image dataset=hateful_memes \
run_type=test checkpoint.resume_zoo=unimodal_image.hateful_memes.images checkpoint.resume_pretrained=False
```
- Result
```latex
val/hateful_memes/cross_entropy: 0.7144, val/total_loss: 0.7144, val/hateful_memes/accuracy: 0.6185, val/hateful_memes/binary_f1: 0.2190, val/hateful_memes/roc_auc: 0.5743

test/hateful_memes/cross_entropy: 0.7415, test/total_loss: 0.7415, test/hateful_memes/accuracy: 0.6060, test/hateful_memes/binary_f1: 0.2216, test/hateful_memes/roc_auc: 0.5494
```

#### **Image-Region**

```latex
mmf_run config=projects/hateful_memes/configs/unimodal/with_features.yaml \
model=unimodal_image dataset=hateful_memes run_type=val \
checkpoint.resume_zoo=unimodal_image.hateful_memes.features \
checkpoint.resume_pretrained=False

mmf_run config=projects/hateful_memes/configs/unimodal/with_features.yaml \
model=unimodal_image dataset=hateful_memes run_type=test \
checkpoint.resume_zoo=unimodal_image.hateful_memes.features \
checkpoint.resume_pretrained=False

mmf_predict config=projects/hateful_memes/configs/unimodal/with_features.yaml model=unimodal_image dataset=hateful_memes \
run_type=test checkpoint.resume_zoo=unimodal_image.hateful_memes.features checkpoint.resume_pretrained=False
```
- Result
```latex
val/hateful_memes/cross_entropy: 0.7382, val/total_loss: 0.7382, val/hateful_memes/accuracy: 0.5759, val/hateful_memes/binary_f1: 0.1331, val/hateful_memes/roc_auc: 0.4734

test/hateful_memes/cross_entropy: 0.6621, test/total_loss: 0.6621, test/hateful_memes/accuracy: 0.6165, test/hateful_memes/binary_f1: 0.3309, test/hateful_memes/roc_auc: 0.5922
```

#### **Text BERT**

```latex
mmf_run config=projects/hateful_memes/configs/unimodal/bert.yaml model=unimodal_text dataset=hateful_memes \
run_type=val checkpoint.resume_file=./save/unimodal_text_final.pth checkpoint.resume_pretrained=False

mmf_run config=projects/hateful_memes/configs/unimodal/bert.yaml model=unimodal_text dataset=hateful_memes \
run_type=test checkpoint.resume_file=./save/unimodal_text_final.pth checkpoint.resume_pretrained=False

mmf_predict config=projects/hateful_memes/configs/unimodal/bert.yaml model=unimodal_text dataset=hateful_memes \
run_type=test checkpoint.resume_file=./save/unimodal_text_final.pth checkpoint.resume_pretrained=False
```
- Result
```latex
val/hateful_memes/cross_entropy: 0.7342, val/total_loss: 0.7342, val/hateful_memes/accuracy: 0.6278, val/hateful_memes/binary_f1: 0.3464, val/hateful_memes/roc_auc: 0.5992

test/hateful_memes/cross_entropy: 0.7204, test/total_loss: 0.7204, test/hateful_memes/accuracy: 0.6395, test/hateful_memes/binary_f1: 0.3632, test/hateful_memes/roc_auc: 0.6304
```

#### **MMBT-Grid**

```latex
mmf_run config=projects/hateful_memes/configs/mmbt/defaults.yaml model=mmbt dataset=hateful_memes \
run_type=val checkpoint.resume_zoo=mmbt.hateful_memes.images checkpoint.resume_pretrained=False

mmf_run config=projects/hateful_memes/configs/mmbt/defaults.yaml model=mmbt dataset=hateful_memes \
run_type=test checkpoint.resume_zoo=mmbt.hateful_memes.images checkpoint.resume_pretrained=False

mmf_predict config=projects/hateful_memes/configs/mmbt/defaults.yaml model=mmbt dataset=hateful_memes \
run_type=test checkpoint.resume_zoo=mmbt.hateful_memes.images checkpoint.resume_pretrained=False
```
- Result
```latex
val/hateful_memes/cross_entropy: 1.3524, val/total_loss: 1.3524, val/hateful_memes/accuracy: 0.6648, val/hateful_memes/binary_f1: 0.3903, val/hateful_memes/roc_auc: 0.6783

test/hateful_memes/cross_entropy: 1.3248, test/total_loss: 1.3248, test/hateful_memes/accuracy: 0.6665, test/hateful_memes/binary_f1: 0.4220, test/hateful_memes/roc_auc: 0.6845
```

#### **ViLBERT**

Note: For this one the pretrained model is somehow not working, I retrained the model use their hyperparameter and get the result

```latex
mmf_run config=projects/hateful_memes/configs/vilbert/defaults.yaml model=vilbert dataset=hateful_memes run_type=val checkpoint.resume_file=./save/vilbert_final.pth checkpoint.resume_pretrained=False

mmf_run config=projects/hateful_memes/configs/vilbert/defaults.yaml model=vilbert dataset=hateful_memes run_type=test checkpoint.resume_file=./save/vilbert_final.pth checkpoint.resume_pretrained=False

mmf_predict config=projects/hateful_memes/configs/vilbert/defaults.yaml model=vilbert dataset=hateful_memes \
run_type=test checkpoint.resume_file=./save/vilbert_final.pth checkpoint.resume_pretrained=False
```
- Result
```latex
val/hateful_memes/cross_entropy: 2.2974, val/total_loss: 2.2974, val/hateful_memes/accuracy: 0.6889, val/hateful_memes/binary_f1: 0.4723, val/hateful_memes/roc_auc: 0.7151

test/hateful_memes/cross_entropy: 2.3136, test/total_loss: 2.3136, test/hateful_memes/accuracy: 0.7030, test/hateful_memes/binary_f1: 0.5037, test/hateful_memes/roc_auc: 0.7352
```

#### **VisualBERT**

```latex
mmf_run config=projects/hateful_memes/configs/visual_bert/direct.yaml \
model=visual_bert dataset=hateful_memes run_type=val \
checkpoint.resume_zoo=visual_bert.finetuned.hateful_memes.direct \
checkpoint.resume_pretrained=False

mmf_run config=projects/hateful_memes/configs/visual_bert/direct.yaml \
model=visual_bert dataset=hateful_memes run_type=test \
checkpoint.resume_zoo=visual_bert.finetuned.hateful_memes.direct \
checkpoint.resume_pretrained=False

mmf_predict config=projects/hateful_memes/configs/visual_bert/direct.yaml model=visual_bert dataset=hateful_memes \
run_type=test checkpoint.resume_zoo=visual_bert.finetuned.hateful_memes.direct checkpoint.resume_pretrained=False
```
- Result
```latex
val/hateful_memes/cross_entropy: 1.2431, val/total_loss: 1.2431, val/hateful_memes/accuracy: 0.6685, val/hateful_memes/binary_f1: 0.4120, val/hateful_memes/roc_auc: 0.6944

test/hateful_memes/cross_entropy: 1.0653, test/total_loss: 1.0653, test/hateful_memes/accuracy: 0.7015, test/hateful_memes/binary_f1: 0.5530, test/hateful_memes/roc_auc: 0.7286
```