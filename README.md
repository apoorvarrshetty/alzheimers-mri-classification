# SYNAPSERA NeuroMemory 🧠

An explainable **research and educational prototype** for three-category classification of 2D brain MRI slices with PyTorch, ResNet18, Streamlit, and Grad-CAM.

> **Not a medical device.** Predictions are research outputs, not an Alzheimer's diagnosis, medical advice, or calibrated risk estimates. Do not use the app for clinical decisions or upload identifiable patient information.

## Dataset and study scope

Dataset: [Images OASIS (Kaggle)](https://www.kaggle.com/datasets/ninadaithal/imagesoasis). The image collection has 86,437 slices derived from 347 identifiable subjects in the filename audit. Only three image labels were modeled:

- Non Demented: 266 subjects, 67,222 slices
- Very mild Dementia: 58 subjects, 13,725 slices
- Mild Dementia: 21 subjects, 5,002 slices

**Moderate Dementia** (2 subjects / 488 slices) was excluded from training/evaluation because it cannot support a meaningful independent subject-level evaluation. These labels are dataset categories, not a validated diagnostic staging system.

The data were split by subject identifier, rather than image slice: training 241 subjects / 60,207 slices; validation 52 / 12,932; held-out test 52 / 12,810. Filename-based grouping prevents overlapping identified subjects across splits, but does not establish the absence of all duplicates, biases, or metadata inconsistencies.

## Model

- Torchvision ResNet18 initialized from ImageNet pretraining during training
- Replaced 3-channel input convolution with a 1-channel convolution initialized using the mean of pretrained channel weights
- Classifier output: 3 logits
- MRI transforms: grayscale, resize to 224 × 224, ToTensor, normalize mean=0.5 / std=0.5
- Weighted sampling by subject and class, Adam optimizer; best checkpoint selected by validation subject-level macro-F1
- FP16 **checkpoint storage** to reduce GitHub file size; weights are loaded into an FP32 model for CPU inference

### Preliminary validation (52 subjects)

| Metric | Original CNN | ResNet18 |
| --- | ---: | ---: |
| Accuracy | 69.23% | 75.00% |
| Macro-F1 | 0.3454 | 0.6131 |
| Balanced accuracy | 40.28% | 70.28% |

ResNet18 recall on validation subjects was 0.78 Non Demented (40 subjects), 0.67 Very mild Dementia (9 subjects), and 0.67 Mild Dementia (only 3 subjects). These are **preliminary validation metrics**, not independent test or external clinical validation. The subject-disjoint held-out test set was evaluated once after model selection. Numerous MRI slices from each subject are strongly correlated; counting them as independent patients would mislead.

### Final held-out test (52 subjects / 12,810 MRI slices)

| Metric | ResNet18 |
| --- | ---: |
| Subject-level accuracy | 73.08% (38/52) |
| Subject-level macro-F1 | 0.5965 |
| Subject-level balanced accuracy | 69.44% |

| Category | Precision | Recall | F1 | Test subjects |
| --- | ---: | ---: | ---: | ---: |
| Non Demented | 0.97 | 0.75 | 0.85 | 40 |
| Very mild Dementia | 0.40 | 0.67 | 0.50 | 9 |
| Mild Dementia | 0.33 | 0.67 | 0.44 | 3 |

These held-out results used the **original FP32 training checkpoint**. The deployed web app uses an FP16-compressed copy, whose full test-set prediction agreement has not yet been measured. This is internal, single-dataset validation and should **not** be interpreted as clinical diagnostic performance.

## Run locally

Python 3.10–3.12 recommended.

```bash
git clone https://github.com/apoorvarrshetty/synapsera-neuromemory.git
cd synapsera-neuromemory
python -m pip install -r requirements.txt
streamlit run app.py
```

The model checkpoint `neuromemory_resnet18_fp16.pth` must be in the repository root, alongside `app.py` and `model.py`.

Upload a **de-identified** JPG or PNG of a 2D MRI slice with comparable image appearance and orientation to the study dataset. Other modalities, orientations, and acquisition conditions are out of distribution. The app currently does **not** verify if an image is a suitable MRI. Its softmax scores are **not calibrated medical probabilities**.

## Explainability

Grad-CAM overlays show which image regions contributed to the **predicted class**. They do not prove a causal relationship or identify Alzheimer pathology. Attributions may highlight scanner artifacts, anatomy boundaries, or other confounding visual cues.

## Repository contents

- `app.py`: Streamlit UI, model inference, Grad-CAM
- `model.py`: ResNet18 definition and preprocessing
- `neuromemory_resnet18_fp16.pth`: compact model checkpoint
- `requirements.txt`: Python dependencies

## Remaining validation

Check duplicate and near-duplicate MRI slices, audit attribution artifacts and incorrect predictions, quantify uncertainty for small classes, compare the deployed FP16 model against the original checkpoint on the locked test subjects without further tuning, and pursue independent external validation.

## License

See [LICENSE](LICENSE) for the source-code license; dataset and model-weight usage may be subject to different upstream terms.
