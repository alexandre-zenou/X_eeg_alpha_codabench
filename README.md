# EEG Alpha Waves – Codabench Challenge (Rodrigues2017)

## Group members
- Alexandre HEYMANN
- Mohamed MOHAMED EL BECHIR
- Noé AMAR
- Olivier Fontaine
- Alexandre Zenou

## Overview

This repository defines a fully working Codabench challenge for a binary
classification task based on EEG recordings.

It includes:
- A complete competition bundle (`bundle.zip`)
- Ingestion and scoring programs
- Prepared dataset splits (train / public test / private test)
- Competition web pages (`pages/`)
- A mandatory starting kit notebook (`starting_kit.ipynb`)
- A reproducible data preparation script (`tools/setup_data.py`)

---

## Dataset and Data Origin

We use the **EEG Alpha Waves** dataset, available in MOABB under the name `Rodrigues2017`.

**Original dataset source (Zenodo):**
[https://zenodo.org/record/2348892](https://zenodo.org/record/2348892)

The dataset is automatically downloaded via MOABB/MNE when running:
```bash
python tools/setup_data.py
```

---

## Task

### Goal
Build a binary classifier from EEG-derived features.

### Feature Extraction

During dataset construction:
- EEG signals are segmented into **2-second windows**.
- For each channel and window, we compute:
  - Mean
  - Standard deviation
  - Alpha band power (8–12 Hz) using Welch PSD integration.

This produces a tabular dataset compatible with standard scikit-learn models.

### Labels

Binary labels are constructed:
- From event annotations when available.
- Otherwise, using a fallback time-based split (early vs late recording).

---

## Evaluation Strategy

### Metric

The challenge uses **Accuracy**, defined as:

$$\text{Accuracy} = \frac{\text{Number of correct predictions}}{\text{Total number of predictions}}$$

Since the dataset is balanced, accuracy is appropriate.

The metric is implemented in:
```
scoring_program/scoring.py
```

### Splitting Strategy

We use a **subject-wise split**:
- Train
- Public test
- Private test

This prevents data leakage across subjects and evaluates generalization to unseen participants.

---

## Submission Format

Participants must submit a single Python file:
```
submission.py
```

Containing:
```python
def get_model():
    return ...
```

The ingestion program will call:
```python
model.fit(X_train, y_train)
model.predict(X_test)
```

---

## Local Usage

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Generate dataset splits**
```bash
python tools/setup_data.py
```

**3. Test ingestion**
```bash
python ingestion_program/ingestion.py \
  --data-dir dev_phase/input_data/ \
  --output-dir ingestion_res/ \
  --submission-dir solution/
```

**4. Test scoring**
```bash
python scoring_program/scoring.py \
  --reference-dir dev_phase/reference_data/ \
  --prediction-dir ingestion_res/ \
  --output-dir scoring_res/
```

**5. Create bundle**
```bash
python tools/create_bundle.py
```

This generates:
```
bundle.zip
```

---

## Deliverables

Final submission includes:
- GitHub repository link
- Codabench competition URL
- Working bundle
- Starting kit notebook
- 

