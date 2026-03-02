# Overview

## Goal
Build a model that predicts a binary label from EEG-derived features computed on short windows.

## Dataset
We use the EEG Alpha Waves dataset (Rodrigues2017), available on Zenodo:
https://zenodo.org/record/2348892

The dataset is downloaded and prepared automatically by `tools/setup_data.py` (via MOABB/MNE).

## Feature extraction
From each 2-second EEG window, we compute simple features per channel:
- mean
- standard deviation
- alpha band power (8–12 Hz)

## Evaluation
Metric: **Accuracy**.

## Splitting strategy
We use a **subject-wise split** into:
- train
- public test
- private test

This helps reduce leakage across subjects and better reflects generalization to unseen participants.