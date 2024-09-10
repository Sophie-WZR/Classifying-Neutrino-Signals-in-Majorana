# Classifying Neutrino Signals in Majorana Demonstrator

This project implements a simple machine learning model to classify whether data points from HPGe detectors indicate the presence of a **neutrino signal** or just **noise**. The model achieves an **ROC AUC score of 0.84**, demonstrating strong predictive performance.

## Overview

High-Purity Germanium (HPGe) detectors are commonly used in astrophysics to detect signals such as neutrinos. This project builds a machine learning classifier that differentiates between genuine neutrino signals and background noise using features extracted from HPGe waveform data.

## Features

The model is trained using a **linear regression** approach with the normal equation to solve for the optimal weights. The performance of the model is evaluated using the **ROC AUC score**, which measures its ability to distinguish between neutrino signals and noise.

### Input Features:
- `tDrift50`: Time drift at 50% amplitude
- `tDrift90`: Time drift at 90% amplitude
- `tDrift100`: Time drift at 100% amplitude
- `blnoise`: Baseline noise level
- `tslope`: Slope of the signal
- `Energy`: Energy of the event
- `Current_Amplitude`: Amplitude of the current

### Output:
- The model outputs a score indicating the likelihood of the presence of a neutrino signal.

## Model

The model uses the following components:
1. **Data Preparation**: The features are extracted and prepared from the input CSV file using `pandas` and `numpy`.
2. **Model Training**: A linear regression model is trained using the **normal equation** to find the optimal weights.
3. **Prediction**: Given new waveform data, the model predicts the score indicating the likelihood of a neutrino signal.
4. **Model Evaluation**: The **ROC AUC** metric is used to evaluate the model's performance on the training set.

## Usage

### 1. Data Preparation

You need a CSV file containing the following columns:
- `tDrift50`
- `tDrift90`
- `tDrift100`
- `blnoise`
- `tslope`
- `Energy`
- `Current_Amplitude`
- `Label` (1 for signal, 0 for noise)

### 2. Run the Model

To run the model, simply execute the `main()` function after ensuring that the required libraries are installed and the CSV data is provided in the correct format.

```bash
python neutrino_signal_classifier.py
