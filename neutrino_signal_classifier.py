"""
Classifying Neutrino Signal from Noise in HPGe Detector
This module predicts whether data points from HPGe detectors indicate the presence of a neutrino signal or just noise.
"""

import numpy as np
import pandas as pd

def prepare_data(filepath):
    """Load and prepare data for training."""
    data = pd.read_csv(filepath)
    X = np.column_stack((
        np.ones(len(data)),  # Intercept
        data['tDrift50'], 
        data['tDrift90'],
        data['tDrift100'],
        data['blnoise'],
        data['tslope'],
        data['Energy'],
        data['Current_Amplitude']
    ))
    Y = data['Label'].values
    return X, Y

def train_model(X, Y):
    """Train the model using normal equation."""
    X_trans = X.T
    return np.linalg.inv(X_trans.dot(X)).dot(X_trans).dot(Y)

def predict(row, weights):
    """Predict the score indicating likelihood of signal presence."""
    formatted_row = np.insert(row, 0, 1)  # Insert 1 at position 0 for the intercept
    return np.dot(weights, formatted_row)

def roc_auc(label, score):
    """Calculate the ROC AUC score."""
    score = np.array(score)
    label = np.array(label)
    tpr = []
    fpr = []
    sigscore = score[label == 1]
    bkgscore = score[label == 0]
    for thr in np.linspace(min(score), max(score), 10000):
        tpr.append(np.sum(sigscore >= thr) / len(sigscore))
        fpr.append(np.sum(bkgscore >= thr) / len(bkgscore))
    return np.trapz(tpr, 1 - np.array(fpr))

def calculate_AUC(df, weights):
    """Compute ROC_AUC_score of the predictions for the dataframe."""
    scores = [predict(df.iloc[i].drop('Label'), weights) for i in range(df.shape[0])]
    return roc_auc(df['Label'].values, scores)

def main():
    """Run the training and prediction process."""
    # Prepare data
    X, Y = prepare_data('training_classification.csv')
    
    # Train model
    weights = train_model(X, Y)
    
    # Load data to predict
    waveforms = pd.read_csv('training_classification.csv')
    
    # Calculate AUC on the training set
    auc_score = calculate_AUC(waveforms, weights)
    print(f"ROC AUC Score: {auc_score}")

if __name__ == '__main__':
    main()
