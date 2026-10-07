# RNN-Based Sequence Classifier

## Problem Statement

Develop an RNN-based sequence classifier and experiment with
different sequence lengths. Demonstrate predictions and apply
the trained model to new sequences.

## Objective

The objective is to build an RNN that learns sequential patterns
and classifies numerical sequences as increasing or decreasing.

## Dataset

The dataset contains numerical sequences with three different
sequence lengths:

- 5
- 10
- 20

Each sequence belongs to one of two classes:

- 0 - Decreasing
- 1 - Increasing

## RNN Architecture

Input Sequence
→ RNN Layer
→ Hidden Representation
→ Fully Connected Layer
→ Binary Classification

## Experiments

The model is trained separately using sequence lengths of 5,
10, and 20.

The accuracy of each experiment is compared.

## New Sequence Prediction

After training, unseen sequences are provided to the model.
The model predicts whether the sequence is increasing or
decreasing.

## Technologies

- Python
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Jupyter Notebook

## Conclusion

The project demonstrates how an RNN can learn temporal patterns
from numerical sequences. Experiments with different sequence
lengths show how sequence size affects classification performance.
The trained model can also classify new unseen sequences.