# Admission Success Predictor 

## Project Overview

The Admission Success Predictor is a machine learning project that estimates the likelihood of a graduate applicant being admitted based on academic profile data such as GRE score, TOEFL score, CGPA, and recommendation strength. The project is designed to support data-driven decision-making for admissions analysis.

## Objectives

Predict admission outcomes using supervised learning

Compare Logistic Regression and Random Forest models

Explain predictions using feature importance and model coefficients

Deliver a reproducible and well-structured ML pipeline

## Dataset

Name: Graduate Admissions Factors Dataset
link: https://www.kaggle.com/datasets/mohansacharya/graduate-admissions

Source: Kaggle

Features:

GRE Score

TOEFL Score

University Rating

SOP

LOR

CGPA

Research

Target: Chance of Admit (converted to binary classification)

## Preprocessing Steps:

Removed identifier column (Serial_No)

Standardized column names

Handled missing values using median imputation

Scaled features for Logistic Regression

## Models Used

Logistic Regression (primary model)

Random Forest Classifier (comparison model)

| Model               | Accuracy | Recall   | F1-Score  |
| ------------------- | -------- | -------- | --------- |
| Logistic Regression | 0.94     | 0.99     | 0.97      |
| Random Forest       | 0.92     | 0.97     | 0.95      |

![alt text](strealit_1.png)
![alt text](streamlit_2.png)