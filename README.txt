CS 6840 / 4840  - Machine Learning
Final Project    :  Heart Stroke Prediction
Group            :  Data Explorers

Team Members:
  Bhargav Yendluri- UID: U01148249
  Kushal Bhandari- UID: U01137652

Dataset: 
Heart Stroke Prediction
Size: 5110 patients dataset
Features: 10 input features + 1 target variable (stroke: 1 = had stroke, 0 = did not)
Challenge: Only 4.87% of patients actually had a stroke, which makes this a class imbalance problem


Description:
For our final project, we made a Heart stroke prediction system using patient data from Kaggle.
The dataset has 5,110 records with features like age, glucose level, BMI, hypertension, and smoking status.
We used three models which are Logistic Regression, Random Forest, and K-Nearest Neighbors.
One of the bigger challenges was the class imbalance since only about approximately 5% of patients in the dataset actually had a stroke,
so we used SMOTE to balance the training data.

Used Models:
  1. Logistic Regression
  2. Random Forest
  3. K-Nearest Neighbors (K=7)

Libraries Used:
pandas: Loading the CSV, data cleaning, building summary tables
numpy: Numerical operations, array math, replacing NaN values
matplotlib: All charts like bar graphs, ROC curves, radar chart, confusion matrices
seaborn: Correlation heatmap styling
scikit-learn: Train/test split, feature scaling, label encoding, all three models, all evaluation metrics
imbalanced-learn: SMOTE oversampling to fix the class imbalance problem

