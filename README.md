# Exploring Fraudulent eCommerce Transactions

This repository contains a Jupyter Notebook that explores a dataset of eCommerce transactions to identify fraudulent activities. The notebook walks through data cleaning, exploratory data analysis, data preprocessing, and the implementation of several machine learning models to classify transactions as either fraudulent or legitimate.

## Dataset

The dataset used in this analysis is `Fraudulent_E-Commerce_Transaction_Data_2.csv`. It contains transactional data along with customer information.

### Dataset Fields

* **Transaction ID**: A unique identifier for each transaction.
* **Customer ID**: A unique identifier for each customer.
* **Transaction Amount**: The monetary value of the transaction.
* **Transaction Date**: The date and time when the transaction occurred.
* **Payment Method**: The method used for payment (e.g., PayPal, credit card, debit card).
* **Product Category**: The category of the product purchased (e.g., electronics, clothing).
* **Quantity**: The number of items purchased in the transaction.
* **Customer Age**: The age of the customer.
* **Customer Location**: The location of the customer.
* **Device Used**: The type of device used to make the transaction (e.g., desktop, mobile, tablet).
* **IP Address**: The IP address from which the transaction was made.
* **Shipping Address**: The address where the items were shipped.
* **Billing Address**: The billing address associated with the payment method.
* **Is Fraudulent**: A binary indicator where 1 represents a fraudulent transaction and 0 represents a legitimate one.
* **Account Age Days**: The age of the customer's account in days.
* **Transaction Hour**: The hour of the day when the transaction was made.

## Exploration and Analysis

The notebook performs the following steps:

1. **Initial Setup**: Imports necessary libraries such as `pandas`, `numpy`, `matplotlib`, and `seaborn`.
2. **Data Loading and Cleaning**: The dataset is loaded, and a basic cleaning process is performed, which includes checking for and handling null values and duplicate entries. Several columns that are not directly useful for modeling (like IDs and addresses) are dropped.
3. **Exploratory Data Analysis (EDA)**:
   * A pie chart visualizes the distribution of fraudulent versus legitimate transactions, highlighting the class imbalance in the dataset. * Pie charts are also used to show the distribution of different payment methods, product categories, and device types.
   * A correlation heatmap is generated to understand the relationships between the different numerical features in the dataset.
4. **Data Preprocessing**:
   * **Label Encoding**: Categorical features like `Product Category`, `Payment Method`, and `Device Used` are converted into numerical format using `LabelEncoder`.
   * **SMOTE (Synthetic Minority Over-sampling Technique)**: To address the imbalance between fraudulent and legitimate transactions, SMOTE is used to oversample the minority class (fraudulent transactions).
5. **Model Implementation**: The dataset is split into training and testing sets. Several classification models are trained and evaluated:
   * Logistic Regression
   * Random Forest
   * Decision Tree
   * Gaussian Naive Bayes
   * Support Vector Machine (SVM)
   * K-Nearest Neighbors (KNN)
   * AdaBoost
   * An extension using a `VotingClassifier` that combines the predictions of a Bagged Random Forest and a Boosted Decision Tree.
6. **Model Evaluation and Comparison**:
   * For each model, a confusion matrix is plotted to visualize its performance.
   * The accuracy, precision, recall, and F1-score are calculated for each model and stored.
   * A final dataframe and bar charts are created to compare the performance of all the implemented models.
7. **Model Saving**: The best-performing model (the Voting Classifier) is saved to a file named `model.sav` using `joblib` for future use.
