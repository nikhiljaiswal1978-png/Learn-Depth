# Task — Loan Approval Analysis

## Learn Depth™ Internship

This project was completed as part of my **Learn Depth™ Internship**.

The objective of this task was to analyze financial information and identify factors associated with **loan approval**. The task focused on exploratory data analysis (EDA), data quality checking, data cleaning, visualization, and basic feature engineering before preparing the dataset for Machine Learning.

## Dataset

The dataset contains financial and demographic information about loan applicants.

### Features

* `age`
* `income_lakh`
* `credit_score`
* `loan_amount_lakh`
* `existing_loans`
* `approved`

Where:

* `income_lakh` represents income in lakhs
* `loan_amount_lakh` represents the requested loan amount in lakhs
* `credit_score` represents the applicant's credit score
* `existing_loans` represents the number of existing loans
* `approved` is the target variable indicating whether the loan was approved

## Tasks Performed

### 1. Dataset Inspection

* Inspected dataset structure
* Examined data types
* Reviewed descriptive statistics
* Checked dataset dimensions

### 2. Data Quality Analysis

Identified:

* Missing values
* Duplicate rows
* Unrealistic ages
* Invalid credit scores
* Negative income values
* Negative existing-loan values

### 3. Exploratory Data Analysis

Created visualizations to analyze:

* Approved vs. rejected applications
* Credit-score distribution
* Income distribution
* Credit score vs. loan approval
* Income vs. loan approval
* Loan amount vs. loan approval

### 4. Correlation Analysis

Created a correlation heatmap to understand relationships between the numerical variables and loan approval.

### 5. Data Cleaning

The dataset was cleaned by:

* Removing duplicate records
* Identifying invalid values
* Handling missing values
* Replacing unrealistic/invalid numerical values appropriately
* Preparing the dataset for further analysis and Machine Learning

### 6. Feature Engineering

Created the following features:

#### `debt_burden`

Represents the relationship between existing loans and income.

#### `loan_to_income`

Represents the requested loan amount relative to the applicant's income.

These features provide additional information that can potentially be useful for understanding loan approval patterns.

### 7. Preparing Data for Machine Learning

The cleaned dataset was divided into:

* **X** — input features
* **y** — target variable (`approved`)

## Project Structure

```text
Task-10-Loan-Approval/
│
├── 10_loan_approval.csv
├── loan_approval_cleaned.csv
├── Loan_Approval_Analysis.ipynb
└── README.md
```

## Tools & Technologies

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Jupyter Notebook

## Key Learning

This task helped me understand that Machine Learning begins well before model training.

Data inspection, identifying data-quality problems, understanding relationships through visualization, making appropriate cleaning decisions, and creating meaningful features are important steps in preparing data for Machine Learning.

The project strengthened my practical understanding of:

**EDA → Data Cleaning → Visualization → Feature Engineering → ML Preparation**

## Internship

Completed as part of my **Learn Depth™ Internship**.

This repository contains my implementation and learning from the assigned task.
