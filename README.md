# [EN] Health Insurance Cross-Sell Prediction

## **🛒 Problem Statement**
The client is an insurance company that provides health insurance to its customers. They need a predictive model to determine whether their existing policyholders will likely purchase vehicle insurance. This will allow the company to optimize its communication strategy and allocate resources efficiently to target the most promising leads.

## **📊 Business Objective**
The objective is to build a model that calculates the customer's propensity to purchase vehicle insurance. This will enable the company to prioritize outreach efforts to the customers most likely to convert, minimizing costs while maximizing sales.

## **📂 Dataset**
The dataset consists of customer demographic information, vehicle details, and policy-related features:

- **Demographics**: Gender, Age, Region Code
- **Vehicle Information**: Vehicle Age, Vehicle Damage History
- **Policy Information**: Annual Premium, Policy Sales Channel
- **Target Variable**: Response (1 = Interested in Vehicle Insurance, 0 = Not Interested)

## **🔄 Project Workflow**

### **1. Data Loading**
The dataset is loaded using Pandas:
```python
import pandas as pd
df_raw = pd.read_csv('train.csv', low_memory=False)
df1 = df_raw.copy()
```

### **2. Data Cleaning & Transformation**
- Standardized column names to snake_case format
- Checked for missing values
- Explored data types and performed necessary conversions

### **3. Exploratory Data Analysis (EDA)**
- Analyzed response variable distribution across different features
- Visualized relationships using histograms, box plots, and bar charts
- Identified key patterns in customer behaviour

### **4. Feature Engineering**
- Created new features where applicable
- Encoded categorical variables using:
  - Label Encoding (e.g., Vehicle Damage, Previously Insured)
  - Target Encoding (e.g., Gender, Region Code, Policy Sales Channel)
  - Ordinal Encoding (e.g., Vehicle Age)
- Scaled numerical variables (Age, Vintage, Annual Premium) using MinMaxScaler and StandardScaler

### **5. Data Splitting & Preparation**
- Split dataset into training (80%) and testing (20%) sets
- Applied feature transformations to both sets
- Ensured there were no missing values post-processing

### **6. Feature Selection**
- Used feature importance techniques (e.g., Extra Trees Classifier) to identify the most relevant features

### **7. Model Development**
Built predictive models using algorithms such as:
- Logistic Regression
- Random Forest
- Gradient Boosting (e.g., XGBoost, LightGBM)

Evaluated models using:
- Accuracy
- Precision, Recall, and F1-score
- ROC-AUC Score

### **8. Model Evaluation & Optimization**
- Fine-tuned hyperparameters using GridSearchCV
- Selected the best-performing model based on business objectives

## **🚀 Deployment**
The model was **deployed as an API on Render**, allowing external access for real-time predictions.

Additionally, the API was integrated with a **Google Sheets spreadsheet** using **Google Apps Script**. This integration enables users to automatically make predictions by inserting new data into the sheet and clicking the prediction button for efficient results.
Link: https://docs.google.com/spreadsheets/d/1DvjzDxXtgxd19dXQTESxDCEhJqUtwvNkuiq-Jy4hUBU/edit?usp=sharing 

## **📈 Results**
- The model provides a ranked list of customers based on their likelihood of purchasing vehicle insurance.
- The business can now focus its marketing efforts on high-propensity leads.

## **🔜 Next Steps**
- Improve model performance by testing hyperparameter optimization techniques.
- Create an interactive dashboard for visualization of predictions.
- Explore new variables that may impact purchase propensity.

## **📂 Repository Structure**
```
|-- data/              # Raw and processed data
|-- notebooks/         # Jupyter notebooks for EDA and modelling
|-- models/            # Trained models and serialized objects
|-- src/               # Scripts for data processing and modelling
|-- README.md          # Project documentation
```

## **🛠 Technologies**
- **Python** (pandas, numpy, sklearn, seaborn, matplotlib)
- **Machine Learning** (Scikit-Learn, XGBoost, LightGBM)
- **Data Visualization** (Seaborn, Matplotlib)

## **👥 Contributor**
Jessica Nadalete - Data Science & Machine Learning Engineer

## **📬 Contact**
For any questions or feedback, reach out via [jessica_nvp@hotmail.com ].

