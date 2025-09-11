# Lung Cancer Prediction Project

This project uses machine learning techniques to analyze and predict lung cancer risk based on survey data. The workflow includes data preprocessing, exploratory data analysis, feature engineering, model training, and evaluation.

## Dataset

- **File:** `survey lung cancer.csv`
- **Features:** Gender, Age, Smoking, Yellow_Fingers, Anxiety, Peer_Pressure, Chronic_Disease, Fatigue, Allergy, Wheezing, Alcohol_Consuming, Coughing, Shortness_of_Breath, Swallowing_Difficulty, Chest_Pain, Lung_Cancer

## Project Workflow

1. **Import Libraries**
    - pandas, numpy, seaborn, matplotlib, scikit-learn, imbalanced-learn

2. **Data Loading and Cleaning**
    - Load the dataset and rename columns
    - Remove duplicates and check for missing values

3. **Exploratory Data Analysis (EDA)**
    - Display data info, statistics, and distributions
    - Visualize feature relationships and class balance

4. **Feature Engineering & Preprocessing**
    - Encode categorical variables using LabelEncoder
    - Create new features if needed
    - Handle imbalanced data using ADASYN
    - Split data into training and test sets
    - Scale features using StandardScaler

5. **Model Training**
    - Train K-Nearest Neighbors (KNN) classifier
    - Train Artificial Neural Network (MLPClassifier)
    - Train Naive Bayes classifier

6. **Evaluation**
    - Evaluate models using accuracy, confusion matrix, and classification report
    - Visualize confusion matrices

## Requirements

See `requirements.txt` for exact package versions.  
Main dependencies:
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn
- imbalanced-learn

## How to Run

1. Install dependencies:
    ```
    pip install -r requirements.txt
    ```
2. Place `survey lung cancer.csv` in the project directory.
3. Open and run `Lung_Cancer.ipynb` in Jupyter Notebook or VS Code.

## Results

- The notebook provides visualizations, model performance metrics, and insights into factors affecting lung cancer risk.

---

**Author:** Liza Thea
