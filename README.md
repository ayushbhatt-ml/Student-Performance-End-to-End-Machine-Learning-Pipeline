# Student Performance Indicator — End-to-End Machine Learning Pipeline

An end-to-end Machine Learning web application designed to predict a student's math performance based on demographic, socioeconomic, and academic attributes. The project features modular component engineering, data preprocessing pipelines, automated hyperparameter optimization across multiple regression algorithms, continuous logging, custom exception tracking, and a web interface built with Flask.

---

## 📋 Table of Contents

- [Overview](#overview)
- [System Architecture & Pipeline Flow](#system-architecture--pipeline-flow)
- [Exploratory Data Analysis (EDA) Summary](#exploratory-data-analysis-eda-summary)
- [Data Transformation & Preprocessing](#data-transformation--preprocessing)
- [Model Training & Evaluation](#model-training--evaluation)
- [Project Directory Structure](#project-directory-structure)
- [Getting Started & Installation](#getting-started--installation)
- [Usage & Web Interface](#usage--web-interface)
- [License](#license)

---

## 📊 Overview

The primary objective of this project is to model student performance outcomes (`math_score`) by processing multi-dimensional inputs including gender, ethnicity, parental education level, lunch status, test preparation completion, and standard academic test scores (`reading_score`, `writing_score`).

Key features of the codebase include:
- Modularized production code under the `src/` directory.
- Centralized custom exception handling logging exact file names and line numbers.
- Detailed timestamped file logging (`logs/`).
- Automated model benchmark and selection via `GridSearchCV`.
- Model artifact persistence (`model.pkl` and `preprocessor.pkl`) using `dill`.

---

## 🔄 System Architecture & Pipeline Flow

```
+------------------+     +--------------------------+     +-------------------------+
|  Raw Data CSV    | --> | Data Ingestion Component  | --> | Artifacts Generation    |
| (data/stud.csv)  |     | (data_ingestion.py)      |     | (train.csv / test.csv)  |
+------------------+     +--------------------------+     +-------------------------+
                                                                       |
                                                                       v
+------------------+     +--------------------------+     +-------------------------+
| Saved Pickles    | <-- | Model Trainer Component  | <-- | Transformation Pipeline |
| (preprocessor,   |     | (model_trainer.py)       |     | (data_transformation.py)|
|  model.pkl)      |     +--------------------------+     +-------------------------+
+------------------+
         |
         v
+------------------+     +--------------------------+
|  Flask Web App   | <-- | Predict Pipeline         |
|   (app.py)       |     | (predict_pipeline.py)    |
+------------------+     +--------------------------+
```

1. **Data Ingestion (`src/components/data_ingestion.py`)**: Loads dataset from raw sources, splits into train/test datasets ($80/20$ split), and saves processed files to the `artifacts/` folder.
2. **Data Transformation (`src/components/data_transformation.py`)**: Assembles feature encoding and numerical normalization pipelines into a unified `ColumnTransformer` saved as `artifacts/preprocessor.pkl`.
3. **Model Training (`src/components/model_trainer.py`)**: Evaluates candidate regressors using `GridSearchCV`, tracks performance via $R^2$ score, and exports the highest-performing model as `artifacts/model.pkl`.
4. **Prediction Pipeline (`src/pipeline/predict_pipeline.py`)**: Reads user inputs through the `CustomData` class, formats input attributes into a Pandas DataFrame, and applies `PredictPipeline` to return model inferences.

---

## 📈 Exploratory Data Analysis (EDA) Summary

The analysis was performed on **1,000 rows and 8 features** with **zero missing or duplicate values**.

### Dataset Features

| Feature Name | Type | Description / Categories |
| :--- | :--- | :--- |
| `gender` | Categorical | `female`, `male` |
| `race_ethnicity` | Categorical | `group A`, `group B`, `group C`, `group D`, `group E` |
| `parental_level_of_education` | Categorical | `bachelor's degree`, `some college`, `master's degree`, `associate's degree`, `high school`, `some high school` |
| `lunch` | Categorical | `standard`, `free/reduced` |
| `test_preparation_course` | Categorical | `none`, `completed` |
| `reading_score` | Numerical | Score ranging from 17 to 100 |
| `writing_score` | Numerical | Score ranging from 10 to 100 |
| **`math_score`** | **Numerical (Target)** | **Score ranging from 0 to 100** |

### Insights

- **Math Performance**: Lowest scoring section overall ($4$ students scored $\le 20$, minimum score of $0$).
- **Reading Performance**: Highest overall performance ($17$ students achieved a perfect score of $100$).
- **Score Distributions**: Overall average scores across subjects display an approximately normal distribution centered between $66$ and $69$.

---

## 🛠️ Data Transformation & Preprocessing

The preprocessing pipeline defined in `DataTransformation.get_data_transformer_object()` implements:

- **Numerical Features** (`writing_score`, `reading_score`):
  - `SimpleImputer(strategy="median")`
  - `StandardScaler()`
- **Categorical Features** (`gender`, `race_ethnicity`, `parental_level_of_education`, `lunch`, `test_preparation_course`):
  - `SimpleImputer(strategy="most_frequent")`
  - `OneHotEncoder()`
  - `StandardScaler(with_mean=False)`

---

## ⚡ Model Training & Evaluation

The candidate algorithms evaluated via multi-fold `GridSearchCV` cross-validation include:
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor (`XGBRegressor`)
- CatBoost Regressor (`CatBoostRegressor`)
- AdaBoost Regressor

The final model selected is the algorithm yielding the highest $R^2$ evaluation score on the test dataset (with a default threshold enforcement of $R^2 \ge 0.60$).

---

## 📁 Project Directory Structure

```text
student-performance-ml/
├── artifacts/                  # Persisted model, preprocessor pickles & splits
│   ├── data.csv
│   ├── train.csv
│   ├── test.csv
│   ├── model.pkl
│   └── preprocessor.pkl
├── data/
│   └── stud.csv                # Raw input dataset
├── notebook/
│   ├── 1. EDA STUDENT PERFORMANCE.ipynb
│   └── 2. MODEL TRAINING.ipynb
├── src/
│   ├── __init__.py
│   ├── components/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py   # Ingestion component
│   │   ├── data_transformation.py # Transformation component
│   │   └── model_trainer.py    # Training & selection component
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── train_pipeline.py   # Training trigger pipeline
│   │   └── predict_pipeline.py # Inference wrapper module
│   ├── exception.py            # Centralized Custom Exception logging
│   ├── logger.py               # Custom logging setup
│   └── utils.py                # Helper utilities (save_object, evaluate_models)
├── templates/
│   ├── index.html              # Web app landing page
│   └── home.html               # Prediction form UI
├── app.py                      # Flask application entry point
├── requirements.txt            # Python dependencies
├── setup.py                    # Package deployment setup script
└── README.md                   # Documentation file
```

---

## 🚀 Getting Started & Installation

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)

### Step 1: Clone Repository & Create Environment

```bash
git clone https://github.com/your-username/student-performance-ml.git
cd student-performance-ml
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Run Training Pipeline

Execute the data ingestion component to trigger preprocessing and model selection:

```bash
python src/components/data_ingestion.py
```

---

## 💻 Usage & Web Interface

Launch the Flask development server:

```bash
python app.py
```

Open your web browser and navigate to:
```text
http://127.0.0.1:5000/predictdata
```

Enter student feature details (gender, race/ethnicity, parent education level, lunch type, test prep course status, reading score, and writing score) to receive real-time predictions for predicted math scores.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
