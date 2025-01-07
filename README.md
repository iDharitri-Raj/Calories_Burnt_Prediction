# Calories Prediction using XGBoost

Predicting calories burned by individuals based on their exercise data is crucial for personalized fitness planning and health monitoring. This project uses machine learning techniques to predict calorie consumption from various features like age, gender, height, weight, and heart rate. The XGBoost Regressor model is used to make predictions, providing valuable insights into caloric expenditure during physical activities.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Data Analysis and Preprocessing](#data-analysis-and-preprocessing)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Results](#results)
- [Usage](#usage)
- [License](#license)

## Overview
This project focuses on predicting the number of calories burned based on a combination of features related to the user and their exercise habits. We used data from two datasets: one containing information about the individual (such as gender, age, height, and weight), and the other describing their exercise routine (such as exercise duration, heart rate, and body temperature). After merging the datasets and performing necessary preprocessing steps, we employed an XGBoost Regressor to predict the target variable—calories burned.

## Dataset
The dataset contains the following columns:

| Column Name     | Description                                             |
|-----------------|---------------------------------------------------------|
| `User_ID`         | Unique identifier for the user                         |
| `Gender`         | Gender of the user (male or female)                    |
| `Age`             | Age of the user                                        |
| `Height`          | Height of the user (in cm)                             |
| `Weight`          | Weight of the user (in kg)                             |
| `Duration`        | Duration of the exercise (in minutes)                  |
| `Heart_Rate`      | Heart rate during the exercise (beats per minute)      |
| `Body_Temp`       | Body temperature during exercise (in Celsius)          |
| `Calories`        | Target variable: Calories burned during the exercise   |

## Installation

Clone the repository:
```bash
git clone https://github.com/iDharitri-Raj/Calories_Burnt_Prediction
```
Navigate to the project directory and install the required dependencies.

## Dependencies
This project requires the following Python libraries:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `xgboost`
- `scikit-learn`

To install the dependencies, run:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

## Data Analysis and Preprocessing

- **Handling Missing Values**: Missing values were imputed or dropped based on the nature of the data.
- **Encoding Categorical Data**: The 'Gender' column was encoded into numerical values (0 for male, 1 for female).
- **Feature Visualization**: Various visualizations were generated to explore the distribution of features such as age, height, weight, and calories.
- **Splitting Data**: The data was split into features (X) and target (Y), followed by a train-test split (80-20 ratio).

## Model Training and Evaluation

- **Model Used**: XGBoost Regressor
- **Hyperparameters**:
  - `n_estimators`: 100
  - `learning_rate`: 0.1
  - `random_state`: 42

- **Performance Metrics**:
  - R² Score on Training Data: 0.9996
  - R² Score on Test Data: 0.9988
  - Mean Absolute Error on Training Data: 0.9322
  - Mean Absolute Error on Test Data: 1.4833

## Results

The XGBoost Regressor performed exceptionally well, achieving a high R² score on both the training and test datasets. The model demonstrated a strong ability to predict calories burned based on the input features. The mean absolute error on the test data was slightly higher, indicating some room for improvement in model accuracy.

## Usage

1. Load the dataset and preprocess it (handle missing values, encode categorical data, etc.).
2. Split the data into training and testing sets.
3. Train the XGBoost Regressor model on the training data.
4. Evaluate the model's performance using R² score and mean absolute error.
5. Use the trained model to predict calories burned for new exercise data.

## License
This project is licensed under the [**MIT License**](LICENSE) 
