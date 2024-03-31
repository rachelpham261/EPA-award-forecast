# EPA Award Value Forecasting for Small and Minority-Owned Businesses

## Project Overview
This project leverages predictive analytics to forecast the potential total value of awards granted by the Environmental Protection Agency (EPA), with a special focus on aiding small, women-owned, and minority-owned businesses in securing EPA contract awards. Our analysis is based on data from the USA Spending's Award Data Archive, focusing on awards from fiscal year 2022 to 2024. We employ two modeling approaches: regression and classification, to predict the potential total value of these awards.

## Data Source
The primary dataset is accessible at [USA Spending's Award Data Archive](https://www.usaspending.gov/award/CONT_AWD_68HE0819F0075_6800_GS00F0002M_4730/).

## Project Structure
The project combines data cleaning, preprocessing, and predictive modeling to analyze the potential total value of EPA awards:

- **Data Cleaning and Preprocessing**: We first clean the dataset and transforms it into a machine learning-ready format. Data processing steps include filtering relevant columns, handling missing values, removing duplicates, and encoding categorical variables.
  
- **Predictive Modeling**:
  - **Regression Analysis**: We use Ridge Regression and XGBoost models to forecast the continuous award value. We perform hyperparameters tuning and performance evaluation thoroughly for each method. We further compare the performance of these models across different numbers of input features.
  - **Classification Analysis**: In addition to regression, we also apply Logistic Regression for classification, categorizing the continuous award value into 3 distinct categories.

## File Descriptions
- `award_data_FY22-24.csv`: Award data, filtered to only include women-owned and minority-owned businesses for fiscal years 2022-2024.
- `clean_data.py`: Cleans the dataset and prepares it for machine learning models. The output of the script is `machine_df.csv` which is then used to train and evaluate different models.
- `ridge_regression.py`: Hyperparameters tuning and performance evaluation for Ridge Regression.
- `xgboost_regression.py`: Hyperparameters tuning and performance evaluation for XGBoost.
- `compare_ridge_xgboost.py`: Compares Ridge and XGBoost model performances.
- `logistic_regression.py`: Implements classification with Logistic Regression.

## Installation

Before running the scripts, ensure you have Python installed on your system. You'll need the following dependencies:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- xgboost
- tqdm

You can install all required packages using the following command:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost tqdm
```

## Usage

To use this project for forecasting the potential total value of EPA awards, follow these steps:

1. **Data Preparation**: Begin with running the data cleaning script to prepare the dataset.
    ```bash
    python clean_data.py
    ```

2. **Model Training and Evaluation**:
    - For **Regression Analysis**, choose between Ridge Regression and XGBoost. To evaluate the performance of these models, run:
        ```bash
        python ridge_regression.py
        ```
        ```bash
        python xgboost_regression.py
        ```
    - After running both regression models, you can compare their performances using:
        ```bash
        python compare_ridge_xgboost.py
        ```
    - For **Classification Analysis** with Logistic Regression, execute:
        ```bash
        python logistic_regression.py
        ```

## Acknowledgements
This project is a collaborative effort with my amazing teammates Tina Brauneck, Tsai Lieh (Dino) Kao, and Riley Nickel.
