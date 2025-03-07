# Heart Disease Prediction

This project aims to predict whether a patient has heart disease based on various input features. The prediction is made using multiple machine learning models, including Logistic Regression, Support Vector Machine, Random Forest Classifier, and K-Nearest Neighbors.

## Project Overview

The project focuses on the following key aspects:

1. **Data Preprocessing:** 
   - Loading and cleaning data.
   - Handling null values, duplicates, and outliers.
   - Feature engineering and encoding categorical features.

2. **Data Exploration:**
   - Visualizing the distribution of features.
   - Checking for imbalanced classes.

3. **Model Building:**
   - Training and testing machine learning models (Logistic Regression, Support Vector Machine, Random Forest Classifier, K-Nearest Neighbors).
   - Evaluating models based on accuracy, precision, recall, and F1-score.
   - Visualizing model performance using a bar chart.

4. **Model Evaluation:**
   - Generating classification reports and confusion matrices to assess model performance.

## Requirements

Before running this project, ensure you have the following Python packages installed:

- numpy
- pandas
- seaborn
- matplotlib
- scikit-learn

You can install the required packages using `pip`:

```bash
pip install numpy pandas seaborn matplotlib scikit-learn
```

## Project Structure

```
├── heart_disease_prediction
│   ├── heart.csv           # Dataset for heart disease prediction
│   └── heart_disease_model.py # Python script containing the code for data processing, feature engineering, and model training
├── README.md               # Project documentation
```

## How to Run

**Run the Python script** to begin the analysis and model training:

```bash
python heart_disease_model.py
```

This will:

- Load and preprocess the dataset.
- Train multiple models.
- Display model evaluation metrics, including accuracy, classification report, confusion matrix, and a comparison plot of model performance.

## Functions Explained

- **`read_data_heart()`**: Reads the heart disease dataset from a CSV file.
- **`check_null_values(df)`**: Checks for any null values in the dataset.
- **`check_statistics(df)`**: Generates basic statistics of the dataset.
- **`data_visualization(df)`**: Visualizes the distribution of features in the dataset.
- **`check_data_is_balance(df)`**: Visualizes the distribution of the target variable to check for class imbalance.
- **`check_duplicates(df)`**: Checks for duplicate rows in the dataset.
- **`drop_duplicates(df)`**: Removes duplicate rows from the dataset.
- **`check_outliers()`**: Visualizes the outliers present in the dataset using a boxplot.
- **`check_correlation()`**: Visualizes the correlation between features using a heatmap.
- **`preprocessing()`**: Encodes categorical features using one-hot encoding.
- **`feature_scaling()`**: Scales numerical features using StandardScaler.
- **`model_building()`**: Trains multiple machine learning models and evaluates their performance. Plots a bar chart comparing model accuracy.

## Model Performance Comparison

The models are evaluated based on the following:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**

A bar plot will be generated to compare the accuracy of the trained models.

## Conclusion

The project demonstrates how to preprocess and visualize heart disease data and apply different machine learning algorithms to predict whether a patient has heart disease. You can experiment with other algorithms or fine-tune the existing ones for improved performance.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

