# Learning Style Classification

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://learning-style-model-az3fasz4y25ytrndpuqyfl.streamlit.app/)

Production-ready machine learning project for classifying student learning styles from academic, behavioral, and environmental features stored in `Student_Performance.xlsx`.

The project includes:

* a training notebook that builds a preprocessing pipeline and trains a balanced `RandomForestClassifier`
* a Streamlit application for dataset inspection and inference using saved artifacts only
* reproducible dependencies and setup instructions

## Live Demo

Streamlit Application:

https://learning-style-model-az3fasz4y25ytrndpuqyfl.streamlit.app/

Try the deployed application directly in your browser without any local setup.

## Project Structure

```text
prototype/
|-- app.py
|-- model_training.ipynb
|-- README.md
|-- requirements.txt
|-- Student_Performance.xlsx
|-- models/
|   |-- learning_style_model.pkl
|   `-- preprocessor.pkl
```

## Features

* Missing value handling for numeric and categorical columns
* Standard scaling for numeric features
* One-hot encoding for categorical features
* Stratified train-test split
* Balanced Random Forest training for class imbalance
* Evaluation with:

  * Accuracy
  * Balanced Accuracy
  * Precision
  * Recall
  * F1 Score
  * ROC-AUC
  * MCC
  * Cohen Kappa
  * Log Loss
* Confusion matrix visualization
* Top feature importance visualization
* Interactive Streamlit dashboard for:

  * Dataset upload
  * Dataset overview
  * Model information
  * Learning style prediction
  * Probability visualization

## Installation

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

## Training the Model

Make sure `Student_Performance.xlsx` is available in the project root, then open and run all cells in `model_training.ipynb`.

Training will:

* Load the dataset
* Build the preprocessing pipeline
* Split the data using stratified train-test split
* Train a balanced `RandomForestClassifier`
* Evaluate model performance
* Save model artifacts

Generated artifacts:

```text
models/
|-- learning_style_model.pkl
`-- preprocessor.pkl
```

## Running the Streamlit App

After the model artifacts have been generated:

```powershell
python -m streamlit run app.py
```

The application supports `.xlsx` and `.csv` uploads and performs inference using the saved model without retraining.

## Expected Workflow

1. Place `Student_Performance.xlsx` in the project root if it is not already available.
2. Install dependencies from `requirements.txt`.
3. Run all cells in `model_training.ipynb`.
4. Confirm that the `models/` directory contains:

   * `learning_style_model.pkl`
   * `preprocessor.pkl`
5. Launch the Streamlit application.
6. Upload a dataset for inspection and analysis.
7. Use the interactive prediction form to classify new student profiles.
8. View prediction probabilities and model insights.

## Model Evaluation Metrics

The notebook evaluates the trained model using:

* Accuracy
* Balanced Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Matthews Correlation Coefficient (MCC)
* Cohen's Kappa
* Log Loss

Additional visualizations include:

* Confusion Matrix
* Classification Report
* Top Feature Importance Plot

## Dependencies

Core libraries used in this project:

* pandas
* numpy
* scikit-learn
* joblib
* matplotlib
* seaborn
* plotly
* streamlit
* openpyxl

Install all dependencies with:

```powershell
pip install -r requirements.txt
```

## Troubleshooting

### Model files not found

If the application reports missing model artifacts:

1. Run all cells in `model_training.ipynb`
2. Verify that the following files exist:

```text
models/learning_style_model.pkl
models/preprocessor.pkl
```

### Dataset upload is empty

* Ensure the uploaded file is not empty
* Verify that the file is a valid `.csv` or `.xlsx`

### Column mismatch during prediction

The uploaded dataset must contain the same feature columns used during training.

* Required columns must be present
* Extra columns are ignored where applicable

### Invalid data types

If numeric fields are interpreted as text:

* Clean the dataset before upload
* Convert values to appropriate numeric formats

### Excel file will not load

Ensure:

```powershell
pip install openpyxl
```

is installed and the Excel file is not corrupted or currently open in another application.

## Notes

* The Streamlit application performs inference only and does not retrain the model.
* Training, evaluation, and artifact generation are handled exclusively by `model_training.ipynb`.
* Saved preprocessing metadata is used to dynamically generate prediction forms and validate user inputs.
* The deployed Streamlit application mirrors the local inference workflow.

## Author

Developed as part of an AIML project focused on Learning Style Classification using supervised machine learning and interactive deployment with Streamlit.
