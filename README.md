# Learning Style Classification

Production-ready machine learning project for classifying student learning styles from academic, behavioral, and environmental features stored in `Student_Performance.xlsx`.

The project includes:

- a training notebook that builds a preprocessing pipeline and trains a balanced `RandomForestClassifier`
- a Streamlit application for dataset inspection and inference using saved artifacts only
- reproducible dependencies and setup instructions

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

- Missing value handling for numeric and categorical columns
- Standard scaling for numeric features
- One-hot encoding for categorical features
- Stratified train-test split
- Balanced Random Forest training for class imbalance
- Evaluation with:
  - Accuracy
  - Balanced Accuracy
  - Precision
  - Recall
  - F1 Score
  - ROC-AUC
  - MCC
  - Cohen Kappa
  - Log Loss
- Confusion matrix visualization
- Top feature importance visualization
- Streamlit workflow for upload, overview, model info, and prediction

## Installation

1. Create and activate a virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies.

```powershell
pip install -r requirements.txt
```

## Training The Model

Make sure `Student_Performance.xlsx` is available in the project root, then open and run all cells in [model_training.ipynb](/C:/Akira/University%20Stuff/Sem%202/1.%20AIML/AI%20Lab/AIML%20Project/prototype/model_training.ipynb).

Training will:

- load the dataset
- build the preprocessing pipeline
- split the data with stratification
- train `RandomForestClassifier(class_weight="balanced")`
- evaluate the model
- save artifacts to:
  - `models/learning_style_model.pkl`
  - `models/preprocessor.pkl`

## Run The Streamlit App

After training artifacts exist:

```powershell
streamlit run app.py
```

The app supports `.xlsx` and `.csv` uploads and does not retrain the model during inference.

## Expected Workflow

1. Place `Student_Performance.xlsx` in the project root if it is not already there.
2. Install dependencies from `requirements.txt`.
3. Run all cells in `model_training.ipynb`.
4. Confirm that the `models/` folder contains both saved `.pkl` files.
5. Launch the Streamlit app with `streamlit run app.py`.
6. Upload a dataset for overview and optional batch prediction.
7. Use the prediction interface to enter a single student profile and view predicted learning style probabilities.

## Dependencies

Core libraries used in this project:

- `pandas`
- `numpy`
- `scikit-learn`
- `joblib`
- `matplotlib`
- `seaborn`
- `plotly`
- `streamlit`
- `openpyxl`

## Troubleshooting

### Model files not found

If the app reports missing model artifacts, run the notebook first and verify:

- `models/learning_style_model.pkl`
- `models/preprocessor.pkl`

### Dataset upload is empty

Use a non-empty `.csv` or `.xlsx` file and ensure the file is readable.

### Column mismatch during prediction

The uploaded dataset must include the same feature columns used during training. Extra columns are ignored, but required feature columns must be present.

### Invalid datatypes

If numeric fields are uploaded as text, clean the dataset or convert those columns before prediction.

### Excel file will not load

Make sure `openpyxl` is installed and the file is not corrupted or locked by another application.

## Notes

- The Streamlit app performs inference only and intentionally does not retrain the model.
- The notebook stores metadata on the saved preprocessor object so the app can rebuild forms and validate inputs consistently.
