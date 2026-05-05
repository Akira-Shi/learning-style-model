# Learning Style Prototype

This prototype is a simple Python-only version of the learning style identification project.

It uses K-Means clustering on the student performance dataset to:

- auto-detect key numeric and categorical columns
- clean missing values
- encode categorical features
- scale the feature set
- train a clustering model
- assign learner-profile labels to discovered clusters
- test prediction on a sample student

## Files

- `learning_style_prototype.py` - main prototype script
- `Student_Performance.xlsx` - small dataset used for training/testing

## Requirements

Install these packages before running:

- `pandas`
- `numpy`
- `scikit-learn`
- `openpyxl`

## Run

From the project root:

```powershell
python prototype\learning_style_prototype.py
```

Or pass a specific dataset:

```powershell
python prototype\learning_style_prototype.py --data "C:\path\to\Student_Performance.xlsx"
```

## Output

The script prints:

- detected columns
- cleaned feature matrix shape
- NaN validation
- cluster counts
- learner-profile interpretation for each cluster
- prediction result for a sample student

## Notes

- This prototype currently uses K-Means only.
- It is intended for experimentation and reporting, separate from the full Streamlit app.
