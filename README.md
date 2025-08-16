# Forest Cover Type Classification

This project uses machine learning models (Random Forest and XGBoost) to classify forest cover types based on cartographic variables. The dataset used is the UCI Forest CoverType dataset.

## Project Structure

- `data/`
  - `covtype.csv` - Preprocessed CSV data file
- `src/`
  - `model.py` - Main script for training, evaluating, and comparing models
- `.gitignore`
- `README.md`

## Requirements

- Python 3.7+
- pandas
- scikit-learn
- matplotlib
- seaborn
- xgboost

Install dependencies with:

```sh
pip install pandas scikit-learn matplotlib seaborn xgboost
```

## Usage

1. Place `covtype.csv` in the `data/` directory.
2. Run the model script:

```sh
python src/model.py
```

This will:
- Load and inspect the data
- Train a Random Forest and XGBoost classifier
- Print accuracy, classification reports, and confusion matrices
- Plot feature importances for both models
- Compare model performance

## Output

- Accuracy and classification metrics for both models
- Confusion matrix plots
- Top 10 feature importance plots for Random Forest and XGBoost

## Data Source

- [UCI Forest CoverType Dataset](https://archive.ics.uci.edu/ml/datasets/covertype)

## License

This project is for educational