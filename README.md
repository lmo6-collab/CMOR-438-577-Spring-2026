# CMOR 438 — Data Science & Machine Learning (Spring 2026)

Author: Lucas Osborn - lmo6@rice.edu

Final project repository for **CMOR 438 / INDE 577 — Data Science and Machine Learning**, taught by **Dr. Randy R. Davila** at Rice University.

This repository contains:

1. **`mlpkg/`** — a custom Python machine-learning package with from-scratch implementations of three classical algorithms (linear regression, k-NN, and the Rosenblatt perceptron).
2. **`tests/`** — a `pytest` suite verifying each algorithm against scikit-learn on standard datasets.
3. **`notebooks/`** — Jupyter notebooks demonstrating a wide range of supervised and unsupervised algorithms applied to real-world public datasets.
4. **`data/`** — datasets (downloaded automatically on first run of each notebook).

---

## Algorithms covered

### Supervised Learning
- **Linear models**: OLS, Ridge, Lasso *(notebook 01 — Auto MPG)*
- **Logistic regression** *(notebook 02 — Heart Disease)*
- **Perceptron** (from scratch) *(notebook 03 — Banknote Authentication)*
- **k-Nearest Neighbors** *(notebook 04 — Glass Identification)*
- **Decision Trees, Random Forests, Gradient Boosting** *(notebook 05 — Titanic)*

### Optimization
- **Gradient Descent** *(notebook 06 — synthetic data + linear regression by GD)*

### Unsupervised Learning
- **K-Means clustering** *(notebook 07 — Wine Quality)*
- **Principal Component Analysis (PCA)** *(notebook 07 — Wine Quality)*

### Software engineering
- Custom `mlpkg` package with sklearn-style API *(notebook 08 — demo)*
- Pytest unit-test suite verifying correctness against scikit-learn

---

## Repository structure

```
CMOR-438-577-Spring-2026/
├── README.md
├── LICENSE
├── requirements.txt
├── setup.py
├── pytest.ini
├── .gitignore
│
├── mlpkg/                             
│   ├── __init__.py
│   ├── metrics.py                    
│   ├── preprocessing.py                
│   ├── supervised/
│   │   ├── __init__.py
│   │   ├── linear_regression.py        
│   │   ├── knn.py                      
│   │   └── perceptron.py               
│   └── unsupervised/                   
│
├── tests/                              
│   ├── test_linear_regression.py
│   ├── test_knn.py
│   ├── test_perceptron.py
│   ├── test_metrics.py
│   └── test_preprocessing.py
│
├── notebooks/                         
│   ├── 01_linear_regression.ipynb
│   ├── 02_logistic_regression.ipynb
│   ├── 03_perceptron.ipynb
│   ├── 04_knn.ipynb
│   ├── 05_trees_and_ensembles.ipynb
│   ├── 06_gradient_descent.ipynb
│   ├── 07_kmeans_and_pca.ipynb
│   └── 08_mlpkg_demo.ipynb
│
└── data/                               
```

---

## Setup

This project uses a Python **virtual environment** for reproducibility — a practice emphasized in INDE 577. The standard-library `venv` tool is recommended.

### Step 1 — Clone the repo
```bash
git clone https://github.com/lmo6-collab/CMOR-438-577-Spring-2026.git
cd CMOR-438-577-Spring-2026
```

### Step 2 — Create and activate a virtual environment
```bash
# Create
python3 -m venv venv

# Activate (macOS / Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### Step 3 — Install dependencies + the local package
```bash
pip install -r requirements.txt
pip install -e .
```

The `-e .` flag installs `mlpkg` in **editable** mode so any code changes are picked up immediately.

### Step 4 — Register a Jupyter kernel (optional)
```bash
pip install ipykernel
python -m ipykernel install --user --name=inde577 --display-name="Python (inde577)"
```

### Step 5 — Launch Jupyter
```bash
jupyter lab    # or: jupyter notebook
```

---

## Running the tests

```bash
pytest -v
```

Each test verifies the from-scratch implementation against scikit-learn's reference, ensuring numerical correctness.

---

## Datasets

All datasets are **downloaded automatically** the first time each notebook runs (cached in `data/`). No manual download required. Sources:

| Notebook | Dataset | Source |
|---|---|---|
| 01 | Auto MPG | UCI Machine Learning Repository |
| 02 | Heart Disease (Cleveland) | UCI Machine Learning Repository |
| 03 | Banknote Authentication | UCI Machine Learning Repository |
| 04 | Glass Identification | UCI Machine Learning Repository |
| 05 | Titanic | datasciencedojo/datasets |
| 06 | (synthetic) | Generated in-notebook |
| 07 | Wine Quality (Red) | UCI Machine Learning Repository |

---

## Quick start (using `mlpkg` directly)

```python
from mlpkg import (
    LinearRegressionScratch, KNNClassifierScratch, PerceptronScratch,
    train_test_split, StandardScaler,
    mean_squared_error, accuracy_score,
)
import numpy as np

# Regression
X = np.random.randn(100, 3)
y = X @ np.array([1.0, 2.0, -1.0]) + 0.5
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegressionScratch().fit(X_tr, y_tr)
print("MSE:", mean_squared_error(y_te, model.predict(X_te)))
print("Coefficients:", model.coef_)
```

---

## Course Information

- **Course**: CMOR 438 / INDE 577 — Data Science and Machine Learning
- **Instructor**: Dr. Randy R. Davila
- **Semester**: Spring 2026
- **Institution**: Rice University

---

## License

MIT — see `LICENSE`.
