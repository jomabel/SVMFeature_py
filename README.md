# 🧠 NSGA2-Multi-objective SVM Classifier with Feature Selection

This project implements the algorithms proposed by Valero et al. (2023) [https://doi.org/10.1016/j.cor.2022.106131] and Alcaraz, J. (2024) [https://doi.org/10.1016/j.cor.2024.106821].

It is a metaheuristic that solves the **soft-margin Support Vector Machine (SVM)** problem when combined with **Feature Selection**. The result of the algorithm is a **Pareto front**, where each solution is an SVM-based classifier that helps decision-makers select the best model for their specific problem.

## 🚀 Key Features

- ⚙️ NSGA-II based metaheuristic
- 📈 Multi-objective optimization
  - Classic SVM objectives: margin distance and epsilon
  - True Positives (TP) and False Positives (FP)
- 🔍 Visualization of the Pareto front and classifier metrics
- 🧬 Analysis of the most frequently selected features and vectors
- 📤 Export of optimal solutions to CSV

## 📦 Installation

To install the library locally using `pip`:

```bash
# Clone the repository
git clone https://github.com/jomabel/SVMFeature_py.git
cd SVMFeature_py

# Install the package in editable mode for development, or just pip install .
pip install -e .

# Install dependencies for the Streamlit app
pip install -r requirements.txt
```

## 💻 Basic Usage

You can use the library in your Python scripts as follows:

```python
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from nsga2_svm_fs import NSGA2_SVMFS

# 1. Load the included dataset
# Example with '09Colon_p.txt': no header, first column is target, first row has feature costs.
data = pd.read_csv("data/09Colon_p.txt", sep=";", header=None)

# 2. Preprocess: drop index, define columns
data = data.drop(columns=[0])
num_cols = data.shape[1]
data.columns = ['y'] + [f'x{i}' for i in range(1, num_cols)]

output_col = data.columns[0]
input_cols = data.columns[1:].tolist()

# 3. Handle costs (first row in this dataset)
costs = data.iloc[0].values[1:].tolist()
data = data.drop(index=data.index[0])

# 4. Normalize features
scaler = MinMaxScaler()
data[input_cols] = scaler.fit_transform(data[input_cols])

# 5. Initialize and run the optimizer
optimizer = NSGA2_SVMFS(
    method="DIST-EPS", # Options: 'DIST-EPS', 'MC'
    data=data,
    costs=costs,
    population_size=10,
    inputs=input_cols,
    output=output_col,
    num_selected_features=5
)

optimizer.run(train="iter", num_iter=20)

# 6. Save and Visualize
optimizer.save_results("09Colon_p", iteration_val=20, iteration_type="iter")
optimizer.draw_solution()
```

## 🛠️ API Reference

### `NSGA2_SVMFS` (Class)

The main class to initialize and run the optimization.

#### `__init__(...)`
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `method` | `str` | Dominance method: `'DIST-EPS'` (Distance-Epsilon) or `'MC'` (Misclassification). |
| `data` | `pd.DataFrame`| The dataset containing features and target. |
| `costs` | `list` | Costs associated with each feature. |
| `population_size`| `int` | Size of the population. |
| `inputs` | `list` | Names of the input feature columns. |
| `output` | `str` | Name of the target column. |
| `num_selected_features`| `int` | Max number of features to select. |
| `allow_clones` | `int` | `0` (default) or `1`. Whether to allow duplicate solutions. |
| `p_mutation` | `float` | Probability of mutation (default `0.7`). |
| `logger` | `fn` | Optional callback function for custom logging. |

#### `run(train='iter', num_iter=10)`
Starts the optimization process.
- `train`: `'iter'` (iterations) or `'sec'` (seconds).
- `num_iter`: Total number of iterations or seconds to run.

#### `draw_solution()`
Generates a scatter plot of the Pareto front (Front 0).

#### `save_results(dataset_name, output_dir="results", ...)`
Exports the current Pareto front to a CSV file.
- `dataset_name`: Name for the output subfolder.
- `output_dir`: Root directory for results.

## 📱 Streamlit App
If you wish to run the interactive web application:
```bash
streamlit run streamlit_app.py
```
