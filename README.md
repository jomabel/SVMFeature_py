# 🧠 SVMFsMeta

This project implements the algorithms proposed by Valero et al. (2023) [https://doi.org/10.1016/j.cor.2022.106131] and Alcaraz, J. (2024) [https://doi.org/10.1016/j.cor.2024.106821].

It is a metaheuristic that solves the **soft-margin Support Vector Machine (SVM)** problem when combined with **Feature Selection**. The result of the algorithm is a **Pareto front**, where each solution is an SVM-based classifier that helps decision-makers select the best model for their specific problem.

## 🚀 Key Features

- ⚙️ NSGA-II based metaheuristic
- 📈 Multi-objective optimization
  - Margin distance and epsilon
  - False Positives (FP) and False Negatives (FN)
- 🔍 Visualization of the Pareto front and classifier metrics
- 🧬 Analysis of the most frequently selected features and vectors
- 📤 Export of optimal solutions to CSV

## 📂 Project Structure

It is fundamental to distinguish that the **main library** is solely the `svmfsmeta` folder. The rest of the files are support tools, examples, or the web application:

### 📗 Core Library (`src/svmfsmeta/`)
This is the **only** code that constitutes the `svmfsmeta` package. This is what gets installed and the only part necessary to integrate the algorithm into other projects:
- `svm_fs_meta.py`: Main `SVMFsMeta` class.
- `population.py` and `solution.py`: NSGA-II algorithm logic.

### 📱 Support Tools and Application
The rest of the repository is **not** part of the core library, but is necessary for the demonstration and examples:
- `streamlit_app.py` and `views/`: Interactive dashboard (requires the library to function).
- `src/utils/visualization.py`: Visualization generation for the dashboard.
- `main.py` and `example_usage.py`: Example scripts for terminal or code execution.

---

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

The library can be used programmatically (see `example_usage.py`) or via the command line (see `main.py`).

```python
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from svmfsmeta import SVMFsMeta

# 1. Load the included dataset
# Example with '09Colon_p.txt': no header, first column is target
data = pd.read_csv("data/09Colon_p.txt", sep=";", header=None)

# 2. Preprocess: drop index, define columns
data = data.drop(columns=[0])
num_cols = data.shape[1]
data.columns = ['y'] + [f'x{i}' for i in range(1, num_cols)]

output_col = data.columns[0]
input_cols = data.columns[1:].tolist()

# 3. Filter binary classification labels (-1 and 1)
data = data[data[output_col].isin([-1, 1])].copy()
data.reset_index(drop=True, inplace=True)

# 4. Normalize features
scaler = MinMaxScaler()
data[input_cols] = scaler.fit_transform(data[input_cols])

# 5. Initialize and run the optimizer
optimizer = SVMFsMeta(
    method="DIST-EPS", # Options: 'DIST-EPS', 'MC'
    data=data,
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

### `SVMFsMeta` (Class)

The main class to initialize and run the optimization.

#### `__init__(...)`
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `method` | `str` | - | Dominance method: `'DIST-EPS'` (Distance-Epsilon) or `'MC'` (Misclassification). |
| `data` | `pd.DataFrame`| - | The dataset containing features and target column. |
| `population_size`| `int` | - | Size of the population. |
| `inputs` | `list` | - | Names of the input feature columns. |
| `output` | `str` | - | Name of the target column. |
| `num_selected_features`| `int` | - | Maximum number of features to be selected. |
| `allow_clones` | `int` | `0` | `0` or `1`. Whether to allow duplicated individuals in the population. |
| `p_mutation` | `float` | `0.7` | Global mutation probability. |
| `p_mutate_individual` | `float` | `0.4` | Probability of mutating an individual. |
| `p_mutate_feature` | `float` | `0.4` | Probability of mutating a feature selection. |
| `p_mutate_coord` | `float` | `0.2` | Probability of mutating a coordinate (hyperparameter). |
| `coord_mutation_type` | `int` | `0` | Coordinate mutation type (0: Scaling, 1: Full replacement). |
| `logger` | `fn` | `None` | Optional callback function for custom logging. |

#### `run(train='sec', num_iter=10)`
Starts the optimization process.
- `train`: `'sec'` (seconds, default) or `'iter'` (iterations).
- `num_iter`: Total number of iterations or seconds to run.

#### `draw_solution()`
Generates a scatter plot of the Pareto front (Front 0).

#### `save_results(dataset_name, output_dir="results", iteration_val=None, iteration_type=None)`
Exports the current Pareto front to a CSV file.
- `dataset_name`: Name for the output subfolder.
- `output_dir`: Root directory for results (default: `"results"`).
- `iteration_val`: Optional iteration value to include in the filename.
- `iteration_type`: Optional iteration type (`'iter'` or `'sec'`) to include in the filename.

## Command Line (main.py)

If you use the `main.py` script, these are the **default values** applied if flags are not provided:

| Option | Default Value | Description |
| :--- | :--- | :--- |
| `--method` | `"DIST-EPS"` | Dominance method (`"DIST-EPS"`, `"MC"`). |
| `--population_size` | `10` | Size of the population. |
| `--num_features` | `5` | Features to be selected. |
| `--iter_type` | `"iter"` | Training mode (`"iter"` or `"sec"`). |
| `--iter_val` | `5` | Iterations or seconds value. |
| `--has_header` | `"no"` | Dataset includes header. |
| `--has_index` | `"yes"` | Dataset includes index column. |
| `--separator` | `","` | CSV separator character. |
| `--normalize` | `"yes"` | Normalize input features. |
| `--allow_clones` | `0` | Allow duplicate individuals. |
| `--p_mutation` | `0.7` | Global mutation probability. |
| `--p_mutate_individual` | `0.4` | Individual mutation probability. |
| `--p_mutate_feature` | `0.4` | Feature mutation probability. |
| `--p_mutate_coord` | `0.2` | Coordinate mutation probability. |
| `--coord_mutation_type` | `0` | 0: Scaling, 1: Full replacement. |

## 📱 Streamlit App

The Streamlit app is a dashboard built **on top** of the library. To run it, you must have the full repository cloned (including the `views/` folder).

1. **Install requirements**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Launch the app**:
   ```bash
   streamlit run streamlit_app.py
   ```
