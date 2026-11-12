# SVM_FS_META

`svmfsmeta` is a Python library for binary soft-margin Support Vector Machine
(SVM) classification with feature selection. It uses NSGA-II to search for a
set of non-dominated classifiers, allowing users to choose a model according
to their preferred trade-off between objectives.

The project is based on the methods described by Valero et al. (2023)
[https://doi.org/10.1016/j.cor.2022.106131] and Alcaraz (2024)
[https://doi.org/10.1016/j.cor.2024.106821].

## Features

- NSGA-II optimization with either objective pair:
  - `DIST-EPS`: maximize the distance between the two margin planes and
    minimize the sum of margin-violation distances (epsilon).
  - `MC`: minimize false negatives (FN) and false positives (FP) separately,
    measured using the intermediate classification plane.
- Classification metrics for each candidate, including accuracy, precision,
  recall, F1, Cohen's kappa, AUC, and a confusion matrix.
- Plotting and CSV export of the first non-dominated front.

## Objective and classification conventions

- Use numeric response labels `-1` and `1`; class `1` is the positive class.
- Every candidate selects **exactly** `num_selected_features` predictors.
- `DIST` is the normalized distance between the class-specific margin planes;
  it is maximized. `EPS` is the sum of the class-specific margin-violation
  distances; it is minimized.
- `MC_POS` is the false-negative count (actual `1`, predicted `-1`), and
  `MC_NEG` is the false-positive count (actual `-1`, predicted `1`). Both are
  minimized. These names are retained for compatibility with the result
  format.
- Predictions use the intermediate plane. A decision score of zero is
  predicted as class `1`. The same predictions are used for MC and the
  classification metrics.
- The confusion matrix has actual classes as rows and predicted classes as
  columns, ordered `-1`, then `1`.
- Front numbers start at zero: `FRONT == 0` is the first non-dominated front.

## Project layout

The installable library is in `src/svmfsmeta/`:

- `svm_fs_meta.py`: public `SVMFsMeta` optimizer interface.
- `population.py`: population generation, NSGA-II ranking, and selection.
- `solution.py`: candidate construction, objective evaluation, and metrics.

The repository also contains `main.py` and `example_usage.py` examples, and a
Streamlit application in `streamlit_app.py` and `views/`. These are support
tools and are not required to use the core library.

---

## 📦 Installation

Install the library and its runtime dependencies with:

```bash
# Clone the repository
git clone https://github.com/jomabel/SVMFeature_py.git
cd SVMFeature_py

# Editable install for development; use `pip install .` for a regular install.
pip install -e .
```

To run the Colon examples, install the optional dependencies (Python 3.10 or later):

```bash
python -m pip install -e ".[examples]"
```

Run Colon from the repository directory:

```bash
python main.py colon --method DIST-EPS --iter_type iter --iter_val 5
```

`main.py` uses Colon when no dataset argument is supplied. You can still pass a
CSV file path instead. Colon is downloaded from `plsgenomics` version `1.5-3`
and cached on first use; R does not need to be installed.

The following installs additional repository dependencies used by the
Streamlit application and development/documentation tools:

```bash
pip install -r requirements.txt
```

## Quick start

This example loads the same `Colon` dataset as the R package: 62 samples and
2,000 predictors. Original class 1 (normal tissue) remains 1, and class 2
(tumor tissue) becomes -1.

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from skdatasets.repositories.cran import fetch_dataset
from svmfsmeta import SVMFsMeta

# Load Colon from the same version of plsgenomics used by the example in R.
colon = fetch_dataset("Colon", "plsgenomics", version="1.5-3")["Colon"]
X = np.asarray(colon["X"], dtype=float)
y = np.where(np.asarray(colon["Y"]).reshape(-1) == 2, -1, 1)
data = pd.DataFrame(X, columns=[f"x{i}" for i in range(1, X.shape[1] + 1)])
data.insert(0, "y", y)

output_col = "y"
input_cols = data.columns.drop(output_col).tolist()

# Optional: scale predictors to [0, 1].
scaler = MinMaxScaler()
data[input_cols] = scaler.fit_transform(data[input_cols])

optimizer = SVMFsMeta(
    method="DIST-EPS",  # "DIST-EPS" or "MC"
    data=data,
    population_size=10,
    inputs=input_cols,
    output=output_col,
    num_selected_features=5,
)

# Run a fixed number of generations.
optimizer.run(train="iter", num_iter=20)

# Inspect and save the first non-dominated front.
pareto_front = optimizer.population.solutions_df
pareto_front = pareto_front[pareto_front["FRONT"] == 0]
optimizer.save_results(
    "colon",
    iteration_val=20,
    iteration_type="iter",
)
optimizer.draw_solution()
```

Optimization is stochastic. Repeated runs are not guaranteed to produce the
same population or Pareto front.

## API reference

### `SVMFsMeta` (Class)

The main class to initialize and run the optimization.

#### `__init__(...)`
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `method` | `str` | Required | Objective pair: `"DIST-EPS"` or `"MC"`. |
| `data` | `pd.DataFrame` | Required | Dataset containing the response and predictors. |
| `population_size` | `int` | Required | Number of candidates in the population; at least 2. |
| `inputs` | `list[str]` | Required | Names of predictor columns. |
| `output` | `str` | Required | Name of the response column. |
| `num_selected_features` | `int` | Required | Exact number of predictors selected in each candidate, from 1 to `len(inputs)`. |
| `allow_clones` | `int` | `0` | `0` forbids duplicate individuals; `1` allows them. |
| `p_mutation` | `float` | `0.7` | Probability of applying mutation to offspring. |
| `p_mutate_individual` | `float` | `0.4` | Probability of mutating an individual. |
| `p_mutate_feature` | `float` | `0.4` | Probability of mutating selected features. |
| `p_mutate_coord` | `float` | `0.2` | Probability of mutating plane coordinates. |
| `coord_mutation_type` | `int` | `0` | `0` scales coordinates; `1` replaces them. |
| `logger` | callback or logger | `None` | Optional custom logging callback or logger. |

The constructor validates configuration before training. `data` must be a
non-empty DataFrame with unique column names and unique row index labels.
`inputs` must be a non-empty list or tuple of unique existing column names,
and must exclude `output`. Selected predictors must contain only finite real
numeric values. The response must contain both numeric labels `-1` and `1`,
with no missing values or other classes. Unused columns are not validated for
numeric content.

Population and feature counts must be integers; booleans are not counts.
All four mutation probabilities must be finite real numbers in `[0, 1]`.
`allow_clones` and `coord_mutation_type` accept integer `0`/`1` or booleans.
`logger` must be `None`, a callable, or an object with a callable `info` method.
Invalid configuration raises `TypeError` or `ValueError` naming the parameter.

### `run(train="sec", num_iter=10)`

Generates the initial population and runs NSGA-II.

- `train="iter"` requires a positive integer `num_iter` and runs that many
  iterations.
- `train="sec"` requires a positive finite number of seconds; fractional
  seconds are accepted. The limit is approximate: initialization happens first
  and an iteration already underway is allowed to finish.

Configuration and the training budget are checked again before generating
candidates or replacing an existing population. Zero, negative, non-finite,
and boolean budgets are rejected; `train` must be `"iter"` or `"sec"`.

The final population is available as `optimizer.population`. Its
`solutions_df` table contains candidate summaries and their `FRONT` values.
Filter `FRONT == 0` to obtain the first non-dominated front.

### `draw_solution()`

Plots the first non-dominated front for the selected objective pair.

### `save_results(dataset_name, output_dir="results", iteration_val=None, iteration_type=None)`

Writes the first non-dominated front to CSV under
`<output_dir>/<dataset_name>/<method>/`. A numeric suffix is added if the
target filename already exists.

- `dataset_name`: directory name for the dataset.
- `output_dir`: root output directory.
- `iteration_val`: optional iteration count or time value included in the
  filename.
- `iteration_type`: optional `"iter"` or `"sec"` value included with
  `iteration_val`.

## Command-line interface

Run `python main.py --help` for the available options. The script accepts a
dataset path followed by optional flags:

```bash
python main.py path/to/data.csv --has_header no --has_index yes --separator "," --normalize yes --method DIST-EPS --population_size 10 --num_features 5 --iter_type iter --iter_val 5
```

The CLI expects the response in the first column after any optional index
column is removed. It filters rows to labels `-1` and `1`; when normalization
is enabled, it applies Min-Max scaling to the predictors.

| Option | Default Value | Description |
| :--- | :--- | :--- |
| `--has_header` | `"no"` | Whether the input file has a header (`yes`/`no`). |
| `--has_index` | `"yes"` | Whether to drop the first input column as an index. |
| `--separator` | `","` | CSV field separator. |
| `--normalize` | `"yes"` | Whether to Min-Max scale predictors. |
| `--method` | `"DIST-EPS"` | Objective pair (`"DIST-EPS"` or `"MC"`). |
| `--population_size` | `10` | Size of the population. |
| `--num_features` | `5` | Exact number of predictors selected per candidate. |
| `--iter_type` | `"iter"` | Training mode (`"iter"` or `"sec"`). |
| `--iter_val` | `5` | Number of iterations or seconds, according to `--iter_type`. |
| `--allow_clones` | `0` | `0` forbids clones; `1` allows them. |
| `--p_mutation` | `0.7` | Global mutation probability. |
| `--p_mutate_individual` | `0.4` | Individual mutation probability. |
| `--p_mutate_feature` | `0.4` | Feature mutation probability. |
| `--p_mutate_coord` | `0.2` | Coordinate mutation probability. |
| `--coord_mutation_type` | `0` | `0` scales coordinates; `1` replaces them. |

When run interactively, the CLI prompts for unspecified options. The defaults
are the same as those listed above.

## Streamlit application

The dashboard requires the full repository. Install the additional
requirements and launch it from the project directory:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
