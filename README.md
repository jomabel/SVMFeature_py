# 🧠 NSGA2-Multi-objective SVM Classifier with Feature Selection

This project implements the algorithms proposed by Valero et al. (2023) [https://doi.org/10.1016/j.cor.2022.106131] and Alcaraz, J. (2024) [https://doi.org/10.1016/j.cor.2024.106821].

It is a metaheuristic that solves the **soft-margin Support Vector Machine (SVM)** problem when combined with **Feature Selection**. The result of the algorithm is a **Pareto front**, where each solution is an SVM-based classifier that helps decision-makers select the best model for their specific problem.

## 🚀 Key Features

- ⚙️ NSGA-II based metaheuristic
- 📈 Multi-objective optimization
  - Classic SVM objectives: margin distance and epsilon
  - True Positives (TP) and False Positives (FP)
  - Classification cost
- 🔍 Visualization of the Pareto front and classifier metrics
- 🧬 Analysis of the most frequently selected features and vectors
- 📤 Export of optimal solutions

## 📦 Installation

To install the library locally using `pip`:

```bash
# Clone the repository
git clone https://github.com/jomabel/SVMFeature_py.git
cd SVMFeature_py

# Install the package
pip install .
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

# 2. Drop the index column and extract feature names and target
data = data.drop(columns=[0])
num_cols = data.shape[1]
column_names = ['y'] + [f'x{i}' for i in range(1, num_cols)]
data.columns = column_names
output_col = data.columns[0]
input_cols = data.columns[1:].tolist()

# 3. Extract costs and remove that row from data
costs = data.iloc[0].values[1:].tolist()
data = data.drop(index=data.index[0])

# 4. Normalize the input features (recommended)
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(data[input_cols])
data_scaled = pd.DataFrame(scaled_features, columns=input_cols)
data_scaled[output_col] = data[output_col].values

# 5. Initialize the optimizer
# Available methods: 'DIST-EPS', 'MC'
optimizer = NSGA2_SVMFS(
    method="DIST-EPS",
    data=data_scaled,
    costs=costs,
    population_size=10,
    inputs=input_cols,
    output=output_col,
    num_selected_features=5,
    # Optional parameters with their default values:
    allow_clones=0,
    p_mutation=0.7,
    p_mutate_individual=0.4,
    p_mutate_feature=0.4,
    p_mutate_coord=0.2,
    coord_mutation_type=0
)

# 6. Run the optimization
# train mode can be 'sec' (seconds) or 'iter' (iterations)
optimizer.run(train="iter", num_iter=5)

# 7. Get and save the Pareto front
results = optimizer.population.solutions_df
pareto_front = results[results.FRONT == 0]
optimizer.save_results("09Colon_p", iteration_val=5, iteration_type="iter")

# 8. Visualize the Pareto front
optimizer.draw_solution()
```

### 📱 Streamlit App
If you wish to run the interactive web application, you can execute the included Streamlit app:
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
