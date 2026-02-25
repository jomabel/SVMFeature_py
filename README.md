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
from nsga2_svm_fs import NSGA2_SVMFS

# 1. Load your dataset
# For example, a CSV where the last column is the target class (-1 or 1)
data = pd.read_csv("your_dataset.csv")

# 2. Define parameters
inputs = ["feature1", "feature2", "feature3", "feature4"]  # List of input feature names
output = "target"  # Name of the target variable column
costs = [1.0, 1.5, 0.8, 2.0]  # Cost array corresponding to each feature
population_size = 50
num_selected_features = 2

# 3. Initialize the optimizer
# Available methods: 'DIST-EPS', 'MC', 'DIST-EPS-COST', 'MC-COST'
optimizer = NSGA2_SVMFS(
    method="DIST-EPS",
    data=data,
    costs=costs,
    population_size=population_size,
    inputs=inputs,
    output=output,
    num_selected_features=num_selected_features,
    # Optional parameters with their default values:
    allow_clones=0,
    p_mutation=0.7,
    p_mutate_individual=0.4,
    p_mutate_feature=0.4,
    p_mutate_coord=0.2,
    coord_mutation_type=0
)

# 4. Run the optimization
# train mode can be 'sec' (seconds) or 'iter' (iterations)
optimizer.run(train="iter", num_iter=20)

# 5. Visualize the Pareto front
optimizer.draw_solution()
```

### 📱 Streamlit App
If you wish to run the interactive web application, you can execute the included Streamlit app:
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
