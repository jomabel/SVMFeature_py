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

### 🔧 How to Run It Locally

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```
