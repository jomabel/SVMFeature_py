from streamlit import title,tabs, warning
from views.simulation import show_simulation
from views.results import show_results

# App title
title("🌱 Multiobjective SVM feature selection via metaheuristics")

tab1, tab2 = tabs(["⚙️ Simulation", "📊 Results"])

with tab1:
    algorithm, method = show_simulation()

with tab2:
    if algorithm:
        show_results(algorithm, method)
    else:
        warning("⚠️ Run a simulation first.")
