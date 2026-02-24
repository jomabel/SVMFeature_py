from ast import literal_eval
from collections import Counter
from streamlit import header, success, dataframe, subheader, plotly_chart, error, columns, markdown
from src.utils.visualization import plot_pareto_front


def show_results(algorithm,method):
    header("Results and visualization")

    # === Results ===
    results = algorithm.population.solutions_df
    pareto_front = results[results.FRONT == 0]

    success("✅ Simulation completed.")
    dataframe(pareto_front)

    # === Visualization ===
    subheader("🌐 Pareto Front Visualization")
    try:
        fig = plot_pareto_front(pareto_front, method)
        plotly_chart(fig)
    except Exception as e:
        error(f"❌ Could not generate the plot: {e}")

    subheader("📊 Most Common Features and Vectors in Pareto Front")

    metrics_cols = ['SOL', 'ACCURACY', 'F1', 'KAPPA', 'AUC', 'COST']

    subheader("📊 Metrics of solutions")

    metrics_cols = [col for col in metrics_cols if col in pareto_front.columns]
    dataframe(pareto_front[metrics_cols])

    # Parse string representations to real lists if needed
    def parse_column_as_list(series):
        if isinstance(series.iloc[0], str):
            return series.apply(literal_eval)
        return series

    # Convert FEATURES and VECTORS columns to list objects
    features_list_series = parse_column_as_list(pareto_front['FEATURES'])
    vectors_list_series = parse_column_as_list(pareto_front['VECTORS'])

    # Count feature frequency (flatten all features into one list)
    feature_counts = Counter(f for features in features_list_series for f in features)
    most_common_features = feature_counts.most_common()

    # Count full vector combinations (as tuples to make them hashable)
    vector_counts = Counter(tuple(vec) for vec in vectors_list_series)
    most_common_vectors = vector_counts.most_common()

    # Crear dos columnas
    col1, col2 = columns(2)

    # Mostrar features a la izquierda
    with col1:
        markdown("### 🧬 Most Frequent Features")
        for feature, count in most_common_features[:10]:
            markdown(f"- **{feature}**: {count} times")

    # Mostrar vectores a la derecha
    with col2:
        markdown("### 🔁 Most Frequent Vectors")
        for vector, count in most_common_vectors[:10]:
            markdown(f"- `{vector}`: {count} times")

