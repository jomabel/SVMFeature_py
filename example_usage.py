import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from svmfsmeta import SVMFsMeta

# 1. Load the included dataset
# Example with '09Colon_p.txt': no header, first column is target.
data = pd.read_csv("data/09Colon_p.txt", sep=";", header=None)

# 2. Drop the index column and extract feature names and target
data = data.drop(columns=[0])
num_cols = data.shape[1]
column_names = ['y'] + [f'x{i}' for i in range(1, num_cols)]
data.columns = column_names
output_col = data.columns[0]
input_cols = data.columns[1:].tolist()
data = data.drop(index=data.index[0])
# 4. Normalize the input features (recommended)
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(data[input_cols])
data_scaled = pd.DataFrame(scaled_features, columns=input_cols)
data_scaled[output_col] = data[output_col].values

# 5. Initialize the optimizer
# Available methods: 'DIST-EPS', 'MC'
optimizer = SVMFsMeta(
    method="DIST-EPS",
    data=data_scaled,
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
print("Pareto front solutions:")
print(pareto_front)
optimizer.save_results("09Colon_p", iteration_val=5, iteration_type="iter")

# 8. Visualize the Pareto front
optimizer.draw_solution()
