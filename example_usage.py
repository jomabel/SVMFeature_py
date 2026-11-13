from main import load_colon
from svmfsmeta import SVMFsMeta
from svmfsmeta.preprocessing import minmax_scale

# Use the same Colon source and response encoding as the R example.
# Install optional data dependencies with: python -m pip install -e ".[examples]"
data = load_colon()
output_col = 'y'
input_cols = data.columns.drop(output_col).tolist()
data_scaled = minmax_scale(data, input_cols)

optimizer = SVMFsMeta(
    method='DIST-EPS', data=data_scaled, population_size=10,
    inputs=input_cols, output=output_col, num_selected_features=5,
)
optimizer.run(train='iter', num_iter=5)
results = optimizer.population.solutions_df
print('Pareto front solutions:')
print(results[results.FRONT == 0])
optimizer.save_results('colon', iteration_val=5, iteration_type='iter')
optimizer.draw_solution()
