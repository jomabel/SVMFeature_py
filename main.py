import sys
import os
from sklearn.preprocessing import MinMaxScaler
from src.nsga2_svm_fs.nsga2_svmfs import NSGA2_SVMFS
from pandas import read_csv, DataFrame

def simulate(method, data_scaled, costs, population_size, input_cols, output_col, num_features, iter_type, iter_val):
    print(f"🚀 Running NSGA-II simulation with method: {method}, population size: {population_size}, number of features: {num_features}, iteration type: {iter_type}, and iteration value: {iter_val}")
    algorithm = NSGA2_SVMFS(
                    method=method,
                    data= data_scaled,
                    costs=costs,
                    population_size=population_size,
                    inputs=input_cols,
                    output=output_col,
                    num_selected_features=num_features,
                )
    
    algorithm.run(train=iter_type, num_iter=iter_val)
    print(f"✅ NSGA-II simulation completed with method: {method}")

    return algorithm, method

def show_results(algorithm, method, dataset_name, population_size, iter_type, iter_val):
    results = algorithm.population.solutions_df
    pareto_front = results[results.FRONT == 0]
    output_dir = f"results/{dataset_name}/{method}/"

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Generate a unique filename based on population size & iteration type and value
    base_filename = f"{population_size}_{iter_type}_{iter_val}"
    extension = ".csv"
    output_file = os.path.join(output_dir, base_filename + extension)
    
    # Check if the file already exists and create a unique filename
    counter = 1
    while os.path.exists(output_file):
        output_file = os.path.join(output_dir, f"{base_filename}_{counter}{extension}")
        counter += 1

    # Save the DataFrame as CSV
    pareto_front.to_csv(output_file, index=False)

    print(f"📁 Pareto front saved in: {output_file}")


DEFAULT_DATA_URL = "https://raw.githubusercontent.com/danielvacarre/NSGAII_SVM_FS/main/data/09Colon_p.txt"
dominance_methods = ["DIST-EPS", "DIST-EPS-COST", "MC", "MC-COST"]

# Extract command line arguments
if len(sys.argv) < 2:
    print("Usage: python main.py <dataset> [has_header] [has_index] [has_costs] [separator] [normalize] [method] [population_size] [num_features] [iter_type] [iter_val]")
    sys.exit(1)

# === CSV Read Options ===
dataset, has_header, has_index, has_costs, separator, normalize =None, "no", "yes", "yes", ",", "yes"
if len(sys.argv) >= 7:
    dataset = sys.argv[1]
    has_header = sys.argv[2].lower() in ["yes", "y"]
    has_index = sys.argv[3].lower() in ["yes", "y"]
    has_costs = sys.argv[4].lower() in ["yes", "y"]
    separator = sys.argv[5] if len(sys.argv) > 5 else ","
    normalize = sys.argv[6].lower() in ["yes", "y"]
else:
    has_header = input("Does the file have a header? (yes/no) [default: no]: ").strip().lower() or "no"
    has_index = input("Does the file have an index column? (yes/no) [default: yes]: ").strip().lower() or "yes"
    has_costs = input("Does the file include a row with feature costs? (yes/no) [default: yes]: ").strip().lower() or "yes"
    separator = input("Select separator (','  ';'  '\t') [default: ',']: ").strip().lower() or ","
    normalize = input("Normalize input features? (yes/no) [default: yes]: ").strip().lower() or "yes"

# CSV parsing config
header_option = 0
if has_header:
    header_option = 0
elif not has_header:
    header_option = None
else:
    print("Invalid response.")
    sys.exit(1)

header_option = 0 if has_header else None

try:
    data = read_csv(dataset, sep=separator, header=header_option)

    if has_index:
        data.drop(columns=[data.columns[0]], inplace=True)

    if header_option is None:
        num_cols = data.shape[1]
        column_names = ['y'] + [f'x{i}' for i in range(1, num_cols)]
        data.columns = column_names
    else:
        data = data[[data.columns[0]] + list(data.columns[1:])]
        
    output_col = data.columns[0]
    input_cols = data.columns[1:].tolist()

    if has_costs:
        costs = data.iloc[0].values[1:].tolist()
        data = data.drop(index=data.index[0])
    else:
        costs = [0.0] * len(input_cols)
        dominance_methods = [method for method in dominance_methods if "COST" not in method]

    if normalize:
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(data[input_cols])
        data_scaled = DataFrame(scaled_features, columns=input_cols)
        data_scaled[output_col] = data[output_col].values
    else:
        data_scaled = data.copy()

    print(f"✅ File loaded successfully.")

except Exception as e:
        print(f"❌ Error while loading the data: {e}")
        sys.exit(1)

# === Algorithm Configuration ===
method, population_size, num_features, iter_type, iter_val = "DIST-EPS", 10, 5, "iter", 5
if len(sys.argv) > 7:
    method = sys.argv[7] if len(sys.argv) > 7 else "DIST-EPS"
    population_size = int(sys.argv[8]) if len(sys.argv) > 8 else 100
    num_features = int(sys.argv[9]) if len(sys.argv) > 9 else 5
    iter_type = sys.argv[10] if len(sys.argv) > 10 else "iter"
    iter_val = int(sys.argv[11]) if len(sys.argv) > 11 else 5
else:
    print("⚙️ ALGORITHM CONFIGURATION")

    # Data input from console
    dataset = input("📂 Dataset path: ").strip()
    if not dataset:
        print(f"❌ No dataset provided. You can download the dataset included in the GitHub repository: {DEFAULT_DATA_URL}")
        sys.exit(1)

    # Method selection
    print(f"⚙️ Available dominance methods: {', '.join(dominance_methods)}")
    method = input("Select dominance method: ").strip().upper()
    if method not in dominance_methods:
        print("Invalid method. Using default: DIST-EPS")
        method = "DIST-EPS"

    # Population
    try:
        population_size = int(input("👥 Population size [5-100, default: 10]: ") or 10)
        if not (5 <= population_size <= 100):
            raise ValueError
    except ValueError:
        print("Invalid input. Using default: 10")
        population_size = 10

    # Number of features
    try:
        max_features = len(input_cols)
        num_features = int(input(f"🔢 Number of features to select [1-{max_features}, default: 5]: ") or 5)
        if not (1 <= num_features <= max_features):
            raise ValueError
    except ValueError:
        print("Invalid input. Using default: 5")
        num_features = 5

    # Iteration type
    iter_type = input("⏱ Training mode (iter/sec) [default: iter]: ").strip().lower() or "iter"
    if iter_type not in ["iter", "sec"]:
        print("Invalid input. Using default: iter")
        iter_type = "iter"
    
    # Iteration value
    try:
        iter_val = int(input(f"Number of {iter_type} [default: 5]: ") or 5)
        if iter_val < 1:
            raise ValueError
    except ValueError:
        print("Invalid input. Using default: 5")
        iter_val = 5

print(f"✅ Algorithm configuration successful.")
# Run the NSGA-II algorithm
algorithm, method = simulate(method, data_scaled, costs, population_size, input_cols, output_col, num_features, iter_type, iter_val)

dataset_name = os.path.splitext(os.path.basename(dataset))[0]
if algorithm is not None:
    print(f"📘 Results will be saved in the 'results' directory under '{dataset_name}/{method}/'.")
    # Save results to CSV
    show_results(algorithm, method, dataset_name, population_size, iter_type, iter_val)
