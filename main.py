import sys
import os
import argparse
from sklearn.preprocessing import MinMaxScaler
from svm_fs_meta import SVMFsMeta
from pandas import read_csv, DataFrame

def simulate(method, data_scaled, population_size, input_cols, output_col, num_features, iter_type, iter_val, **kwargs):
    print(f"Running NSGA-II simulation with method: {method}, population size: {population_size}, number of features: {num_features}, iteration type: {iter_type}, and iteration value: {iter_val}")
    algorithm = SVMFsMeta(
                    method=method,
                    data= data_scaled,
                    population_size=population_size,
                    inputs=input_cols,
                    output=output_col,
                    num_selected_features=num_features,
                    **kwargs
                )
    
    algorithm.run(train=iter_type, num_iter=iter_val)
    print(f"✅ NSGA-II simulation completed with method: {method}")

    return algorithm, method


DEFAULT_DATA_URL = "https://raw.githubusercontent.com/danielvacarre/NSGAII_SVM_FS/main/data/09Colon_p.txt"
dominance_methods_list = ["DIST-EPS", "MC"]

def main():
    dominance_methods = dominance_methods_list.copy()

    parser = argparse.ArgumentParser(description="NSGA-II SVM Feature Selection")
    parser.add_argument("dataset", nargs="?", help="Path to the dataset")
    parser.add_argument("--has_header", type=str, choices=['yes', 'no', 'y', 'n'], help="Does the file have a header? [yes/no]")
    parser.add_argument("--has_index", type=str, choices=['yes', 'no', 'y', 'n'], help="Does the file have an index? [yes/no]")
    parser.add_argument("--separator", type=str, help="Separator (e.g. ',' or ';')")
    parser.add_argument("--normalize", type=str, choices=['yes', 'no', 'y', 'n'], help="Normalize input features? [yes/no]")
    
    parser.add_argument("--method", type=str, choices=dominance_methods, help="Dominance method")
    parser.add_argument("--population_size", type=int, help="Population size")
    parser.add_argument("--num_features", type=int, help="Number of features to select")
    parser.add_argument("--iter_type", type=str, choices=['iter', 'sec'], help="Training mode (iter/sec)")
    parser.add_argument("--iter_val", type=int, help="Number of iterations or seconds")
    
    # Advanced parameters
    parser.add_argument("--allow_clones", type=int, choices=[0, 1], help="Whether to allow duplicated individuals in the population")
    parser.add_argument("--p_mutation", type=float, help="Global mutation probability (default: 0.7)")
    parser.add_argument("--p_mutate_individual", type=float, help="Individual mutation probability (default: 0.4)")
    parser.add_argument("--p_mutate_feature", type=float, help="Feature mutation probability (default: 0.4)")
    parser.add_argument("--p_mutate_coord", type=float, help="Coordinate mutation probability (default: 0.2)")
    parser.add_argument("--coord_mutation_type", type=int, choices=[0, 1], help="Coordinate mutation type (0: Scaling, 1: Full replacement) (default: 0)")

    args = parser.parse_args()

    interactive = len(sys.argv) <= 2 and not any(arg.startswith('--') for arg in sys.argv)

    dataset = args.dataset
    if not dataset:
        if interactive:
            dataset = input("Dataset path: ").strip()
            if not dataset:
                print(f"No dataset provided. You can download the dataset included in the GitHub repository: {DEFAULT_DATA_URL}")
                sys.exit(1)
        else:
            parser.print_help()
            sys.exit(1)

    # Resolve arguments with interactive prompts as fallback if running interactively
    def resolve_bool(arg_val, prompt, default_val):
        if arg_val is not None:
            return arg_val.lower() in ['yes', 'y']
        if interactive:
            user_input = input(f"{prompt} (yes/no) [default: {'yes' if default_val else 'no'}]: ").strip().lower()
            if user_input:
                return user_input in ['yes', 'y']
        return default_val

    has_header = resolve_bool(args.has_header, "Does the file have a header?", False)
    has_index = resolve_bool(args.has_index, "Does the file have an index column?", True)
    separator = args.separator
    if separator is None:
        if interactive:
            separator = input("Select separator (','  ';'  '\\t') [default: ',']: ").strip() or ","
        else:
            separator = ","

    normalize = resolve_bool(args.normalize, "Normalize input features?", True)

    # CSV parsing config
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

        if normalize:
            scaler = MinMaxScaler()
            scaled_features = scaler.fit_transform(data[input_cols])
            data_scaled = DataFrame(scaled_features, columns=input_cols)
            data_scaled[output_col] = data[output_col].values
        else:
            data_scaled = data.copy()

        print(f"File loaded successfully.")

    except Exception as e:
        print(f"Error while loading the data: {e}")
        sys.exit(1)

    # Algorithm configuration
    if interactive:
        print("ALGORITHM CONFIGURATION")

    method = args.method
    if method is None:
        if interactive:
            print(f"Available dominance methods: {', '.join(dominance_methods)}")
            method = input("Select dominance method: ").strip().upper()
            if method not in dominance_methods:
                print("Invalid method. Using default: DIST-EPS")
                method = "DIST-EPS"
        else:
            method = "DIST-EPS"

    population_size = args.population_size
    if population_size is None:
        if interactive:
            try:
                population_size = int(input("👥 Population size [5-100, default: 10]: ") or 10)
            except ValueError:
                population_size = 10
        else:
            population_size = 10

    num_features = args.num_features
    if num_features is None:
        if interactive:
            try:
                max_features = len(input_cols)
                num_features = int(input(f"Number of features to select [1-{max_features}, default: 5]: ") or 5)
            except ValueError:
                num_features = 5
        else:
            num_features = 5

    iter_type = args.iter_type
    if iter_type is None:
        if interactive:
            iter_type = input("Training mode (iter/sec) [default: iter]: ").strip().lower() or "iter"
            if iter_type not in ["iter", "sec"]:
                iter_type = "iter"
        else:
            iter_type = "iter"

    iter_val = args.iter_val
    if iter_val is None:
        if interactive:
            try:
                iter_val = int(input(f"Number of {iter_type} [default: 5]: ") or 5)
            except ValueError:
                iter_val = 5
        else:
            iter_val = 5

    kwargs = {}
    if args.allow_clones is not None: kwargs['allow_clones'] = args.allow_clones
    if args.p_mutation is not None: kwargs['p_mutation'] = args.p_mutation
    if args.p_mutate_individual is not None: kwargs['p_mutate_individual'] = args.p_mutate_individual
    if args.p_mutate_feature is not None: kwargs['p_mutate_feature'] = args.p_mutate_feature
    if args.p_mutate_coord is not None: kwargs['p_mutate_coord'] = args.p_mutate_coord
    if args.coord_mutation_type is not None: kwargs['coord_mutation_type'] = args.coord_mutation_type

    print(f"Algorithm configuration successful.")
    
    # Run the NSGA-II algorithm
    algorithm, method_used = simulate(method, data_scaled, population_size, input_cols, output_col, num_features, iter_type, iter_val, **kwargs)

    dataset_name = os.path.splitext(os.path.basename(dataset))[0]
    if algorithm is not None:
        print(f"Results will be saved in the 'results' directory under '{dataset_name}/{method_used}/'.")
        algorithm.save_results(dataset_name, iteration_val=iter_val, iteration_type=iter_type)

if __name__ == "__main__":
    main()
