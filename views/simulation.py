from sklearn.preprocessing import MinMaxScaler

from src.nsga2_svm_fs.nsga2_svmfs import NSGA2_SVMFS
from streamlit import header, file_uploader, checkbox, subheader, selectbox, number_input, radio, button, empty, \
    success, dataframe, error, stop, spinner, write, expander, slider
from pandas import read_csv, DataFrame

DEFAULT_DATA_URL = "https://raw.githubusercontent.com/danielvacarre/NSGAII_SVM_FS/main/data/09Colon_p.txt"

def show_simulation():

    header("Simulation")

    uploaded_file = file_uploader("📂 Upload your CSV or TXT file", type=["csv", "txt"])
    write("An example dataset is available in: https://github.com/danielvacarre/NSGAII_SVM_FS/blob/main/data/09Colon_p.txt.")

    dominance_methods = ["DIST-EPS", "MC"]

    if uploaded_file:

        # === CSV Read Options ===
        ("⚙️ CSV Read Options")
        has_header = checkbox("Does the file have a header?", value=False)
        has_index = checkbox("Does the file have an index column?", value=True)
        separator = selectbox("Select separator", [",", ";", "\t"], index=1)
        normalize = checkbox("📏 Normalize input features?", value=True)

        # CSV parsing config
        header_option = 0 if has_header else None

        try:
            data = read_csv(uploaded_file, sep=separator, header=header_option)

            if has_index:
                data.drop(columns=[data.columns[0]], inplace=True)

            if not has_header:
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

            success("✅ File loaded successfully.")
            dataframe(data_scaled.head())

        except Exception as e:
            error(f"❌ Error while loading the data: {e}")
            stop()

        # === Algorithm Configuration ===
        method = selectbox("⚙️ Dominance method", dominance_methods)
        population_size = number_input("👥 Population size", min_value=5, value=10, max_value=100)
        num_features = number_input("🔢 Number of features to select", min_value=1, max_value=len(input_cols), value=5)
        iter_type = radio("⏱ Training mode", ["iter", "sec"])
        iter_val = number_input(f"Number of {iter_type}", min_value=1, value=5)

        # === Advanced Parameters ===
        with expander("🛡️ Advanced Settings"):
            allow_clones = selectbox("Allow clones?", [0, 1], index=0, help="Whether to allow duplicated individuals in the population.")
            p_mutation = slider("Global mutation probability", 0.0, 1.0, 0.7)
            p_mutate_individual = slider("Individual mutation probability", 0.0, 1.0, 0.4)
            p_mutate_feature = slider("Feature mutation probability", 0.0, 1.0, 0.4)
            p_mutate_coord = slider("Coordinate mutation probability", 0.0, 1.0, 0.2)
            coord_mutation_type = selectbox("Coordinate mutation type", [0, 1], index=0, help="0: Scaling, 1: Full replacement")

        # === Run svmfsmeta ===
        if button("▶️ Run simulation"):
            with spinner("⏳ Running svmfsmeta..."):
                log_area = empty()
                logs = []

                def log(message: str):
                    logs.append(message)
                    log_area.text_area("📜 Execution log", value="\n".join(logs), height=200)

                algorithm = NSGA2_SVMFS(
                    method=method,
                    data=data_scaled,
                    population_size=population_size,
                    inputs=input_cols,
                    output=output_col,
                    num_selected_features=num_features,
                    allow_clones=allow_clones,
                    p_mutation=p_mutation,
                    p_mutate_individual=p_mutate_individual,
                    p_mutate_feature=p_mutate_feature,
                    p_mutate_coord=p_mutate_coord,
                    coord_mutation_type=coord_mutation_type,
                    logger=log
                )
                algorithm.run(train=iter_type, num_iter=iter_val)

                return algorithm, method

    return None, None
