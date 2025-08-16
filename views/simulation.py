from sklearn.preprocessing import MinMaxScaler

from src.nsga2_svm_fs.nsga2_svmfs import NSGA2_SVMFS
from streamlit import header, file_uploader, checkbox, subheader, selectbox, number_input, radio, button, empty, \
    success, dataframe, error, stop, spinner, write
from pandas import read_csv, DataFrame

DEFAULT_DATA_URL = "https://raw.githubusercontent.com/danielvacarre/NSGAII_SVM_FS/main/data/09Colon_p.txt"

def show_simulation():

    header("Simulation")

    uploaded_file = file_uploader("📂 Upload your CSV or TXT file", type=["csv", "txt"])
    write("An example dataset is available in: https://github.com/danielvacarre/NSGAII_SVM_FS/blob/main/data/09Colon_p.txt.")

    dominance_methods = ["DIST-EPS", "DIST-EPS-COST", "MC", "MC-COST"]

    if uploaded_file:

        # === CSV Read Options ===
        ("⚙️ CSV Read Options")
        has_header = checkbox("Does the file have a header?", value=False)
        has_index = checkbox("Does the file have an index column?", value=True)
        has_costs = checkbox("Does the file include a row with feature costs?", value=True)
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

        # === Run NSGA-II ===
        if button("▶️ Run simulation"):
            with spinner("⏳ Running NSGA-II..."):
                log_area = empty()
                logs = []

                def log(message: str):
                    logs.append(message)
                    log_area.text_area("📜 Execution log", value="\n".join(logs), height=200)

                algorithm = NSGA2_SVMFS(
                    method=method,
                    data= data_scaled,
                    costs=costs,
                    population_size=population_size,
                    inputs=input_cols,
                    output=output_col,
                    num_selected_features=num_features,
                    logger=log
                )
                algorithm.run(train=iter_type, num_iter=iter_val)

                return algorithm, method

    return None, None