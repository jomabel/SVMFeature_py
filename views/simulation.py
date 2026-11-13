from hashlib import sha256

from svmfsmeta.preprocessing import minmax_scale
from pandas import read_csv, DataFrame
from pandas.util import hash_pandas_object
from streamlit import (header, file_uploader, checkbox, selectbox, number_input, radio,
                       button, empty, success, dataframe, error, spinner, write,
                       expander, slider, session_state, cache_data)

from main import COLON_PACKAGE_VERSION, load_colon
from svmfsmeta import SVMFsMeta


RESULT_KEY = '_svmfsmeta_result'


@cache_data(show_spinner=False)
def _load_colon_example():
    return load_colon()


def _clear_result():
    session_state.pop(RESULT_KEY, None)


def show_simulation():
    header('Simulation')
    source = radio('Dataset source', ['Upload CSV/TXT', 'Colon (plsgenomics)'])
    use_colon = source == 'Colon (plsgenomics)'
    dataset_name = 'colon' if use_colon else 'streamlit_upload'

    if use_colon:
        uploaded_file = None
        write(f'Colon from plsgenomics {COLON_PACKAGE_VERSION}: normal tissue = 1, tumor tissue = -1.')
        normalize = checkbox('📏 Normalize input features?', value=True)
    else:
        uploaded_file = file_uploader('📂 Upload your CSV or TXT file', type=['csv', 'txt'])
        if uploaded_file is None:
            _clear_result()
            return None, None, None
        has_header = checkbox('Does the file have a header?', value=False)
        has_index = checkbox('Does the file have an index column?', value=True)
        separator = selectbox('Select separator', [',', ';', '\t'], index=1)
        skip_weights = checkbox(
            'First row contains feature weights/costs', value=False,
            help='Enable only for legacy files whose first data row stores weights rather than a sample.'
        )
        normalize = checkbox('📏 Normalize input features?', value=True)

    try:
        if use_colon:
            with spinner('Loading Colon...'):
                data = _load_colon_example()
        else:
            uploaded_file.seek(0)
            data = read_csv(uploaded_file, sep=separator, header=0 if has_header else None)
            if has_index:
                data = data.iloc[:, 1:].copy()
            if skip_weights:
                data = data.iloc[1:].reset_index(drop=True)
            if not has_header:
                data.columns = ['y'] + [f'x{i}' for i in range(1, data.shape[1])]

        if data.empty or data.shape[1] < 2:
            raise ValueError('The dataset must contain samples, a response column and at least one predictor.')
        output_col = data.columns[0]
        input_cols = data.columns[1:].tolist()
        response = data[output_col]
        if (not response.isin([-1, 1]).all() or set(response.unique()) != {-1, 1} or
                response.value_counts().min() < 2):
            raise ValueError(
                'The response must contain -1 and 1 only, with at least two observations per class. '
                'For legacy files with a weights row, enable First row contains feature weights/costs.'
            )
        if normalize:
            data_scaled = minmax_scale(data, input_cols)
        else:
            data_scaled = data.copy()
        fingerprint = sha256(hash_pandas_object(data_scaled, index=True).values.tobytes()).hexdigest()
    except Exception as exc:
        _clear_result()
        error(f'❌ Error while loading the data: {exc}')
        return None, None, None

    success(f'✅ Dataset loaded: {len(data)} samples, {len(input_cols)} predictors.')
    dataframe(data_scaled.head())

    method = selectbox('⚙️ Dominance method', ['DIST-EPS', 'MC'])
    population_size = number_input('👥 Population size', min_value=5, value=10, max_value=100)
    num_features = number_input('🔢 Number of features to select', min_value=1,
                                max_value=len(input_cols), value=min(5, len(input_cols)))
    iter_type = radio('⏱ Training mode', ['iter', 'sec'])
    iter_val = number_input(f'Number of {iter_type}', min_value=1, value=5)

    with expander('🛡️ Advanced Settings'):
        allow_clones = selectbox('Allow clones?', [0, 1], index=0,
                                 help='Whether to allow duplicate individuals in the population.')
        p_mutation = slider('Global mutation probability', 0.0, 1.0, 0.7)
        p_mutate_individual = slider('Individual mutation probability', 0.0, 1.0, 0.4)
        p_mutate_feature = slider('Feature mutation probability', 0.0, 1.0, 0.4)
        p_mutate_coord = slider('Coordinate mutation probability', 0.0, 1.0, 0.2)
        coord_mutation_type = selectbox('Coordinate mutation type', [0, 1], index=0,
                                        help='0: Scaling, 1: Full replacement')

    request = (dataset_name, normalize, tuple(data_scaled.columns), fingerprint, method, population_size,
               num_features, iter_type, iter_val, allow_clones, p_mutation,
               p_mutate_individual, p_mutate_feature, p_mutate_coord, coord_mutation_type)
    result = session_state.get(RESULT_KEY)
    if result is not None and result['request'] != request:
        _clear_result()

    log_area = empty()
    if button('▶️ Run simulation'):
        _clear_result()
        logs = []
        algorithm = None

        def log(message):
            logs.append(message)
            log_area.code('\n'.join(logs), language='text')

        try:
            with spinner('⏳ Running svmfsmeta...'):
                algorithm = SVMFsMeta(
                    method=method, data=data_scaled, population_size=population_size,
                    inputs=input_cols, output=output_col, num_selected_features=num_features,
                    allow_clones=allow_clones, p_mutation=p_mutation,
                    p_mutate_individual=p_mutate_individual, p_mutate_feature=p_mutate_feature,
                    p_mutate_coord=p_mutate_coord, coord_mutation_type=coord_mutation_type,
                    logger=log
                )
                algorithm.run(train=iter_type, num_iter=iter_val)
            session_state[RESULT_KEY] = {
                'algorithm': algorithm, 'method': method, 'dataset_name': dataset_name,
                'request': request, 'logs': logs,
            }
        except Exception as exc:
            error(f'❌ Error while running the simulation: {exc}')
        finally:
            if algorithm is not None:
                # A saved model must not retain a callback into a previous UI run.
                algorithm.custom_logger = None

    result = session_state.get(RESULT_KEY)
    if result is not None:
        log_area.code('\n'.join(result['logs']), language='text')
        return result['algorithm'], result['method'], result['dataset_name']
    return None, None, None
