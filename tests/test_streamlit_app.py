import importlib.util
import random
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[1]

if (ROOT / 'streamlit_app.py').is_file() and importlib.util.find_spec('streamlit'):
    from streamlit.testing.v1 import AppTest
else:
    AppTest = None


RESULT_KEY = '_svmfsmeta_result'
CSV = ('0;-1;10;100;5;15;7;12\n1;-1;20;200;10;30;14;24\n'
       '2;1;30;300;15;45;21;36\n3;1;40;400;20;60;28;48\n')
SMALL_CSV = '0;-1;.1;.2\n1;-1;.2;.3\n2;1;.8;.7\n3;1;.9;.8\n'


@unittest.skipIf(AppTest is None, 'Streamlit repository files and dependencies are optional')
class TestStreamlitApp(unittest.TestCase):
    def setUp(self):
        state = random.getstate()
        self.addCleanup(random.setstate, state)
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def app(self, csv=CSV):
        script = f'''
import io
import random
import runpy
import sys
from unittest.mock import patch
import pandas as pd
import streamlit as st
sys.path.insert(0, {str(ROOT)!r})
from svmfsmeta import SVMFsMeta
from views.simulation import _load_colon_example
if '_test_ready' not in st.session_state:
    _load_colon_example.clear()
    st.session_state['_test_ready'] = True
    st.session_state['_test_csv'] = {csv!r}
    st.session_state['_test_runs'] = 0
    st.session_state['_test_saves'] = 0
    st.session_state['_test_colon_loads'] = 0

def upload(*args, **kwargs):
    value = st.session_state['_test_csv']
    return None if value is None else io.BytesIO(value.encode('utf-8'))

def colon():
    st.session_state['_test_colon_loads'] += 1
    if st.session_state.get('_test_colon_error', False):
        raise RuntimeError('Loading Colon requires optional dependencies. Install examples.')
    frame = pd.DataFrame({{f'x{{j}}': [(i + 1) * j for i in range(4)] for j in range(1, 7)}})
    frame.insert(0, 'y', [-1, -1, 1, 1])
    return frame

original_run = SVMFsMeta.run
original_save = SVMFsMeta.save_results

def run(algorithm, *args, **kwargs):
    st.session_state['_test_runs'] += 1
    random.seed(2)
    return original_run(algorithm, *args, **kwargs)

def save(algorithm, dataset_name, *args, **kwargs):
    st.session_state['_test_saves'] += 1
    if st.session_state.get('_test_save_error', False):
        raise PermissionError('Test directory is read-only')
    kwargs['output_dir'] = {self.directory.name!r}
    return original_save(algorithm, dataset_name, *args, **kwargs)

with patch('views.simulation.file_uploader', side_effect=upload), \\
     patch('views.simulation.load_colon', side_effect=colon), \\
     patch.object(SVMFsMeta, 'run', new=run), \\
     patch.object(SVMFsMeta, 'save_results', new=save):
    runpy.run_path({str(ROOT / 'streamlit_app.py')!r}, run_name='__main__')
'''
        return AppTest.from_string(script, default_timeout=45).run()

    def assert_clean(self, app):
        self.assertFalse(app.exception, [item.message for item in app.exception])
        self.assertFalse(app.error, [item.value for item in app.error])

    @staticmethod
    def widget(elements, text):
        return next(widget for widget in elements if text in widget.label)

    def train(self, app):
        self.widget(app.selectbox, 'Dominance method').set_value('MC')
        self.widget(app.selectbox, 'Allow clones?').set_value(1)
        self.widget(app.number_input, 'Population size').set_value(5)
        self.widget(app.number_input, 'Number of iter').set_value(1)
        self.widget(app.button, 'Run simulation').click().run()
        self.assert_clean(app)
        self.assertIn(RESULT_KEY, app.session_state)
        return app.session_state[RESULT_KEY]['algorithm']

    def test_empty_app_renders_without_a_dataset(self):
        app = self.app(None)
        self.assert_clean(app)
        self.assertEqual(len(app.tabs), 2)
        self.assertNotIn(RESULT_KEY, app.session_state)

    def test_one_and_two_predictor_uploads_render_and_train(self):
        for csv, count in ((SMALL_CSV, 2), ('0;-1;.1\n1;-1;.2\n2;1;.8\n3;1;.9\n', 1)):
            with self.subTest(predictors=count):
                app = self.app(csv)
                self.assert_clean(app)
                self.assertEqual(self.widget(app.number_input, 'features to select').value, count)
                self.train(app)

    def test_save_keeps_the_model_and_writes_csv_without_retraining(self):
        app = self.app()
        model = self.train(app)
        self.assertIsNone(model.custom_logger)
        for expected_files in (1, 2):
            self.widget(app.button, 'Save results').click().run()
            self.assert_clean(app)
            self.assertIs(app.session_state[RESULT_KEY]['algorithm'], model)
            self.assertEqual(app.session_state['_test_runs'], 1)
            self.assertEqual(app.session_state['_test_saves'], expected_files)
            files = list((Path(self.directory.name) / 'streamlit_upload' / 'MC').glob('*.csv'))
            self.assertEqual(len(files), expected_files)
            self.assertTrue(all(path.stat().st_size > 0 for path in files))

    def test_changing_configuration_clears_the_previous_model(self):
        app = self.app()
        self.train(app)
        self.widget(app.selectbox, 'Dominance method').set_value('DIST-EPS').run()
        self.assert_clean(app)
        self.assertNotIn(RESULT_KEY, app.session_state)
        self.assertEqual(app.session_state['_test_runs'], 1)
        self.assertFalse(any('Save results' in button.label for button in app.button))

    def test_changing_data_clamps_feature_count_and_clears_the_model(self):
        app = self.app()
        self.train(app)
        app.session_state['_test_csv'] = SMALL_CSV
        app.run()
        self.assert_clean(app)
        self.assertNotIn(RESULT_KEY, app.session_state)
        self.assertEqual(self.widget(app.number_input, 'features to select').value, 2)
        app.session_state['_test_csv'] = None
        app.run()
        self.assert_clean(app)
        self.assertNotIn(RESULT_KEY, app.session_state)

    def test_legacy_weights_row_is_skipped_only_when_requested(self):
        app = self.app('0;0;1;1\n' + SMALL_CSV)
        self.assertFalse(app.exception)
        self.assertTrue(any('weights row' in item.value for item in app.error))
        self.widget(app.checkbox, 'First row contains').check().run()
        self.assert_clean(app)
        self.assertEqual(app.dataframe[0].value.shape, (4, 3))
        self.train(app)

    def test_colon_is_cached_and_saves_under_its_own_dataset_name(self):
        app = self.app(None)
        self.widget(app.radio, 'Dataset source').set_value('Colon (plsgenomics)').run()
        self.assert_clean(app)
        self.assertEqual(app.session_state['_test_colon_loads'], 1)
        self.assertEqual(app.dataframe[0].value.shape, (4, 7))
        self.widget(app.checkbox, 'Normalize input').uncheck().run()
        self.assert_clean(app)
        self.assertEqual(app.session_state['_test_colon_loads'], 1)
        model = self.train(app)
        self.widget(app.button, 'Save results').click().run()
        self.assert_clean(app)
        self.assertIs(app.session_state[RESULT_KEY]['algorithm'], model)
        self.assertTrue((Path(self.directory.name) / 'colon' / 'MC' / '5.csv').exists())
        self.assertEqual(app.session_state['_test_colon_loads'], 1)

    def test_colon_loading_errors_are_shown_without_crashing(self):
        app = self.app(None)
        app.session_state['_test_colon_error'] = True
        self.widget(app.radio, 'Dataset source').set_value('Colon (plsgenomics)').run()
        self.assertFalse(app.exception)
        self.assertTrue(any('optional dependencies' in item.value for item in app.error))
        self.assertNotIn(RESULT_KEY, app.session_state)

    def test_missing_predictors_are_reported_before_widgets_are_created(self):
        app = self.app('0;-1\n1;-1\n2;1\n3;1\n')
        self.assertFalse(app.exception)
        self.assertTrue(any('at least one predictor' in item.value for item in app.error))

    def test_training_errors_are_shown_without_crashing(self):
        app = self.app('0;-1;1;1\n1;-1;1;1\n2;1;1;1\n3;1;1;1\n')
        self.widget(app.button, 'Run simulation').click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any('initial population' in item.value for item in app.error))
        self.assertNotIn(RESULT_KEY, app.session_state)

    def test_save_errors_do_not_discard_the_completed_model(self):
        app = self.app()
        model = self.train(app)
        app.session_state['_test_save_error'] = True
        self.widget(app.button, 'Save results').click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any('read-only' in item.value for item in app.error))
        self.assertIs(app.session_state[RESULT_KEY]['algorithm'], model)
        self.assertEqual(app.session_state['_test_runs'], 1)


if __name__ == '__main__':
    unittest.main()
