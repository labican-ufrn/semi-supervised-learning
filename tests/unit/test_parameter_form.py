import unittest
from unittest.mock import patch

from mlabican.gui.parameter_form import ParameterForm
from mlabican.gui.theme import ThemeColor


class TestParameterForm(unittest.TestCase):
    def setUp(self):
        """Initialize the ParameterForm hidden in the background before each test."""
        self.app = ParameterForm()
        self.app.withdraw()
        self.app.update()

    def tearDown(self):
        """Clean up the Tkinter instance after each test."""
        if self.app:
            self.app.destroy()

    def test_initial_ui_state(self):
        """Verify window title, default inputs, and unchecked classifiers."""
        self.assertEqual(
            self.app.title(), 'FlexCon - Configuração de Experimento'
        )
        self.assertEqual(self.app.entry_percentage.get(), '')
        self.assertEqual(self.app.entry_threshold.get(), '')
        self.assertEqual(self.app.entry_cr.get(), '')
        self.assertEqual(self.app.status_label.cget('text'), '')

        expected_classifiers = [
            'KNN',
            'Naive Bayes',
            'Decision Tree',
            'Random Forest',
            'Logistic Regression',
            'Neural Network',
        ]
        self.assertEqual(
            list(self.app.classifiers_vars.keys()), expected_classifiers
        )
        for var in self.app.classifiers_vars.values():
            self.assertFalse(var.get())

    @patch('mlabican.gui.parameter_form.validate_parameters')
    def test_collect_parameters_success(self, mock_validate):
        """Verify valid inputs pass expected arguments and show success message."""
        expected_params = {
            'labeled_percentage': 50.0,
            'threshold': 0.7,
            'cr': 0.3,
            'classifiers': ['KNN', 'Random Forest'],
        }
        mock_validate.return_value = {'parameters': expected_params}

        # Populate form inputs
        self.app.entry_percentage.insert(0, '50')
        self.app.entry_threshold.insert(0, '0.7')
        self.app.entry_cr.insert(0, '0.3')
        self.app.classifiers_vars['KNN'].set(True)
        self.app.classifiers_vars['Random Forest'].set(True)
        self.app.update()

        # Simulate clicking the confirm button
        self.app.button_confirm._command()
        self.app.update()

        # Check that validate_parameters received the exact UI values
        mock_validate.assert_called_once_with(
            '50', '0.7', '0.3', ['KNN', 'Random Forest']
        )

        # Verify stored parameters and status label updates
        self.assertEqual(self.app.parameters, expected_params)
        self.assertEqual(
            self.app.status_label.cget('text'),
            'Parâmetros validados com sucesso!',
        )
        self.assertEqual(
            self.app.status_label.cget('text_color'),
            ThemeColor.SUCCESS.value,
        )

    @patch('mlabican.gui.parameter_form.validate_parameters')
    def test_collect_parameters_displays_validation_errors(self, mock_validate):
        """Verify all field error labels update when validation fails."""
        mock_validate.return_value = {
            'errors': {
                'labeled_percentage': 'Percentual inválido',
                'threshold': 'Threshold deve estar entre 0 e 1',
                'cr': 'CR obrigatório',
                'classifiers': 'Selecione ao menos um classificador',
            }
        }

        self.app.collect_parameters()
        self.app.update()

        self.assertEqual(
            self.app.percentage_error_label.cget('text'),
            'Percentual inválido',
        )
        self.assertEqual(
            self.app.threshold_error_label.cget('text'),
            'Threshold deve estar entre 0 e 1',
        )
        self.assertEqual(
            self.app.cr_error_label.cget('text'), 'CR obrigatório'
        )
        self.assertEqual(
            self.app.classifiers_error_label.cget('text'),
            'Selecione ao menos um classificador',
        )
        self.assertEqual(self.app.status_label.cget('text'), '')
        self.assertFalse(hasattr(self.app, 'parameters'))

    @patch('mlabican.gui.parameter_form.validate_parameters')
    def test_previous_errors_clear_on_resubmit(self, mock_validate):
        """Verify old error messages are wiped out when fixing inputs."""
        # First submission fails
        mock_validate.return_value = {
            'errors': {'labeled_percentage': 'Erro antigo'}
        }
        self.app.collect_parameters()
        self.app.update()
        self.assertEqual(
            self.app.percentage_error_label.cget('text'), 'Erro antigo'
        )

        # Second submission succeeds
        mock_validate.return_value = {'parameters': {'labeled_percentage': 20}}
        self.app.collect_parameters()
        self.app.update()

        self.assertEqual(self.app.percentage_error_label.cget('text'), '')
        self.assertEqual(
            self.app.status_label.cget('text'),
            'Parâmetros validados com sucesso!',
        )
