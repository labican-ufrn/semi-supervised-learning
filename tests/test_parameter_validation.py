from unittest import TestCase

from mlabican.gui.parameter_validation import validate_parameters


class TestParameterValidation(TestCase):

    # Teste 1: todos os parâmetros são válidos.
    def test_valid_parameters(self):
        result = validate_parameters(
            labeled_percentage=50,
            threshold=0.95,
            cr=0.05,
            classifiers=["KNN", "Random Forest"],
        )

        self.assertEqual(result["parameters"], {
            "labeled_percentage": 50.0,
            "threshold": 0.95,
            "cr": 0.05,
            "classifiers": ["KNN", "Random Forest"],
        })

    # Teste 2: percentual de dados rotulados fora do intervalo.
    def test_invalid_labeled_percentage(self):
        for percentage in [-1, 101]:
            with self.subTest(percentage=percentage):
                result = validate_parameters(
                    labeled_percentage=percentage,
                    threshold=0.95,
                    cr=0.05,
                    classifiers=["KNN"],
                )

                self.assertIn("labeled_percentage", result["errors"])

    # Teste 3: threshold fora do intervalo.
    def test_invalid_threshold(self):
        for threshold in [-0.1, 1.1]:
            with self.subTest(threshold=threshold):
                result = validate_parameters(
                    labeled_percentage=50,
                    threshold=threshold,
                    cr=0.05,
                    classifiers=["KNN"],
                )

                self.assertIn("threshold", result["errors"])

    # Teste 4: CR fora do intervalo.
    def test_invalid_cr(self):
        for cr in [-0.1, 1.1]:
            with self.subTest(cr=cr):
                result = validate_parameters(
                    labeled_percentage=50,
                    threshold=0.95,
                    cr=cr,
                    classifiers=["KNN"],
                )

                self.assertIn("cr", result["errors"])

    # Teste 5: campos numéricos vazios.
    def test_empty_numeric_fields(self):
        for field in ["labeled_percentage", "threshold", "cr"]:
            with self.subTest(field=field):
                parameters = {
                    "labeled_percentage": 50,
                    "threshold": 0.95,
                    "cr": 0.05,
                    "classifiers": ["KNN"],
                }

                parameters[field] = ""

                result = validate_parameters(**parameters)

                self.assertIn(field, result["errors"])

    # Teste 6: valores numéricos inválidos.
    def test_non_numeric_values(self):
        result = validate_parameters(
            labeled_percentage="abc",
            threshold=0.95,
            cr=0.05,
            classifiers=["KNN"],
        )

        self.assertIn("labeled_percentage", result["errors"])

    # Teste 7: valores infinitos ou NaN.
    def test_non_finite_values(self):
        for value in [float("inf"), float("-inf"), float("nan")]:
            with self.subTest(value=value):
                result = validate_parameters(
                    labeled_percentage=value,
                    threshold=0.95,
                    cr=0.05,
                    classifiers=["KNN"],
                )

                self.assertIn("labeled_percentage", result["errors"])

    # Teste 8: nenhum classificador selecionado.
    def test_empty_classifiers(self):
        result = validate_parameters(
            labeled_percentage=50,
            threshold=0.95,
            cr=0.05,
            classifiers=[],
        )

        self.assertIn("classifiers", result["errors"])

    # Teste 9: classificador não suportado.
    def test_invalid_classifier(self):
        result = validate_parameters(
            labeled_percentage=50,
            threshold=0.95,
            cr=0.05,
            classifiers=["XGBoost"],
        )

        self.assertIn("classifiers", result["errors"])

    # Teste 10: valores nos limites permitidos.
    def test_boundary_values(self):
        result = validate_parameters(
            labeled_percentage=100,
            threshold=1,
            cr=1,
            classifiers=["KNN"],
        )

        parameters = result["parameters"]

        self.assertEqual(parameters["labeled_percentage"], 100.0)
        self.assertEqual(parameters["threshold"], 1.0)
        self.assertEqual(parameters["cr"], 1.0)

    # Teste 11: aceitar números com até 6 casas decimais.
    def test_valid_decimal_precision(self):
        result = validate_parameters(
            labeled_percentage="50.123456",
            threshold="0.123456",
            cr="0.123456",
            classifiers=["KNN"],
        )

        parameters = result["parameters"]

        self.assertEqual(parameters["labeled_percentage"], 50.123456)
        self.assertEqual(parameters["threshold"], 0.123456)
        self.assertEqual(parameters["cr"], 0.123456)

    # Teste 12: rejeitar números com mais de 6 casas decimais.
    def test_invalid_decimal_precision(self):
        for field in ["labeled_percentage", "threshold", "cr"]:
            with self.subTest(field=field):
                parameters = {
                    "labeled_percentage": "50",
                    "threshold": "0.5",
                    "cr": "0.05",
                    "classifiers": ["KNN"],
                }

                parameters[field] = "0.1234567"

                result = validate_parameters(**parameters)

                self.assertIn(field, result["errors"])
                self.assertRegex(
                    result["errors"][field],
                    "não pode ultrapassar 6 casas decimais",
                )