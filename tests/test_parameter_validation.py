from unittest import TestCase

from gui.parameter_validation import validate_parameters


class TestParameterValidation(TestCase):

    # Teste 1: todos os parâmetros são válidos.
    def test_valid_parameters(self):
        result = validate_parameters(
            labeled_percentage=50,
            threshold=0.95,
            cr=0.05,
            classifiers=["KNN", "Random Forest"],
        )

        self.assertEqual(result, {
            "labeled_percentage": 50.0,
            "threshold": 0.95,
            "cr": 0.05,
            "classifiers": ["KNN", "Random Forest"],
        })

    # Teste 2: percentual de dados rotulados fora do intervalo.
    def test_invalid_labeled_percentage(self):
        for percentage in [-1, 101]:
            with self.subTest(percentage=percentage):
                with self.assertRaises(ValueError):
                    validate_parameters(
                        labeled_percentage=percentage,
                        threshold=0.95,
                        cr=0.05,
                        classifiers=["KNN"],
                    )

    # Teste 3: threshold fora do intervalo.
    def test_invalid_threshold(self):
        for threshold in [-0.1, 1.1]:
            with self.subTest(threshold=threshold):
                with self.assertRaises(ValueError):
                    validate_parameters(
                        labeled_percentage=50,
                        threshold=threshold,
                        cr=0.05,
                        classifiers=["KNN"],
                    )

    # Teste 4: CR fora do intervalo.
    def test_invalid_cr(self):
        for cr in [-0.1, 1.1]:
            with self.subTest(cr=cr):
                with self.assertRaises(ValueError):
                    validate_parameters(
                        labeled_percentage=50,
                        threshold=0.95,
                        cr=cr,
                        classifiers=["KNN"],
                    )

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

                with self.assertRaises(ValueError):
                    validate_parameters(**parameters)

    # Teste 6: valores numéricos inválidos.
    def test_non_numeric_values(self):
        with self.assertRaises(ValueError):
            validate_parameters(
                labeled_percentage="abc",
                threshold=0.95,
                cr=0.05,
                classifiers=["KNN"],
            )

    # Teste 7: valores infinitos ou NaN.
    def test_non_finite_values(self):
        for value in [float("inf"), float("-inf"), float("nan")]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_parameters(
                        labeled_percentage=value,
                        threshold=0.95,
                        cr=0.05,
                        classifiers=["KNN"],
                    )

    # Teste 8: nenhum classificador selecionado.
    def test_empty_classifiers(self):
        with self.assertRaises(ValueError):
            validate_parameters(
                labeled_percentage=50,
                threshold=0.95,
                cr=0.05,
                classifiers=[],
            )

    # Teste 9: classificador não suportado.
    def test_invalid_classifier(self):
        with self.assertRaises(ValueError):
            validate_parameters(
                labeled_percentage=50,
                threshold=0.95,
                cr=0.05,
                classifiers=["XGBoost"],
            )

    # Teste 10: valores nos limites permitidos.
    def test_boundary_values(self):
        result = validate_parameters(
            labeled_percentage=100,
            threshold=1,
            cr=1,
            classifiers=["KNN"],
        )

        self.assertEqual(result["labeled_percentage"], 100.0)
        self.assertEqual(result["threshold"], 1.0)
        self.assertEqual(result["cr"], 1.0)

    # Teste 11: aceitar números com até 6 casas decimais.
    def test_valid_decimal_precision(self):
        result = validate_parameters(
            labeled_percentage="50.123456",
            threshold="0.123456",
            cr="0.123456",
            classifiers=["KNN"],
        )

        self.assertEqual(result["labeled_percentage"], 50.123456)
        self.assertEqual(result["threshold"], 0.123456)
        self.assertEqual(result["cr"], 0.123456)

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

                with self.assertRaisesRegex(
                    ValueError,
                    "não pode ultrapassar 6 casas decimais",
                ):
                    validate_parameters(**parameters)