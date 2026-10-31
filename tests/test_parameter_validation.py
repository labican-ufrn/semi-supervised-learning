import pytest

from gui.parameter_validation import validate_parameters


# Teste 1: todos os parâmetros são válidos.
def test_valid_parameters():
    result = validate_parameters(
        labeled_percentage=50,
        threshold=0.95,
        cr=0.05,
        classifiers=["KNN", "Random Forest"],
    )

    assert result == {
        "labeled_percentage": 50.0,
        "threshold": 0.95,
        "cr": 0.05,
        "classifiers": ["KNN", "Random Forest"],
    }


# Teste 2: percentual de dados rotulados fora do intervalo.
@pytest.mark.parametrize("percentage", [-1, 101])
def test_invalid_labeled_percentage(percentage):
    with pytest.raises(ValueError):
        validate_parameters(
            labeled_percentage=percentage,
            threshold=0.95,
            cr=0.05,
            classifiers=["KNN"],
        )


# Teste 3: threshold fora do intervalo.
@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_invalid_threshold(threshold):
    with pytest.raises(ValueError):
        validate_parameters(
            labeled_percentage=50,
            threshold=threshold,
            cr=0.05,
            classifiers=["KNN"],
        )


# Teste 4: CR fora do intervalo.
@pytest.mark.parametrize("cr", [-0.1, 1.1])
def test_invalid_cr(cr):
    with pytest.raises(ValueError):
        validate_parameters(
            labeled_percentage=50,
            threshold=0.95,
            cr=cr,
            classifiers=["KNN"],
        )


# Teste 5: campos numéricos vazios.
@pytest.mark.parametrize(
    "field",
    ["labeled_percentage", "threshold", "cr"],
)
def test_empty_numeric_fields(field):
    parameters = {
        "labeled_percentage": 50,
        "threshold": 0.95,
        "cr": 0.05,
        "classifiers": ["KNN"],
    }

    parameters[field] = ""

    with pytest.raises(ValueError):
        validate_parameters(**parameters)


# Teste 6: valores numéricos inválidos.
def test_non_numeric_values():
    with pytest.raises(ValueError):
        validate_parameters(
            labeled_percentage="abc",
            threshold=0.95,
            cr=0.05,
            classifiers=["KNN"],
        )


# Teste 7: valores infinitos ou NaN.
@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_non_finite_values(value):
    with pytest.raises(ValueError):
        validate_parameters(
            labeled_percentage=value,
            threshold=0.95,
            cr=0.05,
            classifiers=["KNN"],
        )


# Teste 8: nenhum classificador selecionado.
def test_empty_classifiers():
    with pytest.raises(ValueError):
        validate_parameters(
            labeled_percentage=50,
            threshold=0.95,
            cr=0.05,
            classifiers=[],
        )


# Teste 9: classificador não suportado.
def test_invalid_classifier():
    with pytest.raises(ValueError):
        validate_parameters(
            labeled_percentage=50,
            threshold=0.95,
            cr=0.05,
            classifiers=["XGBoost"],
        )


# Teste 10: valores nos limites permitidos.
def test_boundary_values():
    result = validate_parameters(
        labeled_percentage=100,
        threshold=1,
        cr=1,
        classifiers=["KNN"],
    )

    assert result["labeled_percentage"] == 100.0
    assert result["threshold"] == 1.0
    assert result["cr"] == 1.0
