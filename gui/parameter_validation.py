import math
import re

SUPPORTED_CLASSIFIERS = {
    "KNN",
    "Naive Bayes",
    "Decision Tree",
    "Random Forest",
    "Logistic Regression",
    "Neural Network",
}


def validate_parameters(labeled_percentage, threshold, cr, classifiers):

    if labeled_percentage is None or str(labeled_percentage).strip() == "":
        raise ValueError("O Percentual de dados rotulados não pode ser vazio.")

    if threshold is None or str(threshold).strip() == "":
        raise ValueError("O Threshold não pode ser vazio.")

    if cr is None or str(cr).strip() == "":
        raise ValueError("O CR não pode ser vazio.")

    # aceitar números com até 6 casas decimais
    number_pattern = r"[+-]?\d+(\.\d{1,6})?"

    fields = {
        "Percentual de dados rotulados": labeled_percentage,
        "Threshold": threshold,
        "CR": cr
    }

    for field, value in fields.items():
        value = str(value).strip()  # NOVO: armazenar o valor sem espaços

        # verificar se ultrapassou 6 casas decimais
        if re.search(r"\.\d{7,}", value):
            raise ValueError(
                f"{field} não pode ultrapassar 6 casas decimais."
            )

        # Verificar o formato do número
        if not re.fullmatch(number_pattern, value):
            raise ValueError(f"Digite apenas números no {field}.")

    try:
        LabeledPercentage = float(labeled_percentage)
        Threshold = float(threshold)
        CR = float(cr)

    except (ValueError, TypeError):
        raise ValueError("Digite apenas números.")

    if not all(math.isfinite(value) for value in [LabeledPercentage, Threshold, CR]):
        raise ValueError("Digite apenas números finitos.")

    if not (0 <= LabeledPercentage <= 100):
        raise ValueError("O valor de rotulagem deve estar entre 0 e 100.")

    if not (0 <= Threshold <= 1):
        raise ValueError("O Threshold deve estar entre 0 e 1.")

    if not (0 <= CR <= 1):
        raise ValueError("O valor de CR deve estar entre 0 e 1.")

    if not classifiers:
        raise ValueError("Selecione pelo menos um classificador.")

    invalid_classifiers = set(classifiers) - SUPPORTED_CLASSIFIERS

    if invalid_classifiers:
        raise ValueError(
            f"Classificadores inválidos: {', '.join(invalid_classifiers)}. "
            f"Classificadores suportados: {', '.join(SUPPORTED_CLASSIFIERS)}."
        )

    return {
        "labeled_percentage": LabeledPercentage,
        "threshold": Threshold,
        "cr": CR,
        "classifiers": list(classifiers),
    }