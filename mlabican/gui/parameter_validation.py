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

    errors = {}

    fields = {
        "labeled_percentage": ("Percentual de dados rotulados", labeled_percentage),
        "threshold": ("Threshold", threshold),
        "cr": ("CR", cr)
    }

    number_pattern = r"[+-]?\d+(\.\d{1,6})?"

    numeric_values = {}

    for key, (field_name, value) in fields.items():
        value = str(value).strip()

        if not value:
            errors[key] = f"{field_name} não pode ser vazio."
            continue

        if re.search(r"\.\d{7,}", value):
            errors[key] = f"{field_name} não pode ultrapassar 6 casas decimais."
            continue

        if not re.fullmatch(number_pattern, value):
            errors[key] = f"Digite apenas números no {field_name}."
            continue

        try:
            numeric_value = float(value)

        except (ValueError, TypeError):
            errors[key] = f"Digite apenas números no {field_name}."
            continue

        if not math.isfinite(numeric_value):
            errors[key] = f"{field_name} deve ser um número finito."
            continue

        numeric_values[key] = numeric_value

    if "labeled_percentage" in numeric_values:
        value = numeric_values["labeled_percentage"]

        if not (0 <= value <= 100):
            errors["labeled_percentage"] = (
                "O valor de rotulagem deve estar entre 0 e 100."
            )

    if "threshold" in numeric_values:
        value = numeric_values["threshold"]

        if not (0 <= value <= 1):
            errors["threshold"] = "O Threshold deve estar entre 0 e 1."

    if "cr" in numeric_values:
        value = numeric_values["cr"]

        if not (0 <= value <= 1):
            errors["cr"] = "O valor de CR deve estar entre 0 e 1."

    if not classifiers:
        errors["classifiers"] = "Selecione pelo menos um classificador."

    invalid_classifiers = set(classifiers) - SUPPORTED_CLASSIFIERS

    if invalid_classifiers:
        errors["classifiers"] = (
            f"Classificadores inválidos: {', '.join(invalid_classifiers)}."
        )

    if errors:
        return {"errors": errors}

    return {
        "parameters": {
        "labeled_percentage": numeric_values["labeled_percentage"],
        "threshold": numeric_values["threshold"],
        "cr": numeric_values["cr"],
        "classifiers": list(classifiers),
    }
    }