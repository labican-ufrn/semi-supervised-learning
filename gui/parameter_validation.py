import math

SUPPORTED_CLASSIFIERS = {
    "KNN",
    "Naive Bayes",
    "Decision Tree",
    "Random Forest",
    "Logistic Regression",
    "Neural Network",
}

def validate_parameters(labeled_percentage, threshold, cr, classifiers):
    if labeled_percentage is None or str(labeled_percentage).strip() == "": raise ValueError ("O Percentual de dados rotulados não pode ser vazio.")
    if threshold is None or str(threshold).strip() == "": raise ValueError ("O Threshold não pode ser vazio.")
    if cr is None or str(cr).strip() == "": raise ValueError ("O CR não pode ser vazio.")


    try:
        LabeledPercentage = float(labeled_percentage)
        Threshold = float(threshold)
        CR = float(cr)
    except (ValueError, TypeError): raise ValueError("Digite apenas números")

    if not all (math.isfinite(value) for value in [LabeledPercentage, Threshold, CR]): raise ValueError("Digite apenas números finitos.") 


    if not (0 <= LabeledPercentage <= 100): raise ValueError ("Ovalor de rotulagem deve estar entre 0 e 100.")
    if not (0 <= Threshold <= 1): raise ValueError ("O Threshold deve estar entre 0 e 1.")
    if cr < 0: raise ValueError ("O CR deve ser maior ou igual a 0.") #sem limite superior
    if not classifiers: raise ValueError ("Selecione pelo menos um classificador.")
    invalid_classifiers = set(classifiers) - SUPPORTED_CLASSIFIERS
    if invalid_classifiers: raise ValueError (f"Classificadores inválidos: {', '.join(invalid_classifiers)}. Classificadores suportados: {', '.join(SUPPORTED_CLASSIFIERS)}.")

    return {
        "labeled_percentage": LabeledPercentage,
        "threshold": Threshold,
        "cr": CR,
        "classifiers": list(classifiers),
    }