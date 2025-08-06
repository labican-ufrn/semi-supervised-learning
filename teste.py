import os
from random import choice

from numpy import mean, std, sum, where
from pandas import DataFrame, read_csv
from selfNew import MySelfNew
from selfNewEssemble import MySelfNewEssemble
from selfNewEssembleCP import MySelfNewEssembleCP
from selfOld import MySelfOld
from sklearn import clone
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.utils import select_labels

# Lista dos classificadores candidatos
classifiers = {
    "KNN": KNeighborsClassifier,
    "Naive Bayes": GaussianNB,
    "Decision Tree": DecisionTreeClassifier,
    "Random Forest": RandomForestClassifier,
    "XGBoost": XGBClassifier,
    "Logistic Regression": LogisticRegression,
    "Neural Network": MLPClassifier,
}

# Configuração das repetições e do percentual inicialmente rotulado
n_repeats = 5
initial_label_percentage = 0.25
csv_output = 'resultados_25.csv'

# Pasta onde estão os datasets
dataset_folder = 'datasets'
results_list = []

# Itera sobre todos os arquivos CSV na pasta
for dataset_file in sorted(os.listdir(dataset_folder)):
    if not dataset_file.endswith('.csv'):
        continue
    dataset_path = os.path.join(dataset_folder, dataset_file)

    # Carrega o dataset
    df = read_csv(dataset_path, header=0)
    # Assume que as features estão em todas as colunas, exceto a última, e que a última é o label
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Listas para armazenar os resultados das repetições
    acc_scores = []
    f1_scores = []
    num_labeled_list = []
    num_unlabeled_list = []

    # Repetições para obter média e desvio padrão das métricas
    for repeat in range(n_repeats):
        # Dividir o dataset em treinamento e teste (85% - 15%)
        X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
            X,
            y,
            test_size=0.15,
            random_state=42 + repeat  # variação da semente para cada repetição
        )

        # Para simular poucos dados rotulados no treinamento, separamos 25% dos dados de treinamento para "rotulados"
        # (os demais serão setados para -1 através do select_labels)
        X_train, _, y_train, _ = train_test_split(
            X_train_all,
            y_train_all,
            test_size=0.75,
            random_state=42 + repeat,
            stratify=y_train_all
        )

        # Avalia cada classificador para selecionar o melhor
        eval_results = {name: [] for name in classifiers.keys()}
        for name, model_class in classifiers.items():
            # Configura os parâmetros para alguns classificadores
            if name == "Logistic Regression":
                model = model_class(max_iter=1000)
            elif name == "Neural Network":
                model = model_class(max_iter=3000)
            else:
                model = model_class()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test_all)
            eval_results[name] = accuracy_score(y_test_all, y_pred)

        max_acc = max(eval_results.values())
        best_models = [name for name, acc in eval_results.items() if acc == max_acc]
        best_model_name = choice(best_models)
        best_model_class = classifiers[best_model_name]

        # Cria o comitê
        models = []
        for name, model in classifiers.items():

            if name == "Logistic Regression":
                model = model(max_iter=1000)
            elif name == "Neural Network":
                model = model(max_iter=3000)
            else:
                model = model()

            models.append((name, model))

        # Treina o comitê com o percentual inicial de instâncias rotuladas
        # committee = VotingClassifier(estimators=models, voting='soft', verbose=False)
        # committee.fit(X_train, y_train)

        # Avalia acurácia de cada modelo no conjunto rotulado
        weights = []
        for name, model in models:
            model_clone = clone(model)
            model_clone.fit(X_train, y_train)
            y_pred = model_clone.predict(X_train)
            acc = accuracy_score(y_train, y_pred)
            weights.append(acc)

        # Normaliza os pesos
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        # Cria comitê ponderado
        committee = VotingClassifier(
            estimators=models,
            voting='soft',
            weights=normalized_weights,
            verbose=False
        )

        committee.fit(X_train, y_train)

        # Instancia o especialista utilizando o melhor classificador encontrado
        specialist = MySelfNewEssembleCP(
            base_estimator=best_model_class(),
            committee=committee,
            threshold=0.95,
            max_iter=100,
            silhouette_threshold=-0.2,
            verbose=False  # Defina True para ver mais detalhes na execução
        )

        # Obtém os pseudo-rótulos com apenas 25% inicialmente rotulados
        y_initial = select_labels(y_train_all, X_train_all, initial_label_percentage)
        # Conta instâncias não rotuladas antes do treinamento (usando -1 como indicador de não rótulo)
        num_unlabeled_before = sum(where(y_initial == -1, 1, 0))

        # Treina o especialista com os dados completos (treinamento semi-supervisionado)
        specialist.fit(X_train_all, y_initial)
        y_pseudo = specialist.transduction_
        # Conta quantas instâncias foram rotuladas (rótulo diferente de -1)
        num_labeled = sum(where(y_pseudo != -1, 1, 0))
        num_unlabeled = sum(where(y_pseudo == -1, 1, 0))

        # Prediz no conjunto de teste utilizando o especialista treinado
        y_pred_specialist = specialist.predict(X_test_all)
        acc_final = accuracy_score(y_test_all, y_pred_specialist)
        f1 = f1_score(y_test_all, y_pred_specialist, average='macro')

        # Armazena os resultados da repetição
        acc_scores.append(acc_final)
        f1_scores.append(f1)
        num_labeled_list.append(num_labeled)
        num_unlabeled_list.append(num_unlabeled)

    # Agrega os resultados (média e desvio padrão) a partir das repetições
    result_row = {
        'classifier': best_model_name,
        'dataset': dataset_file,
        'acc_mean': mean(acc_scores),
        'acc_std': std(acc_scores),
        'f1_mean': mean(f1_scores),
        'f1_std': std(f1_scores),
        'initial_label_percentage': initial_label_percentage * 100,  # em %
        'num_labeled': mean(num_labeled_list),   # média das instâncias rotuladas após treinamento
        'num_unlabeled': mean(num_unlabeled_list)  # média das instâncias não rotuladas após treinamento
    }

    # Salva no CSV incrementalmente
    write_header = not os.path.exists(csv_output)
    DataFrame([result_row]).to_csv(
        csv_output,
        mode='a',
        index=False,
        header=write_header
    )


    print(f"\n\n[✓] Finalizado: {dataset_file} | Especialista: {best_model_name}\n\n")
