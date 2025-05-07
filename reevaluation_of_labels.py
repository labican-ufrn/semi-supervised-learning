from random import choice
from numpy import where
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from sklearn.semi_supervised import SelfTrainingClassifier
from selfOld import MySelfOld
from selfNew import MySelfNew
from selfNewEssemble import MySelfNewEssemble


from src.utils import select_labels


# Separa/prepara a base de dados
df = read_csv('datasets/Btsc.csv', header=0)
X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values


# Ao similar a estratégia do boosting. O boosting funciona adicionando
#   mais pesos as instâncias que o classificador 'anterior' errou.
#   Desta forma, é possível utilizar uma estratégia similar para fazer
#   a seleção das instâncias difíceis, tendo como base os outputs
#   parciais das instâncias 'classificadas erradas'.

# Também é possível rodar um algoritmo de agrupamento (e.g. hierárquico)
#   ou outro algoritmo de clusterização, para realizar a escolha das
#   instâncias difíceis.


# Inicializa os classificadores
classifiers = {
    "KNN": KNeighborsClassifier,
    "Naive Bayes": GaussianNB,
    "Decision Tree": DecisionTreeClassifier,
    "Random Forest": RandomForestClassifier,
    "XGBoost": XGBClassifier,
    "Logistic Regression": LogisticRegression,
    "Neural Network": MLPClassifier,
}


# Dividir o dataset (25% rotulados, 75% não rotulados)SelfTrainingClassifier
# Alterei para um split 85-15 para dados de treinamento e teste.
X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
    X,
    y,
    test_size=0.15,
    random_state=42,
)


# Todas as instâncias em X_train estão rotuladas. Isso é injusto para
# nível de comparação dos métodos. Para corrigir isso, eu vou fazer um
# novo split nos dados deixando apenas 25% dos 85% (mesmo que o
# select_labels). A partir deste novo split é que eu faço o treinamento
# e teste do meu algoritmo/comitê.
#
# Nos dados de treinamento eu vou escolher 25% para "ficar com rótulo"
X_train, _, y_train, _ = train_test_split(
    X_train_all,
    y_train_all,
    test_size=0.75,
    random_state=42,
)


print(f"\ntamanho de X_train: {len(X_train)}")
print(f"tamanho de y_train: {len(y_train)}\n")


# Treina os classificadores com o percentual de instâncias inicialmente rotuladas
results = {name: [] for name in classifiers.keys()}

for name, model in classifiers.items():

    if name == "Logistic Regression":
        model = model(max_iter=1000)
    elif name == "Neural Network":
        model = model(max_iter=3000)
    else:
        model = model()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test_all)
    acc = accuracy_score(y_test_all, y_pred)
    results[name].append(acc)
    print(f"{name}: {results[name]}")

# Seleciona o melhor classificador
max_acc = max(results.values())
best_models = [name for name, acc in results.items() if acc == max_acc]
best_model_name = choice(best_models)
best_model = classifiers[best_model_name]


##################################################


# Inicia e treina o especialista com o percentual inicial de instâncias rotuladas
specialist = MySelfNewEssemble(base_estimator=best_model(), threshold=0.95, max_iter=10, silhouette_threshold=-0.2, verbose=True)
y_2 = select_labels(y_train_all, X_train_all, 0.25)

# Contar instâncias não rotuladas antes do treinamento
num_unlabeled_before = sum(where(y_2 == -1, 1, 0))
print(f"\nInstâncias não rotuladas antes do treinamento: {num_unlabeled_before}")

specialist.fit(X_train_all, y_2)

# Contar instâncias rotuladas e não rotuladas após o treinamento
y_pseudo = specialist.transduction_
num_unlabeled_after = sum(where(y_pseudo == -1, 1, 0))
print(f"Instâncias não rotuladas após o treinamento: {num_unlabeled_after}")
print(f"Quantidade de instâncias que foram rotuladas após o treinamento: {num_unlabeled_before - num_unlabeled_after}")

print(
    f'\nTotal de instâncias: {X.shape}'
    f'\nTotal de selecionadas para o experimento: {X_train_all.shape}'
    f'\nTotal de rotuladas que foram para o comitê: {X_train.shape}'
    f'\nTotal de rotuladas que foram para o especialista: {sum(where(y_2 != -1, 1, 0))}'
)

# Armazena as predições do especialista
y_pred = specialist.predict(X_test_all)

# Calcula a acurácia do especialista
accuracy = accuracy_score(y_test_all, y_pred)
print(f"\nAcurácia do Especialista ({best_model_name}): {accuracy:.4f}")

print(f"Critério de parada: {specialist.termination_condition_}")
print(f"Número de iterações realizadas: {specialist.n_iter_}")


##################################################


# # Cria o comitê
# models = []
# for name, model in classifiers.items():

#     if name == "Logistic Regression":
#         model = model(max_iter=1000)
#     elif name == "Neural Network":
#         model = model(max_iter=3000)
#     else:
#         model = model()

#     models.append((name, model))

# ensemble = VotingClassifier(estimators=models, voting='soft', verbose=True)

# # Treina o comitê com o percentual inicial de instâncias rotuladas
# ensemble.fit(X_train, y_train)

# # Armazena as predições do comitê
# y_pred_comite = ensemble.predict(X_test_all)

# # Calcula a acurácia do comitê
# accuracy_comite = accuracy_score(y_test_all, y_pred_comite)
# print(f"\nAcurácia do Comitê ({best_model_name}): {accuracy_comite:.4f}")








# 1. Estamos usando esses classificadores:
#     KNN,
#     Naive Bayes,
#     Decision Tree,
#     Random Forest,
#     XGBoost,
#     Logistic Regression,
#     Neural Network


# 2. Trainamos todos os classificadores com o percentual inicialmente rotulado da base


# 3. Escolhemos aquele com a melhor acurácio para ser o especialista (SelfTraining)


# 4. O especialista usa 0.95 fixo de threshold para incluir as instâncias


# 5. Não estamos obrigando todas as instãncias serem classificadas


# 6. Estamos usando silhouette_threshold=-0.2 (número mágico)


# 7. Ainda não estamos usando comitê nem a sua lógica de votação