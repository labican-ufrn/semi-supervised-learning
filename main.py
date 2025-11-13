import argparse
import os
import warnings
from random import seed

from pandas import read_csv
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB as Naive

from src.ssl.ensemble import Ensemble
from src.ssl.self_flexcon import SelfFlexCon
from src.utils import (
    calculate_mean_stdev,
    list_knn_full,
    list_knn_Prelax,
    list_knn_seeds,
    list_tree,
    result,
    select_labels,
)

warnings.simplefilter("ignore")

parser = argparse.ArgumentParser(description="Escolha um classificador para criar um cômite")
parser.add_argument(
    'classifier',
    metavar='c',
    type=int,
    help='Escolha um classificador para criar um cômite.'
        'Opções: 1 - Naive Bayes, 2 - Tree Decision, 3 - Knn, 4 - Heterogeneous'
)

args = parser.parse_args()

if args.classifier >= 4 or args.classifier < 0:
    print(
        '\nOpção inválida! Escolha corretamente...\n'
        'Opções: 1 - Naive Bayes, 2 - Tree Decision, 3 - Knn, 4 - Heterogeneous\n'
        'Ex: python main.py 1\n'
    )
    exit()

#### Variable initializations...
#datasets_dir = "./datasets"
#datasets = sorted(os.listdir(datasets_dir))
datasets = ["Madelon.csv"]

crs: list[float] = [0.05]
thresholds: list[float] = [0.95]
init_labelled: list[float] = [0.25]
fold_result_acc_final: list[float] = []
fold_result_f1_score_final: list[float] = []

comite_map = {
    1: "Comite_Naive_",
    2: "Comite_Tree_",
    3: "Comite_KNN_",
}
comite: str = comite_map.get(args.classifier, "Comite_Heterogeneo_")

result_folder: str = 'path_for_results'
os.makedirs(result_folder, exist_ok=True)
acc_result_file: str = f'{comite}.csv'
f1_result_file: str = f'{comite}F.csv'

file_path = os.path.join(result_folder, acc_result_file)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write('"ROUNDS", "DATASET", "LABELLED-LEVEL", "ACC", "F1-SCORE"')

for threshold in thresholds:
    for cr in crs:
        for labelled_level in init_labelled:
            for dataset in datasets:

                comite = Ensemble(SelfFlexCon, cr=cr, threshold=threshold)

                df = read_csv(os.path.join('datasets/', dataset), header=0)
                seed(214)
                kfold = StratifiedKFold(n_splits=10)
                _instances = df.iloc[:,:-1].values
                _target = df.iloc[:,-1].values

                # round counter
                rounds = 0
                fold_result_acc = []
                fold_result_f1_score = []

                for train, test in kfold.split(_instances, _target):
                    X_train = _instances[train]
                    X_test = _instances[test]
                    y_train = _target[train]
                    y_test = _target[test]

                    # TODO: Maybe change for line to the following line.
                    # for round, (train, test) in enumerate(kfold.split(_instances, _target)):
                    rounds += 1

                    y = select_labels(y_train, X_train, labelled_level)

                    if (args.classifier == 1) or (args.classifier == 4):
                        for i in range(9):
                            comite.add_classifier(Naive(var_smoothing=float(f'1e-{i}')))

                    if (args.classifier == 2) or (args.classifier == 4):
                        for i in list_tree:
                            comite.add_classifier(i)

                    if (args.classifier == 3) or (args.classifier == 4):
                        if dataset == 'Seeds.csv':
                            for i in list_knn_seeds:
                                comite.add_classifier(i)
                        elif dataset == 'PlanningRelax.csv':
                            for i in list_knn_Prelax:
                                comite.add_classifier(i)
                        else:
                            for i in list_knn_full:
                                comite.add_classifier(i)

                    comite.fit_ensemble(X_train, y)

                    y_pred = comite.predict(X_test)

                    result_acc = accuracy_score(y_test, y_pred)

                    # Adds new accuracy to fold_result_acc
                    fold_result_acc.append(result_acc)

                    # Adds new accuracy to fold_result_acc_final
                    fold_result_acc_final.append(result_acc)

                    # Save data to .csv
                    result_f1 = result(
                        args.classifier,
                        dataset,
                        y_test,
                        y_pred,
                        result_folder,
                        labelled_level,
                        rounds,
                    )

                    fold_result_f1_score.append(result_f1)

                    fold_result_f1_score_final.append(result_f1)

                calculate_mean_stdev(
                    fold_result_acc,
                    args.classifier,
                    labelled_level,
                    result_folder,
                    dataset,
                    fold_result_f1_score,
                )

calculate_mean_stdev(
    fold_result_acc_final,
    args.classifier,
    labelled_level,
    result_folder,
    'FINAL-RESULTS',
    fold_result_f1_score_final
)
