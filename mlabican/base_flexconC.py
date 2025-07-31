import numpy as np
from sklearn.base import clone
from sklearn.utils import safe_mask

from base_flexcons import BaseFlexCon


class BaseFlexConC(BaseFlexCon):
    def __init__(self, base_classifier, cr=0.05, threshold=0.95, verbose=False):
        super().__init__(
            base_classifier=base_classifier,
            threshold=threshold,
            verbose=verbose
        )
        self.cr = cr
        self.threshold = threshold
        self.committee_classifiers = []

    def fit(self, X, y):
        # Inicialização do classificador e acurácia inicial
        labeled_indices = np.where(y != -1)[0]
        unlabeled_indices = np.where(y == -1)[0]

        init_acc = self.train_new_classifier(labeled_indices, X, y)

        for self.n_iter_ in range(self.max_iter):

            # Verifica se ainda há instâncias não rotuladas
            if len(unlabeled_indices) == 0:
                break

            # Fazer previsões e selecionar instâncias
            self.pred_x_it = self.storage_predict(
                unlabeled_indices,
                self.classifier_.predict_proba(X[unlabeled_indices]).max(axis=1),
                self.classifier_.predict(X[unlabeled_indices])
            )
            selected_indices, predictions = self.select_instances_by_rules()

            if len(selected_indices) == 0:
                break

            # Atualizar o conjunto de instâncias rotuladas
            self.add_new_labeled(selected_indices, selected_indices, predictions)

            # Ajuste do threshold baseado na acurácia local e mínima aceitável
            local_measure = self.calc_local_measure(X[safe_mask(X, labeled_indices)], y[labeled_indices], self.classifier_)
            if local_measure > (self.init_acc + 0.01) and (self.threshold - self.cr) > 0.0:
                self.threshold -= self.cr
            elif local_measure < (self.init_acc - 0.01) and (self.threshold + self.cr) <= 1:
                self.threshold += self.cr

            # Re-treinar o classificador com o novo conjunto de dados rotulados
            init_acc = self.train_new_classifier(labeled_indices, X, y)

        return self

    def update_committee(self):
        """
        Adiciona o classificador atual ao comitê para FlexCon-C1(s) e FlexCon-C1(v).
        """
        self.committee_classifiers.append(clone(self.classifier_))

    def label_by_committee_sum(self, X):
        """
        Classificação por soma de probabilidades no comitê (FlexCon-C1(s)).
        """
        probabilities = np.mean([clf.predict_proba(X) for clf in self.committee_classifiers], axis=0)
        return probabilities.argmax(axis=1)

    def label_by_committee_vote(self, X):
        """
        Classificação por voto majoritário no comitê (FlexCon-C1(v)).
        """
        votes = np.array([clf.predict(X) for clf in self.committee_classifiers])
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=votes)

    def label_by_previous_iteration(self, X):
        """
        Classificação pela iteração anterior (FlexCon-C2).
        """
        return self.classifier_.predict(X)
