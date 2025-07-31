from base_flexcons import BaseFlexCon
import numpy as np
from sklearn.utils import safe_mask

class FlexConC(BaseFlexCon):
    def __init__(self, base_classifier, cr=0.05, threshold=0.95, min_precision=0.7, margin=0.1, verbose=False):
        super().__init__(base_classifier=base_classifier, threshold=threshold, verbose=verbose)
        self.cr = cr
        self.min_precision = min_precision
        self.margin = margin

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
            self.pred_x_it = self.storage_predict(unlabeled_indices, self.classifier_.predict_proba(X[unlabeled_indices]).max(axis=1), self.classifier_.predict(X[unlabeled_indices]))
            selected_indices, predictions = self.select_instances_by_rules()

            if len(selected_indices) == 0:
                break

            # Atualizar o conjunto de instâncias rotuladas
            self.add_new_labeled(selected_indices, selected_indices, predictions)

            # Ajuste do threshold baseado na acurácia local e mínima aceitável
            local_measure = self.calc_local_measure(X[safe_mask(X, labeled_indices)], y[labeled_indices], self.classifier_)
            if local_measure > (self.min_precision + self.margin) and (self.threshold - self.cr) > 0.0:
                self.threshold -= self.cr
            elif local_measure < (self.min_precision - self.margin) and (self.threshold + self.cr) <= 1:
                self.threshold += self.cr

            # Re-treinar o classificador com o novo conjunto de dados rotulados
            init_acc = self.train_new_classifier(labeled_indices, X, y)
        
        return self
