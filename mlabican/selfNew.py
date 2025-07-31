import numpy as np
from sklearn.base import clone
from sklearn.utils import check_array
from sklearn.metrics import silhouette_samples


class SelfWithRevaluation:
    def __init__(
        self,
        base_estimator,
        criterion="threshold",
        threshold=0.75,
        k_best=10,
        max_iter=10,
        silhouette_threshold = -0.2,
        verbose=False
    ):
        self.base_estimator = base_estimator
        self.criterion = criterion
        self.threshold = threshold
        self.k_best = k_best
        self.max_iter = max_iter
        self.verbose = verbose
        self.silhouette_threshold = silhouette_threshold
        self.classifier_ = None
        self.transduction_ = None
        self.n_iter_ = 0
        self.termination_condition_ = None

    def fit(self, X, y):
        X = check_array(X)
        self.classifier_ = clone(self.base_estimator)

        # Inicializa a transdução como uma cópia de y
        self.transduction_ = np.copy(y)

        # Inicializa a máscara de rótulos conhecidos
        labeled_mask = np.where(y == -1, False, True)
        X_labeled, y_labeled = X[labeled_mask], y[labeled_mask]

        # Treina inicialmente com os dados rotulados
        self.classifier_.fit(X_labeled, y_labeled)

        # Processo de auto-treinamento
        while not np.all(labeled_mask) and (self.max_iter is None or self.n_iter_ < self.max_iter):
            self.n_iter_ += 1

            # Identifica dados não rotulados
            unlabeled_mask = ~labeled_mask
            X_unlabeled = X[unlabeled_mask]

            if X_unlabeled.shape[0] == 0:
                break  # Todos os dados já estão rotulados

            # Obtém probabilidades preditas
            probs = self.classifier_.predict_proba(X_unlabeled)
            max_probs = probs.max(axis=1)

            # Seleciona instâncias confiáveis
            if self.criterion == "threshold":
                confident_samples = max_probs >= self.threshold
            elif self.criterion == "k_best":
                n_to_select = min(self.k_best, max_probs.shape[0])
                if n_to_select > 0:
                    confident_samples_indices = np.argpartition(-max_probs, n_to_select)[:n_to_select]
                    confident_samples = np.zeros_like(max_probs, dtype=bool)
                    confident_samples[confident_samples_indices] = True
                else:
                    break
            else:
                raise ValueError("Invalid criterion. Choose either 'threshold' or 'k_best'.")

            # Verifica se houve novas amostras confiantes
            if not np.any(confident_samples):
                self.termination_condition_ = "no_change"
                break  # Nenhuma nova amostra foi adicionada

            # Atualiza os rótulos dos dados não rotulados
            new_y = self.classifier_.predict(X_unlabeled[confident_samples])
            indices_unlabeled = np.where(unlabeled_mask)[0]
            indices_confident = indices_unlabeled[confident_samples]

            # Atualiza `transduction_`
            self.transduction_[indices_confident] = new_y
            labeled_mask[indices_confident] = True  # Agora são amostras rotuladas

            # Exibe informações da iteração se verbose=True
            if self.verbose:
                print(f"Iteração {self.n_iter_}: {len(indices_confident)} amostras adicionadas.")

            # Re-treina o classificador com os novos rótulos
            X_labeled, y_labeled = X[labeled_mask], self.transduction_[labeled_mask]

            silhouette_vals = silhouette_samples(X_labeled, y_labeled)

            # Encontra os índices dos pseudo-rótulos que são "fracos"
            weak_pseudo_labels = silhouette_vals < self.silhouette_threshold
            if np.any(weak_pseudo_labels):
                indices_weak = np.where(labeled_mask)[0][weak_pseudo_labels]
                labeled_mask[indices_weak] = False  # Voltam a ser não rotulados
                self.transduction_[indices_weak] = -1  # Reverte para não rotulado

                if self.verbose:
                    print(f"{len(indices_weak)} pseudo-rótulos removidos.")

            self.classifier_.fit(X_labeled, y_labeled)

        # Define a condição final de parada
        if self.n_iter_ == self.max_iter:
            self.termination_condition_ = "max_iter"
        elif np.all(labeled_mask):
            self.termination_condition_ = "all_labeled"

    def predict(self, X):
        if self.classifier_ is None:
            raise ValueError("O classificador ainda não foi treinado.")
        return self.classifier_.predict(X)

    def predict_proba(self, X):
        if self.classifier_ is None:
            raise ValueError("O classificador ainda não foi treinado.")
        return self.classifier_.predict_proba(X)
