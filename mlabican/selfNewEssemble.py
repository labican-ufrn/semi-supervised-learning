import numpy as np
from sklearn.base import clone
from sklearn.utils import check_array
from sklearn.metrics import silhouette_samples


class SelfWithRevaluationEssemble:
    def __init__(
        self,
        base_estimator,
        committee,
        criterion="threshold",
        threshold=0.75,
        k_best=10,
        max_iter=10,
        silhouette_threshold=-0.2,
        verbose=False
    ):
        self.base_estimator = base_estimator
        self.committee = committee
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

        self.transduction_ = np.copy(y)
        labeled_mask = y != -1

        # Treinamento inicial
        self.classifier_.fit(X[labeled_mask], y[labeled_mask])

        while not np.all(labeled_mask) and (self.max_iter is None or self.n_iter_ < self.max_iter):
            self.n_iter_ += 1

            unlabeled_mask = ~labeled_mask
            X_unlabeled = X[unlabeled_mask]

            if X_unlabeled.shape[0] == 0:
                break

            probs = self.classifier_.predict_proba(X_unlabeled)
            max_probs = probs.max(axis=1)

            if self.criterion == "threshold":
                confident_samples = max_probs >= self.threshold
            elif self.criterion == "k_best":
                n_to_select = min(self.k_best, max_probs.shape[0])
                confident_samples = np.zeros_like(max_probs, dtype=bool)
                if n_to_select > 0:
                    confident_indices = np.argpartition(-max_probs, n_to_select)[:n_to_select]
                    confident_samples[confident_indices] = True
            else:
                raise ValueError("Invalid criterion.")

            used_committee = False
            confident_indices = np.array([], dtype=int)

            if not np.any(confident_samples):
                # Silhouette filtering e votação do comitê
                X_labeled = X[labeled_mask]
                y_labeled = self.transduction_[labeled_mask]
                silhouette_vals = silhouette_samples(X_labeled, y_labeled)
                weak_mask = silhouette_vals < self.silhouette_threshold

                if np.any(weak_mask):
                    indices_labeled = np.where(labeled_mask)[0]
                    indices_weak = indices_labeled[weak_mask]
                    labeled_mask[indices_weak] = False
                    self.transduction_[indices_weak] = -1

                    if self.verbose:
                        print(f"{len(indices_weak)} pseudo-rótulos removidos pelo silhouette. Comitê assume.")

                    # Votacao ponderada: especialista + comitê
                    X_weak = X[indices_weak]
                    prob_specialist = self.classifier_.predict_proba(X_weak)
                    prob_committee = self.committee.predict_proba(X_weak)

                    classes = self.classifier_.classes_
                    final_probs = 0.49 * prob_specialist + 0.51 * prob_committee
                    new_labels = classes[np.argmax(final_probs, axis=1)]

                    self.transduction_[indices_weak] = new_labels
                    labeled_mask[indices_weak] = True
                    used_committee = True

                    if self.verbose:
                        print(f"{len(indices_weak)} instâncias rotuladas novamente via comitê.")
                else:
                    self.termination_condition_ = "no_change"
                    if self.verbose:
                        print("Nenhuma instância confiável ou fraca encontrada. Encerrando.")
                    break

            # Atualiza treinamento com todos os rótulos atuais
            X_labeled = X[labeled_mask]
            y_labeled = self.transduction_[labeled_mask]
            self.classifier_.fit(X_labeled, y_labeled)

            if self.verbose:
                print(f"Iteração {self.n_iter_}: {np.sum(confident_samples)} novas amostras confiantes adicionadas.")
                if used_committee:
                    print(f"Iteração {self.n_iter_}: continuando após reclassificação com comitê.")

            indices_unlabeled = np.where(unlabeled_mask)[0]
            confident_indices = indices_unlabeled[confident_samples]

            if confident_indices.size > 0:
                new_y = self.classifier_.predict(X[confident_indices])
                self.transduction_[confident_indices] = new_y
                labeled_mask[confident_indices] = True

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
