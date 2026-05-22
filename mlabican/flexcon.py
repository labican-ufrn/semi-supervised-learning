import numpy as np
import numpy.typing as npt
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.semi_supervised import SelfTrainingClassifier

from mlabican.label.base import LabelingStrategy
from mlabican.label.rules import RuleBasedLabelStrategy
from mlabican.selection.base import SelectionStrategy
from mlabican.selection.rules import Rules
from mlabican.threshold.base import ThresholdStrategy
from mlabican.threshold.classifier import Classifier
from mlabican.utils import get_logger


class FlexCon(SelfTrainingClassifier):
    """
    Implementation of the FlexCon algorithm. Delegates instance
    selection and labeling to injected strategies.

    Classe base do método Flexible Confidence com suporte a estratégias.

    Essa classe age como um orquestrador (Contexto). Ela delega o cálculo
    do limiar, a seleção de instâncias e a rotulação para classes de
    Estratégia injetadas no construtor, respeitando o princípio Open/Closed.

    Parameters
    ----------
        estimator : estimator object
            Classificador que irá ser treinado.
        threshold_strategy : ThresholdStrategy, optional
            Estratégia para atualização do limiar (Default: FlexConCStrategy).
        selection_strategy : SelectionStrategy, optional
            Estratégia para seleção de instâncias (Default: RuleBasedSelectionStrategy).
        labeling_strategy : LabelingStrategy, optional
            Estratégia para definir o rótulo da instância selecionada (Default: RuleBasedLabelingStrategy).
        cr : float, optional
            Taxa de mudança do limiar `threshold`, por default 0.05.
        threshold : float, optional
            Limiar inicial do método, por default 0.95.
        verbose : bool, optional
            Loga alguns dados importantes do método para a tela, por default False.
    """

    def __init__(
        self,
        estimator,
        threshold_strategy: ThresholdStrategy | None = None,
        selection_strategy: SelectionStrategy | None = None,
        labeling_strategy: LabelingStrategy | None = None,
        cr: float = 0.05,
        threshold: float = 0.95,
        max_iter: int = 100,
        verbose: bool = False,
    ):
        super().__init__(
            estimator=estimator,
            threshold=threshold,
            criterion='threshold',
            max_iter=max_iter,
            verbose=verbose,
        )
        # Logger
        self.logger = get_logger(verbose=verbose)

        # State variables
        self.cr: float = cr
        self.old_selected: list = []
        # self.transduction_: list = []
        self.classes_: list = []
        self.termination_condition_: str = ''
        self.pred_1_it: dict = {}
        self.pred_x_it: dict = {}
        self.estimator_ = clone(self.estimator)
        self.n_iter_: int = 0

        # Just type definitions for easy syntax
        self.cl_memory: np.ndarray
        self.labeled_iter_: npt.NDArray[np.int64]
        self.threshold: float

        # Dependency Injection for Strategies
        self.selection_strategy = selection_strategy or Rules()
        self.labeling_strategy = labeling_strategy or RuleBasedLabelStrategy()
        self.threshold_strategy = threshold_strategy or Classifier()

    def __str__(self) -> str:
        return (
            f'Classificador {self.estimator}\n'
            f'Threshold Strategy: {type(self.threshold_strategy).__name__}\n'
            f'Selection Strategy: {type(self.selection_strategy).__name__}\n'
            f'Labeling Strategy: {type(self.labeling_strategy).__name__}\n'
            f'Outros Parâmetro: CR: {self.cr}\t'
            f'Threshold Inicial: {self.threshold}'
        )

    def _log_iteration_stats(
        self, labeled: int, unlabeled: int, msg: str
    ) -> None:
        table = (
            f'\n{"=" * 20}\n'
            f'ITERATION INFO\n'
            f'{"-" * 20}\n'
            f'{self.n_iter_}ª iteration\n'
            f'Labeled instances {labeled:>7}\n'
            f'Remain Unlabeled {unlabeled:>8}\n'
            f'Current thr{self.threshold:>15}\n'
            f'Additional Text:\n{msg}\n'
            f'{"=" * 20}'
        )
        self.logger.info(table)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        The main training loop using the new strategy-based architecture.
        """
        self.logger.info('Start Training!')
        # 1. Initialization
        self.n_iter_ = 0
        has_label = y != -1

        self.transduction_ = np.copy(y)
        self.labeled_iter_ = np.full_like(y, -1)
        self.labeled_iter_[has_label] = 0
        reduce_times = 0

        self.cl_memory = np.zeros((X.shape[0], len(np.unique(y[has_label]))))

        # 2. Main loop
        while (self.n_iter_ < self.max_iter) and not all(has_label):
            self.n_iter_ += 1

            # train
            self.estimator_.fit(X[has_label], self.transduction_[has_label])

            # Predict on unlabeled
            unlabeled = ~has_label
            prob = self.estimator_.predict_proba(X[unlabeled])
            pred = self.estimator_.predict(X[unlabeled])

            # Prepare context for strategies
            # We map local unlabeled indices back to global X indices
            unlabeled_indices = np.where(unlabeled)[0]

            if self.n_iter_ == 1:
                threshold_kwargs = {
                    'init_measure': self.calc_local_measure(
                        X[self.labeled_iter_ == 0], y[self.labeled_iter_ == 0]
                    )
                }
                print(f'init_label: {threshold_kwargs["init_measure"]}')
                self.pred_1_it = self.storage_predict(
                    unlabeled_indices.tolist(),
                    np.max(prob, axis=1).tolist(),
                    pred.tolist(),
                )
                prob_1_it = prob.copy()

            strategy_kwargs = {
                'predictions': pred,
                'prob_1_it': prob_1_it,
                'prob_x_it': prob,
                'pred_1_it': self.pred_1_it,  # Populated in first iteration
                'pred_x_it': self.storage_predict(
                    unlabeled_indices.tolist(),
                    np.max(prob, axis=1).tolist(),
                    pred.tolist(),
                ),
                'cl_memory': self.cl_memory,
            }

            # 3. SELECT (Indices only)
            selected_local = self.selection_strategy.select_instances(
                probabilities=prob, threshold=self.threshold, **strategy_kwargs
            )

            if selected_local.size == 0:
                reduce_times += 1
                self.threshold = float(np.trunc(np.max(prob) * 10**2) / 10**2)
                self.logger.info(
                    f'{self.n_iter_} select 0 instances new thr = {self.threshold} - max prob = {np.max(prob)}'
                )
                # go to init for new iteration.
                continue

            # 3.1 Remap to idx
            selected_global = unlabeled_indices[selected_local].tolist()

            # 4. LABEL (Strategy handles the logic)
            pseudo_labels = self.labeling_strategy.label_instances(
                selected_global, **strategy_kwargs
            )

            # 5. Apply changes
            self.add_new_labeled(selected_global, pseudo_labels)

            # Update Memory & Fit
            self.update_memory(selected_global, pseudo_labels)
            has_label[selected_global] = True

            # 6. Update Threshold (using strategy)
            threshold_kwargs['local_measure'] = self.calc_local_measure(
                X[has_label], y[has_label]
            )
            threshold_kwargs['avg_predict_proba'] = np.average(prob, axis=1)
            threshold_kwargs['coverage'] = len(selected_global) / len(
                unlabeled_indices
            )

            self.threshold = self.threshold_strategy.update_threshold(
                self.threshold, self.cr, **threshold_kwargs
            )
            prob_1_it = np.delete(prob_1_it, selected_local, axis=0)

            if self.verbose:
                self._log_iteration_stats(
                    len(selected_global),
                    len(unlabeled_indices) - len(selected_global),
                    f'Coverage {threshold_kwargs["coverage"]}'
                    f'\nReduce thr {reduce_times} times',
                )
                # print(
                #     f'''ITERATION INFO {self.n_iter_}
                #     Labeled: {}
                #     Unlabeled: {}
                #     THR: {self.threshold}
                #     '''
                # )

        self.classes_ = self.estimator_.classes_
        self.termination_condition_ = (
            'Max iterations'
            if self.n_iter_ == self.max_iter
            else 'Label all instances'
        )

    def calc_local_measure(self, X: np.ndarray, y_true: np.ndarray) -> float:
        """Calcula a eficácia de classificação de um modelo nas instâncias locais."""
        y_pred = self.estimator_.predict(X)
        return accuracy_score(y_true, y_pred)

    # def train_new_classifier(
    #     self, X: np.ndarray, y: np.ndarray, has_label: np.ndarray
    # ) -> float:
    #     """
    #     Treina o modelo inicial apenas com as instâncias rotuladas e retorna
    #     a acurácia inicial.
    #     """
    #     self.transduction_ = np.copy(y)
    #     self.labeled_iter_ = np.full_like(y, -1)
    #     self.labeled_iter_[has_label] = 0
    #     self.init_labeled_ = has_label.copy()

    #     estimator_init = clone(self.estimator)

    #     # L0 - Modelo treinado e classificado com L0
    #     estimator_init.fit(
    #         X[safe_mask(X, has_label)], self.transduction_[has_label]
    #     )

    #     # Acurácia em L0
    #     init_acc = self.calc_local_measure(
    #         X[safe_mask(X, self.init_labeled_)],
    #         y[self.init_labeled_],
    #         estimator_init,
    #     )

    #     return init_acc

    def update_memory(
        self,
        instances: np.ndarray,
        labels: np.ndarray,
        weights: list[float] | None = None,
    ) -> None:
        """
        Atualiza a memória de classificação que rastreia o histórico de
        predições para cada instância.
        """
        if not weights:
            weights = [1 for _ in range(len(instances))]

        for instance, label, weight in zip(instances, labels, weights):
            self.cl_memory[instance][label] += weight

    def storage_predict(
        self, idx: list[int], confidence: list[float], classes: list[int]
    ) -> dict[int, dict[str, float]]:
        """
        Armazena a predição e confiança de cada instância para uso nas regras
        das iterações subsequentes.
        """
        memo = {}
        for i, conf, label in zip(idx, confidence, classes):
            memo[i] = {'confidence': conf, 'classes': label}
        return memo

    def add_new_labeled(
        self, selected_full: np.ndarray, pred: np.ndarray
    ) -> None:
        """
        Atualiza os arrays de transdução e controle com as novas instâncias
        rotuladas na iteração atual.
        """
        self.transduction_[selected_full] = pred
        self.labeled_iter_[selected_full] = self.n_iter_
