"""
self_train_committee.py [Votação ponderada do comitê]

Classificador Self-Training com comitê para Votação Ponderada
e estratégia de reavaliação ImmediateRevaluationStrategy.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.semi_supervised import SelfTrainingClassifier


class SelfTraining(SelfTrainingClassifier):
    """
    Parameters
    ----------
    estimator : estimator object
        Classificador base (precisa ter ``predict_proba``, porque usamos a
        probabilidade como medida de confiança).
    committee : estimator object, optional
        Committee ensemble for weighted voting revaluation. Defaults to RandomForestClassifier(n_estimators=10).
    

    revaluation_strategy : RevaluationStrategy, optional
        Objeto com ``revaluate(instances, labels, labeled_mask, **kwargs)``
        que devolve ``(labels, labeled_mask)``. Chamado ao fim de cada
        iteração. Se None, não há reavaliação (self-training puro, que
        serve de baseline para comparar).
    
    threshold : float, optional
        Confiança mínima para aceitar um pseudo-rótulo.
    max_iter : int, optional
        Número máximo de iterações.
    verbose : int, optional
        0 = silencioso (sem logs); 1 = exibição padrão (resumo do progresso).
    """

    def __init__(
        self,
        estimator,
        committee=None, # Adiciona o comitê como argumento
        revaluation_strategy=None,
        threshold: float = 0.95,
        max_iter: int = 100,
        verbose: int = 0,
    ):
        super().__init__(
            estimator=estimator,
            threshold=threshold,
            criterion='threshold',
            max_iter=max_iter,
            verbose=verbose,
        )
        self.committee = committee # Armazena o comitê
        self.revaluation_strategy = revaluation_strategy
        self.revaluate_counter = 0


    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SelfTraining':
        X = np.asarray(X)
        y = np.asarray(y)

        has_label = y != -1

        if self.verbose >= 1:
            print(
                f'[SELF-TRAIN-COMMITTEE] fit recebeu X={X.shape} y={y.shape} | '
                f'rotulados={int(has_label.sum())} '
                f'sem rótulo={int((~has_label).sum())}'
            )

        self.estimator_ = clone(self.estimator)
        self.transduction_ = np.copy(y)

        # INSTANCIA E TREINA O COMITÊ INICIALMENTE
        if self.committee is not None:
            self.committee_ = clone(self.committee)
        else:
            self.committee_ = RandomForestClassifier(n_estimators=10, random_state=42)

        self.committee_.fit(X[has_label], self.transduction_[has_label])

        self.labeled_iter_ = np.full(y.shape, -1, dtype=np.int64)
        self.labeled_iter_[has_label] = 0

        self.n_iter_ = 0
        self.termination_condition_ = 'max_iter'
        while self.n_iter_ < self.max_iter and not has_label.all():
            self.n_iter_ += 1

            # TODO(refactor): Subsubstituir 'estado_antes' por um detector de ciclos global (ex: historico_estados).
            # O objetivo era registrar o estado de (has_label, transduction) para interromper o Self-Training e
            # evitar o "ping-pong" (ciclo de remoção via reavaliação e re-aceitação no ciclo seguinte).
            # Falhou porque comparava apenas a iteração imediatamente anterior; como a remoção e a re-aceitação
            # ocorrem em iterações diferentes, o estado mudava a cada passo e a igualdade simples nunca batia.
            # Uma possível solução seria salvar os hashes dos estados anteriores em um conjunto/lista.
           
            # estado_antes = (has_label.copy(), self.transduction_.copy())

           
            # 1) TREINAR só com o que tem rótulo
            self.estimator_.fit(X[has_label], self.transduction_[has_label])
            self.committee_.fit(X[has_label], self.transduction_[has_label])

            # 2) PREDIZER nas instâncias sem rótulo
            unlabeled_idx = np.where(~has_label)[0]
            prob = self.estimator_.predict_proba(X[unlabeled_idx])
            pred = self.estimator_.classes_[np.argmax(prob, axis=1)]

            # 3) SELECIONAR por confiança
            selected = prob.max(axis=1) > self.threshold

            if not selected.any():
                # print('Select 0 instances, doing revaluation...\n\n\n')
                if self.revaluation_strategy:
                    # print('Revaluation...\n\n\n')
                    revaluate = False
                    n_antes = int(has_label.sum())

                    # Salva cópia temporária dos rótulos para comparar se o comitê alterou algo
                    labels_antes = self.transduction_.copy()

                    self.transduction_, has_label = self.revaluation_strategy.revaluate(
                        X,
                        self.transduction_,
                        has_label,
                        estimator=self.estimator_,
                        committee=self.committee_, # Adiciona o comitê como argumento para a reavaliação
                        labeled_iter=self.labeled_iter_,
                    )
                    
                    if has_label.any():
                        self.revaluate_counter += 1

                        self.labeled_iter_[~has_label] = -1
                        revaluate = True

                        

                    if not revaluate:
                        self.termination_condition_ = 'no_change'
                        # print('Não quis fazer a reavaliação, tô cansado, flw!')


            # 4) ROTULAR
            chosen = unlabeled_idx[selected]
            self.transduction_[chosen] = pred[selected]
            self.labeled_iter_[chosen] = self.n_iter_
            has_label[chosen] = True

            
            # 5) VERIFICAÇÃO DE ESTAGNAÇÃO POR IGLUANDADE DE ESTADO (DESATIVADO)

            # if np.array_equal(has_label, estado_antes[0]) and np.array_equal(
            #     self.transduction_, estado_antes[1]
            # ):
            #     self.termination_condition_ = 'no_change'
            #     break

        if has_label.all():
            self.termination_condition_ = 'all_labeled'


        if self.verbose >= 1:
            print(f'[SELF-TRAIN-COMMITTEE] Fim do treinamento: {self.revaluate_counter} reavaliações executadas.')
        
        # Treino FINAL com o conjunto de rótulos consolidado
        self.estimator_.fit(X[has_label], self.transduction_[has_label])
        self.classes_ = self.estimator_.classes_

        return self