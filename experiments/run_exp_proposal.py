"""
run_exp_proposal.py

Semi-supervised benchmark runner using SelfTraining (self_train_proposal.py)
with Silhouette-based pseudo-label revaluation (DeferredRevaluationStrategy).

Features:
  - Loop order strictly follows experimental design:
      for pct in percentages:
          for seed in seeds:
              for fold in StratifiedKFold:
  - Interactive tqdm progress bar with live metrics, ETA, and speed
  - Full support for both Binary and Multi-class datasets
  - Robust specialist selection: Adaptive KNN (n_neighbors) and cross-validation 
    fallbacks designed for small or class-imbalanced datasets (e.g., PlanningRelax).
  - Independent output files per dataset:
      * results/<dataset>.csv
  - Instant disk flushing after every single fold (zero data loss on interruption)
  - Robust exception handling (folds/datasets continue even if an error occurs)
  - Metrics: Accuracy, F1-Score (macro), ROC-AUC (ovr for multiclass), % labeled,
    iterations, termination condition, and elapsed time.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from random import choice
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

# Library imports
from mlabican.revaluation.metrics import SilhouetteMetric
from mlabican.revaluation.strategies import DeferredRevaluationStrategy
from mlabican.utils import select_labels
from mlabican.self_train_proposal import SelfTraining

# Optional XGBoost support
try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# ==============================================================
# Configuration Defaults
# ==============================================================

DATASETS_DIR = Path("datasets")
OUTPUT_DIR = Path("results")

THRESHOLD = 0.95            # Confidence threshold (fixed)
SILHOUETTE_THRESHOLD = -0.2 # Silhouette cutoff for pruning weak pseudo-labels
MAX_ITER = 100              # Maximum self-training iterations

# Candidate base classifiers for specialist selection
CLASSIFIERS: dict[str, Any] = {
    "KNN": KNeighborsClassifier,
    "Naive Bayes": GaussianNB,
    "Decision Tree": DecisionTreeClassifier,
    "Random Forest": RandomForestClassifier,
    "Logistic Regression": LogisticRegression,
    "Neural Network": MLPClassifier,
}

if HAS_XGBOOST:
    CLASSIFIERS["XGBoost"] = XGBClassifier

def _dist(y):
    vals, counts = np.unique(y, return_counts=True)
    return dict(zip(vals.tolist(), counts.tolist()))

# ==============================================================
# Helper Functions
# ==============================================================

def fit_predict(X_train, y_semi, X_test, best_name, best_cls):
    # A estratégia é injetada: o SelfTraining não conhece a Silhouette,
    # só recebe um objeto com .revaluate(...)
    reval = DeferredRevaluationStrategy(
        metric=SilhouetteMetric(),
        threshold=SILHOUETTE_THRESHOLD,
        protect_initial=True,
    )
    model = SelfTraining(
        estimator=instantiate_classifier(best_name, best_cls),
        revaluation_strategy=reval,
        threshold=THRESHOLD,
        max_iter=MAX_ITER,
        verbose=1,  # 0 = silencioso | 1 = exibe resumo do progresso do Self-Training
    )
    model.fit(X_train, y_semi)  # só X_train e y_semi (com -1)
    return model, model.predict(X_test), model.predict_proba(X_test)

def instantiate_classifier(name: str, cls: Any) -> Any:
    """Instantiate a base classifier with standard hyperparameters."""
    if name == "Logistic Regression":
        return cls(max_iter=1000)
    if name == "Neural Network":
        return cls(max_iter=3000)
    if name == "XGBoost":
        return cls(eval_metric="logloss", verbosity=0)
    return cls()


def select_specialist(
    X_labeled: np.ndarray,
    y_labeled: np.ndarray,
) -> tuple[str, Any]:
    """
    Avalia os classificadores candidatos usando APENAS as instâncias rotuladas
    do treino via validação cruzada interna. Robusto contra baixos volumes de dados.
    """
    scores: dict[str, float] = {}

    n_samples = len(y_labeled)
    n_classes = len(np.unique(y_labeled))

    # Se houver menos de 2 amostras por classe ou amostragem insuficiente de 2 classes, cai para Decision Tree como fallback
    if n_samples < 2 or n_classes < 2:
        return "Decision Tree", DecisionTreeClassifier

    # Ajusta cv_splits dinamicamente com base na menor classe disponível
    class_counts = np.bincount(y_labeled) if np.issubdtype(y_labeled.dtype, np.integer) else pd.Series(y_labeled).value_counts().values
    min_class_samples = class_counts.min() if len(class_counts) > 0 else 1

    cv_splits = max(2, min(3, min_class_samples))

    for name, cls in CLASSIFIERS.items():
        try:
            # Instanciação adaptativa para o KNN não estourar n_neighbors
            if name == "KNN":
                k_adj = max(1, min(5, n_samples - 1))
                model = cls(n_neighbors=k_adj)
            else:
                model = instantiate_classifier(name, cls)
            
            cv_scores = cross_val_score(model, X_labeled, y_labeled, cv=cv_splits, scoring="accuracy")

            # Filtra retornos numéricos inválidos (nan/inf)
            valid_scores = [s for s in cv_scores if not np.isnan(s)]
            if valid_scores:
                scores[name] = float(np.mean(valid_scores))

        except Exception:
            continue

    # Fallback seguro caso a validação cruzada falhe para todos os modelos
    if not scores:
        return "Decision Tree", DecisionTreeClassifier

    max_acc = max(scores.values())
    candidates = [name for name, acc in scores.items() if acc == max_acc]

    best_name = choice(candidates)
    return best_name, CLASSIFIERS[best_name]


def normalize_labels(y: np.ndarray) -> np.ndarray:
    """Encode arbitrary labels into contiguous integers (0, 1, ..., C-1)."""
    classes = np.unique(y)
    mapping = {c: i for i, c in enumerate(classes)}
    return np.array([mapping[v] for v in y], dtype=int)


def compute_auc(y_true: np.ndarray, proba: np.ndarray, n_classes: int) -> float | None:
    """Compute ROC-AUC robustly for both binary and multi-class problems."""
    try:
        if n_classes == 2:
            if proba.ndim == 2 and proba.shape[1] >= 2:
                return float(roc_auc_score(y_true, proba[:, 1]))
            return float(roc_auc_score(y_true, proba))
        else:
            return float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro"))
    except Exception:
        # Fallback if a fold lacks instances of a class in y_test
        return None


# ==============================================================
# Dataset Evaluation Pipeline
# ==============================================================

def evaluate_dataset(
    dataset_name: str,
    X: np.ndarray,
    y: np.ndarray,
    percentages: list[float],
    seeds: list[int],
    n_folds: int,
    dataset_idx: int,
    total_datasets: int,
) -> None:
    """
    Run cross-validation for a dataset, following the exact experimental loop:
        1. Fix pct in percentages
        2. Vary seed in seeds
        3. Run folds in StratifiedKFold
    Saves each fold immediately to results/<dataset>.csv.
    """
    n_classes = len(np.unique(y))
    dataset_type = "binary" if n_classes == 2 else f"multiclass ({n_classes} classes)"

    target_csv = OUTPUT_DIR / f"{dataset_name}.csv"
    total_folds_dataset = len(percentages) * len(seeds) * n_folds

    pbar = tqdm(
        total=total_folds_dataset,
        desc=f"[{dataset_idx}/{total_datasets}] {dataset_name} ({dataset_type})",
        unit="fold",
        dynamic_ncols=True,
        leave=True,
        file=sys.stdout,
    )

    for pct in percentages:
        for seed in seeds:
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

            for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
                X_train_fold = X[train_idx]
                y_train_fold = y[train_idx]
                X_test = X[test_idx]
                y_test = y[test_idx]

                n_train = len(y_train_fold)
                n_test = len(y_test)

                # if DEBUG_TEST == 1:
                #     n_total = len(y)
                #     tqdm.write(f"\n[T1] fold {fold}/{n_folds} | dataset={n_total} instâncias")
                #     tqdm.write(f"[T1] treino={n_train} ({n_train / n_total:.1%}) | teste={n_test} ({n_test / n_total:.1%})")
                #     tqdm.write(f"[T1] índices em comum treino/teste: {np.intersect1d(train_idx, test_idx).size} (deve ser 0)")
                #     tqdm.write(f"[T1] y treino: {_dist(y_train_fold)} | y teste: {_dist(y_test)}")

                # Mask training labels: keep pct% labeled, rest become -1
                np.random.seed(seed + fold)
                y_semi = select_labels(y_train_fold.copy(), pct)
                unlabeled_before = int(np.sum(y_semi == -1))

                # if DEBUG_TEST == 2:
                #     n_lab = int((y_semi != -1).sum())
                #     tqdm.write(f"\n[T2] pct={pct:.0%} fold={fold}/{n_folds} | treino={n_train}")
                #     tqdm.write(f"[T2] rotulados={n_lab} ({n_lab / n_train:.1%}) | sem rótulo={unlabeled_before} ({unlabeled_before / n_train:.1%})")
                #     tqdm.write(f"[T2] esperado: {pct:.0%} rotulados / {1 - pct:.0%} sem rótulo")
                #     tqdm.write(f"[T2] rotulados por classe: {_dist(y_semi[y_semi != -1])} | treino todo: {_dist(y_train_fold)}")

                # Seleciona especialista usando APENAS a parcela rotulada do treino (Split 2)
                labeled_indices = np.where(y_semi != -1)[0]
                best_name, best_cls = select_specialist(
                    X_train_fold[labeled_indices],
                    y_semi[labeled_indices],
                )
                # if DEBUG_TEST == 1:
                #     tqdm.write(f"[T1] select_specialist usou {len(labeled_indices)} rotulados do treino '{best_name}'")

                first_write = not target_csv.exists()
                t0 = time.time()

                base_row = {
                    "dataset": dataset_name,
                    "type": "binary" if n_classes == 2 else "multiclass",
                    "n_classes": n_classes,
                    "pct": pct,
                    "seed": seed,
                    "fold": fold,
                    "train_size": n_train,
                    "test_size": n_test,
                    "specialist": best_name,
                    "unlabeled_before": unlabeled_before,
                }

                last_acc = None

                try:
                    model, y_pred, proba = fit_predict(
                        X_train_fold, y_semi.copy(), X_test, best_name, best_cls
                    )

                    # if DEBUG_TEST == 1:
                    #     tqdm.write(f"[T1] modelo viu {len(model.transduction_)} instâncias no fit (esperado = treino = {n_train})")
                    #     tqdm.write(f"[T1] predição em {len(y_pred)} instâncias (esperado = teste = {n_test})")

                    acc = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average="macro")
                    auc = compute_auc(y_test, proba, n_classes)
                    last_acc = acc

                    unlabeled_after = int(np.sum(model.transduction_ == -1))
                    pct_labeled = (len(y_semi) - unlabeled_after) / len(y_semi)

                    row = {
                        **base_row,
                        "unlabeled_after": unlabeled_after,
                        "acc": round(acc, 6),
                        "f1": round(f1, 6),
                        "auc": round(auc, 6) if auc is not None else None,
                        "pct_labeled": round(pct_labeled, 6),
                        "n_iter": model.n_iter_,
                        "termination_condition": model.termination_condition_,
                        "elapsed_s": round(time.time() - t0, 3),
                        "error": None,
                    }
                except Exception as e:
                    tqdm.write(
                        f"[{dataset_name}] pct={pct:.0%} seed={seed} "
                        f"fold={fold}/{n_folds} [ERROR]: {e}",
                        file=sys.stdout,
                    )
                    row = {
                        **base_row,
                        "acc": None,
                        "f1": None,
                        "auc": None,
                        "elapsed_s": round(time.time() - t0, 3),
                        "error": str(e),
                    }

                pd.DataFrame([row]).to_csv(
                    target_csv, mode="a", header=first_write, index=False
                )

                postfix = {
                    "pct": f"{pct:.0%}",
                    "seed": seed,
                    "fold": f"{fold}/{n_folds}",
                }
                if last_acc is not None:
                    postfix["acc"] = f"{last_acc:.4f}"
                pbar.set_postfix(postfix)
                pbar.update(1)

    pbar.close()


# ==============================================================
# CLI and Entrypoint
# ==============================================================

def list_datasets() -> None:
    """Print all available CSV datasets with sample counts and classes."""
    csvs = sorted(DATASETS_DIR.glob("*.csv"))
    if not csvs:
        print(f"No CSV datasets found in '{DATASETS_DIR}/'.")
        return

    print(f"\nDatasets in '{DATASETS_DIR}/' ({len(csvs)} total):\n")
    for p in csvs:
        try:
            df = pd.read_csv(p, header=0)
            y = df.iloc[:, -1].values
            n_classes = len(np.unique(y))
            tag = "BINARY" if n_classes == 2 else f"{n_classes} classes (MULTICLASS)"
            print(f"  {p.name:<35} [{tag:<26}] ({len(df)} samples, {df.shape[1]-1} features)")
        except Exception as e:
            print(f"  {p.name:<35} [ERROR: {e}]")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Self-Training semi-supervised benchmark (Proposta - Classificador Único) com reavaliação por Silhueta."
    )
    parser.add_argument(
        "--datasets", nargs="*", metavar="NAME",
        help="Dataset name(s) (with or without .csv). Omit to run on all datasets."
    )
    parser.add_argument(
        "--pct", nargs="+", type=float, default=[0.05, 0.10, 0.15, 0.20, 0.25],
        metavar="P", help="Initial labeled data percentages (default: 0.05 0.10 0.15 0.20 0.25)."
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[42, 7, 13, 21, 99],
        metavar="S", help="Seeds for reproducibility (default: 42 7 13 21 99)."
    )
    parser.add_argument(
        "--folds", type=int, default=10,
        help="Number of StratifiedKFold splits (default: 10)."
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available datasets and exit."
    )
    args = parser.parse_args()

    if args.list:
        list_datasets()
        return

    # Target datasets
    if args.datasets:
        targets = args.datasets
    else:
        targets = [p.stem for p in sorted(DATASETS_DIR.glob("*.csv"))]

    OUTPUT_DIR.mkdir(exist_ok=True)

    print("\n" + "="*65)
    print(" BENCHMARK: Self-Training (Proposta - Classificador Único) ")
    print("="*65)
    print(f"Loop order        : pct -> seed -> fold (StratifiedKFold)")
    print(f"Folds             : {args.folds} folds")
    print(f"Percentages (pct) : {[f'{p:.0%}' for p in args.pct]}")
    print(f"Seeds             : {args.seeds}")
    print(f"Outputs Directory : {OUTPUT_DIR}/<dataset>.csv")
    print(f"Total Datasets    : {len(targets)}")
    print("="*65 + "\n")

    total_start = time.time()

    for i, ds_name in enumerate(targets, start=1):
        filename_csv = ds_name if ds_name.endswith(".csv") else f"{ds_name}.csv"
        path = DATASETS_DIR / filename_csv
        if not path.exists():
            print(f"[WARNING] Dataset not found: {path.name}. Skipping.\n")
            continue

        df = pd.read_csv(path, header=0)
        X = df.iloc[:, :-1].values
        y = normalize_labels(df.iloc[:, -1].values)

        ds_start = time.time()
        evaluate_dataset(
            dataset_name=path.stem,
            X=X,
            y=y,
            percentages=args.pct,
            seeds=args.seeds,
            n_folds=args.folds,
            dataset_idx=i,
            total_datasets=len(targets),
        )
        tqdm.write(
            f">>> Finished {path.stem} in {time.time() - ds_start:.1f}s\n",
            file=sys.stdout,
        )

    print("="*65)
    print(f"ALL DATASETS COMPLETED IN {time.time() - total_start:.1f}s")
    print(f"All separate CSV files saved to: {OUTPUT_DIR}/")
    print("="*65 + "\n")


if __name__ == "__main__":
    main()