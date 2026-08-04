"""
Modelo Random Forest para classificação morfológica de galáxias.
"""

from typing import Dict, Tuple

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


DEFAULT_RANDOM_FOREST_PARAMS = {
    "n_estimators": 101,
    "criterion": "gini",
    "max_features": "sqrt",
    "max_depth": 25,
    "class_weight": "balanced",
}


def create_random_forest_model(
    random_state: int = 42,
    **overrides,
) -> RandomForestClassifier:
    """
    Cria um classificador Random Forest com parâmetros baseados no código da Bea.

    O random_state aqui controla a aleatoriedade interna do Random Forest.
    A separação treino/validação/teste deve vir dos manifestos do projeto.
    """
    if isinstance(random_state, bool) or not isinstance(random_state, int):
        raise TypeError("random_state deve ser um número inteiro.")

    params = DEFAULT_RANDOM_FOREST_PARAMS.copy()
    params.update(overrides)
    params["random_state"] = random_state

    return RandomForestClassifier(**params)


def train_random_forest(
    model: RandomForestClassifier,
    features_train: pd.DataFrame,
    target_train: pd.Series,
) -> RandomForestClassifier:
    """
    Treina o Random Forest com o conjunto de treino.
    """
    if not isinstance(model, RandomForestClassifier):
        raise TypeError(
            "model deve ser uma instância de RandomForestClassifier."
        )

    if not isinstance(features_train, pd.DataFrame):
        raise TypeError("features_train deve ser um pandas.DataFrame.")

    if not isinstance(target_train, pd.Series):
        raise TypeError("target_train deve ser um pandas.Series.")

    if len(features_train) != len(target_train):
        raise ValueError(
            "features_train e target_train devem possuir o mesmo tamanho."
        )

    model.fit(features_train, target_train)

    return model


def evaluate_random_forest(
    model: RandomForestClassifier,
    features: pd.DataFrame,
    target: pd.Series,
) -> dict:
    """
    Avalia o Random Forest em um conjunto de dados.
    """
    if not isinstance(model, RandomForestClassifier):
        raise TypeError(
            "model deve ser uma instância de RandomForestClassifier."
        )

    if not isinstance(features, pd.DataFrame):
        raise TypeError("features deve ser um pandas.DataFrame.")

    if not isinstance(target, pd.Series):
        raise TypeError("target deve ser um pandas.Series.")

    if len(features) != len(target):
        raise ValueError(
            "features e target devem possuir o mesmo tamanho."
        )

    predictions = model.predict(features)
    labels = sorted(target.unique().tolist())

    return {
        "accuracy": accuracy_score(target, predictions),
        "precision_macro": precision_score(
            target,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "recall_macro": recall_score(
            target,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "f1_macro": f1_score(
            target,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "f1_weighted": f1_score(
            target,
            predictions,
            average="weighted",
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            target,
            predictions,
            labels=labels,
        ).tolist(),
        "classification_report": classification_report(
            target,
            predictions,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
    }


def train_and_evaluate_random_forest(
    split_data: Dict[str, Tuple[pd.DataFrame, pd.Series]],
    random_state: int = 42,
    **model_overrides,
) -> dict:
    """
    Treina no conjunto de treino e avalia em validação e teste.

    Espera receber split_data no formato produzido por apply_split_manifest().
    """
    required_keys = {
        "train",
        "validation",
        "test",
    }

    if not isinstance(split_data, dict):
        raise TypeError("split_data deve ser um dicionário.")

    if set(split_data.keys()) != required_keys:
        raise ValueError(
            "split_data deve conter exatamente as chaves "
            "'train', 'validation' e 'test'."
        )

    features_train, target_train = split_data["train"]
    features_validation, target_validation = split_data["validation"]
    features_test, target_test = split_data["test"]

    model = create_random_forest_model(
        random_state=random_state,
        **model_overrides,
    )

    trained_model = train_random_forest(
        model=model,
        features_train=features_train,
        target_train=target_train,
    )

    return {
        "model_params": trained_model.get_params(),
        "validation": evaluate_random_forest(
            model=trained_model,
            features=features_validation,
            target=target_validation,
        ),
        "test": evaluate_random_forest(
            model=trained_model,
            features=features_test,
            target=target_test,
        ),
    }