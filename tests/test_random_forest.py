import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.models.random_forest import (
    DEFAULT_RANDOM_FOREST_PARAMS,
    create_random_forest_model,
    evaluate_random_forest,
    train_and_evaluate_random_forest,
    train_random_forest,
)


def create_features() -> pd.DataFrame:
    """Cria features simples com classes separáveis."""
    return pd.DataFrame(
        {
            "gg": [
                0.10,
                0.12,
                0.14,
                0.70,
                0.72,
                0.74,
                0.40,
                0.42,
                0.44,
            ],
            "m20": [
                -1.90,
                -1.85,
                -1.80,
                -0.70,
                -0.65,
                -0.60,
                -1.30,
                -1.25,
                -1.20,
            ],
            "cc": [
                4.00,
                4.10,
                4.20,
                1.30,
                1.40,
                1.50,
                2.50,
                2.60,
                2.70,
            ],
        }
    )


def create_target() -> pd.Series:
    """Cria classes morfológicas simples."""
    return pd.Series(
        [1, 1, 1, 2, 2, 2, 3, 3, 3],
        name="type",
    )


def create_split_data() -> dict:
    """Cria splits mínimos para treino, validação e teste."""
    features = create_features()
    target = create_target()

    return {
        "train": (
            features.iloc[[0, 1, 3, 4, 6, 7]].reset_index(drop=True),
            target.iloc[[0, 1, 3, 4, 6, 7]].reset_index(drop=True),
        ),
        "validation": (
            features.iloc[[2, 5, 8]].reset_index(drop=True),
            target.iloc[[2, 5, 8]].reset_index(drop=True),
        ),
        "test": (
            features.iloc[[0, 3, 6]].reset_index(drop=True),
            target.iloc[[0, 3, 6]].reset_index(drop=True),
        ),
    }


def test_create_random_forest_model_uses_default_params() -> None:
    """O modelo deve usar os parâmetros baseados no código da Bea."""
    model = create_random_forest_model(random_state=123)

    assert isinstance(model, RandomForestClassifier)
    assert model.n_estimators == DEFAULT_RANDOM_FOREST_PARAMS["n_estimators"]
    assert model.criterion == DEFAULT_RANDOM_FOREST_PARAMS["criterion"]
    assert model.max_features == DEFAULT_RANDOM_FOREST_PARAMS["max_features"]
    assert model.max_depth == DEFAULT_RANDOM_FOREST_PARAMS["max_depth"]
    assert model.class_weight == DEFAULT_RANDOM_FOREST_PARAMS["class_weight"]
    assert model.random_state == 123


def test_create_random_forest_model_accepts_overrides() -> None:
    """Parâmetros podem ser ajustados quando necessário."""
    model = create_random_forest_model(
        random_state=123,
        n_estimators=10,
        max_depth=5,
    )

    assert model.n_estimators == 10
    assert model.max_depth == 5


def test_create_random_forest_model_rejects_invalid_random_state() -> None:
    """random_state deve ser inteiro."""
    with pytest.raises(TypeError):
        create_random_forest_model(random_state="123")


def test_train_random_forest_returns_trained_model() -> None:
    """O treino deve retornar o próprio modelo treinado."""
    model = create_random_forest_model(random_state=123)

    trained_model = train_random_forest(
        model=model,
        features_train=create_features(),
        target_train=create_target(),
    )

    assert trained_model is model
    assert hasattr(trained_model, "classes_")


def test_train_random_forest_rejects_invalid_model() -> None:
    """O modelo deve ser uma instância de RandomForestClassifier."""
    with pytest.raises(TypeError):
        train_random_forest(
            model=object(),
            features_train=create_features(),
            target_train=create_target(),
        )


def test_train_random_forest_rejects_different_lengths() -> None:
    """features_train e target_train devem ter o mesmo tamanho."""
    model = create_random_forest_model(random_state=123)

    with pytest.raises(ValueError):
        train_random_forest(
            model=model,
            features_train=create_features(),
            target_train=pd.Series([1, 2]),
        )


def test_evaluate_random_forest_returns_metrics() -> None:
    """A avaliação deve retornar métricas principais."""
    model = create_random_forest_model(random_state=123)

    trained_model = train_random_forest(
        model=model,
        features_train=create_features(),
        target_train=create_target(),
    )

    metrics = evaluate_random_forest(
        model=trained_model,
        features=create_features(),
        target=create_target(),
    )

    assert "accuracy" in metrics
    assert "precision_macro" in metrics
    assert "recall_macro" in metrics
    assert "f1_macro" in metrics
    assert "f1_weighted" in metrics
    assert "confusion_matrix" in metrics
    assert "classification_report" in metrics


def test_train_and_evaluate_random_forest_returns_validation_and_test() -> None:
    """O fluxo de alto nível deve avaliar validação e teste."""
    results = train_and_evaluate_random_forest(
        split_data=create_split_data(),
        random_state=123,
        n_estimators=20,
    )

    assert "model_params" in results
    assert "validation" in results
    assert "test" in results
    assert "accuracy" in results["validation"]
    assert "accuracy" in results["test"]


def test_train_and_evaluate_random_forest_rejects_invalid_split_keys() -> None:
    """split_data deve conter train, validation e test."""
    with pytest.raises(ValueError):
        train_and_evaluate_random_forest(
            split_data={
                "train": create_split_data()["train"],
                "test": create_split_data()["test"],
            },
            random_state=123,
        )