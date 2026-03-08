import json
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from ..models.suspicion_model import SuspicionScorer
from ..config import (
    RESULTS_PATH,
    LABELS_PATH,
    BIN_DIR,
    SUSPICION_MODEL_PATH,
)


# Stage 2 mapping (enunciado):
# 0 -> INFRINGEMENT_DISCARDED (label_id=7)
# 1 -> INFRINGEMENT_VALIDATED, INFRINGEMENT_CONFIRMED,
#      CONFIRMATION_ON_HOLD, INFRINGEMENT_VERIFIED, CONFIRMATION_DISCARDED
NEGATIVE_LABEL = 7
POSITIVE_LABELS = {5, 9, 10, 15, 16}
ALLOWED_STAGE2_LABELS = {NEGATIVE_LABEL, *POSITIVE_LABELS}
MODEL_METADATA_PATH = BIN_DIR / "suspicionScorer_metadata.json"
SPLIT_TEST_SIZE = 0.2
SPLIT_RANDOM_STATE = 42
SPLIT_STRATIFY = True


def main() -> None:
    result_df = pd.read_csv(
        RESULTS_PATH,
        sep="\t",
        header=None,
        names=["title", "label_id"],
        encoding="utf-8",
    )
    labels_df = pd.read_csv(LABELS_PATH, sep="\t", encoding="utf-8")

    stage2_df = result_df[result_df["label_id"].isin(ALLOWED_STAGE2_LABELS)].copy()
    stage2_df["suspicion_target"] = (stage2_df["label_id"] != NEGATIVE_LABEL).astype(int)

    print("=== Datos Stage 2 ===")
    print(f"Total original: {len(result_df)}")
    print(f"Usados en Stage 2: {len(stage2_df)}")
    print(f"Descartados por etiqueta no definida en Stage 2: {len(result_df) - len(stage2_df)}")
    print("Distribucion target Stage 2 (0=not suspicious, 1=suspicious):")
    print(stage2_df["suspicion_target"].value_counts().sort_index())

    X_train_titles, X_test_titles, y_train, y_test = train_test_split(
        stage2_df["title"],
        stage2_df["suspicion_target"],
        test_size=SPLIT_TEST_SIZE,
        random_state=SPLIT_RANDOM_STATE,
        stratify=stage2_df["suspicion_target"] if SPLIT_STRATIFY else None,
    )

    model = SuspicionScorer(similarity_top_k=3)
    model.fit(X_train_titles, y_train)

    BIN_DIR.mkdir(parents=True, exist_ok=True)

    TRAIN_SPLIT_PATH = BIN_DIR / "suspicion_scorer_train_split.tsv"
    VALIDATION_SPLIT_PATH = BIN_DIR / "suspicion_scorer_validation_split.tsv"

    train_split_df = stage2_df.loc[X_train_titles.index, ["title", "label_id"]].copy()
    train_split_df["target"] = y_train
    train_split_df.to_csv(TRAIN_SPLIT_PATH, sep="\t", index=False, encoding="utf-8")

    validation_split_df = stage2_df.loc[X_test_titles.index, ["title", "label_id"]].copy()
    validation_split_df["target"] = y_test
    validation_split_df.to_csv(VALIDATION_SPLIT_PATH, sep="\t", index=False, encoding="utf-8")

    y_pred = model.predict(X_test_titles)
    y_score = model.predict_proba(X_test_titles)[:, 1]

    precision = float(precision_score(y_test, y_pred))
    recall = float(recall_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))
    class_metrics = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        output_dict=True,
        zero_division=0,
    )

    print("\n=== Evaluacion Stage 2 (TF-IDF + similarity features) en test ===")
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1:", round(f1, 4))
    print("ROC-AUC:", round(roc_auc_score(y_test, y_score), 4))
    print("PR-AUC:", round(average_precision_score(y_test, y_score), 4))

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    cm_df = pd.DataFrame(
        cm,
        index=["real_not_suspicious(0)", "real_suspicious(1)"],
        columns=["pred_not_suspicious(0)", "pred_suspicious(1)"],
    )
    tn, fp, fn, tp = cm.ravel()
    print("\nMatriz de confusion (filas=real, columnas=pred):")
    print(cm_df)
    print(f"\nTN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    joblib.dump(model, SUSPICION_MODEL_PATH)
    print(f"\nModelo Stage 2 guardado en: {SUSPICION_MODEL_PATH.resolve()}")

    print("\nMapeo labels Stage 2 usado:")
    print(f"Negativa ({NEGATIVE_LABEL}) -> INFRINGEMENT_DISCARDED")
    print(f"Positivas {sorted(POSITIVE_LABELS)} -> suspicious")
    print("\nCatalogo labels disponible en labels.tsv:")
    print(labels_df[["incidentstatusid", "name"]])

    metadata_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": {
            "suspicion_model_path": str(SUSPICION_MODEL_PATH),
            "suspicion_embedding_path": str(SUSPICION_MODEL_PATH),
        },
        "suspicion_scorer": {
            "data": {
                "source_files": {
                    "results_tsv": str(RESULTS_PATH),
                    "labels_tsv": str(LABELS_PATH),
                },
                "training_data_tsv": str(TRAIN_SPLIT_PATH),
                "validation_data_tsv": str(VALIDATION_SPLIT_PATH),
                "target_definition": (
                    "Use labels {7,5,9,10,15,16}; "
                    "suspicion_target = (label_id != 7).astype(int)"
                ),
                "split_config": {
                    "method": "train_test_split",
                    "test_size": SPLIT_TEST_SIZE,
                    "random_state": SPLIT_RANDOM_STATE,
                    "stratify": SPLIT_STRATIFY,
                },
                "train_rows": int(len(X_train_titles)),
                "validation_rows": int(len(X_test_titles)),
                "train_positive_rows": int(y_train.sum()),
                "train_negative_rows": int(len(y_train) - y_train.sum()),
                "validation_positive_rows": int(y_test.sum()),
                "validation_negative_rows": int(len(y_test) - y_test.sum()),
            },
            "metrics": {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "confusion_matrix": {
                    "labels": [0, 1],
                    "matrix": cm.tolist(),
                    "tn": int(tn),
                    "fp": int(fp),
                    "fn": int(fn),
                    "tp": int(tp),
                },
                "per_class": {
                    "class_0": {
                        "precision": float(class_metrics["0"]["precision"]),
                        "recall": float(class_metrics["0"]["recall"]),
                        "f1": float(class_metrics["0"]["f1-score"]),
                        "support": int(class_metrics["0"]["support"]),
                    },
                    "class_1": {
                        "precision": float(class_metrics["1"]["precision"]),
                        "recall": float(class_metrics["1"]["recall"]),
                        "f1": float(class_metrics["1"]["f1-score"]),
                        "support": int(class_metrics["1"]["support"]),
                    },
                },
            },
        },
    }

    with MODEL_METADATA_PATH.open("w", encoding="utf-8") as fp:
        json.dump(metadata_payload, fp, ensure_ascii=False, indent=2)
    print(f"\nMetadata guardada en: {MODEL_METADATA_PATH.resolve()}")


if __name__ == "__main__":
    main()
