import json
from datetime import datetime, timezone

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from ..config import (
    RESULTS_PATH,
    LABELS_PATH,
    BIN_DIR,
    ASSET_MODEL_PATH,
)
from ..models.asset_classificator_model import AssetClassificatorModel

MODEL_METADATA_PATH = BIN_DIR / "assetClassificator_metadata.json"
SPLIT_TEST_SIZE = 0.2
SPLIT_RANDOM_STATE = 42
SPLIT_STRATIFY = True

result_df = pd.read_csv(
    RESULTS_PATH,
    sep="\t",
    header=None,
    names=["title", "label_id"],
    encoding="utf-8"
)

labels_df = pd.read_csv(
    LABELS_PATH,
    sep="\t", 
    encoding="utf-8")

titles = result_df["title"]
y = (result_df["label_id"] != 4).astype(int)

X_train_titles, X_test_titles, y_train, y_test = train_test_split(
    titles,
    y,
    test_size=SPLIT_TEST_SIZE,
    random_state=SPLIT_RANDOM_STATE,
    stratify=y if SPLIT_STRATIFY else None,
)

model = AssetClassificatorModel()
model.fit(X_train_titles, y_train)

tfidvect = model.vectorizer
X_train = tfidvect.transform(X_train_titles)
X_test = tfidvect.transform(X_test_titles)
vocab = tfidvect.vocabulary_                  # dict: token -> columna
print("tam vocab:", len(vocab))
print("primeros 20 tokens:", list(vocab.keys())[:20])
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

BIN_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(model, ASSET_MODEL_PATH)
print(f"Modelo Stage 1 guardado en: {ASSET_MODEL_PATH.resolve()}")

TRAIN_SPLIT_PATH = BIN_DIR / "asset_classifier_train_split.tsv"
VALIDATION_SPLIT_PATH = BIN_DIR / "asset_classifier_validation_split.tsv"

train_split_df = result_df.loc[X_train_titles.index, ["title", "label_id"]].copy()
train_split_df["target"] = y_train
train_split_df.to_csv(TRAIN_SPLIT_PATH, sep="\t", index=False, encoding="utf-8")

validation_split_df = result_df.loc[X_test_titles.index, ["title", "label_id"]].copy()
validation_split_df["target"] = y_test
validation_split_df.to_csv(VALIDATION_SPLIT_PATH, sep="\t", index=False, encoding="utf-8")

y_pred = model.predict(X_test_titles) #clase predita per cada mostra
y_score = model.predict_proba(X_test_titles)[:, 1] #prob de classe 1 per cada mostra

precision = float(precision_score(y_test, y_pred))
recall = float(recall_score(y_test, y_pred))
f1 = float(f1_score(y_test, y_pred))
roc_auc = float(roc_auc_score(y_test, y_score))
pr_auc = float(average_precision_score(y_test, y_score))
class_metrics = classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    output_dict=True,
    zero_division=0,
)

print("\n=== Evaluacion en test ===")
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1:", round(f1, 4))
print("ROC-AUC:", round(roc_auc, 4))
print("PR-AUC:", round(pr_auc, 4))
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
cm_df = pd.DataFrame(
    cm,
    index=["real_non_asset(0)", "real_asset(1)"],
    columns=["pred_non_asset(0)", "pred_asset(1)"],
)
tn, fp, fn, tp = cm.ravel()
print("\nMatriz de confusion (filas=real, columnas=pred):")
print(cm_df)
print(f"\nTN={tn}  FP={fp}  FN={fn}  TP={tp}")
print("\nClassification report:")
print(classification_report(y_test, y_pred, digits=4, zero_division=0))

metadata_payload = {
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "artifacts": {
        "asset_model_path": str(ASSET_MODEL_PATH),
        "asset_embedding_path": str(ASSET_MODEL_PATH),
    },
    "asset_classifier": {
        "data": {
            "source_files": {
                "results_tsv": str(RESULTS_PATH),
                "labels_tsv": str(LABELS_PATH),
            },
            "training_data_tsv": str(TRAIN_SPLIT_PATH),
            "validation_data_tsv": str(VALIDATION_SPLIT_PATH),
            "target_definition": "y = (label_id != 4).astype(int)",
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
    }
}

with MODEL_METADATA_PATH.open("w", encoding="utf-8") as fp:
    json.dump(metadata_payload, fp, ensure_ascii=False, indent=2)
print(f"\nMetadata guardada en: {MODEL_METADATA_PATH.resolve()}")
