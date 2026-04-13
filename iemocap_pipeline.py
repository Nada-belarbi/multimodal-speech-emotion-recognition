from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import warnings

import kagglehub
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import clone
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    LeaveOneGroupOut,
    RandomizedSearchCV,
    train_test_split,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

EMOTION_ID_TO_NAME = {
    0: "happy",
    1: "sad",
    2: "neutral",
    3: "angry",
    4: "excited",
    5: "frustrated",
}

GENDER_MAP = {"M": "male", "F": "female"}


@dataclass
class IEMOCAPBundle:
    utterance_ids: dict
    utterance_genders: dict
    emotion_ids: dict
    audio_features: dict
    auxiliary_features: dict
    dense_features: dict
    transcripts: dict
    provided_train_dialogues: list
    provided_test_dialogues: list


def resolve_iemocap_pickle_path(download_if_missing: bool = True) -> Path:
    """Locate the KaggleHub IEMOCAP feature bundle without creating a local data/ directory."""
    candidates = [
        Path.cwd() / "IEMOCAP_features.pkl",
        Path.home() / ".cache" / "kagglehub" / "datasets" / "columbine" / "iemocap" / "versions" / "2" / "IEMOCAP_features.pkl",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    if download_if_missing:
        downloaded_root = Path(kagglehub.dataset_download("columbine/iemocap"))
        candidate = downloaded_root / "IEMOCAP_features.pkl"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "IEMOCAP_features.pkl was not found. KaggleHub download may have failed."
    )


def load_iemocap_bundle(download_if_missing: bool = True) -> IEMOCAPBundle:
    """Load the KaggleHub IEMOCAP multimodal bundle."""
    bundle_path = resolve_iemocap_pickle_path(download_if_missing=download_if_missing)
    payload = pd.read_pickle(bundle_path)

    if not isinstance(payload, list) or len(payload) != 9:
        raise ValueError(
            f"Unexpected bundle format in {bundle_path}. Expected a list of length 9, got {type(payload)}."
        )

    return IEMOCAPBundle(*payload)


def _parse_sample_id(sample_id: str) -> dict:
    sample_id = str(sample_id)
    parts = sample_id.split("_")
    conversation_id = "_".join(parts[:-1]) if len(parts) > 1 else sample_id
    utterance_suffix = parts[-1] if parts else sample_id
    session_id = sample_id[:5] if sample_id.startswith("Ses") else "unknown"
    speaker_tag = utterance_suffix[0] if utterance_suffix else "U"

    return {
        "conversation_id": conversation_id,
        "session_id": session_id,
        "speaker_role": speaker_tag,
        "speaker_id": f"{session_id}_{speaker_tag}",
        "speaker_gender": GENDER_MAP.get(speaker_tag, "unknown"),
        "conversation_type": "scripted" if "script" in sample_id.lower() else "improvised",
    }


def build_central_dataframe(download_if_missing: bool = True) -> pd.DataFrame:
    """Flatten the KaggleHub bundle into a single analysis table.

    The public KaggleHub release provides precomputed feature vectors and transcripts,
    so `audio_path` is kept as missing rather than forcing a local `data/` directory.
    """
    bundle = load_iemocap_bundle(download_if_missing=download_if_missing)
    rows = []

    for dialogue_id, sample_ids in bundle.utterance_ids.items():
        genders = bundle.utterance_genders.get(dialogue_id, [])
        emotions = bundle.emotion_ids.get(dialogue_id, [])
        audio_vectors = bundle.audio_features.get(dialogue_id, [])
        aux_vectors = bundle.auxiliary_features.get(dialogue_id, [])
        dense_vectors = bundle.dense_features.get(dialogue_id, [])
        transcripts = bundle.transcripts.get(dialogue_id, [])

        available_lengths = [
            len(sample_ids),
            len(genders),
            len(emotions),
            len(audio_vectors),
            len(aux_vectors),
            len(dense_vectors),
            len(transcripts),
        ]
        n_items = min(available_lengths) if available_lengths else 0

        if len(set(available_lengths)) != 1:
            warnings.warn(
                f"Length mismatch in {dialogue_id}: {available_lengths}. Truncating to {n_items} aligned samples."
            )

        for idx in range(n_items):
            sample_id = str(sample_ids[idx])
            parsed = _parse_sample_id(sample_id)
            emotion_id = int(emotions[idx])
            transcript = str(transcripts[idx]).strip()
            audio_vector = np.asarray(audio_vectors[idx], dtype=np.float32)
            aux_vector = np.asarray(aux_vectors[idx], dtype=np.float32)
            dense_vector = np.asarray(dense_vectors[idx], dtype=np.float32)

            rows.append(
                {
                    "sample_id": sample_id,
                    "conversation_id": parsed["conversation_id"],
                    "dialogue_id": dialogue_id,
                    "session_id": parsed["session_id"],
                    "speaker_id": parsed["speaker_id"],
                    "speaker_role": parsed["speaker_role"],
                    "gender": GENDER_MAP.get(str(genders[idx]), parsed["speaker_gender"]),
                    "emotion_id": emotion_id,
                    "emotion": EMOTION_ID_TO_NAME.get(emotion_id, f"class_{emotion_id}"),
                    "intensity": "unknown",
                    "transcript": transcript,
                    "audio_path": pd.NA,
                    "conversation_type": parsed["conversation_type"],
                    "is_scripted": parsed["conversation_type"] == "scripted",
                    "provided_split": "test" if dialogue_id in bundle.provided_test_dialogues else "train",
                    "audio_features": audio_vector,
                    "auxiliary_features": aux_vector,
                    "dense_features": dense_vector,
                    "audio_feature_dim": int(audio_vector.shape[0]),
                    "aux_feature_dim": int(aux_vector.shape[0]),
                    "dense_feature_dim": int(dense_vector.shape[0]),
                    "transcript_length_chars": len(transcript),
                    "transcript_length_words": len(transcript.split()),
                }
            )

    df = pd.DataFrame(rows).sort_values(["session_id", "dialogue_id", "sample_id"]).reset_index(drop=True)
    return df


def audit_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Audit the flattened table and log issues instead of failing on the first bad sample."""
    issue_rows = []

    for _, row in df.iterrows():
        sample_id = row["sample_id"]

        if not str(row.get("transcript", "")).strip():
            issue_rows.append({"sample_id": sample_id, "reason": "empty_transcript"})

        audio_vector = np.asarray(row.get("audio_features"), dtype=np.float32)
        if audio_vector.size == 0:
            issue_rows.append({"sample_id": sample_id, "reason": "missing_audio_features"})
        elif not np.isfinite(audio_vector).all():
            issue_rows.append({"sample_id": sample_id, "reason": "non_finite_audio_features"})

        if pd.isna(row.get("emotion")):
            issue_rows.append({"sample_id": sample_id, "reason": "missing_emotion_label"})

        if pd.isna(row.get("speaker_id")):
            issue_rows.append({"sample_id": sample_id, "reason": "missing_speaker_id"})

    duplicate_ids = df[df["sample_id"].duplicated()]["sample_id"].tolist()
    for sample_id in duplicate_ids:
        issue_rows.append({"sample_id": sample_id, "reason": "duplicate_sample_id"})

    issues = pd.DataFrame(issue_rows)
    if issues.empty:
        summary = pd.DataFrame(
            [{"reason": "no_issues_detected", "count": 0}],
            columns=["reason", "count"],
        )
        return df.copy(), summary, issues

    issue_summary = issues["reason"].value_counts().rename_axis("reason").reset_index(name="count")
    problematic_ids = set(issues["sample_id"].tolist())
    clean_df = df.loc[~df["sample_id"].isin(problematic_ids)].reset_index(drop=True)
    return clean_df, issue_summary, issues


def dataset_overview_table(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "num_samples": len(df),
                "num_dialogues": df["dialogue_id"].nunique(),
                "num_speakers": df["speaker_id"].nunique(),
                "num_emotions": df["emotion"].nunique(),
                "avg_transcript_length_words": round(df["transcript_length_words"].mean(), 2),
                "median_transcript_length_words": round(df["transcript_length_words"].median(), 2),
            }
        ]
    )


def stack_feature_column(df: pd.DataFrame, column: str) -> np.ndarray:
    return np.vstack(df[column].apply(lambda value: np.asarray(value, dtype=np.float32)).tolist())


def encode_labels(labels: Iterable[str]) -> tuple[np.ndarray, LabelEncoder]:
    encoder = LabelEncoder()
    y = encoder.fit_transform(list(labels))
    return y, encoder


def build_tfidf_features(
    train_texts: Iterable[str],
    test_texts: Iterable[str],
    max_features: int = 2000,
    ngram_range: tuple[int, int] = (1, 2),
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, TfidfVectorizer]:
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, lowercase=True)
    X_train = vectorizer.fit_transform(list(train_texts))
    X_test = vectorizer.transform(list(test_texts))
    return X_train, X_test, vectorizer


def build_sentence_embeddings(texts: Iterable[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(model_name)
    embeddings = encoder.encode(list(texts), show_progress_bar=False)
    return np.asarray(embeddings, dtype=np.float32)


def build_transformer_cls_embeddings(
    texts: Iterable[str],
    model_name: str = "distilbert-base-uncased",
    batch_size: int = 32,
    max_length: int = 128,
) -> np.ndarray:
    """Build CLS embeddings directly from a transformer encoder."""
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except Exception as exc:
        raise ImportError("transformers and torch are required for CLS embeddings.") from exc

    cleaned_texts = [str(text) if text is not None else "" for text in texts]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    batches = []
    with torch.no_grad():
        for start in range(0, len(cleaned_texts), batch_size):
            batch = cleaned_texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = model(**encoded)
            cls_batch = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy().astype(np.float32)
            batches.append(cls_batch)

    return np.vstack(batches)


def scale_feature_blocks(train_block, test_block, with_mean: bool | None = None):
    """Fit scaling on the training block only and apply it to train/test."""
    is_sparse = sparse.issparse(train_block) or sparse.issparse(test_block)
    scaler = StandardScaler(with_mean=False if is_sparse else (True if with_mean is None else with_mean))
    scaled_train = scaler.fit_transform(train_block)
    scaled_test = scaler.transform(test_block)
    return scaled_train, scaled_test, scaler


def _reduce_feature_block(train_block, test_block, n_components: int = 64, random_state: int = 42):
    max_components = min(train_block.shape[1], max(1, train_block.shape[0] - 1))
    effective_components = max(1, min(n_components, max_components))

    if sparse.issparse(train_block):
        reducer = TruncatedSVD(n_components=effective_components, random_state=random_state)
    else:
        reducer = PCA(n_components=effective_components, random_state=random_state)

    reduced_train = reducer.fit_transform(train_block)
    reduced_test = reducer.transform(test_block)
    return reduced_train, reduced_test, reducer


def build_cross_modal_interaction_features(
    train_audio,
    test_audio,
    train_text,
    test_text,
    n_components: int = 64,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a cross-modal block with scaled modality features and interaction terms."""
    audio_train_scaled, audio_test_scaled, _ = scale_feature_blocks(train_audio, test_audio)
    text_train_scaled, text_test_scaled, _ = scale_feature_blocks(train_text, test_text)

    shared_components = min(
        n_components,
        audio_train_scaled.shape[1],
        text_train_scaled.shape[1],
        max(1, audio_train_scaled.shape[0] - 1),
    )

    audio_train_red, audio_test_red, _ = _reduce_feature_block(
        audio_train_scaled,
        audio_test_scaled,
        n_components=shared_components,
        random_state=random_state,
    )
    text_train_red, text_test_red, _ = _reduce_feature_block(
        text_train_scaled,
        text_test_scaled,
        n_components=shared_components,
        random_state=random_state,
    )

    train_block = np.hstack(
        [
            audio_train_red,
            text_train_red,
            audio_train_red * text_train_red,
            np.abs(audio_train_red - text_train_red),
        ]
    ).astype(np.float32)
    test_block = np.hstack(
        [
            audio_test_red,
            text_test_red,
            audio_test_red * text_test_red,
            np.abs(audio_test_red - text_test_red),
        ]
    ).astype(np.float32)
    return train_block, test_block


def fuse_feature_blocks(*blocks):
    cleaned_blocks = [block for block in blocks if block is not None]
    if not cleaned_blocks:
        raise ValueError("At least one feature block is required for fusion.")

    if any(sparse.issparse(block) for block in cleaned_blocks):
        sparse_blocks = [block if sparse.issparse(block) else sparse.csr_matrix(block) for block in cleaned_blocks]
        return sparse.hstack(sparse_blocks).tocsr()

    return np.hstack(cleaned_blocks)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }


def get_models(random_state: int = 42) -> dict[str, object]:
    rf = RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )

    mlp = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=False)),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(256, 128),
                    activation="relu",
                    alpha=1e-4,
                    batch_size=64,
                    learning_rate_init=1e-3,
                    early_stopping=True,
                    validation_fraction=0.15,
                    max_iter=300,
                    random_state=random_state,
                ),
            ),
        ]
    )

    return {"RandomForest": rf, "MLP": mlp}


def compare_standard_vs_group_split(
    X,
    y: np.ndarray,
    groups: Iterable[str],
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Compare an optimistic standard split against a speaker-independent grouped split."""
    groups = np.asarray(list(groups))
    models = get_models(random_state=random_state)
    results = []
    details = {}

    idx = np.arange(len(y))
    idx_train_std, idx_test_std = train_test_split(
        idx,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    idx_train_grp, idx_test_grp = next(gss.split(idx, y, groups=groups))

    split_map = {
        "Approach 1 — Standard stratified split": (idx_train_std, idx_test_std),
        "Approach 2 — Speaker-independent group split": (idx_train_grp, idx_test_grp),
    }

    for split_name, (train_idx, test_idx) in split_map.items():
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        for model_name, model in models.items():
            fitted = clone(model)
            fitted.fit(X_train, y_train)
            y_pred_train = fitted.predict(X_train)
            y_pred_test = fitted.predict(X_test)

            row = {
                "split": split_name,
                "model": model_name,
                "train_accuracy": accuracy_score(y_train, y_pred_train),
                **compute_metrics(y_test, y_pred_test),
            }
            results.append(row)
            details[(split_name, model_name)] = {
                "model": fitted,
                "y_true": y_test,
                "y_pred": y_pred_test,
                "train_indices": train_idx,
                "test_indices": test_idx,
            }

    results_df = pd.DataFrame(results).sort_values(["split", "macro_f1"], ascending=[True, False])
    return results_df, details


def evaluate_models_with_groupkfold(
    X,
    y: np.ndarray,
    groups: Iterable[str],
    modality_name: str,
    random_state: int = 42,
    n_splits: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run GroupKFold evaluation and return fold-level metrics and predictions."""
    groups = np.asarray(list(groups))
    unique_groups = np.unique(groups)
    effective_splits = min(n_splits, len(unique_groups))
    if effective_splits < 2:
        raise ValueError("At least two distinct speaker groups are required for GroupKFold.")

    splitter = GroupKFold(n_splits=effective_splits)
    models = get_models(random_state=random_state)

    metric_rows = []
    prediction_rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(np.arange(len(y)), y, groups=groups), start=1):
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        for model_name, model in models.items():
            fitted = clone(model)
            fitted.fit(X_train, y_train)
            y_pred_train = fitted.predict(X_train)
            y_pred_test = fitted.predict(X_test)

            metric_rows.append(
                {
                    "modality": modality_name,
                    "model": model_name,
                    "fold": fold_idx,
                    "train_accuracy": accuracy_score(y_train, y_pred_train),
                    **compute_metrics(y_test, y_pred_test),
                }
            )

            for sample_position, true_value, pred_value in zip(test_idx, y_test, y_pred_test):
                prediction_rows.append(
                    {
                        "sample_index": int(sample_position),
                        "modality": modality_name,
                        "model": model_name,
                        "fold": fold_idx,
                        "y_true": int(true_value),
                        "y_pred": int(pred_value),
                    }
                )

    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def evaluate_tfidf_modalities_with_groupkfold(
    texts: Iterable[str],
    y: np.ndarray,
    groups: Iterable[str],
    modality_name: str,
    audio_block=None,
    random_state: int = 42,
    n_splits: int = 5,
    max_features: int = 2000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate TF-IDF text-only or audio+TF-IDF fusion inside each GroupKFold split."""
    texts = np.asarray(list(texts), dtype=object)
    groups = np.asarray(list(groups))
    unique_groups = np.unique(groups)
    effective_splits = min(n_splits, len(unique_groups))
    if effective_splits < 2:
        raise ValueError("At least two distinct speaker groups are required for GroupKFold.")

    splitter = GroupKFold(n_splits=effective_splits)
    models = get_models(random_state=random_state)

    metric_rows = []
    prediction_rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(texts, y, groups=groups), start=1):
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            lowercase=True,
        )
        X_train_text = vectorizer.fit_transform(texts[train_idx])
        X_test_text = vectorizer.transform(texts[test_idx])

        if audio_block is not None:
            X_train = fuse_feature_blocks(audio_block[train_idx], X_train_text)
            X_test = fuse_feature_blocks(audio_block[test_idx], X_test_text)
        else:
            X_train = X_train_text
            X_test = X_test_text

        y_train = y[train_idx]
        y_test = y[test_idx]

        for model_name, model in models.items():
            fitted = clone(model)
            fitted.fit(X_train, y_train)
            y_pred_train = fitted.predict(X_train)
            y_pred_test = fitted.predict(X_test)

            metric_rows.append(
                {
                    "modality": modality_name,
                    "model": model_name,
                    "fold": fold_idx,
                    "train_accuracy": accuracy_score(y_train, y_pred_train),
                    **compute_metrics(y_test, y_pred_test),
                }
            )

            for sample_position, true_value, pred_value in zip(test_idx, y_test, y_pred_test):
                prediction_rows.append(
                    {
                        "sample_index": int(sample_position),
                        "modality": modality_name,
                        "model": model_name,
                        "fold": fold_idx,
                        "y_true": int(true_value),
                        "y_pred": int(pred_value),
                    }
                )

    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def evaluate_models_with_loso(
    X,
    y: np.ndarray,
    groups: Iterable[str],
    modality_name: str,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate the standard models with Leave-One-Speaker-Out splits."""
    groups = np.asarray(list(groups))
    if len(np.unique(groups)) < 2:
        raise ValueError("At least two distinct speaker groups are required for LOSO.")

    splitter = LeaveOneGroupOut()
    models = get_models(random_state=random_state)
    metric_rows = []
    prediction_rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(np.arange(len(y)), y, groups=groups), start=1):
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        held_out_group = str(np.unique(groups[test_idx])[0])

        for model_name, model in models.items():
            fitted = clone(model)
            fitted.fit(X_train, y_train)
            y_pred_train = fitted.predict(X_train)
            y_pred_test = fitted.predict(X_test)

            metric_rows.append(
                {
                    "modality": modality_name,
                    "model": model_name,
                    "fold": fold_idx,
                    "held_out_group": held_out_group,
                    "train_accuracy": accuracy_score(y_train, y_pred_train),
                    **compute_metrics(y_test, y_pred_test),
                }
            )

            for sample_position, true_value, pred_value in zip(test_idx, y_test, y_pred_test):
                prediction_rows.append(
                    {
                        "sample_index": int(sample_position),
                        "modality": modality_name,
                        "model": model_name,
                        "fold": fold_idx,
                        "held_out_group": held_out_group,
                        "y_true": int(true_value),
                        "y_pred": int(pred_value),
                    }
                )

    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def evaluate_tfidf_modalities_with_loso(
    texts: Iterable[str],
    y: np.ndarray,
    groups: Iterable[str],
    modality_name: str,
    audio_block=None,
    random_state: int = 42,
    max_features: int = 2000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate TF-IDF text-only or audio+TF-IDF fusion under LOSO."""
    texts = np.asarray(list(texts), dtype=object)
    groups = np.asarray(list(groups))
    if len(np.unique(groups)) < 2:
        raise ValueError("At least two distinct speaker groups are required for LOSO.")

    splitter = LeaveOneGroupOut()
    models = get_models(random_state=random_state)
    metric_rows = []
    prediction_rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(texts, y, groups=groups), start=1):
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            lowercase=True,
        )
        X_train_text = vectorizer.fit_transform(texts[train_idx])
        X_test_text = vectorizer.transform(texts[test_idx])

        if audio_block is not None:
            X_train = fuse_feature_blocks(audio_block[train_idx], X_train_text)
            X_test = fuse_feature_blocks(audio_block[test_idx], X_test_text)
        else:
            X_train = X_train_text
            X_test = X_test_text

        y_train = y[train_idx]
        y_test = y[test_idx]
        held_out_group = str(np.unique(groups[test_idx])[0])

        for model_name, model in models.items():
            fitted = clone(model)
            fitted.fit(X_train, y_train)
            y_pred_train = fitted.predict(X_train)
            y_pred_test = fitted.predict(X_test)

            metric_rows.append(
                {
                    "modality": modality_name,
                    "model": model_name,
                    "fold": fold_idx,
                    "held_out_group": held_out_group,
                    "train_accuracy": accuracy_score(y_train, y_pred_train),
                    **compute_metrics(y_test, y_pred_test),
                }
            )

            for sample_position, true_value, pred_value in zip(test_idx, y_test, y_pred_test):
                prediction_rows.append(
                    {
                        "sample_index": int(sample_position),
                        "modality": modality_name,
                        "model": model_name,
                        "fold": fold_idx,
                        "held_out_group": held_out_group,
                        "y_true": int(true_value),
                        "y_pred": int(pred_value),
                    }
                )

    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def _evaluate_adaptive_weighted_fusion(
    audio_block,
    text_block,
    y: np.ndarray,
    groups: Iterable[str],
    splitter,
    modality_name: str,
    random_state: int = 42,
    weight_grid: Iterable[float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    groups = np.asarray(list(groups))
    weight_candidates = np.asarray(list(weight_grid) if weight_grid is not None else np.linspace(0.1, 0.9, 9))
    models = get_models(random_state=random_state)

    metric_rows = []
    prediction_rows = []
    weight_rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(np.arange(len(y)), y, groups=groups), start=1):
        X_audio_train = audio_block[train_idx]
        X_audio_test = audio_block[test_idx]
        X_text_train = text_block[train_idx]
        X_text_test = text_block[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        groups_train = groups[train_idx]

        if len(np.unique(groups_train)) >= 2:
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
            sub_train_idx, val_idx = next(gss.split(np.arange(len(y_train)), y_train, groups=groups_train))
        else:
            stratify = y_train if len(np.unique(y_train)) > 1 else None
            sub_train_idx, val_idx = train_test_split(
                np.arange(len(y_train)),
                test_size=0.2,
                random_state=random_state,
                stratify=stratify,
            )

        for model_name, model in models.items():
            audio_model_val = clone(model).fit(X_audio_train[sub_train_idx], y_train[sub_train_idx])
            text_model_val = clone(model).fit(X_text_train[sub_train_idx], y_train[sub_train_idx])

            audio_proba_val = audio_model_val.predict_proba(X_audio_train[val_idx])
            text_proba_val = text_model_val.predict_proba(X_text_train[val_idx])

            best_alpha = 0.5
            best_val_f1 = -np.inf
            for alpha in weight_candidates:
                fused_val_pred = np.argmax(alpha * audio_proba_val + (1.0 - alpha) * text_proba_val, axis=1)
                current_f1 = f1_score(y_train[val_idx], fused_val_pred, average="macro")
                if current_f1 > best_val_f1:
                    best_val_f1 = current_f1
                    best_alpha = float(alpha)

            final_audio_model = clone(model).fit(X_audio_train, y_train)
            final_text_model = clone(model).fit(X_text_train, y_train)

            fused_train_proba = (
                best_alpha * final_audio_model.predict_proba(X_audio_train)
                + (1.0 - best_alpha) * final_text_model.predict_proba(X_text_train)
            )
            fused_test_proba = (
                best_alpha * final_audio_model.predict_proba(X_audio_test)
                + (1.0 - best_alpha) * final_text_model.predict_proba(X_text_test)
            )

            y_pred_train = np.argmax(fused_train_proba, axis=1)
            y_pred_test = np.argmax(fused_test_proba, axis=1)

            metric_rows.append(
                {
                    "modality": modality_name,
                    "model": model_name,
                    "fold": fold_idx,
                    "train_accuracy": accuracy_score(y_train, y_pred_train),
                    "selected_audio_weight": best_alpha,
                    "selected_text_weight": 1.0 - best_alpha,
                    **compute_metrics(y_test, y_pred_test),
                }
            )
            weight_rows.append(
                {
                    "modality": modality_name,
                    "model": model_name,
                    "fold": fold_idx,
                    "selected_audio_weight": best_alpha,
                    "selected_text_weight": 1.0 - best_alpha,
                    "validation_macro_f1": best_val_f1,
                }
            )

            for sample_position, true_value, pred_value in zip(test_idx, y_test, y_pred_test):
                prediction_rows.append(
                    {
                        "sample_index": int(sample_position),
                        "modality": modality_name,
                        "model": model_name,
                        "fold": fold_idx,
                        "y_true": int(true_value),
                        "y_pred": int(pred_value),
                    }
                )

    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows), pd.DataFrame(weight_rows)


def evaluate_adaptive_weighted_fusion_with_groupkfold(
    audio_block,
    text_block,
    y: np.ndarray,
    groups: Iterable[str],
    modality_name: str,
    random_state: int = 42,
    n_splits: int = 5,
    weight_grid: Iterable[float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    groups = np.asarray(list(groups))
    unique_groups = np.unique(groups)
    effective_splits = min(n_splits, len(unique_groups))
    if effective_splits < 2:
        raise ValueError("At least two distinct speaker groups are required for GroupKFold.")

    splitter = GroupKFold(n_splits=effective_splits)
    return _evaluate_adaptive_weighted_fusion(
        audio_block,
        text_block,
        y,
        groups,
        splitter,
        modality_name=modality_name,
        random_state=random_state,
        weight_grid=weight_grid,
    )


def evaluate_adaptive_weighted_fusion_with_loso(
    audio_block,
    text_block,
    y: np.ndarray,
    groups: Iterable[str],
    modality_name: str,
    random_state: int = 42,
    weight_grid: Iterable[float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    groups = np.asarray(list(groups))
    if len(np.unique(groups)) < 2:
        raise ValueError("At least two distinct speaker groups are required for LOSO.")

    splitter = LeaveOneGroupOut()
    return _evaluate_adaptive_weighted_fusion(
        audio_block,
        text_block,
        y,
        groups,
        splitter,
        modality_name=modality_name,
        random_state=random_state,
        weight_grid=weight_grid,
    )


def _evaluate_cross_modal_fusion(
    audio_block,
    text_block,
    y: np.ndarray,
    groups: Iterable[str],
    splitter,
    modality_name: str,
    random_state: int = 42,
    n_components: int = 64,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = np.asarray(list(groups))
    models = get_models(random_state=random_state)
    metric_rows = []
    prediction_rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(np.arange(len(y)), y, groups=groups), start=1):
        X_train, X_test = build_cross_modal_interaction_features(
            audio_block[train_idx],
            audio_block[test_idx],
            text_block[train_idx],
            text_block[test_idx],
            n_components=n_components,
            random_state=random_state,
        )
        y_train = y[train_idx]
        y_test = y[test_idx]

        for model_name, model in models.items():
            fitted = clone(model)
            fitted.fit(X_train, y_train)
            y_pred_train = fitted.predict(X_train)
            y_pred_test = fitted.predict(X_test)

            metric_rows.append(
                {
                    "modality": modality_name,
                    "model": model_name,
                    "fold": fold_idx,
                    "train_accuracy": accuracy_score(y_train, y_pred_train),
                    **compute_metrics(y_test, y_pred_test),
                }
            )

            for sample_position, true_value, pred_value in zip(test_idx, y_test, y_pred_test):
                prediction_rows.append(
                    {
                        "sample_index": int(sample_position),
                        "modality": modality_name,
                        "model": model_name,
                        "fold": fold_idx,
                        "y_true": int(true_value),
                        "y_pred": int(pred_value),
                    }
                )

    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def evaluate_cross_modal_fusion_with_groupkfold(
    audio_block,
    text_block,
    y: np.ndarray,
    groups: Iterable[str],
    modality_name: str,
    random_state: int = 42,
    n_splits: int = 5,
    n_components: int = 64,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = np.asarray(list(groups))
    unique_groups = np.unique(groups)
    effective_splits = min(n_splits, len(unique_groups))
    if effective_splits < 2:
        raise ValueError("At least two distinct speaker groups are required for GroupKFold.")

    splitter = GroupKFold(n_splits=effective_splits)
    return _evaluate_cross_modal_fusion(
        audio_block,
        text_block,
        y,
        groups,
        splitter,
        modality_name=modality_name,
        random_state=random_state,
        n_components=n_components,
    )


def evaluate_cross_modal_fusion_with_loso(
    audio_block,
    text_block,
    y: np.ndarray,
    groups: Iterable[str],
    modality_name: str,
    random_state: int = 42,
    n_components: int = 64,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = np.asarray(list(groups))
    if len(np.unique(groups)) < 2:
        raise ValueError("At least two distinct speaker groups are required for LOSO.")

    splitter = LeaveOneGroupOut()
    return _evaluate_cross_modal_fusion(
        audio_block,
        text_block,
        y,
        groups,
        splitter,
        modality_name=modality_name,
        random_state=random_state,
        n_components=n_components,
    )


def _evaluate_cnn_on_splits(
    X: np.ndarray,
    y: np.ndarray,
    groups: Iterable[str],
    splitter,
    modality_name: str,
    random_state: int = 42,
    epochs: int = 15,
    batch_size: int = 64,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = np.asarray(list(groups))
    metric_rows = []
    prediction_rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(np.arange(len(y)), y, groups=groups), start=1):
        metrics, y_pred = train_simple_audio_cnn(
            X[train_idx],
            y[train_idx],
            X[test_idx],
            y[test_idx],
            random_state=random_state,
            epochs=epochs,
            batch_size=batch_size,
        )

        metric_rows.append(
            {
                "modality": modality_name,
                "model": "SimpleAudioCNN",
                "fold": fold_idx,
                "train_accuracy": np.nan,
                **metrics,
            }
        )

        for sample_position, true_value, pred_value in zip(test_idx, y[test_idx], y_pred):
            prediction_rows.append(
                {
                    "sample_index": int(sample_position),
                    "modality": modality_name,
                    "model": "SimpleAudioCNN",
                    "fold": fold_idx,
                    "y_true": int(true_value),
                    "y_pred": int(pred_value),
                }
            )

    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def evaluate_cnn_with_groupkfold(
    X: np.ndarray,
    y: np.ndarray,
    groups: Iterable[str],
    modality_name: str = "Audio only (CNN)",
    random_state: int = 42,
    n_splits: int = 5,
    epochs: int = 15,
    batch_size: int = 64,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = np.asarray(list(groups))
    unique_groups = np.unique(groups)
    effective_splits = min(n_splits, len(unique_groups))
    if effective_splits < 2:
        raise ValueError("At least two distinct speaker groups are required for GroupKFold.")

    splitter = GroupKFold(n_splits=effective_splits)
    return _evaluate_cnn_on_splits(
        X,
        y,
        groups,
        splitter,
        modality_name=modality_name,
        random_state=random_state,
        epochs=epochs,
        batch_size=batch_size,
    )


def evaluate_cnn_with_loso(
    X: np.ndarray,
    y: np.ndarray,
    groups: Iterable[str],
    modality_name: str = "Audio only (CNN LOSO)",
    random_state: int = 42,
    epochs: int = 15,
    batch_size: int = 64,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = np.asarray(list(groups))
    if len(np.unique(groups)) < 2:
        raise ValueError("At least two distinct speaker groups are required for LOSO.")

    splitter = LeaveOneGroupOut()
    return _evaluate_cnn_on_splits(
        X,
        y,
        groups,
        splitter,
        modality_name=modality_name,
        random_state=random_state,
        epochs=epochs,
        batch_size=batch_size,
    )


def get_search_spaces(random_state: int = 42) -> dict[str, tuple[object, dict]]:
    rf = RandomForestClassifier(class_weight="balanced", random_state=random_state, n_jobs=-1)
    rf_space = {
        "n_estimators": [200, 300, 500],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
    }

    mlp = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=False)),
            (
                "mlp",
                MLPClassifier(
                    early_stopping=True,
                    validation_fraction=0.15,
                    max_iter=300,
                    random_state=random_state,
                ),
            ),
        ]
    )
    mlp_space = {
        "mlp__hidden_layer_sizes": [(128,), (256,), (256, 128), (128, 64)],
        "mlp__alpha": [1e-5, 1e-4, 1e-3],
        "mlp__batch_size": [32, 64, 128],
        "mlp__learning_rate_init": [1e-4, 5e-4, 1e-3],
        "mlp__activation": ["relu", "tanh"],
    }

    return {"RandomForest": (rf, rf_space), "MLP": (mlp, mlp_space)}


def run_group_random_search(
    model_name: str,
    X,
    y: np.ndarray,
    groups: Iterable[str],
    n_iter: int = 10,
    n_splits: int = 5,
    random_state: int = 42,
):
    estimator, search_space = get_search_spaces(random_state=random_state)[model_name]
    groups = np.asarray(list(groups))
    effective_splits = min(n_splits, len(np.unique(groups)))
    cv = GroupKFold(n_splits=effective_splits)

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=search_space,
        n_iter=n_iter,
        scoring="f1_macro",
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    search.fit(X, y, groups=groups)
    return search


def summarize_cv_results(metrics_df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "train_accuracy",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "precision_macro",
        "recall_macro",
    ]
    summary = metrics_df.groupby(["modality", "model"])[numeric_cols].agg(["mean", "std"])
    return summary.round(4)


def classification_report_dataframe(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> pd.DataFrame:
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    return pd.DataFrame(report).T


def compute_factor_metrics(
    df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    factor_col: str,
) -> pd.DataFrame:
    merged = predictions_df.merge(
        df.reset_index().rename(columns={"index": "sample_index"})[["sample_index", factor_col]],
        on="sample_index",
        how="left",
    )

    rows = []
    for (modality, model, factor_value), group in merged.groupby(["modality", "model", factor_col]):
        rows.append(
            {
                "modality": modality,
                "model": model,
                factor_col: factor_value,
                "num_samples": len(group),
                **compute_metrics(group["y_true"].to_numpy(), group["y_pred"].to_numpy()),
            }
        )

    return pd.DataFrame(rows).sort_values(["modality", "model", factor_col]).reset_index(drop=True)


def find_most_confused_pairs(cm: np.ndarray, class_names: list[str], top_k: int = 8) -> list[dict]:
    confusions = []
    for i, true_name in enumerate(class_names):
        for j, pred_name in enumerate(class_names):
            if i != j and cm[i, j] > 0:
                confusions.append(
                    {
                        "true_label": true_name,
                        "predicted_label": pred_name,
                        "count": int(cm[i, j]),
                    }
                )

    confusions.sort(key=lambda item: item["count"], reverse=True)
    return confusions[:top_k]


def log_results_to_mlflow(
    experiment_name: str,
    run_name: str,
    metrics: dict,
    params: dict | None = None,
    tags: dict | None = None,
) -> bool:
    try:
        import mlflow

        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=run_name):
            for key, value in (params or {}).items():
                mlflow.log_param(key, value)
            for key, value in metrics.items():
                mlflow.log_metric(key, float(value))
            if tags:
                mlflow.set_tags(tags)
        return True
    except Exception as exc:
        warnings.warn(f"MLflow logging skipped: {exc}")
        return False


def train_simple_audio_cnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    random_state: int = 42,
    epochs: int = 15,
    batch_size: int = 64,
) -> tuple[dict, np.ndarray]:
    """Train a lightweight 1D CNN on static audio vectors as a deep-learning baseline."""
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as exc:
        raise ImportError("PyTorch is required for the CNN baseline.") from exc

    torch.manual_seed(random_state)
    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)

    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-6
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    n_classes = int(np.max(y_train)) + 1

    train_ds = TensorDataset(
        torch.tensor(X_train[:, None, :]),
        torch.tensor(y_train, dtype=torch.long),
    )
    test_x = torch.tensor(X_test[:, None, :])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = nn.Sequential(
        nn.Conv1d(1, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool1d(2),
        nn.Conv1d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),
        nn.Dropout(0.3),
        nn.Linear(64, n_classes),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(test_x)
        y_pred = torch.argmax(logits, dim=1).cpu().numpy()

    return compute_metrics(y_test, y_pred), y_pred
