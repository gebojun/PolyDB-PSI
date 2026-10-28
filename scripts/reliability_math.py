"""Numerical definitions shared by V11 training and reliability scoring."""

from __future__ import annotations

import numpy as np


def sample_std(values: np.ndarray, axis: int = 1) -> np.ndarray:
    """Sample standard deviation for five repeated predictions (ddof=1)."""
    arr = np.asarray(values, dtype=float)
    valid_n = np.sum(np.isfinite(arr), axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.nanstd(arr, axis=axis, ddof=1)
    return np.where(valid_n >= 2, result, np.nan)


def normalize_by_observed_std(values: np.ndarray, observed_std: float) -> np.ndarray:
    """Normalize without adding a unit-dependent absolute epsilon."""
    arr = np.asarray(values, dtype=float)
    if not np.isfinite(observed_std) or observed_std <= 0:
        return np.full(arr.shape, np.nan, dtype=float)
    return arr / observed_std


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return float("nan")
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    true = np.asarray(y_true[mask], dtype=float)
    pred = np.asarray(y_pred[mask], dtype=float)
    if len(true) < 2:
        return float("nan")
    denominator = float(np.sum((true - np.mean(true)) ** 2))
    if denominator <= 0:
        return float("nan")
    return float(1.0 - np.sum((true - pred) ** 2) / denominator)


def nrmse(y_true: np.ndarray, y_pred: np.ndarray, observed_std: float) -> float:
    if not np.isfinite(observed_std) or observed_std <= 0:
        return float("nan")
    return rmse(y_true, y_pred) / observed_std


def tanimoto_similarity_to_references(
    query_fp: np.ndarray,
    reference_fps: np.ndarray,
) -> np.ndarray:
    """Binary Tanimoto similarity c / (a + b - c)."""
    query = np.asarray(query_fp, dtype=np.uint8)
    references = np.asarray(reference_fps, dtype=np.uint8)
    intersection = np.sum(references & query, axis=1, dtype=np.int32)
    union = (
        np.sum(references, axis=1, dtype=np.int32)
        + int(np.sum(query, dtype=np.int32))
        - intersection
    )
    return np.divide(
        intersection,
        union,
        out=np.zeros(len(references), dtype=float),
        where=union > 0,
    )


def deterministic_top_indices(similarities: np.ndarray, k: int) -> np.ndarray:
    """Sort by similarity descending, then by reference position ascending."""
    values = np.asarray(similarities)
    if k >= len(values):
        return np.lexsort((np.arange(len(values)), -values))
    threshold = np.partition(values, len(values) - k)[len(values) - k]
    above = np.flatnonzero(values > threshold)
    tied = np.flatnonzero(values == threshold)
    selected = np.concatenate([above, tied[: k - len(above)]])
    return selected[np.lexsort((selected, -values[selected]))]


def deterministic_top_indices_at_threshold(
    similarities: np.ndarray,
    k: int,
    threshold: float,
) -> np.ndarray:
    """Deterministic top-k when the kth-value threshold is already known."""
    values = np.asarray(similarities)
    above = np.flatnonzero(values > threshold)
    tied = np.flatnonzero(values == threshold)
    selected = np.concatenate([above, tied[: k - len(above)]])
    return selected[np.lexsort((selected, -values[selected]))]


def local_risk_from_neighbors(
    query_fps: np.ndarray,
    predicted_values: np.ndarray,
    reference_fps: np.ndarray,
    reference_values: np.ndarray,
    k: int,
) -> dict[str, np.ndarray]:
    """Compute SI-defined mean similarity, D, A, G, and local statistics."""
    queries = np.asarray(query_fps, dtype=np.uint8)
    predictions = np.asarray(predicted_values, dtype=float)
    refs = np.asarray(reference_fps, dtype=np.uint8)
    ref_y = np.asarray(reference_values, dtype=float)

    n_query = len(queries)
    outputs = {
        "mean_tanimoto_similarity": np.full(n_query, np.nan, dtype=float),
        "tanimoto_distance": np.full(n_query, np.nan, dtype=float),
        "local_mean": np.full(n_query, np.nan, dtype=float),
        "local_std": np.full(n_query, np.nan, dtype=float),
        "local_z": np.full(n_query, np.nan, dtype=float),
        "local_consistency_risk": np.full(n_query, np.nan, dtype=float),
        "structure_local_risk": np.full(n_query, np.nan, dtype=float),
    }
    if len(refs) == 0:
        return outputs

    k_eff = min(int(k), len(refs))
    for row, (query, prediction) in enumerate(zip(queries, predictions)):
        similarities = tanimoto_similarity_to_references(query, refs)
        neighbor_idx = deterministic_top_indices(similarities, k_eff)

        neighbor_sims = similarities[neighbor_idx]
        neighbor_values = ref_y[neighbor_idx]
        mean_similarity = float(np.mean(neighbor_sims))
        local_mean = float(np.mean(neighbor_values))
        local_std = (
            float(np.std(neighbor_values, ddof=1))
            if len(neighbor_values) >= 2
            else float("nan")
        )

        if np.isfinite(local_std) and local_std > 0:
            local_z = float(abs(prediction - local_mean) / local_std)
        elif np.isfinite(local_std) and local_std == 0:
            local_z = 0.0 if prediction == local_mean else float("inf")
        else:
            local_z = float("nan")

        distance = float(np.sqrt(2.0 * (1.0 - mean_similarity)))
        consistency = float(mean_similarity * max(local_z - 1.0, 0.0))
        combined = float(np.sqrt((distance**2 + consistency**2) / 2.0))

        outputs["mean_tanimoto_similarity"][row] = mean_similarity
        outputs["tanimoto_distance"][row] = distance
        outputs["local_mean"][row] = local_mean
        outputs["local_std"][row] = local_std
        outputs["local_z"][row] = local_z
        outputs["local_consistency_risk"][row] = consistency
        outputs["structure_local_risk"][row] = combined

    return outputs


def aggregate_repeat_risk(
    repeat_components: list[dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Average repeat-specific structure/local evidence component-wise."""
    if not repeat_components:
        return {}
    result: dict[str, np.ndarray] = {}
    for key in repeat_components[0]:
        matrix = np.column_stack([component[key] for component in repeat_components])
        valid_n = np.sum(~np.isnan(matrix), axis=1)
        sums = np.nansum(matrix, axis=1)
        result[key] = np.divide(
            sums,
            valid_n,
            out=np.full(matrix.shape[0], np.nan, dtype=float),
            where=valid_n > 0,
        )
    return result


def pack_binary_fingerprints(fingerprints: np.ndarray) -> np.ndarray:
    """Pack 167 binary MACCS columns into three uint64 words per row."""
    binary = np.asarray(fingerprints, dtype=np.uint8)
    packed_bytes = np.packbits(binary, axis=1, bitorder="little")
    padding = (-packed_bytes.shape[1]) % 8
    if padding:
        packed_bytes = np.pad(packed_bytes, ((0, 0), (0, padding)))
    return np.ascontiguousarray(packed_bytes).view(np.uint64)


def top_tanimoto_candidates(
    query_fps: np.ndarray,
    reference_fps: np.ndarray,
    n_candidates: int,
    query_chunk_size: int = 128,
    progress_callback=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Find exact top candidates using chunked packed-bit Tanimoto similarity."""
    queries = pack_binary_fingerprints(query_fps)
    references = pack_binary_fingerprints(reference_fps)
    if len(references) == 0:
        shape = (len(queries), 0)
        return np.empty(shape, dtype=np.int32), np.empty(shape, dtype=np.float32)
    if not hasattr(np, "bitwise_count"):
        raise RuntimeError("V11 reliability scoring requires NumPy >= 2.0")

    candidate_count = min(int(n_candidates), len(references))
    indices = np.empty((len(queries), candidate_count), dtype=np.int32)
    scores = np.empty((len(queries), candidate_count), dtype=np.float64)
    ref_popcount = np.bitwise_count(references).sum(axis=1, dtype=np.int16)

    for start in range(0, len(queries), query_chunk_size):
        stop = min(start + query_chunk_size, len(queries))
        query_chunk = queries[start:stop]
        intersections = np.bitwise_count(
            np.bitwise_and(query_chunk[:, None, :], references[None, :, :])
        ).sum(axis=2, dtype=np.int16)
        query_popcount = np.bitwise_count(query_chunk).sum(axis=1, dtype=np.int16)
        unions = query_popcount[:, None] + ref_popcount[None, :] - intersections
        similarities = np.divide(
            intersections,
            unions,
            out=np.zeros(intersections.shape, dtype=np.float64),
            where=unions > 0,
        )

        if candidate_count == len(references):
            selected = np.vstack(
                [
                    deterministic_top_indices(row, candidate_count)
                    for row in similarities
                ]
            )
        else:
            threshold_position = len(references) - candidate_count
            thresholds = np.partition(
                similarities, threshold_position, axis=1
            )[:, threshold_position]
            selected = np.vstack(
                [
                    deterministic_top_indices_at_threshold(
                        row, candidate_count, threshold
                    )
                    for row, threshold in zip(similarities, thresholds)
                ]
            )
        indices[start:stop] = selected
        scores[start:stop] = np.take_along_axis(similarities, selected, axis=1)
        if progress_callback is not None:
            progress_callback(stop, len(queries))

    return indices, scores


def local_risk_from_candidates(
    query_fps: np.ndarray,
    predicted_values: np.ndarray,
    reference_fps: np.ndarray,
    reference_values: np.ndarray,
    candidate_indices: np.ndarray,
    candidate_similarities: np.ndarray,
    reference_fold_ids: np.ndarray,
    query_fold_ids: np.ndarray,
    k: int,
) -> dict[str, np.ndarray]:
    """Compute exact fold-restricted local risks from a verified candidate pool."""
    predictions = np.asarray(predicted_values, dtype=float)
    ref_y = np.asarray(reference_values, dtype=float)
    ref_folds = np.asarray(reference_fold_ids, dtype=int)
    query_folds = np.asarray(query_fold_ids, dtype=int)
    candidates = np.asarray(candidate_indices, dtype=np.int32).copy()
    similarities = np.asarray(candidate_similarities, dtype=float).copy()
    n_query = len(predictions)

    allowed = ref_folds[candidates] != query_folds[:, None]
    allowed_rank = np.cumsum(allowed, axis=1)
    selected = allowed & (allowed_rank <= k)
    selected_count = np.sum(selected, axis=1)
    fallback_rows = np.flatnonzero(selected_count < k)

    if len(fallback_rows):
        # This path is exact and normally unused. It protects correctness if a
        # highly unusual fold assignment excludes nearly the entire candidate pool.
        for row in fallback_rows:
            valid_refs = np.flatnonzero(ref_folds != query_folds[row])
            all_sims = tanimoto_similarity_to_references(
                query_fps[row], reference_fps[valid_refs]
            )
            k_eff = min(k, len(valid_refs))
            if k_eff == 0:
                continue
            selected_local = deterministic_top_indices(all_sims, k_eff)
            exact_refs = valid_refs[selected_local]
            exact_sims = all_sims[selected_local]
            candidates[row, :k_eff] = exact_refs
            similarities[row, :k_eff] = exact_sims
            selected[row, :] = False
            selected[row, :k_eff] = True
            selected_count[row] = k_eff

    safe_count = np.maximum(selected_count, 1)
    neighbor_values = ref_y[candidates]
    mean_similarity = np.sum(
        np.where(selected, similarities, 0.0), axis=1
    ) / safe_count
    local_mean = np.sum(
        np.where(selected, neighbor_values, 0.0), axis=1
    ) / safe_count
    squared_deviation = np.where(
        selected, (neighbor_values - local_mean[:, None]) ** 2, 0.0
    )
    local_variance = np.divide(
        np.sum(squared_deviation, axis=1),
        selected_count - 1,
        out=np.full(n_query, np.nan, dtype=float),
        where=selected_count >= 2,
    )
    local_std = np.sqrt(local_variance)

    local_z = np.full(n_query, np.nan, dtype=float)
    positive_std = np.isfinite(local_std) & (local_std > 0)
    local_z[positive_std] = (
        np.abs(predictions[positive_std] - local_mean[positive_std])
        / local_std[positive_std]
    )
    zero_std = np.isfinite(local_std) & (local_std == 0)
    local_z[zero_std] = np.where(
        predictions[zero_std] == local_mean[zero_std], 0.0, np.inf
    )

    distance = np.sqrt(np.maximum(0.0, 2.0 * (1.0 - mean_similarity)))
    excess = np.maximum(local_z - 1.0, 0.0)
    consistency = np.zeros(n_query, dtype=float)
    np.multiply(
        mean_similarity,
        excess,
        out=consistency,
        where=mean_similarity != 0.0,
    )
    combined = np.sqrt((distance**2 + consistency**2) / 2.0)

    return {
        "mean_tanimoto_similarity": mean_similarity,
        "tanimoto_distance": distance,
        "local_mean": local_mean,
        "local_std": local_std,
        "local_z": local_z,
        "local_consistency_risk": consistency,
        "structure_local_risk": combined,
        "n_neighbors": selected_count.astype(np.int16),
        "n_candidate_fallbacks": np.full(n_query, len(fallback_rows), dtype=np.int32),
    }


def reliability_score(
    property_nrmse: float,
    ensemble_std_normalized: np.ndarray,
    tanimoto_distance: np.ndarray,
) -> np.ndarray:
    """Canonical V11 score:

    R_i = 1 - [f(NRMSE_j) + f(sigma_i) + f(D_i * sigma_i)] / 3,
    where f(x) = x / (1 + x).
    """
    sigma = np.asarray(ensemble_std_normalized, dtype=float)
    distance = np.asarray(tanimoto_distance, dtype=float)
    if not np.isfinite(property_nrmse) or property_nrmse < 0:
        return np.full(np.broadcast(sigma, distance).shape, np.nan, dtype=float)
    if np.any(sigma < 0) or np.any(distance < 0):
        raise ValueError("Reliability risks must be non-negative.")

    model_risk = property_nrmse / (1.0 + property_nrmse)
    ensemble_risk = sigma / (1.0 + sigma)
    distance_sigma = distance * sigma
    interaction_risk = distance_sigma / (1.0 + distance_sigma)
    return 1.0 - (model_risk + ensemble_risk + interaction_risk) / 3.0
