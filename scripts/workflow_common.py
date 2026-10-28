"""
共享工具函数库

提供数据加载、特征构建、模型训练等共享功能。
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from rdkit import Chem, DataStructs, rdBase
    from rdkit.Chem import MACCSkeys

    rdBase.DisableLog("rdApp.*")
    RDKIT_AVAILABLE = True
except Exception:
    Chem = None
    DataStructs = None
    MACCSkeys = None
    RDKIT_AVAILABLE = False

# 添加config到路径
SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_DIR = SCRIPT_DIR.parent / "config"
if str(CONFIG_DIR) not in sys.path:
    sys.path.insert(0, str(CONFIG_DIR))

from workflow_config import (
    EXCLUDED_INPUT_FEATURES,
    FP_COLS,
    FP_N_BITS,
    K_NEAREST_NEIGHBORS,
    SMILES_COL,
)


# ============================================================
# 文件I/O
# ============================================================
def ensure_dir(path: Path) -> Path:
    """确保目录存在"""
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Any, filepath: Path) -> None:
    """保存JSON"""
    ensure_dir(filepath.parent)
    temporary = filepath.with_suffix(filepath.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    temporary.replace(filepath)


def load_json(filepath: Path) -> Any:
    """加载JSON"""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_pickle(obj: Any, filepath: Path) -> None:
    """保存pickle"""
    with open(filepath, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(filepath: Path) -> Any:
    """加载pickle"""
    with open(filepath, "rb") as f:
        return pickle.load(f)


# ============================================================
# SMILES与MACCS指纹
# ============================================================
def mol_from_psmiles(smiles: object):
    """从pSMILES字符串构建RDKit分子对象"""
    if not RDKIT_AVAILABLE or pd.isna(smiles):
        return None
    text = str(smiles).strip()
    if not text:
        return None
    # 替换通配符原子为*
    text = text.replace("[*]", "*")
    try:
        return Chem.MolFromSmiles(text)
    except Exception:
        return None


def maccs_fingerprint(mol) -> np.ndarray | None:
    """计算MACCS指纹（167位）"""
    if not RDKIT_AVAILABLE or mol is None:
        return None
    try:
        fp = MACCSkeys.GenMACCSKeys(mol)
        arr = np.zeros(FP_N_BITS, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except Exception:
        return None


def compute_maccs_fingerprints(df: pd.DataFrame, smiles_col: str = SMILES_COL) -> pd.DataFrame:
    """为DataFrame批量计算MACCS指纹"""
    fps = []
    valid_mask = []

    for smiles in df[smiles_col]:
        mol = mol_from_psmiles(smiles)
        fp = maccs_fingerprint(mol)
        if fp is not None:
            fps.append(fp)
            valid_mask.append(True)
        else:
            fps.append(np.zeros(FP_N_BITS, dtype=np.uint8))
            valid_mask.append(False)

    fp_df = pd.DataFrame(fps, columns=FP_COLS, index=df.index)
    fp_df["maccs_valid"] = valid_mask

    return fp_df


def tanimoto_similarity(fp1: np.ndarray, fp2: np.ndarray) -> float:
    """计算两个MACCS指纹的Tanimoto相似度"""
    if fp1 is None or fp2 is None:
        return 0.0

    intersection = np.sum(fp1 & fp2)
    union = np.sum(fp1 | fp2)

    if union == 0:
        return 0.0

    return float(intersection) / float(union)


def compute_nearest_neighbor_distance(
    query_fps: np.ndarray,
    reference_fps: np.ndarray,
    k: int = K_NEAREST_NEIGHBORS,
) -> np.ndarray:
    """
    计算每个query到reference集合的平均k近邻Tanimoto距离

    Args:
        query_fps: (n_query, 167) 查询指纹
        reference_fps: (n_reference, 167) 参考指纹
        k: 近邻数

    Returns:
        (n_query,) 平均Tanimoto距离
    """
    distances = []

    for qfp in query_fps:
        similarities = []
        for rfp in reference_fps:
            sim = tanimoto_similarity(qfp, rfp)
            similarities.append(sim)

        # 取top-k相似度
        similarities = sorted(similarities, reverse=True)[:k]
        avg_similarity = np.mean(similarities) if similarities else 0.0

        # SI definition: D = sqrt(2 * (1 - mean Tanimoto similarity)).
        distance = np.sqrt(2.0 * (1.0 - avg_similarity))
        distances.append(distance)

    return np.array(distances)


# ============================================================
# 特征矩阵构建
# ============================================================
def filter_feature_columns(df: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    """过滤掉全NaN或标准差为0的特征列"""
    usable = []
    for col in feature_cols:
        if col not in df.columns:
            continue
        if df[col].notna().sum() == 0:
            continue
        if df[col].dtype in [np.float32, np.float64, float]:
            if df[col].std() == 0:
                continue
        usable.append(col)
    return usable


def make_feature_matrix(df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """
    构建特征矩阵，保留NaN（TabPFN可以处理）

    Args:
        df: 数据框
        feature_cols: 特征列名

    Returns:
        (n_samples, n_features) 特征矩阵
    """
    X = df[feature_cols].to_numpy(dtype=np.float32)
    return X


def property_columns(df: pd.DataFrame, exclude_cols: set[str] | None = None) -> list[str]:
    """
    获取所有性质列（非SMILES、非MACCS、非ID列）

    Args:
        df: 数据框
        exclude_cols: 需要排除的列

    Returns:
        性质列名列表
    """
    if exclude_cols is None:
        exclude_cols = set()

    exclude_cols = exclude_cols | {SMILES_COL, "maccs_valid", "polymer_id", "entry_id", "missing_rate"}
    exclude_cols = exclude_cols | set(FP_COLS)

    prop_cols = []
    for col in df.columns:
        if col in exclude_cols:
            continue
        if col.startswith(("maccs_", "fp_")):
            continue
        prop_cols.append(col)

    return prop_cols


# ============================================================
# 性质分类
# ============================================================
CATEGORY_KEYWORDS = {
    "thermal": ["Tg", "Td", "Tm", "Tc", "temp", "CTE", "Cp", "thermal", "heat", "softening", "HDT", "Vicat", "brittleness", "LC_transition", "Avrami", "cryst"],
    "mechanical": ["tensile", "modulus", "elongation", "strength", "DMA", "shear", "fiber", "flexural", "compressive", "hardness", "impact", "bulk", "compressibility"],
    "electrical": ["conductivity", "resistivity", "dielectric", "bandgap", "tandelta"],
    "optical": ["refractive", "Transmittance"],
    "solution": ["viscosity", "density", "specific_vol", "cohesive", "solubility", "Rg", "virial", "LCST", "UCST", "Hansen"],
    "stability": ["water_absorption", "oxygen_index", "swelling", "IEC", "G_value", "radiation"],
    "surface": ["surface_tension", "contact_angle", "interfacial"],
    "transport": ["perm", "diffusion", "solubility"],
}


def infer_category(property_name: str) -> str:
    """根据性质名称推断类别（优先使用xlsx权威映射，回退到关键词）"""
    try:
        from workflow_config import PROPERTY_CATEGORY_MAP
        if property_name in PROPERTY_CATEGORY_MAP:
            return PROPERTY_CATEGORY_MAP[property_name]
    except ImportError:
        pass

    prop_lower = property_name.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in prop_lower:
                return category
    return "unknown"


def get_same_category_properties(target_property: str, all_properties: list[str]) -> list[str]:
    """获取同类别的其他性质"""
    target_category = infer_category(target_property)

    same_category = []
    for prop in all_properties:
        if prop == target_property:
            continue
        if prop in EXCLUDED_INPUT_FEATURES:
            continue
        if infer_category(prop) == target_category:
            same_category.append(prop)

    return same_category


# ============================================================
# 可靠性评分组件
# ============================================================
def compute_ensemble_dispersion_normalized(
    predictions: np.ndarray,  # shape: (n_samples, n_models)
    observed_std: float,
) -> np.ndarray:
    """
    计算标准化ensemble离散度

    Args:
        predictions: (n_samples, n_models) 多模型预测
        observed_std: 观测值的标准差

    Returns:
        (n_samples,) 标准化离散度
    """
    if not np.isfinite(observed_std) or observed_std <= 0:
        return np.full(len(predictions), np.nan, dtype=float)
    ensemble_std = np.std(predictions, axis=1, ddof=1)
    normalized = ensemble_std / observed_std
    return normalized


def compute_local_structural_risk(
    query_distance: np.ndarray,  # (n_samples,)
    local_prediction_deviation: np.ndarray | None = None,  # (n_samples,)
    tolerance_cz: float = 1.0,
) -> np.ndarray:
    """
    计算结构-局部适用域风险

    Args:
        query_distance: 到训练集的平均k近邻距离
        local_prediction_deviation: 预测值与局部均值的标准化偏离（可选）
        tolerance_cz: 局部一致性容忍度

    Returns:
        (n_samples,) 结构风险分数
    """
    # 基础结构风险 = 距离
    structural_risk = query_distance

    # 如果有局部偏离信息，加入局部一致性惩罚
    if local_prediction_deviation is not None:
        # 超过tolerance_cz标准差的部分产生惩罚
        deviation_penalty = np.maximum(local_prediction_deviation - tolerance_cz, 0)

        # 距离越近，局部冲突权重越大
        avg_similarity = 1 - (query_distance**2) / 2
        local_risk = avg_similarity * deviation_penalty

        # 综合风险（均方根）
        combined_risk = np.sqrt((structural_risk**2 + local_risk**2) / 2)
        return combined_risk

    return structural_risk


def compute_reliability_score(
    property_nrmse: float,
    ensemble_std_normalized: np.ndarray,  # (n_samples,)
    tanimoto_distance: np.ndarray,  # (n_samples,)
) -> np.ndarray:
    """
    计算 V11 统一可靠性分数。

    R_i = 1 - [NRMSE_j/(1+NRMSE_j) + sigma_i/(1+sigma_i)
               + (D_i*sigma_i)/(1+D_i*sigma_i)] / 3

    Args:
        property_nrmse: 性质级归一化RMSE
        ensemble_std_normalized: (n_samples,) 标准化ensemble离散度
        tanimoto_distance: (n_samples,) fold-matched Tanimoto距离

    Returns:
        (n_samples,) 可靠性分数 [0, 1]
    """
    sigma = np.asarray(ensemble_std_normalized, dtype=float)
    distance = np.asarray(tanimoto_distance, dtype=float)
    model_risk = property_nrmse / (1.0 + property_nrmse)
    ensemble_risk = sigma / (1.0 + sigma)
    interaction = distance * sigma
    interaction_risk = interaction / (1.0 + interaction)
    return 1.0 - (model_risk + ensemble_risk + interaction_risk) / 3.0


# ============================================================
# 日志与进度
# ============================================================
def log_message(message: str, log_file: Path | None = None) -> None:
    """打印并记录日志"""
    print(message, flush=True)

    if log_file is not None:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(message + "\n")


def format_time(seconds: float) -> str:
    """格式化时间"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        return f"{seconds/3600:.2f}h"
