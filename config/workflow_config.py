"""Configuration for the self-contained V11 repeated cross-fitting workflow."""

from __future__ import annotations

from pathlib import Path


# ============================================================
# 路径配置（对齐v8风格）
# ============================================================
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = ROOT / "input_data"
DEFAULT_OUTPUT_DIR = ROOT / "outputs"

# Self-contained inputs copied from the latest V10 bundle.
LOCAL_MERGED_DATASET = DEFAULT_INPUT_DIR / "merged dataset" / "dataset_v10_maccs_drop30_invalidpsmiles.csv"
LOCAL_CHECKPOINT = ROOT / "tabpfn-v3-regressor-v3_default.ckpt"
SOURCE_MERGED_DATASET = LOCAL_MERGED_DATASET
SOURCE_CHECKPOINT = LOCAL_CHECKPOINT
DEFAULT_CHECKPOINT = SOURCE_CHECKPOINT

# 输出子目录
OOF_PREDICTIONS_DIR = DEFAULT_OUTPUT_DIR / "01_oof_predictions"
OOF_RELIABILITY_DIR = DEFAULT_OUTPUT_DIR / "02_oof_reliability_scores"
MISSING_PREDICTIONS_DIR = DEFAULT_OUTPUT_DIR / "03_missing_predictions"
P0_FIGURES_DIR = DEFAULT_OUTPUT_DIR / "04_p0_figures"
LOGS_DIR = DEFAULT_OUTPUT_DIR / "logs"


# ============================================================
# 实验标签
# ============================================================
VERSION = "v11_repeated5fold_cv_reliability_validation"


# ============================================================
# Repeated cross-fitting configuration. Repeat 0 reproduces the latest V10
# split seed; the other repeats add independent deterministic partitions.
# ============================================================
N_FOLDS = 5
N_REPEATS = 5
CV_RANDOM_STATE = 42
REPEAT_RANDOM_STATES = [42, 43, 44, 45, 46]
MODEL_RANDOM_STATE = 0


# ============================================================
# MACCS指纹配置（对齐v8风格命名）
# ============================================================
MACCS_N_BITS = 167
MACCS_PREFIX = "maccs_"
FP_N_BITS = MACCS_N_BITS  # 向后兼容
FP_PREFIX = MACCS_PREFIX
FP_COLS = [f"{MACCS_PREFIX}{idx:04d}" for idx in range(MACCS_N_BITS)]
SMILES_COL = "smiles"  # v10实际使用smiles（小写）


# ============================================================
# 从原cleaned_v10继承的性质过滤配置
# ============================================================
# 排除的辅助列
EXCLUDED_PROPERTIES = ["missing_rate"]

EXPLICIT_OBSERVED_ONLY_TARGETS: dict[str, str] = {
    "conductivity_inv_ohm_cm": "data_conflict_and_redundant_with_volume_resistivity",
    "G_value_events_100eV": "sample_size_lt_50_stability",
    "radiation_resistance_Mrad": "sample_size_lt_50_stability",
    "interfacial_tension_mN_m": "sample_size_lt_50_surface",
    "LCST_K": "nonpositive_test_skill_missing_solution_context",
    "Rg_nm": "nonpositive_test_skill_missing_solution_context",
    "UCST_K": "sample_size_lt_50_solution",
}

EXPLICIT_IMPUTATION_TARGETS: dict[str, str] = {
    "IEC_meq_g": "user_restored_for_v10_low_n_but_acceptable_for_labeled_imputation",
}

EXCLUDED_INPUT_FEATURES: set[str] = {
    "conductivity_inv_ohm_cm",
}

CATEGORY_MIN_OBSERVED: dict[str, int] = {
    "mechanical": 50,
    "thermal": 50,
    "stability": 50,
    "surface": 50,
}

DEFAULT_MIN_OBSERVED = 50
MIN_ELIGIBLE_OBSERVED = 50


# ============================================================
# 测试运行配置（3个代表性性质）
# ============================================================
TEST_PROPERTIES = [
    "Tg_K",                  # 观测数最多（~24719）
    "dielectric_constant",   # 中等（~579）
    "thermal_cond_W_mK",     # 稀疏（~151）
]


# ============================================================
# 可靠性评分配置
# ============================================================
RELIABILITY_HIGH_THRESHOLD = 0.80
RELIABILITY_MEDIUM_THRESHOLD = 0.60
K_NEAREST_NEIGHBORS = 5
LOCAL_CONSISTENCY_TOLERANCE_CZ = 1.0
NEIGHBOR_CANDIDATE_MULTIPLIER = 8


# ============================================================
# 性质分类关键词（对齐v10 common.py）
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


# ============================================================
# 辅助函数（对齐v10逻辑）
# ============================================================
from dataclasses import dataclass


@dataclass(frozen=True)
class ScopeDecision:
    category: str
    property_name: str
    n_observed_total: int
    n_missing_total: int
    min_observed_required: int
    scope: str
    reason: str


def minimum_observed_for_category(category: str) -> int:
    """获取类别的最小观测数要求"""
    return CATEGORY_MIN_OBSERVED.get(category, DEFAULT_MIN_OBSERVED)


def decide_scope(
    category: str,
    property_name: str,
    n_observed_total: int,
    n_missing_total: int
) -> ScopeDecision:
    """
    决定某性质是否进入填补范围（与原cleaned_v10一致）

    Returns:
        ScopeDecision对象，scope字段为：
        - "observed_only": 仅保留观测，不填补
        - "candidate_for_imputation": 候选填补目标
    """
    min_required = minimum_observed_for_category(category)

    # 显式排除的性质
    if property_name in EXPLICIT_OBSERVED_ONLY_TARGETS:
        return ScopeDecision(
            category=category,
            property_name=property_name,
            n_observed_total=n_observed_total,
            n_missing_total=n_missing_total,
            min_observed_required=min_required,
            scope="observed_only",
            reason=EXPLICIT_OBSERVED_ONLY_TARGETS[property_name],
        )

    # 显式包含的性质（即使观测数略低）
    if property_name in EXPLICIT_IMPUTATION_TARGETS:
        return ScopeDecision(
            category=category,
            property_name=property_name,
            n_observed_total=n_observed_total,
            n_missing_total=n_missing_total,
            min_observed_required=min_required,
            scope="candidate_for_imputation",
            reason=EXPLICIT_IMPUTATION_TARGETS[property_name],
        )

    # 观测数不足
    if n_observed_total < min_required:
        return ScopeDecision(
            category=category,
            property_name=property_name,
            n_observed_total=n_observed_total,
            n_missing_total=n_missing_total,
            min_observed_required=min_required,
            scope="observed_only",
            reason=f"category_min_observed_not_met_{n_observed_total}_lt_{min_required}",
        )

    # 通过所有检查
    return ScopeDecision(
        category=category,
        property_name=property_name,
        n_observed_total=n_observed_total,
        n_missing_total=n_missing_total,
        min_observed_required=min_required,
        scope="candidate_for_imputation",
        reason="passed_v10_scope_gate",
    )

# 自动生成：从xlsx权威映射（覆盖infer_category的关键词逻辑）
PROPERTY_CATEGORY_MAP = {
    "Avrami_n": "thermal",
    "CO2_perm_Barrer": "transport",
    "CTE_ppm_K": "thermal",
    "Cp_cal_g_C": "thermal",
    "DMA_loss_GPa": "mechanical",
    "DMA_storage_GPa": "mechanical",
    "DMA_tandelta": "mechanical",
    "G_value_events_100eV": "stability",
    "H2_perm_Barrer": "transport",
    "HDT_K": "thermal",
    "Hansen_d_J_cm3_0p5": "solution",
    "Hansen_h_J_cm3_0p5": "solution",
    "Hansen_p_J_cm3_0p5": "solution",
    "IEC_meq_g": "stability",
    "LCST_K": "solution",
    "LC_transition_K": "thermal",
    "O2_perm_Barrer": "transport",
    "Rg_nm": "solution",
    "Tc_K": "thermal",
    "Td_K": "thermal",
    "Td_loss_pct": "thermal",
    "Tg_K": "thermal",
    "Tm_K": "thermal",
    "Transmittance_400nm_pct": "optical",
    "UCST_K": "solution",
    "Vicat_K": "thermal",
    "bandgap_eV": "electrical",
    "brittleness_temp_K": "thermal",
    "bulk_modulus_GPa": "mechanical",
    "cohesive_energy_cal_cm3": "solution",
    "compressibility_1_GPa": "mechanical",
    "compressive_modulus_GPa": "mechanical",
    "compressive_strength_MPa": "mechanical",
    "conductivity_inv_ohm_cm": "electrical",
    "contact_angle_degree": "surface",
    "cryst_halftime_s": "thermal",
    "cryst_heat_cal_g": "thermal",
    "crystal_growth_nm_s": "thermal",
    "density_g_cm3": "solution",
    "dielectric_constant": "electrical",
    "dielectric_loss": "electrical",
    "dielectric_tandelta": "electrical",
    "elongation_break_pct": "mechanical",
    "elongation_yield_pct": "mechanical",
    "fiber_elongation_pct": "mechanical",
    "fiber_modulus_g_denier": "mechanical",
    "fiber_strength_g_denier": "mechanical",
    "flexural_break_MPa": "mechanical",
    "flexural_loss_GPa": "mechanical",
    "flexural_modulus_GPa": "mechanical",
    "flexural_storage_GPa": "mechanical",
    "flexural_tandelta": "mechanical",
    "fusion_heat_kcal_g": "thermal",
    "gas_diffusion_cm2_s": "transport",
    "gas_perm": "transport",
    "gas_solubility": "transport",
    "hardness_MPa": "mechanical",
    "impact_strength_kJ_m2": "mechanical",
    "interfacial_tension_mN_m": "surface",
    "intrinsic_viscosity_dL_g": "solution",
    "oxygen_index_pct": "stability",
    "radiation_resistance_Mrad": "stability",
    "refractive_index": "optical",
    "second_virial_cm3_mol_g2": "solution",
    "shear_storage_GPa": "mechanical",
    "shear_tandelta": "mechanical",
    "softening_temp_K": "thermal",
    "solubility_param_J_cm3_0p5": "solution",
    "specific_vol_cm3_g": "solution",
    "surface_tension_mN_m": "surface",
    "swelling_pct": "stability",
    "tensile_break_GPa": "mechanical",
    "tensile_modulus_GPa": "mechanical",
    "tensile_yield_GPa": "mechanical",
    "thermal_cond_W_mK": "thermal",
    "thermal_diffusivity_m2_s": "thermal",
    "volume_resistivity_ohm_cm": "electrical",
    "water_absorption_pct": "stability",
}
