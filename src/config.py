"""
src/config.py
=============
Loads configs/settings.yaml and exposes a fully-typed ProjectConfig dataclass.

Usage
-----
    from src.config import get_config
    cfg = get_config()
    print(cfg.training.test_size)       # 0.20
    print(cfg.data.product_names)       # ['FSV CMSI', 'FSV Credit Card', ...]
    print(cfg.paths.abs('raw_data'))    # /absolute/path/to/data/raw/member_sample.csv

Why a dataclass instead of a plain dict?
-----------------------------------------
cfg['training']['test_sze'] fails at runtime in production.
cfg.training.test_sze fails at import time, caught by your IDE immediately.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Resolve paths relative to the repo root (two levels above this file)
_ROOT        = Path(__file__).resolve().parents[1]
_CONFIG_PATH = _ROOT / "configs" / "settings.yaml"


# ── Sub-config dataclasses ────────────────────────────────────────────────────

@dataclass
class PathsConfig:
    raw_data:        str
    processed_data:  str
    household_data:  str
    cluster_data:    str
    recommendations: str
    model_dir:       str
    report_dir:      str

    def abs(self, key: str) -> Path:
        """Return an absolute Path for any path key, resolved from the repo root."""
        return _ROOT / getattr(self, key)


@dataclass
class DataConfig:
    min_product_penetration: float
    member_status_keep:      List[str]
    product_cols:            List[str]
    exclude_cols:            List[str]
    income_map:              Dict[str, int]
    credit_map:              Dict[str, int]
    children_map:            Dict[str, int]

    @property
    def product_names(self) -> List[str]:
        """Clean product names: 'FSV Credit Card Flag' → 'FSV Credit Card'."""
        return [c.replace(" Flag", "") for c in self.product_cols]

    @property
    def col_to_name(self) -> Dict[str, str]:
        """Map raw column name → clean product name."""
        return dict(zip(self.product_cols, self.product_names))


@dataclass
class TrainingConfig:
    test_size: float
    val_size:  float
    cv_folds:  int


@dataclass
class ClassificationConfig:
    class_weight:           str
    primary_metric:         str
    lift_target_percentile: float
    xgboost:                Dict[str, Any]
    random_forest:          Dict[str, Any]


@dataclass
class RegressionConfig:
    target_col:   str
    ridge_alphas: List[float]
    lasso_alphas: List[float]
    random_forest: Dict[str, Any]


@dataclass
class ClusteringConfig:
    k_range: List[int]
    final_k: int


@dataclass
class PlottingConfig:
    dpi:    int
    colors: Dict[str, str]


@dataclass
class ProjectConfig:
    name:           str
    version:        str
    random_seed:    int
    description:    str
    paths:          PathsConfig
    data:           DataConfig
    training:       TrainingConfig
    classification: ClassificationConfig
    regression:     RegressionConfig
    clustering:     ClusteringConfig
    plotting:       PlottingConfig


# ── Loader ────────────────────────────────────────────────────────────────────

def get_config(path: Optional[Path] = None) -> ProjectConfig:
    """
    Load and validate settings.yaml; return a ProjectConfig instance.

    Parameters
    ----------
    path : optional override path to a different YAML file
           (useful for testing with a minimal config)
    """
    config_path = path or _CONFIG_PATH
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    proj = raw["project"]

    return ProjectConfig(
        name        = proj["name"],
        version     = proj["version"],
        random_seed = proj["random_seed"],
        description = proj.get("description", ""),
        paths          = PathsConfig(**raw["paths"]),
        data           = DataConfig(**raw["data"]),
        training       = TrainingConfig(**raw["training"]),
        classification = ClassificationConfig(**raw["classification"]),
        regression     = RegressionConfig(**raw["regression"]),
        clustering     = ClusteringConfig(**raw["clustering"]),
        plotting       = PlottingConfig(**raw["plotting"]),
    )