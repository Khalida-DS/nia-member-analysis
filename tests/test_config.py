"""
tests/test_config.py
====================
Validate that settings.yaml loads correctly and all required fields exist.
Run: pytest tests/test_config.py -v
"""
import pytest
from src.config import get_config, ProjectConfig


def test_loads_without_error():
    cfg = get_config()
    assert cfg is not None


def test_is_project_config():
    assert isinstance(get_config(), ProjectConfig)


def test_random_seed_is_int():
    assert isinstance(get_config().random_seed, int)


def test_nine_products():
    cfg = get_config()
    assert len(cfg.data.product_cols)  == 9
    assert len(cfg.data.product_names) == 9


def test_product_col_to_name():
    cfg = get_config()
    m   = cfg.data.col_to_name
    assert m["FSV Credit Card Flag"] == "FSV Credit Card"
    assert m["INS Client Flag"]       == "INS Client"


def test_income_map_has_all_bands():
    cfg = get_config()
    assert "Under 10K" in cfg.data.income_map
    assert "250K+"     in cfg.data.income_map
    assert len(cfg.data.income_map) == 15


def test_credit_map_has_all_bands():
    cfg = get_config()
    assert "800+"     in cfg.data.credit_map
    assert "499 & Less" in cfg.data.credit_map


def test_split_sizes_sum_to_less_than_one():
    cfg = get_config()
    assert cfg.training.test_size + cfg.training.val_size < 1.0


def test_paths_are_strings():
    cfg = get_config()
    assert isinstance(cfg.paths.raw_data,  str)
    assert isinstance(cfg.paths.model_dir, str)


def test_clustering_final_k_in_range():
    cfg = get_config()
    assert cfg.clustering.final_k in cfg.clustering.k_range


def test_product_names_property():
    cfg = get_config()
    names = cfg.data.product_names
    assert "FSV Credit Card" in names
    assert "INS Client"      in names
    # None should have " Flag" suffix
    for name in names:
        assert "Flag" not in name