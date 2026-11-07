"""
Resolve LiveSQLBench knowledge/semantics sources for a reference database.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional
import json


DB_TO_LIVESQL_DIR: Dict[str, str] = {
    "archeology": "archeology_scan",
    "baseball": "baseball",
    "sports": "sports_events",
    "labor": "labor_certification_applications",
    "households": "households",
    "fake_account": "fake_account",
    "cross_border": "cross_border",
    "mental_health": "mental_health",
    "insider_trading": "insider_trading",
    "hulushows": "hulushows",
    "cold_chain": "cold_chain_pharma_compliance",
    "cybermarket": "cybermarket_pattern",
    "etf": "exchange_traded_funds",
    "museum": "museum_artifact",
    "planets": "planets_data",
    "virtual_idol": "virtual_idol",
    # RDS database name differs from the LiveSQLBench directory for this DB.
    "robot_fault": "robot_fault_prediction",
}


def get_repo_root() -> Path:
    """Return the repository root (assumes this file is under src/benchmark/)."""
    return Path(__file__).resolve().parents[2]


def get_livesqlbench_root(repo_root: Optional[Path] = None) -> Path:
    """
    Return the LiveSQLBench assets root.

    Override with `LIVESQLBENCH_ROOT` when the KB/semantics live outside the repo
    (e.g., on a shared/remote mount).
    """
    env_root = (os.environ.get("LIVESQLBENCH_ROOT") or "").strip()
    if env_root:
        root = Path(env_root).expanduser()
        if not root.is_absolute():
            root = (repo_root or get_repo_root()) / root
        return root
    return (repo_root or get_repo_root()) / "livesqlbench-base-full-v1"


def get_livesql_dir(reference_database: Optional[str], repo_root: Optional[Path] = None) -> Optional[str]:
    """
    Map a scenario `reference_database` name to a directory under the LiveSQLBench root.

    - Prefer explicit aliases in `DB_TO_LIVESQL_DIR` (e.g., `archeology` -> `archeology_scan`).
    - Fallback: allow using the LiveSQLBench directory name directly (e.g., `solar_panel`).
    """
    if not reference_database:
        return None
    mapped = DB_TO_LIVESQL_DIR.get(reference_database)
    if mapped:
        return mapped
    base = get_livesqlbench_root(repo_root)
    direct = base / reference_database
    return reference_database if direct.is_dir() else None


def resolve_kb_file(reference_database: Optional[str], repo_root: Optional[Path] = None) -> Optional[Path]:
    dir_name = get_livesql_dir(reference_database, repo_root)
    if not dir_name:
        return None
    base = get_livesqlbench_root(repo_root)
    return base / dir_name / f"{dir_name}_kb.jsonl"


def resolve_column_meanings_file(reference_database: Optional[str], repo_root: Optional[Path] = None) -> Optional[Path]:
    dir_name = get_livesql_dir(reference_database, repo_root)
    if not dir_name:
        return None
    base = get_livesqlbench_root(repo_root)
    return base / dir_name / f"{dir_name}_column_meaning_base.json"


def load_base_column_meanings(reference_database: Optional[str], repo_root: Optional[Path] = None) -> Optional[Dict[str, object]]:
    path = resolve_column_meanings_file(reference_database, repo_root)
    if not path or not path.exists():
        return None
    with open(path, "r") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else None
