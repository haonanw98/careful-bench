import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import _evaluation_pattern_matches


def test_forbidden_sql_pattern_supports_regex_prefix() -> None:
    assert _evaluation_pattern_matches(
        r"regex:\bcbp_\d+_[0-9a-f]{10}\b",
        "SELECT * FROM public.cbp_15_abcdef1234",
    )
    assert _evaluation_pattern_matches(
        r"regex:\bpg_inherits\b",
        "SELECT * FROM pg_inherits",
    )


def test_forbidden_sql_pattern_regex_prefix_does_not_match_normal_parent_query() -> None:
    assert not _evaluation_pattern_matches(
        r"regex:\bcbp_\d+_[0-9a-f]{10}\b",
        "SELECT * FROM orders WHERE orderdate >= DATE '2026-02-01'",
    )
