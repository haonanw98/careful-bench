from decimal import Decimal

from src.benchmark.evaluator import compare_query_results


def test_decimal_scientific_notation_matches_plain_decimal() -> None:
    actual = [(Decimal("1.41E+3"), "Mountain-200 Black, 42", Decimal("2508291.0800"))]
    expected = [(Decimal("1410"), "Mountain-200 Black, 42", Decimal("2508291.08"))]

    comparison = compare_query_results(actual, expected, order_matters=True)

    assert comparison["matches"], comparison


def test_decimal_scientific_notation_matches_integer() -> None:
    actual = [(Decimal("1.41E+3"), "Mountain-200 Black, 42", Decimal("10"))]
    expected = [(1410, "Mountain-200 Black, 42", Decimal("10.0"))]

    comparison = compare_query_results(actual, expected, order_matters=True)

    assert comparison["matches"], comparison
