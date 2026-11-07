"""
Query result comparison using Spider's evaluation methodology.

This module implements the same result equality checking as Spider's test-suite-sql-eval,
which checks if two query results are semantically equivalent by:
1. Comparing row counts and column counts
2. Trying different column permutations to account for column reordering
3. Supporting both ordered (with ORDER BY) and unordered (set-based) comparison
4. Handling multiset semantics for duplicate rows

Reference: https://github.com/taoyds/test-suite-sql-eval
"""

from typing import List, Tuple, Set, Dict, Any, Union
from itertools import product
from collections import defaultdict
from decimal import Decimal, InvalidOperation
import random
from ..logger import get_logger

logger = get_logger(__name__)


def permute_tuple(element: Tuple, perm: Tuple) -> Tuple:
    """Permute a tuple according to a given permutation."""
    assert len(element) == len(perm)
    return tuple([element[i] for i in perm])


def unorder_row(row: Tuple) -> Tuple:
    """Sort elements within a row to create a normalized representation."""
    return tuple(sorted(row, key=stable_sort_key))


def decimal_sort_text(value: Decimal) -> str:
    """Return a canonical decimal string that is stable across display formats."""
    normalized = value.normalize()
    if normalized == normalized.to_integral_value():
        return format(normalized, "f")
    return format(normalized, "f").rstrip("0").rstrip(".")


def stable_sort_key(value: Any) -> Tuple[str, str]:
    """Sort normalized SQL values without depending on Decimal display notation."""
    if isinstance(value, Decimal):
        return (type(value).__name__, decimal_sort_text(value))
    return (type(value).__name__, str(value))


def quick_rej(result1: List[Tuple], result2: List[Tuple], order_matters: bool) -> bool:
    """
    Quick rejection test for result equality.

    Checks if result1 and result2 have the same bag of unordered rows,
    which is a necessary condition for equivalence.

    Args:
        result1: First result set as list of tuples
        result2: Second result set as list of tuples
        order_matters: Whether row order should be preserved

    Returns:
        True if results could be equal, False if definitely not equal
    """
    s1 = [unorder_row(row) for row in result1]
    s2 = [unorder_row(row) for row in result2]
    if order_matters:
        return s1 == s2
    else:
        return set(s1) == set(s2)


def multiset_eq(l1: List, l2: List) -> bool:
    """
    Check if two lists represent the same multiset (bag).

    Returns True if l1 and l2 contain the same elements with the same frequencies.
    """
    if len(l1) != len(l2):
        return False
    d = defaultdict(int)
    for e in l1:
        d[e] = d[e] + 1
    for e in l2:
        d[e] = d[e] - 1
        if d[e] < 0:
            return False
    return True


def get_constraint_permutation(tab1_sets_by_columns: List[Set], result2: List[Tuple]):
    """
    Generate constrained column permutations to test.

    This reduces the search space of column permutations by sampling rows
    and checking which column mappings are possible.
    """
    num_cols = len(result2[0])
    perm_constraints = [{i for i in range(num_cols)} for _ in range(num_cols)]
    if num_cols <= 3:
        return product(*perm_constraints)

    # Sample 20 rows to constrain the permutation space
    for _ in range(20):
        random_tab2_row = random.choice(result2)

        for tab1_col in range(num_cols):
            for tab2_col in set(perm_constraints[tab1_col]):
                if random_tab2_row[tab2_col] not in tab1_sets_by_columns[tab1_col]:
                    perm_constraints[tab1_col].remove(tab2_col)
    return product(*perm_constraints)


def result_eq(result1: List[Tuple], result2: List[Tuple], order_matters: bool) -> bool:
    """
    Check whether two query result sets are semantically equivalent.

    This is the main comparison function that checks if two result sets represent
    the same data, accounting for:
    - Different column orderings
    - Different row orderings (when order_matters=False)
    - Multiset semantics (duplicate rows)

    Args:
        result1: First result set as list of tuples (rows)
        result2: Second result set as list of tuples (rows)
        order_matters: Whether row order should be preserved (True for ORDER BY queries)

    Returns:
        True if results are semantically equivalent, False otherwise
    """
    # Empty results are equal
    if len(result1) == 0 and len(result2) == 0:
        return True

    # Different number of rows = different results
    if len(result1) != len(result2):
        return False

    num_cols = len(result1[0])

    # Different number of columns = different results
    if len(result2[0]) != num_cols:
        return False

    # Quick rejection: check if results have same bag of unordered rows
    if not quick_rej(result1, result2, order_matters):
        return False

    # Now we need to find a column permutation such that result1 == result2
    # Build sets of values for each column in result1
    tab1_sets_by_columns = [{row[i] for row in result1} for i in range(num_cols)]

    # Try all possible column permutations that might make result1 == result2
    for perm in get_constraint_permutation(tab1_sets_by_columns, result2):
        if len(perm) != len(set(perm)):
            # Skip invalid permutations (duplicate column mappings)
            continue

        if num_cols == 1:
            result2_perm = result2
        else:
            result2_perm = [permute_tuple(element, perm) for element in result2]

        if order_matters:
            # For ORDER BY queries, row order must match exactly
            if result1 == result2_perm:
                return True
        else:
            # For non-ORDER BY queries, use bag semantics
            if set(result1) == set(result2_perm) and multiset_eq(result1, result2_perm):
                return True

    return False


def normalize_result_value(value: Any) -> Any:
    """Normalize one SQL result value for stable exact comparison."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, Decimal):
        return value.normalize()
    if isinstance(value, int):
        return Decimal(value).normalize()
    if isinstance(value, float):
        try:
            return Decimal(str(value)).normalize()
        except InvalidOperation:
            return str(value)
    return str(value)


def normalize_result_row(row: Union[Dict[str, Any], Tuple, List]) -> Tuple:
    """
    Normalize a result row to a tuple for comparison.

    Handles different row formats:
    - Dict with column names: Extract values in sorted column order
    - Tuple or list: Convert to tuple, normalizing None and stringifying values

    Args:
        row: A row from query results

    Returns:
        Normalized tuple representation
    """
    if isinstance(row, dict):
        # Extract values in sorted column order for consistent comparison
        return tuple(normalize_result_value(v) for k, v in sorted(row.items()))
    elif isinstance(row, (list, tuple)):
        # Convert to tuple, normalizing values
        return tuple(normalize_result_value(v) for v in row)
    else:
        # Unknown format, try to convert directly
        return tuple([normalize_result_value(row)])


def compare_query_results(actual_result: Union[List, Dict],
                          expected_result: Union[List, Dict],
                          order_matters: bool = False) -> Dict[str, Any]:
    """
    Compare actual and expected query results using Spider's methodology.

    Supports two input formats:
    1. List of rows (tuples/lists/dicts)
    2. Dict with 'rows', 'columns', and optional metadata

    Args:
        actual_result: Actual query result
        expected_result: Expected query result
        order_matters: Whether row order should be preserved (auto-detected from query if possible)

    Returns:
        Dict with:
            - 'matches': bool indicating if results match
            - 'reason': str describing the comparison result
            - 'details': dict with additional comparison details
    """
    try:
        # Extract rows from different formats
        if isinstance(expected_result, dict):
            expected_rows = expected_result.get('rows', [])
        else:
            expected_rows = expected_result

        if isinstance(actual_result, dict):
            actual_rows = actual_result.get('rows', [])
        else:
            actual_rows = actual_result

        # Normalize rows to tuples
        expected_tuples = [normalize_result_row(row) for row in expected_rows]
        actual_tuples = [normalize_result_row(row) for row in actual_rows]

        # Compare using Spider's result_eq
        matches = result_eq(actual_tuples, expected_tuples, order_matters)

        details = {
            'actual_row_count': len(actual_tuples),
            'expected_row_count': len(expected_tuples),
            'order_matters': order_matters
        }

        if matches:
            return {
                'matches': True,
                'reason': f'Results match ({len(expected_tuples)} rows)',
                'details': details
            }
        else:
            # Provide helpful reason for mismatch
            if len(actual_tuples) != len(expected_tuples):
                reason = f'Row count mismatch: actual={len(actual_tuples)}, expected={len(expected_tuples)}'
            elif actual_tuples and expected_tuples and len(actual_tuples[0]) != len(expected_tuples[0]):
                reason = f'Column count mismatch: actual={len(actual_tuples[0])}, expected={len(expected_tuples[0])}'
            else:
                reason = 'Row values do not match (different data or ordering)'

            details['sample_actual'] = actual_tuples[:3] if actual_tuples else []
            details['sample_expected'] = expected_tuples[:3] if expected_tuples else []

            return {
                'matches': False,
                'reason': reason,
                'details': details
            }

    except Exception as e:
        logger.error(f"Error comparing query results: {e}")
        return {
            'matches': False,
            'reason': f'Comparison error: {str(e)}',
            'details': {'error': str(e)}
        }
