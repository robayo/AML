import pandas as pd
from  scipy import stats
import numpy as np
def corr_with_binary_labels(
        df: pd.DataFrame,
        target: str,          # name of the categorical column
        positive: str,        # value to be taken as 1
        negative: str         # value to be taken as 0
) -> pd.Series:
    """
    Strength of association between a binary indicator
    [target == positive] vs [target == negative]  and every
    other column in `df`.

    • Rows where `target` is neither positive nor negative are dropped.
    • Numeric predictors → point-biserial correlation (Pearson on 0/1).
    • Categorical predictors → Cramér’s V.

    Returns a Series sorted high → low (absolute strength).
    """
    # ── 1. Filter rows that have either positive or negative label ──
    mask = df[target].isin([positive, negative])
    if mask.sum() == 0:
        raise ValueError("No rows match the given positive/negative labels.")

    # Binary 0/1 vector
    bin_target = (df.loc[mask, target] == positive).astype(int)

    out = {}

    # ── 2. Loop over the other columns ──
    for col in df.columns:
        if col == target:
            continue

        y = df.loc[mask, col]   # same subset for the predictor

        # Drop rows where either series is NaN
        valid = bin_target.notna() & y.notna()
        if valid.sum() < 2:
            score = np.nan
        else:
            score, _ = stats.pearsonr(bin_target[valid], y[valid])



        out[col] = score

    # Return strongest → weakest (use abs() so negative values rank high too)
    return (pd.Series(out)
              .reindex(out)          # preserve insertion order
              .sort_values(key=np.abs, ascending=False))
