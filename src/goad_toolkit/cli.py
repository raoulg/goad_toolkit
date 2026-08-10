"""Command line entry point: fit the distribution registry to one column."""

import argparse
from pathlib import Path
from typing import List

import numpy as np

from goad_toolkit.analytics import DistributionFitter, FitResult, Result
from goad_toolkit.filehandler import reader_for


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="goad",
        description=(
            "Fit every distribution in the registry to one column of a csv or "
            "parquet file and print the fits, ranked by log-likelihood."
        ),
    )
    parser.add_argument("file", type=Path, help="path to a .csv or .parquet file")
    parser.add_argument("column", help="name of the column to fit")
    parser.add_argument(
        "--discrete",
        action="store_true",
        help="fit the discrete families instead of the continuous ones",
    )
    parser.add_argument(
        "--criterion",
        default="combined",
        choices=["likelihood", "ks", "combined"],
        help="which fits to mark as best (default: combined)",
    )
    parser.add_argument(
        "--method", default="mle", help="scipy fitting method (default: mle)"
    )
    return parser.parse_args(argv)


def load_column(file: Path, column: str) -> np.ndarray:
    """Read one column and return it as a float array without missing values."""
    if not file.exists():
        raise SystemExit(f"No such file: {file}")
    data = reader_for(file)(file)
    if column not in data.columns:
        raise SystemExit(
            f"Column '{column}' not found. Available: {list(data.columns)}"
        )
    return data[column].dropna().to_numpy()


def format_results(results: List[Result]) -> str:
    """Render fits as a table, best log-likelihood first, failures last."""
    successes = [fit for fit in results if isinstance(fit, FitResult)]
    failures = [fit for fit in results if not isinstance(fit, FitResult)]
    successes.sort(
        key=lambda fit: fit.log_likelihood
        if fit.log_likelihood is not None
        else -np.inf,
        reverse=True,
    )

    header = f"{'distribution':<14}{'loglik':>14}{'ks p':>10}  best"
    lines = [header, "-" * len(header)]
    for fit in successes:
        loglik = "n/a" if fit.log_likelihood is None else f"{fit.log_likelihood:.2f}"
        ks = "n/a" if fit.kstest is None else f"{fit.kstest.p_value:.4f}"
        marks = []
        if fit.best_likelihood:
            marks.append("likelihood")
        if fit.best_ks:
            marks.append("ks")
        row = f"{fit.distribution:<14}{loglik:>14}{ks:>10}  {', '.join(marks)}"
        lines.append(row.rstrip())
    for fit in failures:
        lines.append(f"{fit.distribution:<14}{'failed: ' + fit.message}")
    return "\n".join(lines)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    values = load_column(args.file, args.column)
    fitter = DistributionFitter()
    results = fitter.fit(
        values, discrete=args.discrete, method=args.method, criterion=args.criterion
    )
    print(f"\n{len(values)} values from {args.file}:{args.column}\n")
    print(format_results(results))


if __name__ == "__main__":
    main()
