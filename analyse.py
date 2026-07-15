"""
Student Performance - Exploratory Data Analysis
------------------------------------------------
Analyzes student marks, attendance, and their relationship using
descriptive statistics and visualizations.

Usage:
    python analyse.py
    python analyse.py --input data.csv --output-dir plots
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser(description="EDA on student performance data.")
    parser.add_argument("--input", default="data.csv", help="Path to input CSV file")
    parser.add_argument("--output-dir", default="output", help="Directory to save plots")
    parser.add_argument("--show", action="store_true", help="Display plots interactively")
    return parser.parse_args()


def load_data(filepath: str) -> pd.DataFrame:
    """Load the dataset and validate required columns exist."""
    path = Path(filepath)
    if not path.exists():
        sys.exit(f"Error: file not found -> {filepath}")

    df = pd.read_csv(path)

    required_cols = {"Name", "Marks", "Attendance"}
    missing = required_cols - set(df.columns)
    if missing:
        sys.exit(f"Error: missing required column(s): {missing}")

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values and duplicate rows."""
    before = len(df)

    df = df.drop_duplicates()
    null_counts = df.isnull().sum()

    if null_counts.any():
        print("\nMissing values found:")
        print(null_counts[null_counts > 0])
        df = df.dropna(subset=["Marks", "Attendance"])

    removed = before - len(df)
    if removed:
        print(f"Removed {removed} duplicate/invalid row(s).")

    return df.reset_index(drop=True)


def summarize(df: pd.DataFrame) -> None:
    """Print dataset preview and descriptive statistics."""
    print("\n" + "=" * 50)
    print("DATASET PREVIEW")
    print("=" * 50)
    print(df.head())

    print("\n" + "=" * 50)
    print("STATISTICAL SUMMARY")
    print("=" * 50)
    print(df.describe())

    print("\n" + "=" * 50)
    print("KEY METRICS")
    print("=" * 50)
    print(f"Average Marks:      {df['Marks'].mean():.2f}")
    print(f"Median Marks:       {df['Marks'].median():.2f}")
    print(f"Highest Scorer:     {df.loc[df['Marks'].idxmax(), 'Name']} ({df['Marks'].max()})")
    print(f"Lowest Scorer:      {df.loc[df['Marks'].idxmin(), 'Name']} ({df['Marks'].min()})")
    print(f"Average Attendance: {df['Attendance'].mean():.2f}")

    correlation = df["Attendance"].corr(df["Marks"])
    print(f"Attendance-Marks Correlation: {correlation:.2f}")


def plot_marks_bar(df: pd.DataFrame, output_dir: Path, show: bool) -> None:
    """Bar chart of marks per student."""
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Name", y="Marks", data=df, hue="Name", palette="viridis", legend=False)
    plt.xlabel("Students")
    plt.ylabel("Marks")
    plt.title("Marks of Students")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "marks_bar_chart.png", dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_attendance_vs_marks(df: pd.DataFrame, output_dir: Path, show: bool) -> None:
    """Scatter plot examining the attendance-marks relationship."""
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x="Attendance", y="Marks", data=df, s=80, color="darkblue")
    sns.regplot(x="Attendance", y="Marks", data=df, scatter=False, color="red")
    plt.xlabel("Attendance (%)")
    plt.ylabel("Marks")
    plt.title("Attendance vs Marks")
    plt.tight_layout()
    plt.savefig(output_dir / "attendance_vs_marks.png", dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: Path, show: bool) -> None:
    """Correlation heatmap of numeric columns."""
    plt.figure(figsize=(6, 5))
    numeric_df = df.select_dtypes(include="number")
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=150)
    if show:
        plt.show()
    plt.close()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.input)
    df = clean_data(df)

    summarize(df)

    print(f"\nGenerating visualizations -> saving to '{output_dir}/'")
    plot_marks_bar(df, output_dir, args.show)
    plot_attendance_vs_marks(df, output_dir, args.show)
    plot_correlation_heatmap(df, output_dir, args.show)

    print("Done. Plots saved as PNG files.")


if __name__ == "__main__":
    main()
