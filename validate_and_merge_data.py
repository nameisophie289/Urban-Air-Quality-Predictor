"""
Data Validation and Merging Script
Validates raw pollutant and weather data, then merges them into a single dataset.

Input:
  - data/raw/openaq_hourly/combined_*.csv (pollutant data)
  - data/raw/weather_hourly/weather_combined_*.csv (weather data)

Output:
  - data/processed/pollution_weather_merged.csv (merged dataset)
  - data/processed/data_quality_report.txt (validation report)
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_validation.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def find_latest_file(directory: str, pattern: str) -> Optional[Path]:
    """
    Find the latest file matching a pattern in a directory
    
    Args:
        directory: Directory to search
        pattern: Glob pattern to match
        
    Returns:
        Path to the latest file, or None if not found
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        return None
    
    files = list(dir_path.glob(pattern))
    if not files:
        return None
    
    # Sort by modification time (most recent first)
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return files[0]


def load_pollutant_data() -> Tuple[Optional[pd.DataFrame], str]:
    """
    Load pollutant data from raw folder
    
    Returns:
        Tuple of (DataFrame or None, status message)
    """
    logger.info("Loading pollutant data...")
    
    # Find the combined file
    filepath = find_latest_file("data/raw/openaq_hourly", "combined_*.csv")
    
    if filepath is None:
        return None, "No pollutant data file found in data/raw/openaq_hourly/"
    
    try:
        df = pd.read_csv(filepath)
        df["datetime"] = pd.to_datetime(df["datetime"])
        logger.info(f"Loaded pollutant data from: {filepath}")
        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        return df, f"Loaded from {filepath}"
    except Exception as e:
        return None, f"Error loading pollutant data: {e}"


def load_weather_data() -> Tuple[Optional[pd.DataFrame], str]:
    """
    Load weather data from raw folder
    
    Returns:
        Tuple of (DataFrame or None, status message)
    """
    logger.info("Loading weather data...")
    
    # Find the combined file
    filepath = find_latest_file("data/raw/weather_hourly", "weather_combined_*.csv")
    
    if filepath is None:
        return None, "No weather data file found in data/raw/weather_hourly/"
    
    try:
        df = pd.read_csv(filepath)
        df["datetime"] = pd.to_datetime(df["datetime"])
        logger.info(f"Loaded weather data from: {filepath}")
        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        return df, f"Loaded from {filepath}"
    except Exception as e:
        return None, f"Error loading weather data: {e}"


def validate_data(df: pd.DataFrame, name: str) -> dict:
    """
    Validate a DataFrame and return quality metrics
    
    Args:
        df: DataFrame to validate
        name: Name for logging
        
    Returns:
        Dictionary with validation metrics
    """
    logger.info(f"\nValidating {name}...")
    
    metrics = {
        "name": name,
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "missing_values": {},
        "missing_pct": {},
        "duplicates": 0,
        "date_range": None,
        "hourly_gaps": 0,
    }
    
    # Check missing values
    for col in df.columns:
        missing = df[col].isna().sum()
        metrics["missing_values"][col] = missing
        metrics["missing_pct"][col] = (missing / len(df)) * 100
    
    # Check for duplicate timestamps
    if "datetime" in df.columns:
        metrics["duplicates"] = df["datetime"].duplicated().sum()
        metrics["date_range"] = (df["datetime"].min(), df["datetime"].max())
        
        # Check for hourly gaps
        df_sorted = df.sort_values("datetime")
        time_diffs = df_sorted["datetime"].diff().dropna()
        expected_diff = pd.Timedelta(hours=1)
        gaps = (time_diffs != expected_diff).sum()
        metrics["hourly_gaps"] = gaps
    
    # Log summary
    logger.info(f"  Rows: {metrics['rows']:,}")
    logger.info(f"  Columns: {metrics['columns']}")
    logger.info(f"  Duplicates: {metrics['duplicates']}")
    logger.info(f"  Hourly gaps: {metrics['hourly_gaps']}")
    
    # Log missing values
    logger.info(f"  Missing values:")
    for col, pct in metrics["missing_pct"].items():
        if pct > 0:
            logger.info(f"    {col}: {pct:.1f}%")
    
    return metrics


def merge_data(
    df_pollutant: pd.DataFrame,
    df_weather: pd.DataFrame,
    timezone: str = "America/Los_Angeles"
) -> pd.DataFrame:
    """
    Merge pollutant and weather data on datetime
    
    The pollutant data is in UTC, weather data is in local timezone.
    We convert both to local timezone for merging.
    
    Args:
        df_pollutant: Pollutant DataFrame
        df_weather: Weather DataFrame
        timezone: Local timezone for alignment
        
    Returns:
        Merged DataFrame
    """
    logger.info("\nMerging pollutant and weather data...")
    
    # Make copies
    df_poll = df_pollutant.copy()
    df_weath = df_weather.copy()
    
    # Check if pollutant datetime has timezone info
    if df_poll["datetime"].dt.tz is not None:
        # Convert UTC to local timezone
        df_poll["datetime_local"] = df_poll["datetime"].dt.tz_convert(timezone).dt.tz_localize(None)
    else:
        # Assume it's already local or naive
        df_poll["datetime_local"] = df_poll["datetime"]
    
    # Weather datetime should already be in local timezone (naive)
    if df_weath["datetime"].dt.tz is not None:
        df_weath["datetime"] = df_weath["datetime"].dt.tz_localize(None)
    
    # Merge on local datetime
    df_merged = pd.merge(
        df_poll,
        df_weath,
        left_on="datetime_local",
        right_on="datetime",
        how="inner",
        suffixes=("", "_weather")
    )
    
    # Clean up columns
    cols_to_drop = ["datetime_local", "datetime_weather"]
    df_merged = df_merged.drop(columns=[c for c in cols_to_drop if c in df_merged.columns])
    
    # Sort by datetime
    df_merged = df_merged.sort_values("datetime").reset_index(drop=True)
    
    logger.info(f"  Merged shape: {df_merged.shape}")
    logger.info(f"  Date range: {df_merged['datetime'].min()} to {df_merged['datetime'].max()}")
    
    return df_merged


def generate_quality_report(
    pollutant_metrics: dict,
    weather_metrics: dict,
    merged_metrics: dict,
    output_path: str
) -> None:
    """
    Generate a data quality report
    
    Args:
        pollutant_metrics: Validation metrics for pollutant data
        weather_metrics: Validation metrics for weather data
        merged_metrics: Validation metrics for merged data
        output_path: Path to save the report
    """
    report_lines = [
        "=" * 80,
        "DATA QUALITY REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 80,
        "",
        "=" * 40,
        "POLLUTANT DATA",
        "=" * 40,
        f"Rows: {pollutant_metrics['rows']:,}",
        f"Columns: {pollutant_metrics['columns']}",
        f"Duplicates: {pollutant_metrics['duplicates']}",
        f"Hourly gaps: {pollutant_metrics['hourly_gaps']}",
        "",
        "Missing values:",
    ]
    
    for col, pct in pollutant_metrics["missing_pct"].items():
        report_lines.append(f"  {col}: {pct:.2f}%")
    
    report_lines.extend([
        "",
        "=" * 40,
        "WEATHER DATA",
        "=" * 40,
        f"Rows: {weather_metrics['rows']:,}",
        f"Columns: {weather_metrics['columns']}",
        f"Duplicates: {weather_metrics['duplicates']}",
        f"Hourly gaps: {weather_metrics['hourly_gaps']}",
        "",
        "Missing values:",
    ])
    
    for col, pct in weather_metrics["missing_pct"].items():
        if pct > 0:
            report_lines.append(f"  {col}: {pct:.2f}%")
    
    if all(pct == 0 for pct in weather_metrics["missing_pct"].values()):
        report_lines.append("  None")
    
    report_lines.extend([
        "",
        "=" * 40,
        "MERGED DATA",
        "=" * 40,
        f"Rows: {merged_metrics['rows']:,}",
        f"Columns: {merged_metrics['columns']}",
        f"Column names: {', '.join(merged_metrics['column_names'])}",
        "",
        "=" * 80,
    ])
    
    # Write report
    with open(output_path, "w") as f:
        f.write("\n".join(report_lines))
    
    logger.info(f"\nQuality report saved to: {output_path}")


def main():
    """Main function to validate and merge data"""
    
    logger.info("=" * 80)
    logger.info("DATA VALIDATION AND MERGING")
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    
    # Create output directory
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df_pollutant, poll_status = load_pollutant_data()
    df_weather, weather_status = load_weather_data()
    
    # Check if both loaded successfully
    if df_pollutant is None:
        logger.error(f"Failed to load pollutant data: {poll_status}")
        return False
    
    if df_weather is None:
        logger.error(f"Failed to load weather data: {weather_status}")
        return False
    
    # Validate data
    pollutant_metrics = validate_data(df_pollutant, "Pollutant Data")
    weather_metrics = validate_data(df_weather, "Weather Data")
    
    # Merge data
    df_merged = merge_data(df_pollutant, df_weather)
    
    # Validate merged data
    merged_metrics = validate_data(df_merged, "Merged Data")
    
    # Save merged data
    merged_filepath = output_dir / "pollution_weather_merged.csv"
    df_merged.to_csv(merged_filepath, index=False)
    logger.info(f"\n✓ Merged data saved to: {merged_filepath}")
    
    # Generate quality report
    report_filepath = output_dir / "data_quality_report.txt"
    generate_quality_report(pollutant_metrics, weather_metrics, merged_metrics, str(report_filepath))
    
    # Print summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Pollutant records: {pollutant_metrics['rows']:,}")
    logger.info(f"Weather records: {weather_metrics['rows']:,}")
    logger.info(f"Merged records: {merged_metrics['rows']:,}")
    logger.info(f"Final columns: {merged_metrics['columns']}")
    logger.info("")
    logger.info(f"Output files:")
    logger.info(f"  - {merged_filepath}")
    logger.info(f"  - {report_filepath}")
    logger.info("")
    logger.info("✓ Data validation and merging complete!")
    logger.info("=" * 80)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
