"""
Open-Meteo API Client Module
Provides interface for collecting historical and recent weather data from Open-Meteo API
"""

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
from tqdm import tqdm

# Import config - handle both direct and module import
try:
    import config
except ImportError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    import config

logger = logging.getLogger(__name__)

# Default hourly weather variables from config
DEFAULT_HOURLY_VARIABLES = config.WEATHER_VARIABLES_HOURLY


class OpenMeteoClient:
    """
    Client for interacting with Open-Meteo API

    Supports both historical (archive) and recent (forecast) weather data collection.
    No API key required - Open-Meteo is free and open.
    """

    # API endpoints from config
    ARCHIVE_API_URL = config.OPEN_METEO_ARCHIVE_URL
    FORECAST_API_URL = config.OPEN_METEO_FORECAST_URL

    def __init__(self, timezone: str = None):
        """
        Initialize Open-Meteo API client

        Args:
            timezone: Timezone for weather data (default from config)
        """
        self.timezone = timezone or config.OPEN_METEO_TIMEZONE
        self.session = requests.Session()

    def get_hourly_weather(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
        variables: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Get historical hourly weather data for a location

        Args:
            latitude: Latitude of the location
            longitude: Longitude of the location
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            variables: List of weather variables to fetch (uses defaults if None)

        Returns:
            DataFrame with hourly weather data
        """
        if variables is None:
            variables = DEFAULT_HOURLY_VARIABLES

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(variables),
            "timezone": self.timezone,
        }

        try:
            logger.info(f"Fetching weather data for ({latitude}, {longitude})")
            logger.info(f"Date range: {start_date} to {end_date}")

            response = self.session.get(self.ARCHIVE_API_URL, params=params)
            response.raise_for_status()
            data = response.json()

            # Extract hourly data
            hourly_data = data.get("hourly", {})

            if not hourly_data:
                logger.warning("No hourly data returned")
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(hourly_data)
            df["datetime"] = pd.to_datetime(df["time"])
            df = df.drop(columns=["time"])

            # Reorder columns with datetime first
            cols = ["datetime"] + [c for c in df.columns if c != "datetime"]
            df = df[cols]

            logger.info(f"Fetched {len(df)} hours of weather data")
            return df

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching weather data: {e}")
            return pd.DataFrame()

    def get_recent_weather(
        self,
        latitude: float,
        longitude: float,
        past_days: int = 7,
        variables: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Get recent weather data using the Forecast API

        The forecast API can provide more up-to-date data than the archive API.

        Args:
            latitude: Latitude of the location
            longitude: Longitude of the location
            past_days: Number of past days to fetch (max 92)
            variables: List of weather variables to fetch (uses defaults if None)

        Returns:
            DataFrame with hourly weather data
        """
        if variables is None:
            variables = DEFAULT_HOURLY_VARIABLES

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": ",".join(variables),
            "timezone": self.timezone,
            "past_days": min(past_days, 92),
            "forecast_days": 0,  # Only historical, no forecast
        }

        try:
            logger.info(f"Fetching recent weather data (past {past_days} days)")
            logger.info(f"Location: ({latitude}, {longitude})")

            response = self.session.get(self.FORECAST_API_URL, params=params)
            response.raise_for_status()
            data = response.json()

            # Extract hourly data
            hourly_data = data.get("hourly", {})

            if not hourly_data:
                logger.warning("No hourly data returned")
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(hourly_data)
            df["datetime"] = pd.to_datetime(df["time"])
            df = df.drop(columns=["time"])

            # Reorder columns
            cols = ["datetime"] + [c for c in df.columns if c != "datetime"]
            df = df[cols]

            logger.info(f"Fetched {len(df)} hours of recent weather data")
            return df

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching recent weather data: {e}")
            return pd.DataFrame()

    def collect_historical_data_batch(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
        output_dir: str = "data/raw/weather_hourly",
        chunk_days: int = 30,
        variables: Optional[List[str]] = None,
        resume: bool = True,
    ) -> pd.DataFrame:
        """
        Collect historical weather data in chunks with checkpointing

        Args:
            latitude: Latitude of the location
            longitude: Longitude of the location
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            output_dir: Directory to save checkpoint files
            chunk_days: Number of days per chunk (default: 30)
            variables: List of weather variables to fetch
            resume: If True, skip already collected chunks

        Returns:
            Combined DataFrame with all weather data
        """
        if variables is None:
            variables = DEFAULT_HOURLY_VARIABLES

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate date chunks
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        date_chunks = []
        current = start
        while current < end:
            chunk_end = min(current + timedelta(days=chunk_days), end)
            date_chunks.append((current, chunk_end))
            current = chunk_end

        logger.info(f"Total chunks to collect: {len(date_chunks)}")
        logger.info(f"Date range: {start_date} to {end_date}")
        logger.info(f"Location: ({latitude}, {longitude})")
        logger.info(f"Variables: {len(variables)} weather parameters")
        logger.info(f"Output directory: {output_path.absolute()}")

        all_data = []

        # Process each chunk
        for chunk_start, chunk_end in tqdm(date_chunks, desc="Collecting weather data"):
            chunk_start_str = chunk_start.strftime("%Y-%m-%d")
            chunk_end_str = chunk_end.strftime("%Y-%m-%d")

            # Generate checkpoint filename
            chunk_filename = f"weather_{chunk_start.strftime('%Y%m%d')}_{chunk_end.strftime('%Y%m%d')}.csv"
            chunk_filepath = output_path / chunk_filename

            # Skip if already collected and resume is enabled
            if resume and chunk_filepath.exists():
                logger.info(
                    f"Loading cached data for {chunk_start_str} to {chunk_end_str}"
                )
                try:
                    chunk_df = pd.read_csv(chunk_filepath)
                    chunk_df["datetime"] = pd.to_datetime(chunk_df["datetime"])
                    all_data.append(chunk_df)
                    continue
                except Exception as e:
                    logger.warning(f"Failed to load cached file: {e}. Re-fetching...")

            # Fetch weather data for this chunk
            logger.info(
                f"Fetching weather data for {chunk_start_str} to {chunk_end_str}"
            )

            try:
                chunk_df = self.get_hourly_weather(
                    latitude=latitude,
                    longitude=longitude,
                    start_date=chunk_start_str,
                    end_date=chunk_end_str,
                    variables=variables,
                )

                if not chunk_df.empty:
                    # Save checkpoint
                    chunk_df.to_csv(chunk_filepath, index=False)
                    logger.info(
                        f"  ✓ Saved {len(chunk_df)} records to {chunk_filename}"
                    )
                    all_data.append(chunk_df)
                else:
                    logger.warning(
                        f"  ✗ No data for {chunk_start_str} to {chunk_end_str}"
                    )

                time.sleep(0.5)  # Rate limiting between chunks

            except Exception as e:
                logger.error(f"  ✗ Error fetching chunk: {e}")
                continue

        # Combine all chunks
        if all_data:
            logger.info(f"\nCombining {len(all_data)} chunks...")
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=["datetime"])
            combined_df = combined_df.sort_values("datetime").reset_index(drop=True)

            # Save final combined file
            final_filepath = (
                output_path / f"weather_combined_{start_date}_{end_date}.csv"
            )
            combined_df.to_csv(final_filepath, index=False)

            logger.info(f"\n✓ Weather data collection complete!")
            logger.info(f"Total records: {len(combined_df)}")
            logger.info(
                f"Date range: {combined_df['datetime'].min()} to {combined_df['datetime'].max()}"
            )
            logger.info(f"Saved to: {final_filepath.absolute()}")

            return combined_df
        else:
            logger.error("No weather data collected!")
            return pd.DataFrame()
