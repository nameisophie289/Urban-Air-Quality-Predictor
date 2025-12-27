"""
OpenAQ API v3 Client Module
Provides flexible interface for collecting air quality data from OpenAQ
"""

import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


class OpenAQClient:
    """
    Client for interacting with OpenAQ API v3

    Supports flexible parameter-based data collection for any air quality pollutant.
    """

    def __init__(self, api_key: str, base_url: str = "https://api.openaq.org/v3"):
        """
        Initialize OpenAQ API client

        Args:
            api_key: OpenAQ API key (required for v3)
            base_url: API base URL (default: https://api.openaq.org/v3)
        """
        self.base_url = base_url
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update(
            {"X-API-Key": api_key, "Accept": "application/json"}
        )

    def make_request(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        method: str = "GET",
    ) -> Dict:
        """
        General function to make requests to OpenAQ API

        Args:
            endpoint: API endpoint path (e.g., '/locations', '/sensors/2150/hours')
            params: Query parameters as dictionary
            method: HTTP method (default: GET)

        Returns:
            Response JSON as dictionary

        Raises:
            requests.exceptions.RequestException: If request fails
        """
        # Construct full URL
        if endpoint.startswith("/"):
            url = f"{self.base_url}{endpoint}"
        else:
            url = f"{self.base_url}/{endpoint}"

        try:
            if method.upper() == "GET":
                response = self.session.get(url, params=params)
            elif method.upper() == "POST":
                response = self.session.post(url, json=params)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"Error making request to {endpoint}: {e}")
            raise

    def get_locations_by_coordinates(
        self,
        latitude: float,
        longitude: float,
        radius: int = 25000,
        limit: int = 10,
    ) -> List[Dict]:
        """
        Get monitoring locations near coordinates for a specific parameter

        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            parameter_id: OpenAQ parameter ID (e.g.,2 for PM2.5, 1 for PM10)
            radius: Search radius in meters (max 25000)
            limit: Maximum number of locations to return
        Returns:
            List of location dictionaries
        """
        url = f"{self.base_url}/locations"
        params = {
            "coordinates": f"{latitude},{longitude}",
            "radius": min(radius, 25000),
            "limit": limit,
            "order_by": "id",
            "sort_order": "asc",
        }

        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            return data.get("results", [])
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching locations: {e}")
            return []

    def get_sensor_hourly_data(
        self, sensors_id: int, date_from: str, date_to: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get hourly data for a specific sensor for a date range

        Args:
            sensors_id: OpenAQ sensor ID
            date_from: Start date (YYYY-MM-DD)
            date_to: End date (YYYY-MM-DD). If None, fetches single day

        Returns:
            DataFrame with columns: datetime, value
        """
        url = f"{self.base_url}/sensors/{sensors_id}/hours"
        all_data = []

        # Parse dates
        start_date_obj = datetime.strptime(date_from, "%Y-%m-%d")
        if date_to is None:
            end_date_obj = start_date_obj + timedelta(days=1)
        else:
            end_date_obj = datetime.strptime(date_to, "%Y-%m-%d") + timedelta(days=1)

        # Calculate number of hours in the date range and set limit accordingly
        # Add buffer to ensure we get all records
        hours_in_range = int((end_date_obj - start_date_obj).total_seconds() / 3600)
        limit = max(hours_in_range + 24, 200)  # Add buffer, minimum 200

        params = {
            "datetime_from": f"{date_from}T00:00:00Z",
            "datetime_to": f"{end_date_obj.strftime('%Y-%m-%d')}T00:00:00Z",
            "limit": limit,
        }

        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            results = data.get("results", [])

            # Extract datetime and value from nested structure
            for record in results:
                try:
                    datetime_utc = (
                        record.get("period", {}).get("datetimeFrom", {}).get("utc")
                    )
                    value = record.get("value")

                    if datetime_utc and value is not None:
                        all_data.append({"datetime": datetime_utc, "value": value})
                except Exception as e:
                    logger.warning(f"Error parsing record: {e}")
                    continue

            # time.sleep(0.1)  # Rate limiting

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching sensor {sensors_id} data: {e}")

        # Convert to DataFrame
        if all_data:
            df = pd.DataFrame(all_data)
            df["datetime"] = pd.to_datetime(df["datetime"])
            return df
        else:
            return pd.DataFrame()

    def collect_parameter_data(
        self,
        latitude: float,
        longitude: float,
        parameter_id: int,
        parameter_name: str,
        datetime_from: str,
        datetime_to: str,
        radius: int = 25000,
        max_sensors: int = 1,
    ) -> pd.DataFrame:
        """
        Collect data for a single parameter near a location

        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            parameter_id: OpenAQ parameter ID
            parameter_name: Human-readable parameter name (for logging)
            datetime_from: Start date (YYYY-MM-DD)
            datetime_to: End date (YYYY-MM-DD)
            radius: Search radius in meters (max 25000)
            max_sensors: Maximum number of sensors to collect from

        Returns:
            DataFrame with columns: datetime, value, location_id, location_name, sensors_id, sensor_name
        """
        logger.info(f"Searching for {parameter_name} locations...")
        locations = self.get_locations_by_coordinates(
            latitude, longitude, parameter_id, radius
        )

        if not locations:
            logger.warning(f"No locations found for {parameter_name}")
            return pd.DataFrame()

        logger.info(
            f"Found {len(locations)} locations with {parameter_name} monitoring"
        )

        # Find location with active sensors
        location = None
        for loc in locations:
            sensors = loc.get("sensors", [])
            if sensors:
                location = loc
                break

        if not location:
            logger.warning(
                f"No locations with active sensors found for {parameter_name}"
            )
            return pd.DataFrame()

        location_name = location.get("name", "Unknown")
        location_id = location.get("id", "Unknown")
        sensors = location.get("sensors", [])

        logger.info(f"Selected location: {location_name} (ID: {location_id})")
        logger.info(f"Available sensors: {len(sensors)}")

        # Collect data from sensors
        all_sensor_data = []

        for sensor in sensors[:max_sensors]:
            sensors_id = sensor.get("id")
            sensor_name = sensor.get("name", "Unknown")

            logger.info(f"  Collecting from sensor: {sensor_name} (ID: {sensors_id})")

            df = self.get_sensor_hourly_data(sensors_id, datetime_from, datetime_to)

            if not df.empty:
                logger.info(f"  ✓ Collected {len(df)} hourly records")
                df["location_id"] = location_id
                df["location_name"] = location_name
                df["sensors_id"] = sensors_id
                df["sensor_name"] = sensor_name
                all_sensor_data.append(df)
            else:
                logger.warning(f"  ✗ No data available")

        if all_sensor_data:
            combined_df = pd.concat(all_sensor_data, ignore_index=True)
            logger.info(f"Total {parameter_name} records collected: {len(combined_df)}")
            return combined_df
        else:
            return pd.DataFrame()

    def collect_multi_parameter_data(
        self,
        latitude: float,
        longitude: float,
        parameters: Dict[str, Dict],
        datetime_from: str,
        datetime_to: str,
        radius: int = 25000,
    ) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
        """
        Collect data for multiple parameters and create combined dataset

        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            parameters: Dictionary of parameters to collect
                       Format: {"pm25": {"id": 2, "name": "PM2.5", "units": "µg/m³"}, ...}
            datetime_from: Start date (YYYY-MM-DD)
            datetime_to: End date (YYYY-MM-DD)
            radius: Search radius in meters

        Returns:
            Tuple of (individual_data_dict, combined_dataframe)
            - individual_data_dict: Dict mapping parameter keys to their DataFrames
            - combined_dataframe: Wide-format DataFrame with all parameters by datetime
        """
        all_parameter_data = {}

        for param_key, param_info in parameters.items():
            logger.info(f"\n{'='*80}")
            logger.info(
                f"Collecting {param_info['name']} ({param_info.get('units', 'N/A')})"
            )
            logger.info(f"Parameter ID: {param_info['id']}")
            logger.info(f"{'='*80}")

            df = self.collect_parameter_data(
                latitude=latitude,
                longitude=longitude,
                parameter_id=param_info["id"],
                parameter_name=param_info["name"],
                datetime_from=datetime_from,
                datetime_to=datetime_to,
                radius=radius,
            )

            if not df.empty:
                all_parameter_data[param_key] = df
                time.sleep(0.1)  # Rate limiting between parameters

        # Create combined wide-format DataFrame
        combined_data = []

        for param_key, df in all_parameter_data.items():
            if not df.empty:
                param_df = df[["datetime", "value"]].copy()
                param_df = param_df.groupby("datetime")["value"].mean().reset_index()
                param_df.columns = ["datetime", param_key]
                combined_data.append(param_df)

        if combined_data:
            combined_df = combined_data[0]
            for df in combined_data[1:]:
                combined_df = pd.merge(combined_df, df, on="datetime", how="outer")
            combined_df = combined_df.sort_values("datetime")
        else:
            combined_df = pd.DataFrame()

        return all_parameter_data, combined_df

    def collect_historical_data_batch(
        self,
        parameters: Dict[str, Dict],
        start_date: str,
        end_date: str,
        output_dir: str = "data/raw",
        chunk_days: int = 7,
        resume: bool = True,
    ) -> pd.DataFrame:
        """
        Collect historical data in weekly chunks with checkpointing

        Args:
            parameters: Dictionary of parameters with sensors_id
                       Format: {"pm25": {"sensors_id": 123, "name": "PM2.5", ...}, ...}
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            output_dir: Directory to save checkpoint files
            chunk_days: Number of days per chunk (default: 7 for weekly)
            resume: If True, skip already collected chunks

        Returns:
            Combined DataFrame with all parameters
        """
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
        logger.info(
            f"Parameters: {', '.join([p['name'] for p in parameters.values()])}"
        )
        logger.info(f"Output directory: {output_path.absolute()}")

        all_data = []

        # Process each chunk
        for chunk_start, chunk_end in tqdm(date_chunks, desc="Collecting data"):
            chunk_start_str = chunk_start.strftime("%Y-%m-%d")
            chunk_end_str = chunk_end.strftime("%Y-%m-%d")

            # Generate checkpoint filename
            chunk_filename = (
                f"{chunk_start.strftime('%Y%m%d')}_{chunk_end.strftime('%Y%m%d')}.csv"
            )
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

            # Collect data for this chunk
            logger.info(f"Fetching data for {chunk_start_str} to {chunk_end_str}")
            chunk_data = {}

            for param_key, param_info in parameters.items():
                sensors_id = param_info.get("sensors_id")
                if not sensors_id:
                    logger.warning(f"No sensors_id for {param_key}, skipping")
                    continue

                try:
                    df = self.get_sensor_hourly_data(
                        sensors_id=sensors_id,
                        date_from=chunk_start_str,
                        date_to=chunk_end_str,
                    )

                    if not df.empty:
                        # Rename value column to parameter name
                        df = df.rename(columns={"value": param_key})
                        chunk_data[param_key] = df[["datetime", param_key]]
                        logger.info(f"  ✓ {param_info['name']}: {len(df)} records")
                    else:
                        logger.warning(f"  ✗ {param_info['name']}: No data")

                    time.sleep(0.2)  # Rate limiting between parameters

                except Exception as e:
                    logger.error(f"  ✗ {param_info['name']}: Error - {e}")
                    continue

            # Merge all parameters for this chunk
            if chunk_data:
                chunk_df = list(chunk_data.values())[0]
                for df in list(chunk_data.values())[1:]:
                    chunk_df = pd.merge(chunk_df, df, on="datetime", how="outer")

                chunk_df = chunk_df.sort_values("datetime").reset_index(drop=True)

                # Save checkpoint
                chunk_df.to_csv(chunk_filepath, index=False)
                logger.info(f"  Saved checkpoint: {chunk_filename}")

                all_data.append(chunk_df)
            else:
                logger.warning(
                    f"No data collected for chunk {chunk_start_str} to {chunk_end_str}"
                )

            time.sleep(0.5)  # Rate limiting between chunks

        # Combine all chunks
        if all_data:
            logger.info(f"\nCombining {len(all_data)} chunks...")
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df = combined_df.sort_values("datetime").reset_index(drop=True)

            # Save final combined file
            final_filepath = output_path / f"combined_{start_date}_{end_date}.csv"
            combined_df.to_csv(final_filepath, index=False)
            logger.info(f"\n✓ Data collection complete!")
            logger.info(f"Total records: {len(combined_df)}")
            logger.info(
                f"Date range: {combined_df['datetime'].min()} to {combined_df['datetime'].max()}"
            )
            logger.info(f"Saved to: {final_filepath.absolute()}")

            return combined_df
        else:
            logger.error("No data collected!")
            return pd.DataFrame()
