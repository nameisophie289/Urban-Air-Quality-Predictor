"""
Simultaneous Data Collection Script
Collects both pollutant (OpenAQ) and weather (Open-Meteo) data in parallel.
All raw data is saved to data/raw/ folder.
"""

import concurrent.futures
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import config
from src.api_clients.openaq_client import OpenAQClient
from src.api_clients.openmeteo_client import OpenMeteoClient

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_collection.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def collect_pollutant_data():
    """
    Collect pollutant data from OpenAQ API
    
    Returns:
        Tuple of (success: bool, filepath: str or None, records: int)
    """
    logger.info("="*60)
    logger.info("Starting POLLUTANT data collection (OpenAQ)")
    logger.info("="*60)
    
    try:
        # Initialize client
        client = OpenAQClient(api_key=config.OPENAQ_API_KEY)
        
        # Collection parameters
        logger.info(f"Location: {config.OPENAQ_DATA_LOCATION['name']}")
        logger.info(f"Date Range: {config.START_DATE} to {config.END_DATE}")
        logger.info(f"Parameters: {', '.join([p['name'] for p in config.OPENAQ_PARAMETERS.values()])}")
        
        # Collect data
        df = client.collect_historical_data_batch(
            parameters=config.OPENAQ_PARAMETERS,
            start_date=config.START_DATE,
            end_date=config.END_DATE,
            output_dir="data/raw/openaq_hourly",
            chunk_days=7,
            resume=True
        )
        
        if not df.empty:
            filepath = f"data/raw/openaq_hourly/combined_{config.START_DATE}_{config.END_DATE}.csv"
            logger.info(f"✓ Pollutant data collection complete: {len(df)} records")
            return True, filepath, len(df)
        else:
            logger.error("✗ No pollutant data collected")
            return False, None, 0
            
    except Exception as e:
        logger.error(f"✗ Pollutant data collection failed: {e}")
        return False, None, 0


def collect_weather_data():
    """
    Collect weather data from Open-Meteo API
    
    Returns:
        Tuple of (success: bool, filepath: str or None, records: int)
    """
    logger.info("="*60)
    logger.info("Starting WEATHER data collection (Open-Meteo)")
    logger.info("="*60)
    
    try:
        # Initialize client
        client = OpenMeteoClient(timezone="America/Los_Angeles")
        
        # Get location from pollutant data source (same location)
        latitude = config.OPENAQ_DATA_LOCATION["coordinates"]["latitude"]
        longitude = config.OPENAQ_DATA_LOCATION["coordinates"]["longitude"]
        
        logger.info(f"Location: {config.OPENAQ_DATA_LOCATION['name']}")
        logger.info(f"Coordinates: ({latitude}, {longitude})")
        logger.info(f"Date Range: {config.START_DATE} to {config.END_DATE}")
        
        # Collect data
        df = client.collect_historical_data_batch(
            latitude=latitude,
            longitude=longitude,
            start_date=config.START_DATE,
            end_date=config.END_DATE,
            output_dir="data/raw/weather_hourly",
            chunk_days=30,  # Larger chunks for weather (no rate limit issues)
            resume=True
        )
        
        if not df.empty:
            filepath = f"data/raw/weather_hourly/weather_combined_{config.START_DATE}_{config.END_DATE}.csv"
            logger.info(f"✓ Weather data collection complete: {len(df)} records")
            return True, filepath, len(df)
        else:
            logger.error("✗ No weather data collected")
            return False, None, 0
            
    except Exception as e:
        logger.error(f"✗ Weather data collection failed: {e}")
        return False, None, 0


def collect_all_data_parallel():
    """
    Collect both pollutant and weather data in parallel
    """
    logger.info("="*80)
    logger.info("SIMULTANEOUS DATA COLLECTION")
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)
    logger.info("")
    
    # Run both collections in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        # Submit both tasks
        pollutant_future = executor.submit(collect_pollutant_data)
        weather_future = executor.submit(collect_weather_data)
        
        # Wait for both to complete
        pollutant_result = pollutant_future.result()
        weather_result = weather_future.result()
    
    # Print summary
    logger.info("")
    logger.info("="*80)
    logger.info("COLLECTION SUMMARY")
    logger.info("="*80)
    
    pollutant_success, pollutant_path, pollutant_records = pollutant_result
    weather_success, weather_path, weather_records = weather_result
    
    logger.info(f"\nPollutant Data:")
    if pollutant_success:
        logger.info(f"  ✓ Status: SUCCESS")
        logger.info(f"  ✓ Records: {pollutant_records:,}")
        logger.info(f"  ✓ File: {pollutant_path}")
    else:
        logger.info(f"  ✗ Status: FAILED")
    
    logger.info(f"\nWeather Data:")
    if weather_success:
        logger.info(f"  ✓ Status: SUCCESS")
        logger.info(f"  ✓ Records: {weather_records:,}")
        logger.info(f"  ✓ File: {weather_path}")
    else:
        logger.info(f"  ✗ Status: FAILED")
    
    logger.info("")
    logger.info("="*80)
    
    if pollutant_success and weather_success:
        logger.info("✓ ALL DATA COLLECTION COMPLETE!")
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Run 'python validate_and_merge_data.py' to validate and merge the data")
        logger.info("  2. Merged data will be saved to data/processed/")
    else:
        logger.info("✗ Some collections failed. Check logs for details.")
    
    logger.info("="*80)
    
    return pollutant_success and weather_success


def collect_all_data_sequential():
    """
    Collect both pollutant and weather data sequentially
    (Alternative if parallel collection causes issues)
    """
    logger.info("="*80)
    logger.info("SEQUENTIAL DATA COLLECTION")
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)
    logger.info("")
    
    # Collect pollutant data first
    pollutant_result = collect_pollutant_data()
    
    logger.info("")
    
    # Then collect weather data
    weather_result = collect_weather_data()
    
    # Print summary (same as parallel)
    logger.info("")
    logger.info("="*80)
    logger.info("COLLECTION SUMMARY")
    logger.info("="*80)
    
    pollutant_success, pollutant_path, pollutant_records = pollutant_result
    weather_success, weather_path, weather_records = weather_result
    
    logger.info(f"\nPollutant Data:")
    if pollutant_success:
        logger.info(f"  ✓ Status: SUCCESS")
        logger.info(f"  ✓ Records: {pollutant_records:,}")
        logger.info(f"  ✓ File: {pollutant_path}")
    else:
        logger.info(f"  ✗ Status: FAILED")
    
    logger.info(f"\nWeather Data:")
    if weather_success:
        logger.info(f"  ✓ Status: SUCCESS")
        logger.info(f"  ✓ Records: {weather_records:,}")
        logger.info(f"  ✓ File: {weather_path}")
    else:
        logger.info(f"  ✗ Status: FAILED")
    
    logger.info("")
    logger.info("="*80)
    
    if pollutant_success and weather_success:
        logger.info("✓ ALL DATA COLLECTION COMPLETE!")
    else:
        logger.info("✗ Some collections failed. Check logs for details.")
    
    logger.info("="*80)
    
    return pollutant_success and weather_success


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect pollutant and weather data")
    parser.add_argument(
        "--mode",
        choices=["parallel", "sequential", "pollutant", "weather"],
        default="parallel",
        help="Collection mode: parallel (default), sequential, pollutant-only, or weather-only"
    )
    
    args = parser.parse_args()
    
    if args.mode == "parallel":
        success = collect_all_data_parallel()
    elif args.mode == "sequential":
        success = collect_all_data_sequential()
    elif args.mode == "pollutant":
        success, _, _ = collect_pollutant_data()
    elif args.mode == "weather":
        success, _, _ = collect_weather_data()
    
    sys.exit(0 if success else 1)
