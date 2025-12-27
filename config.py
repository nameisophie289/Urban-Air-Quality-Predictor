"""
Configuration file for data collection
"""

# Target location for air quality prediction (California locations)
LOCATION = {
    "name": "Sydney",
    "country": "AU",
    "latitude": -33.8688,
    "longitude": 151.2093,
}

# Date range for historical data (1 month: Nov 1 to Dec 1, 2025)
START_DATE = "2025-06-01"
END_DATE = "2025-12-01"

# Data collection granularity
DATA_GRANULARITY = "hourly"  # Options: "hourly", "daily"

# OpenAQ API settings
OPENAQ_BASE_URL = "https://api.openaq.org/v3"
OPENAQ_API_KEY = "165b273174cbaf081286128ecdaa706aaf9bbf802ce815e87bbe41424fe6c9cc"

# OpenAQ Parameters to collect (flexible - add/remove as needed)
# ALL SENSORS FROM ONE LOCATION: Sydney - Rozelle (ID: 417)
# Government monitor with all parameters available
# Verified working sensors as of 2025-12-27
OPENAQ_PARAMETERS = {
    "pm25": {
        "id": 2,
        "name": "PM2.5",
        "units": "µg/m³",
        "purpose": "Target",
        "sensors_id": 21610,  # Sydney - Rozelle (Gov Monitor)
    },
    "pm10": {
        "id": 1,
        "name": "PM10",
        "units": "µg/m³",
        "purpose": "Strong predictor",
        "sensors_id": 4805,  # Sydney - Rozelle (Gov Monitor)
    },
    "no2": {
        "id": 7,
        "name": "NO2",
        "units": "ppm",
        "purpose": "Traffic pollution",
        "sensors_id": 4810,  # Sydney - Rozelle (Gov Monitor)
    },
    "so2": {
        "id": 9,
        "name": "SO2",
        "units": "ppm",
        "purpose": "Industrial source",
        "sensors_id": 4807,  # Sydney - Rozelle (Gov Monitor)
    },
    "co": {
        "id": 8,
        "name": "CO",
        "units": "ppm",
        "purpose": "Combustion",
        "sensors_id": 4820,  # Sydney - Rozelle (Gov Monitor)
    },
    "o3": {
        "id": 10,
        "name": "O3",
        "units": "ppm",
        "purpose": "Atmospheric chemistry",
        "sensors_id": 728,  # Sydney - Rozelle (Gov Monitor)
    },
}

# Data source location (for reference)
OPENAQ_DATA_LOCATION = {
    "id": 417,
    "name": "Sydney - Rozelle",
    "coordinates": {"latitude": -33.8658, "longitude": 151.1625},
    "provider": "Australia - New South Wales",
    "is_government_monitor": True,
}


# Default location for data collection
DEFAULT_LOCATION = {"name": "Sydney", "latitude": -33.8688, "longitude": 151.2093}

# Open-Meteo API settings
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_TIMEZONE = "Australia/Sydney"

# Hourly weather variables relevant for air quality prediction
WEATHER_VARIABLES_HOURLY = [
    "temperature_2m",  # Temperature at 2m height (°C)
    "relative_humidity_2m",  # Relative humidity at 2m (%)
    "dew_point_2m",  # Dew point at 2m (°C)
    "apparent_temperature",  # Feels like temperature (°C)
    "precipitation",  # Precipitation (mm)
    "rain",  # Rain (mm)
    "pressure_msl",  # Sea level pressure (hPa)
    "surface_pressure",  # Surface pressure (hPa)
    "cloud_cover",  # Cloud cover (%)
    "wind_speed_10m",  # Wind speed at 10m (km/h)
    "wind_direction_10m",  # Wind direction at 10m (°)
    "wind_gusts_10m",  # Wind gusts at 10m (km/h)
    "is_day",  # Is day (0/1)
    "sunshine_duration",  # Sunshine duration (s)
]


# Data storage paths
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
