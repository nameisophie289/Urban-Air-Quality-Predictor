# Urban Air Quality Prediction: PM2.5 Forecasting Using LSTM Networks

## Project Report

**Author:** Sophie Lam  
**Location:** Sydney - Rozelle, NSW, Australia  
**Date Range:** June 1, 2025 - December 1, 2025  

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Introduction](#introduction)
3. [Data Collection](#data-collection)
4. [Data Processing and Quality](#data-processing-and-quality)
5. [Model Architecture](#model-architecture)
6. [Results and Analysis](#results-and-analysis)
7. [Conclusions and Future Work](#conclusions-and-future-work)
8. [Technical Appendix](#technical-appendix)

---

## Executive Summary

This project develops an LSTM-based neural network for hourly PM2.5 (fine particulate matter) prediction in the Sydney urban area. The system ingests six months of air quality measurements and meteorological data, processes and validates the data pipeline, and trains a deep learning model for short-term air quality forecasting.

### Key Results

| Metric | Training Set | Test Set |
|--------|-------------|----------|
| **MAE** | 3.41 µg/m³ | 5.93 µg/m³ |
| **RMSE** | 5.88 µg/m³ | 8.09 µg/m³ |
| **R²** | 0.74 | 0.17 |

The model demonstrates strong fitting on training data but exhibits a generalization gap on unseen data, indicating opportunities for further optimization.

---

## Introduction

### Background

Air quality monitoring and forecasting are critical for public health, urban planning, and environmental policy. PM2.5 (particulate matter with diameter ≤ 2.5 micrometers) is a key indicator of air pollution, capable of penetrating deep into the respiratory system and causing serious health effects.

### Objectives

1. **Data Pipeline Development**: Create an automated system to collect air quality and weather data from multiple APIs
2. **Data Quality Assurance**: Implement validation and preprocessing pipelines for reliable model training
3. **Predictive Modeling**: Develop an LSTM neural network capable of forecasting hourly PM2.5 concentrations
4. **Operational Forecasting**: Generate 24-hour ahead predictions for practical applications

### Study Area

- **Location:** Sydney - Rozelle (Government Monitor)
- **Coordinates:** 33.8658°S, 151.1625°E
- **Data Provider:** Australia - New South Wales (Government-operated monitoring station)
- **Temporal Coverage**: 2025-06-01 07:00 UTC to 2025-12-01 23:00 UTC

### PM2.5 Time Series Overview

The following visualization shows PM2.5 concentration patterns over the last month of the study period:

![PM2.5 Time Series - Last Month](../charts/pm25_timeseries_last_month.png)

*Figure 1: PM2.5 concentration time series showing daily and hourly variations in air quality.*

---

## Data Collection

### Data Sources

#### 1. OpenAQ API v3 - Air Quality Data

OpenAQ provides open-access air quality data from government-operated monitoring stations worldwide.

**Collected Parameters:**

| Parameter | ID | Units | Purpose | Sensor ID |
|-----------|-----|-------|---------|-----------|
| PM2.5 | 2 | µg/m³ | Target variable | 21610 |
| PM10 | 1 | µg/m³ | Strong predictor | 4805 |
| NO₂ | 7 | ppm | Traffic pollution indicator | 4810 |
| SO₂ | 9 | ppm | Industrial source indicator | 4807 |
| CO | 8 | ppm | Combustion indicator | 4820 |
| O₃ | 10 | ppm | Atmospheric chemistry | 728 |

#### 2. Open-Meteo API - Weather Data

Open-Meteo provides free historical and forecast weather data without API key requirements.

**Collected Variables:**

| Category | Variables |
|----------|-----------|
| **Temperature** | temperature_2m, dew_point_2m, apparent_temperature |
| **Humidity** | relative_humidity_2m |
| **Precipitation** | precipitation, rain |
| **Pressure** | pressure_msl, surface_pressure |
| **Wind** | wind_speed_10m, wind_direction_10m, wind_gusts_10m |
| **Other** | cloud_cover, is_day, sunshine_duration |

### Data Collection Pipeline

The project implements a robust data collection system with the following features:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Data Collection Architecture                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌─────────────┐         ┌─────────────────────────────────────┐   │
│   │   OpenAQ    │────────▶│     openaq_client.py                │   │
│   │   API v3    │         │  - Hourly pollutant data            │   │
│   └─────────────┘         │  - Chunked collection (7-day)       │   │
│                           │  - Checkpoint/resume support         │   │
│                           └──────────────┬──────────────────────┘   │
│                                          │                           │
│   ┌─────────────┐         ┌──────────────▼──────────────────────┐   │
│   │ Open-Meteo  │────────▶│     openmeteo_client.py             │   │
│   │    API      │         │  - Hourly weather data              │   │
│   └─────────────┘         │  - Chunked collection (30-day)      │   │
│                           │  - Archive & forecast APIs           │   │
│                           └──────────────┬──────────────────────┘   │
│                                          │                           │
│                           ┌──────────────▼──────────────────────┐   │
│                           │       collect_data.py               │   │
│                           │  - Parallel/sequential modes        │   │
│                           │  - Concurrent data fetching          │   │
│                           │  - Logging and progress tracking     │   │
│                           └─────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Collection Features

1. **Parallel Execution**: Uses `ThreadPoolExecutor` for simultaneous API requests
2. **Chunked Processing**: Breaks long date ranges into manageable chunks (7 days for pollutants, 30 days for weather)
3. **Checkpoint/Resume**: Saves intermediate results and supports resuming interrupted collections
4. **Rate Limiting**: Implements delays between API calls to respect rate limits
5. **Error Handling**: Robust exception handling with detailed logging

---

## Data Processing and Quality

### Validation Pipeline

The `validate_and_merge_data.py` script performs comprehensive data quality checks:

```
┌────────────────────────────────────────────────────────────────┐
│                   Data Validation Pipeline                      │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  data/raw/openaq_hourly/    ──┐                                │
│  combined_*.csv                │                                │
│                                ▼                                │
│                    ┌───────────────────────┐                   │
│                    │   Load & Validate     │                   │
│                    │   - Missing values    │                   │
│                    │   - Duplicate check   │                   │
│                    │   - Hourly gap check  │                   │
│                    └──────────┬────────────┘                   │
│                               │                                 │
│  data/raw/weather_hourly/  ───┤                                │
│  weather_combined_*.csv       │                                │
│                               ▼                                 │
│                    ┌───────────────────────┐                   │
│                    │   Merge on Datetime   │                   │
│                    │   - UTC to Local TZ   │                   │
│                    │   - Inner join         │                   │
│                    └──────────┬────────────┘                   │
│                               │                                 │
│                               ▼                                 │
│                    ┌───────────────────────┐                   │
│                    │       Outputs          │                   │
│                    │ - Merged CSV          │                   │
│                    │ - Quality Report      │                   │
│                    └───────────────────────┘                   │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### Data Quality Report Summary

#### Pollutant Data

| Metric | Value |
|--------|-------|
| Total Records | 4,938 |
| Columns | 7 |
| Duplicate Timestamps | 611 |
| Hourly Gaps | 638 |

**Missing Value Rates:**

| Parameter | Missing % |
|-----------|-----------|
| datetime | 0.00% |
| PM2.5 | 2.88% |
| PM10 | 0.77% |
| NO₂ | 2.47% |
| SO₂ | 2.55% |
| CO | 3.24% |
| O₃ | 1.96% |

#### Weather Data

| Metric | Value |
|--------|-------|
| Total Records | 4,416 |
| Columns | 15 |
| Duplicate Timestamps | 0 |
| Hourly Gaps | 0 |
| Missing Values | None |

#### Merged Dataset

| Metric | Value |
|--------|-------|
| **Final Records** | 4,931 |
| **Total Features** | 21 |
| **Date Range** | 2025-06-01 07:00 UTC to 2025-12-01 23:00 UTC |

### Missing Data Handling Strategy

1. **Linear Interpolation**: Applied on time index for small gaps (< 3.3%)
2. **Forward/Backward Fill**: Used for edge cases
3. **Result**: Zero null values in processed dataset

---

## Model Architecture

### Feature Engineering

#### Input Features (24 total)

| Category | Features |
|----------|----------|
| **Pollutants (5)** | PM10, NO₂, SO₂, CO, O₃ |
| **Weather (14)** | temperature_2m, relative_humidity_2m, dew_point_2m, apparent_temperature, precipitation, rain, pressure_msl, surface_pressure, cloud_cover, wind_speed_10m, wind_direction_10m, wind_gusts_10m, is_day, sunshine_duration |
| **Temporal (5)** | hour, day_of_week, month, hour_sin, hour_cos |

#### Target Variable
- **PM2.5** (µg/m³)

#### Feature Scaling
- MinMaxScaler applied to all features and target, normalizing to [0, 1] range

### LSTM Network Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    LSTM Model Architecture                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Input Layer                                                    │
│   └── Shape: (24 timesteps × 24 features)                       │
│                                                                  │
│   LSTM Layer 1                                                   │
│   └── Units: 64, return_sequences=True                          │
│                                                                  │
│   Dropout Layer 1                                                │
│   └── Rate: 0.2                                                  │
│                                                                  │
│   LSTM Layer 2                                                   │
│   └── Units: 32                                                  │
│                                                                  │
│   Dropout Layer 2                                                │
│   └── Rate: 0.2                                                  │
│                                                                  │
│   Dense Layer 1                                                  │
│   └── Units: 32, Activation: ReLU                               │
│                                                                  │
│   Dropout Layer 3                                                │
│   └── Rate: 0.2                                                  │
│                                                                  │
│   Output Layer                                                   │
│   └── Units: 1 (PM2.5 prediction)                               │
│                                                                  │
│   Total Trainable Parameters: ~36,000                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| **Optimizer** | Adam |
| **Learning Rate** | 1e-3 |
| **Loss Function** | Mean Squared Error (MSE) |
| **Metrics** | MAE, MSE |
| **Batch Size** | Default (32) |
| **Max Epochs** | 100 |
| **Validation Split** | 20% of training data |

### Regularization Strategies

| Technique | Configuration |
|-----------|---------------|
| **Dropout** | 0.2 after each LSTM and Dense layer |
| **EarlyStopping** | Patience: 15 epochs, restores best weights |
| **ReduceLROnPlateau** | Factor: 0.5, Patience: 5 epochs |

### Training Progress

![Training Loss Curves](../charts/training_loss_curves.png)

*Figure 3: Training and validation loss curves showing model convergence over epochs.*

### Sequence Configuration

| Parameter | Value |
|-----------|-------|
| **Lookback Window** | 24 hours |
| **Prediction Horizon** | 1 hour ahead |
| **Total Sequences** | 4,907 |
| **Train/Test Split** | 70% / 30% (temporal order preserved) |
| **Training Samples** | 3,434 |
| **Test Samples** | 1,473 |

---

## Results and Analysis

### Model Performance

#### Training vs Test Metrics

| Metric | Training | Test | Gap |
|--------|----------|------|-----|
| **MAE** | 3.41 µg/m³ | 5.93 µg/m³ | +2.52 |
| **RMSE** | 5.88 µg/m³ | 8.09 µg/m³ | +2.21 |
| **R²** | 0.74 | 0.17 | -0.57 |

#### Actual vs Predicted Comparison

![Actual vs Predicted PM2.5](../charts/actual_vs_predicted.png)

*Figure 4: Comparison of actual PM2.5 values against model predictions for both training and test sets.*

### Interpretation

#### Strengths
- **Training Performance**: R² of 0.74 indicates the model captures 74% of PM2.5 variance in training data
- **Low Training MAE**: Average prediction error of 3.41 µg/m³ is acceptable for air quality applications
- **Stable Training**: Loss/MAE curves show convergence without obvious overfitting spikes

#### Challenges
- **Generalization Gap**: Significant performance degradation on test data
- **Positive but Low Test R²**: Model performance (R²=0.17) is better than mean prediction but captures limited variability on unseen data
- **Underfitting/Distribution Shift**: Model captures structural patterns but misses variability in unseen periods

### Feature Correlations

Based on correlation heatmap analysis:

![Feature Correlation Heatmap](../charts/correlation_heatmap.png)

*Figure 2: Correlation heatmap showing relationships between all features. PM10 shows the strongest correlation with PM2.5 (0.87).*

| Feature | Correlation with PM2.5 |
|---------|------------------------|
| PM10 | **0.87** (strongest) |
| CO | Moderate positive |
| NO₂ | Moderate positive |
| SO₂ | Moderate positive |
| Wind Speed | Weak negative |
| Rain | Weak negative |

### 24-Hour Forecast Sample

Using the latest 24-hour window for next-day prediction:

![24-Hour Forecast](../charts/24hour_forecast.png)

*Figure 5: 24-hour ahead PM2.5 forecast based on the most recent data, overlaid with recent historical values.*

| Metric | Value |
|--------|-------|
| **Predicted Range** | 6.6 - 7.2 µg/m³ |
| **Mean Prediction** | ~7.0 µg/m³ |
| **Standard Deviation** | 0.18 µg/m³ |
| **AQI Category** | "Good" |

---

## Conclusions and Future Work

### Key Findings

1. **Data Pipeline Success**: Robust collection and validation pipeline successfully integrates air quality and weather data from multiple sources
2. **Model Architecture**: LSTM with dropout regularization shows promise for time-series air quality prediction
3. **Training Effectiveness**: Model learns meaningful patterns from the training data
4. **Generalization Challenge**: Significant work needed to improve test set performance

### Limitations

1. **Temporal Distribution Shift**: The test period may contain different pollution patterns (seasonal, events)
2. **Limited Feature Engineering**: Additional temporal and spatial features could improve predictions
3. **Single Location**: Model trained on one monitoring station may not generalize to other areas
4. **Six-Month Window**: Longer historical data could capture more seasonal patterns