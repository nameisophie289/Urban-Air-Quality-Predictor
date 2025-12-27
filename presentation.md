# PM2.5 LSTM Notebook – Presentation Script

**What we built**
- Goal: Predict hourly PM2.5 using an LSTM fed by six months of pollution + weather data.
- Coverage: 2025-06-01 07:00 UTC to 2025-12-01 23:00 UTC; 4,931 hourly rows, 21 columns (PM, gases, weather, timestamp).
- Toolkit: Pandas/NumPy for prep, MinMax scaling, TensorFlow/Keras LSTM, sklearn metrics; plots with Matplotlib/Seaborn.

**Data checks**
- Missingness: PM/gases had small gaps (<=3.3%). Strategy—linear interpolation on the time index, then forward/backfill; result: zero nulls.
- Sanity peek: Basic info/describe confirm dtypes and ranges; sample rows show aligned pollution and weather readings.
- Trends: Last-month PM2.5 line shows day-to-day swings; correlation heatmap shows PM10 strongest at 0.87, moderate positives for CO/NO2/SO2, weak negatives for wind/rain.

**Features and target**
- Inputs (24): 5 pollutants (pm10, no2, so2, co, o3), 14 weather vars (temperature, humidity, dew point, apparent temp, precip/rain, pressure, cloud cover, wind speed/direction/gusts, is_day, sunshine), and 5 time encodings (hour, day_of_week, month, hour_sin, hour_cos).
- Target: PM2.5.
- Scaling: MinMax on all features and target to [0, 1].

**Sequence setup**
- Lookback window: 24 hours to predict the next hour.
- Generated 4,907 sequences shaped (24 time steps × 24 features); targets aligned one step ahead.
- Train/test split: 70/30 in time order (3,434 train; 1,473 test).

**Model**
- Architecture: LSTM(64, return_sequences) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(32, relu) → Dropout(0.2) → Dense(1).
- Params: ~36k trainable. Optimizer: Adam lr=1e-3. Loss: MSE. Metrics: MAE, MSE.
- Regularization: EarlyStopping (patience 15, restore best) and ReduceLROnPlateau (factor 0.5, patience 5).

**Training signal**
- Fit up to 100 epochs with 20% of train as validation; callbacks stopped early and reduced LR as needed.
- Loss/MAE curves stabilize; no obvious overfit spike, but gap remains on held-out data.

**Performance**
- Train: MAE 3.48 µg/m³, RMSE 5.42, R² 0.78.
- Test: MAE 6.14 µg/m³, RMSE 9.20, R² -0.08 → underfitting/generalization gap.
- Interpretation: Model captures structure but misses variability in unseen period.

**Predictions & visuals**
- Actual vs. predicted plots for the last ~3 months of train and test show tighter fit on train, looser on test.
- Next-day forecast: Using the latest 24-hour window, predicted the next 24 hours: range 6.6–7.2 µg/m³ (mean ~7.0, std 0.18), labeled “Good”. Overlaid on the last 72 hours of history.

**Takeaways and next moves**
- Model is lightweight and stable but not generalizing well. Next steps: tune LSTM units/dropout/LR, extend window or add seasonal signals (e.g., holidays, monthly sin/cos), try seq2seq/Temporal Convolutional/Transformer baselines, and consider ensembling with gradient boosting on lagged features.
