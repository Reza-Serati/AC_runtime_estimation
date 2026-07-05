# Ambient Sensing Dataset

This directory contains the ambient sensing dataset used in the accompanying publication.

## File

```
all_sensor_data.csv
```

Each row corresponds to a single one-minute observation collected from an ambient sensor installed within a hotel room.

## Variables

| Column | Description |
|---------|-------------|
| X.timestamp | Observation timestamp |
| sensor.temperature | Indoor temperature (°C) |
| sensor.humidity | Indoor relative humidity (%) |
| label | Air-conditioner operating state (0 = Off, 1 = On) |
| sensor.number | Unique room/sensor identifier |

## Sampling Frequency

One observation per minute.

## Data Size

One row represents one timestamp from one room.

## Usage

The dataset can be used directly with the training pipeline contained in:

```
R/ac_estimation.R
```

No additional preprocessing is required beyond the steps implemented in the repository.

## Citation

If you use this dataset in academic work, please cite the accompanying paper.