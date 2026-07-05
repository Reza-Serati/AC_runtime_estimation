# Ambient Sensing Dataset

This directory contains the ambient sensing dataset accompanying the paper:

> **Deep Sequence Learning Models for Hotel Room Air Conditioner Runtime Estimation via Ambient Sensing**

## Overview

The dataset is organised into **individual CSV files**, with **one file per room (sensor)**. This structure keeps each file small enough for GitHub while preserving the complete dataset.

Each file contains one-minute ambient sensing observations collected from a single hotel room.

## Directory Structure

```
data/
├── README.md
└── by_room/
    ├── sensor_101.csv
    ├── sensor_102.csv
    ├── sensor_103.csv
    ├── ...
    └── sensor_XXX.csv
```

Each `sensor_XXX.csv` file contains data from a single room/sensor.

## Data Format

Each row represents one observation collected at a one-minute interval.

| Column | Description |
|---------|-------------|
| `X.timestamp` | Timestamp of the observation |
| `sensor.temperature` | Indoor air temperature (°C) |
| `sensor.humidity` | Indoor relative humidity (%) |
| `label` | Air-conditioner operating state (0 = Off, 1 = On) |
| `sensor.number` | Unique room/sensor identifier |

## Example

| X.timestamp | sensor.temperature | sensor.humidity | label | sensor.number |
|--------------|------------------:|---------------:|------:|--------------:|
|2024-04-02 11:41:00|22.297|56.5|0|203|
|2024-04-02 11:42:00|22.406|56.5|0|203|

## Sampling Frequency

- **Sampling interval:** 1 minute
- **One file per room/sensor**

## Usage

The training pipeline automatically reads the room-level CSV files during preprocessing. Each file follows the same schema and can also be analysed independently.

## Citation

If you use this dataset in your research, please cite the accompanying publication:

> Serati, R., *Deep Sequence Learning Models for Hotel Room Air Conditioner Runtime Estimation via Ambient Sensing*.

## License

This dataset is distributed under the same license as the source code in this repository.