# Hotel Room Air Conditioner Runtime Estimation via Ambient Sensing

This repository contains the official implementation and dataset accompanying the paper:

> **Deep Sequence Learning Models for Hotel Room Air Conditioner Runtime Estimation via Ambient Sensing**

## Overview

This repository provides:

- The complete R implementation used for model development.
- The ambient sensing dataset used in the study.
- Training and evaluation scripts.
- Helper functions for preprocessing, feature engineering, model training, prediction, and evaluation.

The objective is to estimate hotel-room air-conditioner runtime using only unobtrusive ambient sensing measurements.

---

## Repository Structure

```
.
├── R/
│   ├── ac_estimation.R
│   └── ac_helper_code.R
│
├── data/
│   ├── all_sensor_data.csv
│   └── README.md
│
├── models/
├── outputs/
│
├── LICENSE
└── README.md
```

---

## Dataset

The repository includes the complete ambient sensing dataset used in this study.

Each row represents one timestamped observation from an individual room sensor.

Example:

| Timestamp | Temperature | Humidity | Label | Sensor |
|-----------|------------:|---------:|------:|---------|
|2024-04-02 11:41:00|22.297|56.5|0|203|

---

## Input Variables

| Column | Description |
|---------|-------------|
| X.timestamp | Timestamp of the observation |
| sensor.temperature | Indoor air temperature (°C) |
| sensor.humidity | Relative humidity (%) |
| label | Air-conditioner state (0 = Off, 1 = On) |
| sensor.number | Unique room/sensor identifier |

---

## Requirements

Developed using R.

Required packages include:

- keras
- tensorflow
- data.table
- lubridate
- caret
- reticulate
- tidyverse

---

## Running

1. Clone the repository.

2. Open

```
R/ac_estimation.R
```

3. Update the data path if required.

4. Run the training script.


---

## License

Released under the MIT License.