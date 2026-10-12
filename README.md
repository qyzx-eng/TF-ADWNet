# TF-ADWNet

Implementation of **TF-ADWNet: A Time-Frequency Adaptive Dynamic Weighted Network for Long-Term Time Series Forecasting**.

## Requirements

- Python: 3.12
- Other dependencies: see `environment.yml`.

Install the dependencies:
```bash
pip install -r environment.yml
```
Datasets

The experiments use ETTh1, ETTh2, ETTm1, ETTm2, Electricity, Traffic, and Weather.

Place the dataset files in the corresponding folders under dataset/:

TF-ADWNet/
└── dataset/
    ├── electricity/
    ├── ETT-small/
    ├── traffic/
    └── weather/

Dataset

Directory

ETTh1, ETTh2, ETTm1, ETTm2

dataset/ETT-small/

Electricity

dataset/electricity/

Traffic

dataset/traffic/

Weather

dataset/weather/

Before running an experiment:

Download and extract the required dataset.

Place its CSV file in the corresponding directory.

Set the dataset directory and file name in the experiment script or configuration.

Run the experiment from the project root directory.

Keep the directory names and capitalization consistent with the structure above.
