# TF-ADWNet

Implementation of **TF-ADWNet: A Time-Frequency Adaptive Dynamic Weighted Network for Long-Term Time Series Forecasting**.

## Requirements

The implementation was tested with:

- Python 3.12.8
- NumPy 2.0.1
- PyTorch 2.4.1
- PyWavelets 1.8.0

Other dependencies are specified in `environment.yml`.

Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate TF-ADWNet
```

## Datasets

The experiments use ETTh1, ETTh2, ETTm1, ETTm2, Electricity, Traffic and Weather.

Download and extract the datasets, then place their CSV files in the following directories:

```text
TF-ADWNet/
└── dataset/
    ├── electricity/
    ├── ETT-small/
    ├── traffic/
    └── weather/
```

- `ETT-small/`: ETTh1, ETTh2, ETTm1, and ETTm2.
- `electricity/`: Electricity.
- `traffic/`: Traffic.
- `weather/`: Weather.

Before running an experiment, configure the dataset directory and CSV file name in the corresponding experiment script. Run the scripts from the project root directory.
