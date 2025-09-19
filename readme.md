# Risk-Neutral Density (RND) Estimation for Equity Risk Premium Analysis

This is a part of the ERP Study that I was working on. In the project, I extracted the Risk-Neutral Density (RND) from NIFTY 50 options data using the Breeden-Litzenberger methodology, and then subtracted the risk neutral mean return from the physical mean return estimated using Multistaged Gordon Growth Model.

## Overview

The Risk-Neutral Density represents the market's implied probability distribution of future asset prices, derived from option prices. This implementation uses:

1. **Breeden-Litzenberger Formula**: Extracts RND from option prices using second derivatives
2. **Volatility Surface Smoothing**: Uses spline interpolation to create smooth implied volatility curves
3. **Tail Modeling**: Employs Generalized Extreme Value (GEV) distribution for modeling extreme price movements
4. **Statistical Moments**: Calculates mean, variance, skewness, and kurtosis of the risk-neutral distribution

## Features

- Extract RND from real options market data
- Filter for Out-The-Money (OTM) options to reduce noise
- Smooth volatility surface using univariate splines
- Model extreme tails using GEV distribution
- Calculate risk-neutral statistical moments
- Compare extracted RND with theoretical lognormal distribution
- Batch processing for multiple expiry dates

## Requirements

```python
pandas
numpy
matplotlib
scipy
seaborn
openpyxl  # for Excel file handling
```

## Installation

```bash
git clone https://github.com/yourusername/rnd-erp-estimation.git
cd rnd-erp-estimation
pip install -r requirements.txt
```

## Data Format

The input Excel file should contain options data with the following columns:

| Column Name | Description |
|-------------|-------------|
| Date | Trading date |
| Expiry | Option expiry date |
| Strike Price | Option strike price |
| Underlying Value | Current price of underlying asset |
| Risk Free Rate | Risk-free interest rate |
| Dividend Yield (%) | Dividend yield percentage |
| Time to Maturity | Time to expiry in years |
| Option type | "CE" for Call, "PE" for Put |
| Settle Price | Option settlement/closing price |
| LTP | Last traded price (fallback for strike) |

## Usage

### Basic Usage

```python
import pandas as pd
from rnd_estimation import extract_rnd, GEV_fit

# Load your options data
df = pd.read_excel("your_options_data.xlsx")

# Extract RND for specific date and expiry
date = "2025-08-29"
expiry = "2025-09-09"
density_df, option_count = extract_rnd(df, date, expiry)

# Fit GEV distribution for tail modeling
if option_count >= 5:  # Minimum options required
    final_density = GEV_fit(density_df)
```

### Batch Processing

The main script processes all available expiry dates and generates summary statistics:

```python
python rnd_estimation.py
```

This will create `RND_results.xlsx` containing risk-neutral moments for all processed expiries.

## Methodology

### 1. Data Filtering
- Filters for Out-The-Money options (calls above spot, puts below spot)
- Removes duplicate strikes
- Requires minimum 5 options per expiry

### 2. Implied Volatility Calculation
- Uses Black-Scholes-Merton model
- Employs Brent's root-finding method
- Handles both calls and puts

### 3. Volatility Surface Smoothing
- Creates log-moneyness grid
- Applies univariate spline (degree 4) to smooth volatility surface
- Interpolates over fine strike grid

### 4. RND Extraction
- Applies Breeden-Litzenberger formula: `RND = e^(rT) * ∂²C/∂K²`
- Uses three-point central finite difference for second derivative
- Normalizes density to integrate to 1

### 5. Tail Modeling
- Fits GEV distribution to left and right tails
- Extends density beyond available option strikes
- Captures crash risk (left tail) and rally risk (right tail)

## Output

The script generates:

1. **RND_results.xlsx**: Summary statistics including:
   - Risk-neutral mean
   - Forward price
   - Standard deviation
   - Skewness
   - Kurtosis
   - Number of options used

2. **Visualization**: Comparison plots of extracted RND vs. theoretical lognormal distribution

## Key Functions

### `extract_rnd(df, date, expiry)`
Main function that extracts RND from options data.

**Parameters:**
- `df`: DataFrame with options data
- `date`: Trading date
- `expiry`: Option expiry date

**Returns:**
- `df_int`: DataFrame with interpolated density
- `filtered_count`: Number of options used

### `GEV_fit(df_int)`
Fits Generalized Extreme Value distribution to model tails.

**Parameters:**
- `df_int`: DataFrame from extract_rnd function

**Returns:**
- `density`: Combined density with GEV tails

## Known Limitations

1. **Minimum Data Requirement**: Requires at least 5 options per expiry
2. **Smoothing Parameter**: Spline smoothing uses fixed parameters that may not be optimal for all datasets
3. **Tail Fitting**: GEV fitting can be sensitive to boundary conditions
4. **Error Handling**: Basic error handling in batch processing mode

## Applications

This RND estimation can be used for:
- Equity Risk Premium calculation
- Volatility surface analysis
- Risk management and hedging
- Academic research on option-implied distributions
- Market sentiment analysis

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add improvement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

- Breeden, D. T., & Litzenberger, R. H. (1978). Prices of state-contingent claims implicit in option prices
- Bliss, R. R., & Panigirtzoglou, N. (2004). Option-implied risk aversion estimates
- Jackwerth, J. C. (2000). Recovering risk aversion from option prices and realized returns

## Contact

For questions or suggestions, please open an issue or contact [vikigamer@gmail.com]
