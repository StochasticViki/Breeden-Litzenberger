"""
Risk-Neutral Density (RND) Estimation from Options Data
======================================================

This module implements the Breeden-Litzenberger methodology for extracting
risk-neutral probability densities from option prices. The implementation
includes volatility surface smoothing and tail modeling using Generalized
Extreme Value (GEV) distribution.

Author: [Your Name]
Date: [Date]
Version: 1.0

References:
- Breeden, D. T., & Litzenberger, R. H. (1978)
- Bliss, R. R., & Panigirtzoglou, N. (2004)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, lognorm
from scipy.interpolate import UnivariateSpline
from scipy.optimize import brentq, fsolve, leastsq
import seaborn as sns
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def extract_rnd(df, date, expiry):
    """
    Extract Risk-Neutral Density from options data using Breeden-Litzenberger methodology.
    
    This function processes options data to derive the market-implied probability
    distribution of future asset prices. It filters for Out-The-Money options,
    smooths the volatility surface, and calculates the risk-neutral density.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Options data containing columns:
        ['Date', 'Expiry', 'Strike Price', 'Underlying Value', 'Risk Free Rate',
         'Dividend Yield (%)', 'Time to Maturity', 'Option type', 'Settle Price', 'LTP']
    date : str
        Trading date to filter data
    expiry : str
        Option expiry date to filter data
    
    Returns:
    --------
    df_int : pandas.DataFrame
        DataFrame containing interpolated strikes, implied volatilities, BSM prices,
        and calculated risk-neutral density
    filtered_count : int
        Number of options remaining after filtering
    
    Notes:
    ------
    - Requires minimum 5 options for reliable density estimation
    - Uses OTM options only: calls above spot, puts below spot
    - Applies univariate spline smoothing to volatility surface
    - Calculates density using three-point central finite difference
    """
    
    # Data filtering and preparation
    df = df[df["Expiry"] == expiry]
    df = df[df["Date"] == date]
    raw_count = df.shape[0]
    df = df.sort_values('Strike Price').drop_duplicates('Strike Price')
    
    # Get spot price
    s0 = df['Underlying Value'].iloc[0]
    
    # Filter for Out-The-Money options to reduce noise
    otm_calls = (df["Strike Price"] >= s0) & (df["Option type"] == "CE")
    otm_puts = (df["Strike Price"] <= s0) & (df["Option type"] == "PE")
    
    df = df[otm_calls | otm_puts]
    filtered_count = df.shape[0]
    
    # Return empty result if insufficient data
    if filtered_count < 5:
        return pd.DataFrame(), filtered_count
    
    def bsm_price(S, K, T, r, q, sigma, option_type):
        """
        Black-Scholes-Merton option pricing formula.
        
        Parameters:
        -----------
        S : float
            Current stock price
        K : float
            Strike price
        T : float
            Time to maturity in years
        r : float
            Risk-free rate
        q : float
            Dividend yield
        sigma : float
            Volatility
        option_type : str
            "CE" for call, "PE" for put
        
        Returns:
        --------
        float
            Option price
        """
        if sigma <= 0:
            return np.nan
        
        d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)

        if option_type.upper() == "CE":  # Call
            return S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        else:  # Put
            return K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)

    def bsm_call_price(S, K, T, r, q, sigma):
        """
        Black-Scholes call option price - used for Breeden-Litzenberger formula.
        
        Note: We use call prices for all strikes in B-L formula, converting puts
        via put-call parity if needed.
        """
        d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        return S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

    def implied_volatility(S, K, T, r, q, settlement_price, option_type):
        """
        Calculate implied volatility using Brent's root-finding method.
        
        Parameters:
        -----------
        S, K, T, r, q : float
            Standard BSM parameters
        settlement_price : float
            Observed option price
        option_type : str
            "CE" or "PE"
        
        Returns:
        --------
        float
            Implied volatility, or NaN if calculation fails
        """
        try:
            # Define objective function for root finding
            obj = lambda sigma: bsm_price(S, K, T, r, q, sigma, option_type) - settlement_price
            
            # Search for implied volatility between 0.01% and 500%
            iv = brentq(obj, 1e-6, 5.0, maxiter=500)
            return iv
        except Exception:
            return np.nan
    
    # Calculate implied volatilities for all options
    ivs = []
    for _, row in df.iterrows():
        iv = implied_volatility(
            S=row["Underlying Value"],
            K=row["Strike Price"] if row["Strike Price"] else row["LTP"],
            T=row["Time to Maturity"],
            r=row["Risk Free Rate"],
            q=row["Dividend Yield (%)"],
            settlement_price=row["Settle Price"],
            option_type=row["Option type"]
        )
        ivs.append(iv if iv is not np.nan else np.nan)
    
    df["Implied Volatility (%)"] = ivs
    
    # Calculate log-moneyness for volatility surface interpolation
    df["Log Moneyness"] = np.log(df["Strike Price"]/df["Underlying Value"])
    df = df.sort_values('Log Moneyness')
    
    # Create volatility surface using spline interpolation
    x = df['Log Moneyness'].values
    y = df['Implied Volatility (%)'].values
    us = UnivariateSpline(x, y, k=4, s=len(x))  # Degree 4 spline with smoothing
    
    # Generate fine grid for density calculation
    lm_grid = np.arange(x[0], x[-1], 0.1/100)
    strike_grid = np.exp(lm_grid) * s0  # Convert back to strike prices
    iv_grid = us(lm_grid)
    
    # Create interpolated DataFrame
    df_int = pd.DataFrame({
        'K': strike_grid, 
        'sigma': iv_grid, 
        "log moneyness": lm_grid
    })
    
    # Add market parameters
    df_int['r'] = df['Risk Free Rate'].iloc[0]
    df_int['q'] = df['Dividend Yield (%)'].iloc[0]
    df_int['tau'] = df['Time to Maturity'].iloc[0]
    df_int['S'] = df['Underlying Value'].iloc[0]
    df_int['cp_flag'] = np.where(df_int['K'] < df_int['S'], 'PE', 'CE')
    
    # Calculate call prices for all strikes (Breeden-Litzenberger requires calls)
    df_int['bsm_price'] = bsm_call_price(
        df_int['S'], 
        df_int['K'],
        df_int['tau'], 
        df_int['r'], 
        df_int['q'], 
        df_int['sigma']
    )
    
    # Prepare for finite difference calculation
    df_int['l_bsm_price'] = df_int['bsm_price'].shift(1)  # Price at K-1
    df_int['f_bsm_price'] = df_int['bsm_price'].shift(-1)  # Price at K+1
    df_int['l_K'] = df_int['K'].shift(1)  # Strike K-1
    df_int['f_K'] = df_int['K'].shift(-1)  # Strike K+1
    df_int = df_int.sort_values("K")
    
    # Calculate Risk-Neutral Density using Breeden-Litzenberger formula:
    # RND = e^(rT) * ∂²C/∂K²
    # Using three-point central finite difference for second derivative
    df_int['density'] = np.exp(df_int['r']*df_int['tau']) * (
        df_int['f_bsm_price'] - 2*df_int['bsm_price'] + df_int['l_bsm_price']
    ) / ((df_int['f_K'] - df_int['K']) * (df_int['K'] - df_int['l_K']))
    
    # Clean negative densities (can occur due to numerical issues)
    df_int.loc[df_int['density'] < 0, 'density'] = 0
    df_int = df_int.dropna(subset=['density'])
    
    # Calculate moneyness and prepare additional columns
    df_int['moneyness'] = round(df_int['K']/df_int['S'], 2) - 1
    df_int['f_density'] = df_int['density'].shift(-1)
    
    # Calculate cumulative distribution (Risk-Neutral Probability)
    # RNP = e^(rT) * ∂C/∂K + 1
    df_int['cumulative'] = np.exp(df_int['r']*df_int['tau']) * (
        df_int['f_bsm_price'] - df_int['l_bsm_price']
    ) / (df_int['f_K'] - df_int['l_K']) + 1
    
    return df_int, filtered_count


def GEV_fit(df_int):
    """
    Fit Generalized Extreme Value (GEV) distribution to model density tails.
    
    Since OTM options can be noisy due to low trading volume and wide bid-ask spreads,
    this function extends the density using GEV distribution for both left (crash risk)
    and right (rally risk) tails.
    
    Parameters:
    -----------
    df_int : pandas.DataFrame
        Output from extract_rnd function containing density estimates
    
    Returns:
    --------
    density : pandas.DataFrame
        Combined density with GEV-modeled tails, columns ['K', 'density']
    
    Notes:
    ------
    - Uses method of moments to fit GEV parameters
    - Left tail models crash risk (extreme downward moves)
    - Right tail models rally risk (extreme upward moves)
    - May fail for insufficient or poor quality boundary data
    """
    
    def equations(p, *con):
        """System of equations for GEV parameter estimation."""
        mu, sigma, eta = p
        s1, s2, s3, c1, c2, c3 = con
        return (
            np.exp(-(1+eta*((s1-mu)/sigma))**(-1/eta)) - c1,  # RNP match
            (np.exp(1+eta)**((-1/eta)-1)) * np.exp(-(1+eta*((s2-mu)/sigma))**(-1/eta)) - c2,  # RND match
            (np.exp(1+eta)**((-1/eta)-1)) * np.exp(-(1+eta*((s3-mu)/sigma))**(-1/eta)) - c3   # Another RND match
        )

    def GEV(st, mu, sigma, eta):
        """Generalized Extreme Value probability density function."""
        return (np.exp(1+eta)**((-1/eta)-1)) * np.exp(-(1+eta*((st-mu)/sigma))**(-1/eta))

    # Fit left tail (crash risk)
    c1, s1 = 1 + df_int.iloc[1, :]['cumulative'], df_int.iloc[1, :]['K']
    c2, s2 = df_int.iloc[1, :]['density'], df_int.iloc[1, :]['K']
    c3, s3 = df_int.iloc[2, :]['density'], df_int.iloc[2, :]['K']

    con = (s1, s2, s3, c1, c2, c3)
    con = [round(i, 6) for i in con]
    con = tuple(con)

    # Solve for GEV parameters using least squares initialization
    init, n = leastsq(equations, (0.5, 0.5, 0.5), args=con)
    mu, sigma, eta = fsolve(equations, init, args=con)
    
    # Generate left tail density
    left_tail = [GEV(i, mu, sigma, eta) for i in np.arange(0.1, s1, 0.1)]
    lt = pd.DataFrame({'K': np.arange(0.1, s1, 0.1), 'density': left_tail})
    lt = lt.dropna()

    # Fit right tail (rally risk)
    c1, s1 = 1 - df_int.iloc[-2, :]['cumulative'], df_int.iloc[-2, :]['K']
    c2, s2 = df_int.iloc[-2, :]['density'], df_int.iloc[-2, :]['K']
    c3, s3 = df_int.iloc[-3, :]['density'], df_int.iloc[-3, :]['K']

    con = (s1, s2, s3, c1, c2, c3)
    con = [round(i, 6) for i in con]
    con = tuple(con)

    init, n = leastsq(equations, (1, 1, 0.1), args=con)
    mu, sigma, eta = fsolve(equations, init, args=con)
    
    # Generate right tail density
    right_tail = [GEV(i, mu, sigma, eta) for i in np.arange(s1, 1.2*df_int['S'].iloc[0], 10)]
    rt = pd.DataFrame({'K': np.arange(s1, 1.2*df_int['S'].iloc[0], 10), 'density': right_tail})
    rt = rt.dropna()

    # Combine all density components
    df_int = df_int.dropna(subset=['density'])
    density = pd.concat([lt, df_int[['K', 'density']], rt])
    
    return density


def plot_rnd_comparison(K, pdf, mean_RN, sd_RN):
    """
    Plot extracted RND against theoretical lognormal distribution.
    
    Parameters:
    -----------
    K : array-like
        Strike prices
    pdf : array-like
        Risk-neutral probability density
    mean_RN : float
        Risk-neutral mean
    sd_RN : float
        Risk-neutral standard deviation
    """
    # Calculate lognormal parameters from moments
    var_RN = sd_RN**2
    mu_ln = np.log(mean_RN**2 / np.sqrt(var_RN + mean_RN**2))
    sigma_ln = np.sqrt(np.log(1 + var_RN / mean_RN**2))
    
    # Generate lognormal PDF
    ln_pdf = lognorm.pdf(K, s=sigma_ln, scale=np.exp(mu_ln))
    ln_pdf = ln_pdf / np.trapz(ln_pdf, K)  # Normalize
    
    # Create comparison plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(K, pdf, color='tab:green', lw=2, label='Extracted RND')
    ax.plot(K, ln_pdf, color='tab:red', ls='--', lw=1.8, label='Lognormal fit (same moments)')
    ax.axvline(mean_RN, color='k', linestyle=':', label=f'RN mean = {mean_RN:.0f}')
    ax.set_xlabel('Strike Price')
    ax.set_ylabel('Density')
    ax.set_title('Risk-Neutral Density vs Lognormal Benchmark')
    ax.set_xlim(mean_RN - 4*sd_RN, mean_RN + 4*sd_RN)
    ax.set_ylim(0, max(pdf.max(), ln_pdf.max())*1.2)
    ax.legend()
    
    plt.tight_layout()
    return fig


def main():
    """
    Main execution function - processes all expiry dates and generates results.
    """
    print("Starting Risk-Neutral Density Estimation...")
    print("=" * 50)
    
    # Load data
    try:
        df = pd.read_excel("RND Workings.xlsx", sheet_name="all_data")
        print(f"Loaded data with {len(df)} rows")
    except FileNotFoundError:
        print("Error: 'RND Workings.xlsx' file not found!")
        print("Please ensure the data file is in the current directory.")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Get unique expiry dates
    dates = df["Expiry"].value_counts().sort_index().index
    print(f"Processing {len(dates)} expiry dates...")
    
    # Initialize results DataFrame
    results_df = pd.DataFrame(columns=[
        'Expiry', 'Mean', "Forward", 'Std Dev', 'Skew', 'Kurtosis', 'Counts'
    ])
    
    successful_extractions = 0
    
    for i, date in enumerate(dates):
        try:
            print(f"Processing expiry {i+1}/{len(dates)}: {date}")
            
            # Extract RND
            den, count = extract_rnd(df, "2025-08-29", date)
            
            if count < 5:
                print(f"  Insufficient options ({count} < 5), skipping...")
                continue
            
            if den.empty or np.trapz(den['density'], den['K']) <= 0:
                print(f"  Invalid density calculation, skipping...")
                continue
            
            # Calculate forward price
            F = den["S"].iloc[0] * np.exp((den["r"].iloc[0] - den["q"].iloc[0]) * den["tau"].iloc[0])
            
            # Fit GEV tails
            den = GEV_fit(den)
            
            # Extract density data
            K = den['K'].values
            pdf = den['density'].values
            
            # Normalize density
            area = np.trapz(pdf, K)
            if area <= 0:
                print(f"  Non-positive area ({area}), skipping...")
                continue
            
            pdf = pdf / area
            
            # Calculate risk-neutral moments
            mean_RN = np.trapz(K * pdf, K)
            var_RN = np.trapz((K - mean_RN)**2 * pdf, K)
            sd_RN = np.sqrt(var_RN)
            skew_RN = np.trapz((K - mean_RN)**3 * pdf, K) / (sd_RN**3) if sd_RN > 0 else np.nan
            kurt_RN = np.trapz((K - mean_RN)**4 * pdf, K) / (sd_RN**4) if sd_RN > 0 else np.nan
            
            # Store results
            new_row = pd.DataFrame([{
                'Expiry': date,
                'Mean': mean_RN,
                'Forward': F,
                'Std Dev': sd_RN,
                'Skew': skew_RN,
                'Kurtosis': kurt_RN,
                'Counts': count
            }])
            results_df = pd.concat([results_df, new_row], ignore_index=True)
            
            successful_extractions += 1
            print(f"  Success! Mean: {mean_RN:.2f}, Std Dev: {sd_RN:.2f}, Skew: {skew_RN:.3f}")
            
        except Exception as e:
            print(f"  Error processing {date}: {str(e)}")
            # Add empty row for failed processing
            new_row = pd.DataFrame([{
                'Expiry': date,
                'Mean': "",
                'Forward': "",
                'Std Dev': "",
                'Skew': "",
                'Kurtosis': "",
                'Counts': ""
            }])
            results_df = pd.concat([results_df, new_row], ignore_index=True)
    
    # Save results
    try:
        results_df.to_excel("RND_results.xlsx", index=False)
        print(f"\nResults saved to 'RND_results.xlsx'")
        print(f"Successfully processed {successful_extractions}/{len(dates)} expiries")
    except Exception as e:
        print(f"Error saving results: {e}")
    
    print("\nRisk-Neutral Density estimation completed!")


if __name__ == "__main__":
    main()