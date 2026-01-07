import numpy as np
from scipy.stats import norm

def calculate_merton_pd(E, D, r, sigma_e, T):
    """
    Merton Model Probability of Default (PD) calculation.
    E: Equity Value, D: Face Value of Debt, r: Risk-free rate
    sigma_e: Equity Volatility, T: Time to Maturity
    """
    # Asset value (V) and Asset volatility (sigma_v) estimation
    V = E + D
    sigma_v = (E / V) * sigma_e
    
    # Distance to Default (d2)
    d2 = (np.log(V / D) + (r - 0.5 * sigma_v**2) * T) / (sigma_v * np.sqrt(T))
    
    # Probability of Default
    pd = norm.cdf(-d2)
    return pd

# Portfolio simulation
if __name__ == "__main__":
    print(f"Sample PD: {calculate_merton_pd(2.5e6, 1.25e6, 0.05, 0.30, 1.0):.4%}")
