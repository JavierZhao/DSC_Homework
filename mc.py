import numpy as np
import math
import matplotlib.pyplot as plt
from iminuit import Minuit
def background(E_min, E_max, n_bins, B):
    '''
    Input:
        E_min, E_max: min and max of the energy range
        n_bins: number of bins within the energy range
        B: background event rate, unit: number of events
    Output:
        bin_centers: array contains the central energy of the energy bins
        bkg_counts: array contains the expected number of backgrounds in each energy bin
    '''
    # Implement your code below
    # Calculate bin edges and centers
    bin_edges = np.linspace(E_min, E_max, n_bins + 1)
    energies = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    # Distribute background events evenly over the bins
    bkg_counts = np.full(n_bins, B / n_bins)
    return energies, bkg_counts
E_min = 100
E_max = 300
n_bins = 40
B = 2000
energies, bkg_counts = background(E_min, E_max, n_bins, B)
assert len(energies) == n_bins
assert math.isclose(np.sum(bkg_counts), B, abs_tol=1.0)

from scipy.stats import norm
def signal(E_min, E_max, n_bins, E_0, sigma, S):
    '''
    Input:
        E_min, E_max: min and max of the energy range
        n_bins: number of bins within the energy range
        B: background event rate, unit: number of events
        E_0: the energy of the monoenergetic signal
        sigma: the width of the signal energy spectrum (assuming gaussian)
        S: total number of signal events
    Output:
        bin_centers: array contains the central energy of the energy bins
        signal_counts: array contains the expected number of signals in each energy bin
    '''
    # Implement your code below
    bin_edges = np.linspace(E_min, E_max, n_bins + 1)
    energies = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    # Compute the probability for a signal event to fall in each bin using the Gaussian CDF.
    bin_probs = norm.cdf(bin_edges[1:], loc=E_0, scale=sigma) - norm.cdf(bin_edges[:-1], loc=E_0, scale=sigma)
    signal_counts = S * bin_probs
    return energies, signal_counts
E_0 = 250
sigma = 10
S = 50
energies, signal_counts = signal(E_min, E_max, n_bins, E_0, sigma, S)
assert len(energies) == n_bins
assert math.isclose(np.sum(signal_counts), S, abs_tol=0.1)

def generate_toy_mc(E_min, E_max, n_bins, E_0, sigma, S, B):
    '''
    Input:
        E_min, E_max: min and max of the energy range
        n_bins: number of bins within the energy range
        E_0: energy of the monoenergetic signal
        sigma: width of the signal energy spectrum (assuming Gaussian)
        S: total number of signal eventss
        B: total number of background events
    Output:
        toy_data: simulated events in each bin, signal_observed + background_observed
    '''
    # Implement your code below
    energies, bkg_counts = background(E_min, E_max, n_bins, B)
    energies, signal_counts = signal(E_min, E_max, n_bins, E_0, sigma, S)
    
    # Draw signal counts first, then background counts
    signal_observed = np.random.poisson(signal_counts)
    bkg_observed = np.random.poisson(bkg_counts)
    
    
    toy_data = signal_observed + bkg_observed
    return toy_data
# Set random seed for reproducibility
np.random.seed(42)

observed_counts = generate_toy_mc(E_min, E_max, n_bins, E_0, sigma, S, B)
expected_counts = signal_counts+bkg_counts
assert observed_counts[0] == 46
assert observed_counts[-1] == 64
assert np.sum(observed_counts) == 2087

# Define the chi-square function, make use of the functions you defined previously
def chi2(S_fit, B_fit):
    '''
    Input:
        S: total number of signal events
        B: total number of background events
    Output:
        Chi square
    '''
    # Implement your code below
    energies, bkg_fit = background(E_min, E_max, n_bins, B_fit)
    energies, signal_fit = signal(E_min, E_max, n_bins, E_0, sigma, S_fit)
    expected_fit = bkg_fit + signal_fit
    
    # Compute the chi-square value:
    chi2_value = np.sum((observed_counts - expected_fit)**2 / expected_fit)
    
    return chi2_value

chi2_test = chi2(50, 2000)
assert math.isclose(chi2_test, 48.4744, abs_tol=1e-2)