import math
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy.stats import skew

# CO2 density calculation imports
try:
    import CoolProp.CoolProp as CP
    COOLPROP_AVAILABLE = True
except ImportError:
    COOLPROP_AVAILABLE = False
    st.warning("CoolProp not available. CO2 density calculation from P-T will be disabled. Install with: pip install coolprop")

# -----------------------------
# Geometric Correction Factor Calculator Functions
# -----------------------------

def get_gcf_lookup_table():
    """Create the geometric correction factor lookup table based on the Excel analysis"""
    
    # Define the Res Tk/Closure ratios (x-axis) - FROM EXCEL ARK2 SHEET
    # Complete sequence from 1.0 to 0.0 with 0.01 intervals
    res_tk_closure_ratios = np.array([
        1.0, 0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.9, 0.89, 0.88, 0.87, 0.86, 0.85, 0.84, 0.83, 0.82, 0.81,
        0.8, 0.79, 0.78, 0.77, 0.76, 0.75, 0.74, 0.73, 0.72, 0.71, 0.7, 0.69, 0.68, 0.67, 0.66, 0.65, 0.64, 0.63, 0.62, 0.61,
        0.6, 0.59, 0.58, 0.57, 0.56, 0.55, 0.54, 0.53, 0.52, 0.51, 0.5, 0.49, 0.48, 0.47, 0.46, 0.45, 0.44, 0.43, 0.42, 0.41,
        0.4, 0.39, 0.38, 0.37, 0.36, 0.35, 0.34, 0.33, 0.32, 0.31, 0.3, 0.29, 0.28, 0.27, 0.26, 0.25, 0.24, 0.23, 0.22, 0.21,
        0.2, 0.19, 0.18, 0.17, 0.16, 0.15, 0.14, 0.13, 0.12, 0.11, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.0
    ])
    
    # Define GCF values for different shape types and L/W ratios - FROM EXCEL ARK2 SHEET
    # CORRECTED: As Reservoir Thickness/Closure Ratio increases, GCF decreases
    
    # Shape 1: Dome/Cone/Pyramid (L/W=1) - UPDATED FROM CSV
    gcf_dome = np.array([
        0.335, 0.3385, 0.342, 0.3455, 0.349, 0.3525, 0.356, 0.3595, 0.363, 0.3665, 0.37, 0.3745, 0.379, 0.3835, 0.388, 0.3925, 0.397, 0.4015, 0.406, 0.4105,
        0.415, 0.42, 0.425, 0.43, 0.435, 0.44, 0.445, 0.45, 0.455, 0.46, 0.465, 0.471, 0.477, 0.483, 0.489, 0.495, 0.501, 0.507, 0.513, 0.519,
        0.525, 0.53175, 0.5385, 0.54525, 0.552, 0.55875, 0.5655, 0.57225, 0.579, 0.58575, 0.5925, 0.59925, 0.606, 0.61275, 0.6195, 0.62625, 0.633, 0.63975, 0.6465, 0.65325,
        0.66, 0.668, 0.676, 0.684, 0.692, 0.7, 0.708, 0.716, 0.724, 0.732, 0.74, 0.74875, 0.7575, 0.76625, 0.775, 0.78375, 0.7925, 0.80125, 0.81, 0.81875,
        0.8275, 0.83675, 0.846, 0.85525, 0.8645, 0.87375, 0.883, 0.89225, 0.9015, 0.91075, 0.92, 0.928, 0.936, 0.944, 0.952, 0.96, 0.968, 0.976, 0.984, 0.992, 1.0
    ])
    
    # Shape 2: Anticline (L/W=2,5,10) - UPDATED FROM CSV
    gcf_anticline_2 = np.array([
        0.42, 0.424, 0.428, 0.432, 0.436, 0.44, 0.444, 0.448, 0.452, 0.456, 0.46, 0.4645, 0.469, 0.4735, 0.478, 0.4825, 0.487, 0.4915, 0.496, 0.5005,
        0.505, 0.51, 0.515, 0.52, 0.525, 0.53, 0.535, 0.54, 0.545, 0.55, 0.555, 0.5605, 0.566, 0.5715, 0.577, 0.5825, 0.588, 0.5935, 0.599, 0.6045,
        0.61, 0.616, 0.622, 0.628, 0.634, 0.64, 0.646, 0.652, 0.658, 0.664, 0.67, 0.676, 0.682, 0.688, 0.694, 0.7, 0.706, 0.712, 0.718, 0.724,
        0.73, 0.73687, 0.74374, 0.75061, 0.75748, 0.76435, 0.77122, 0.77809, 0.78496, 0.79183, 0.7987, 0.80558, 0.81246, 0.81934, 0.82622, 0.8331, 0.83998, 0.84686, 0.85374, 0.86062,
        0.8675, 0.87425, 0.881, 0.88775, 0.8945, 0.90125, 0.908, 0.91475, 0.9215, 0.92825, 0.935, 0.9415, 0.948, 0.9545, 0.961, 0.9675, 0.974, 0.9805, 0.987, 0.9935, 1.0
    ])
    
    gcf_anticline_5 = np.array([
        0.465, 0.46925, 0.4735, 0.47775, 0.482, 0.48625, 0.4905, 0.49475, 0.499, 0.50325, 0.5075, 0.512, 0.5165, 0.521, 0.5255, 0.53, 0.5345, 0.539, 0.5435, 0.548,
        0.5525, 0.5575, 0.5625, 0.5675, 0.5725, 0.5775, 0.5825, 0.5875, 0.5925, 0.5975, 0.6025, 0.60775, 0.613, 0.61825, 0.6235, 0.62875, 0.634, 0.63925, 0.6445, 0.64975,
        0.655, 0.6605, 0.666, 0.6715, 0.677, 0.6825, 0.688, 0.6935, 0.699, 0.7045, 0.71, 0.7155, 0.721, 0.7265, 0.732, 0.7375, 0.743, 0.7485, 0.754, 0.7595,
        0.765, 0.770935, 0.77687, 0.782805, 0.78874, 0.794675, 0.80061, 0.806545, 0.81248, 0.818415, 0.82435, 0.83029, 0.83623, 0.84217, 0.84811, 0.85405, 0.85999, 0.86593, 0.87187, 0.87781,
        0.88375, 0.889625, 0.8955, 0.901375, 0.90725, 0.913125, 0.919, 0.924875, 0.93075, 0.936625, 0.9425, 0.94825, 0.954, 0.95975, 0.9655, 0.97125, 0.977, 0.98275, 0.9885, 0.99425, 1.0
    ])
    
    gcf_anticline_10 = np.array([
        0.51, 0.5145, 0.519, 0.5235, 0.528, 0.5325, 0.537, 0.5415, 0.546, 0.5505, 0.555, 0.5595, 0.564, 0.5685, 0.573, 0.5775, 0.582, 0.5865, 0.591, 0.5955,
        0.6, 0.605, 0.61, 0.615, 0.62, 0.625, 0.63, 0.635, 0.64, 0.645, 0.65, 0.655, 0.66, 0.665, 0.67, 0.675, 0.68, 0.685, 0.69, 0.695,
        0.7, 0.705, 0.71, 0.715, 0.72, 0.725, 0.73, 0.735, 0.74, 0.745, 0.75, 0.755, 0.76, 0.765, 0.77, 0.775, 0.78, 0.785, 0.79, 0.795,
        0.8, 0.805, 0.81, 0.815, 0.82, 0.825, 0.83, 0.835, 0.84, 0.845, 0.85, 0.855, 0.86, 0.865, 0.87, 0.875, 0.88, 0.885, 0.89, 0.895,
        0.9, 0.905, 0.91, 0.915, 0.92, 0.925, 0.93, 0.935, 0.94, 0.945, 0.95, 0.955, 0.96, 0.965, 0.97, 0.975, 0.98, 0.985, 0.99, 0.995, 1.0
    ])
    
    # Shape 3: Flat-top Dome (L/W=1) - UPDATED FROM CSV
    gcf_flat_dome = np.array([
        0.59, 0.59425, 0.5985, 0.60275, 0.607, 0.61125, 0.6155, 0.61975, 0.624, 0.62825, 0.6325, 0.63675, 0.641, 0.64525, 0.6495, 0.65375, 0.658, 0.66225, 0.6665, 0.67075,
        0.675, 0.679, 0.683, 0.687, 0.691, 0.695, 0.699, 0.703, 0.707, 0.711, 0.715, 0.71925, 0.7235, 0.72775, 0.732, 0.73625, 0.7405, 0.74475, 0.749, 0.75325,
        0.7575, 0.76175, 0.766, 0.77025, 0.7745, 0.77875, 0.783, 0.78725, 0.7915, 0.79575, 0.8, 0.804, 0.808, 0.812, 0.816, 0.82, 0.824, 0.828, 0.832, 0.836,
        0.84, 0.84425, 0.8485, 0.85275, 0.857, 0.86125, 0.8655, 0.86975, 0.874, 0.87825, 0.8825, 0.8865, 0.8905, 0.8945, 0.8985, 0.9025, 0.9065, 0.9105, 0.9145, 0.9185,
        0.9225, 0.92675, 0.931, 0.93525, 0.9395, 0.94375, 0.948, 0.95225, 0.9565, 0.96075, 0.965, 0.9685, 0.972, 0.9755, 0.979, 0.9825, 0.986, 0.9895, 0.993, 0.9965, 1.0
    ])
    
    # Shape 4: Flat-top Anticline (L/W=2,5,10) - UPDATED FROM CSV
    gcf_flat_anticline_2 = np.array([
        0.675, 0.6785, 0.682, 0.6855, 0.689, 0.6925, 0.696, 0.6995, 0.703, 0.7065, 0.71, 0.713, 0.716, 0.719, 0.722, 0.725, 0.728, 0.731, 0.734, 0.737,
        0.74, 0.7435, 0.747, 0.7505, 0.754, 0.7575, 0.761, 0.7645, 0.768, 0.7715, 0.775, 0.7785, 0.782, 0.7855, 0.789, 0.7925, 0.796, 0.7995, 0.803, 0.8065,
        0.81, 0.8135, 0.817, 0.8205, 0.824, 0.8275, 0.831, 0.8345, 0.838, 0.8415, 0.845, 0.84825, 0.8515, 0.85475, 0.858, 0.86125, 0.8645, 0.86775, 0.871, 0.87425,
        0.8775, 0.88075, 0.884, 0.88725, 0.8905, 0.89375, 0.897, 0.90025, 0.9035, 0.90675, 0.91, 0.9135, 0.917, 0.9205, 0.924, 0.9275, 0.931, 0.9345, 0.938, 0.9415,
        0.945, 0.948, 0.951, 0.954, 0.957, 0.96, 0.963, 0.966, 0.969, 0.972, 0.975, 0.9775, 0.98, 0.9825, 0.985, 0.9875, 0.99, 0.9925, 0.995, 0.9975, 1.0
    ])
    
    gcf_flat_anticline_5 = np.array([
        0.7075, 0.71075, 0.714, 0.71725, 0.7205, 0.72375, 0.727, 0.73025, 0.7335, 0.73675, 0.74, 0.743, 0.746, 0.749, 0.752, 0.755, 0.758, 0.761, 0.764, 0.767,
        0.77, 0.77325, 0.7765, 0.77975, 0.783, 0.78625, 0.7895, 0.79275, 0.796, 0.79925, 0.8025, 0.80575, 0.809, 0.81225, 0.8155, 0.81875, 0.822, 0.82525, 0.8285, 0.83175,
        0.835, 0.83825, 0.8415, 0.84475, 0.848, 0.85125, 0.8545, 0.85775, 0.861, 0.86425, 0.8675, 0.87025, 0.873, 0.87575, 0.8785, 0.88125, 0.884, 0.88675, 0.8895, 0.89225,
        0.895, 0.897875, 0.90075, 0.903625, 0.9065, 0.909375, 0.91225, 0.915125, 0.918, 0.920875, 0.92375, 0.92675, 0.92975, 0.93275, 0.93575, 0.93875, 0.94175, 0.94475, 0.94775, 0.95075,
        0.95375, 0.956375, 0.959, 0.961625, 0.96425, 0.966875, 0.9695, 0.972125, 0.97475, 0.977375, 0.98, 0.982, 0.984, 0.986, 0.988, 0.99, 0.992, 0.994, 0.996, 0.998, 1.0
    ])
    
    gcf_flat_anticline_10 = np.array([
        0.74, 0.743, 0.746, 0.749, 0.752, 0.755, 0.758, 0.761, 0.764, 0.767, 0.77, 0.773, 0.776, 0.779, 0.782, 0.785, 0.788, 0.791, 0.794, 0.797,
        0.8, 0.803, 0.806, 0.809, 0.812, 0.815, 0.818, 0.821, 0.824, 0.827, 0.83, 0.833, 0.836, 0.839, 0.842, 0.845, 0.848, 0.851, 0.854, 0.857, 0.86,
        0.863, 0.866, 0.869, 0.872, 0.875, 0.878, 0.881, 0.884, 0.887, 0.89, 0.893, 0.896, 0.899, 0.902, 0.905, 0.908, 0.911, 0.914, 0.917, 0.92,
        0.923, 0.926, 0.929, 0.932, 0.935, 0.938, 0.941, 0.944, 0.947, 0.95, 0.953, 0.955, 0.957, 0.959, 0.961, 0.963, 0.965, 0.967, 0.969, 0.971, 0.973,
        0.975, 0.97675, 0.9785, 0.98025, 0.982, 0.98375, 0.9855, 0.98725, 0.989, 0.99075, 0.9925, 0.99375, 0.995, 0.99625, 0.9975, 0.99875, 1.0, 1.0, 1.0, 1.0, 1.0
    ])
    
    # Shape 5: Block (L/W=W) - Always 1.0
    gcf_block = np.array([1.0] * len(res_tk_closure_ratios))
    
    # Create lookup dictionary
    lookup_table = {
        (1, 1): gcf_dome,
        (2, 2): gcf_anticline_2,
        (2, 5): gcf_anticline_5,
        (2, 10): gcf_anticline_10,
        (3, 1): gcf_flat_dome,
        (4, 2): gcf_flat_anticline_2,
        (4, 5): gcf_flat_anticline_5,
        (4, 10): gcf_flat_anticline_10,
        (5, 1): gcf_block  # Block uses L/W=1 for lookup
    }
    
    return res_tk_closure_ratios, lookup_table

def interpolate_gcf(shape_type, lw_ratio, res_tk_closure_ratio):
    """Interpolate GCF value based on shape type, L/W ratio, and Res Tk/Closure ratio"""
    
    res_tk_closure_ratios, lookup_table = get_gcf_lookup_table()
    
    # Get the appropriate GCF curve
    if (shape_type, lw_ratio) in lookup_table:
        gcf_curve = lookup_table[(shape_type, lw_ratio)]
    else:
        # For shape type 5 (Block), use L/W=1
        if shape_type == 5:
            gcf_curve = lookup_table[(5, 1)]
        else:
            # Default to dome if not found
            gcf_curve = lookup_table[(1, 1)]
    
    # Interpolate to find GCF value
    if res_tk_closure_ratio <= 0:
        return gcf_curve[0]
    elif res_tk_closure_ratio >= 1:
        return gcf_curve[-1]
    else:
        # Since res_tk_closure_ratios is in descending order (1.0 to 0.0),
        # we need to handle the search differently
        # Find the index where the ratio would be inserted to maintain descending order
        for i in range(len(res_tk_closure_ratios)):
            if res_tk_closure_ratios[i] <= res_tk_closure_ratio:
                idx = i
                break
        else:
            idx = len(res_tk_closure_ratios) - 1
        
        # Handle edge cases
        if idx == 0:
            return gcf_curve[0]
        elif idx >= len(res_tk_closure_ratios) - 1:
            return gcf_curve[-1]
        else:
            # Linear interpolation
            x0, x1 = res_tk_closure_ratios[idx-1], res_tk_closure_ratios[idx]
            y0, y1 = gcf_curve[idx-1], gcf_curve[idx]
            return y0 + (y1 - y0) * (res_tk_closure_ratio - x0) / (x1 - x0)


# -----------------------------
# Sampling utilities
# -----------------------------

def sample_uniform(low: float, high: float, n: int) -> np.ndarray:
	low, high = float(low), float(high)
	if high < low:
		low, high = high, low
	return np.random.uniform(low, high, size=n)


def sample_triangular(low: float, mode: float, high: float, n: int) -> np.ndarray:
	low, high = float(low), float(high)
	if high < low:
		low, high = high, low
	mode = min(max(mode, low), high)
	return np.random.triangular(low, mode, high, size=n)


def sample_pert(min_v: float, mode_v: float, max_v: float, n: int, lam: float = 4.0) -> np.ndarray:
	a, b, c = float(min_v), float(mode_v), float(max_v)
	if b < a:
		b = a
	if c < b:
		c = b
	if c == a:
		return np.full(n, a)
	alpha = 1.0 + lam * (b - a) / (c - a)
	beta = 1.0 + lam * (c - b) / (c - a)
	y = np.random.beta(alpha, beta, size=n)
	return a + y * (c - a)


def sample_lognormal_mean_sd(mean: float, sd: float, n: int) -> np.ndarray:
	mean = float(mean)
	sd = float(sd)
	if mean <= 0 or sd <= 0:
		return np.full(n, max(mean, 1e-9))
	sigma2 = math.log(1.0 + (sd * sd) / (mean * mean))
	sigma = math.sqrt(sigma2)
	mu = math.log(mean) - 0.5 * sigma2
	return np.random.lognormal(mean=mu, sigma=sigma, size=n)


def sample_beta_subjective(mode_v: float, mean_v: float, min_v: float, max_v: float, n: int) -> np.ndarray:
	a, b = float(min_v), float(max_v)
	mu_x = float(mean_v)
	mode_x = float(mode_v)

	if b < a:
		a, b = b, a
	if b == a:
		return np.full(n, a)

	mu = (mu_x - a) / (b - a)
	mo = (mode_x - a) / (b - a)

	mu = min(max(mu, 1e-6), 1 - 1e-6)
	mo = min(max(mo, 1e-6), 1 - 1e-6)

	den = (mu - mo)
	if abs(den) < 1e-8:
		s = 6.0
	else:
		s = (1.0 - 2.0 * mo) / den
		if s <= 2.0001:
			s = 2.0001 + (2.0 - s) + 2.0

	alpha = mu * s
	beta = s - alpha

	alpha = max(alpha, 1e-3)
	beta = max(beta, 1e-3)

	y = np.random.beta(alpha, beta, size=n)
	return a + y * (b - a)


# -----------------------------
# Correlation sampling utilities
# -----------------------------

def apply_correlation(x: np.ndarray, y: np.ndarray, correlation: float) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Apply correlation between two arrays using Cholesky decomposition.
	
	Args:
		x: First array of samples
		y: Second array of samples  
		correlation: Correlation coefficient between -1 and 1
		
	Returns:
		Tuple of (correlated_x, correlated_y)
	"""
	if abs(correlation) < 1e-6:
		return x, y
	
	# Ensure correlation is within bounds
	correlation = max(-0.99, min(0.99, correlation))
	
	# Create correlation matrix
	corr_matrix = np.array([[1.0, correlation], [correlation, 1.0]])
	
	# Cholesky decomposition
	try:
		L = np.linalg.cholesky(corr_matrix)
	except np.linalg.LinAlgError:
		# If correlation matrix is not positive definite, use SVD
		U, s, Vt = np.linalg.svd(corr_matrix)
		L = U @ np.sqrt(np.diag(s)) @ Vt
	
	# Generate correlated normal random variables
	z = np.random.standard_normal((2, len(x)))
	correlated_normal = L @ z
	
	# Transform to uniform [0,1] using normal CDF
	from scipy.stats import norm
	u1 = norm.cdf(correlated_normal[0])
	u2 = norm.cdf(correlated_normal[1])
	
	# Apply inverse CDF to get correlated samples with original distributions
	# For x: use empirical inverse CDF
	x_sorted = np.sort(x)
	x_indices = (u1 * (len(x) - 1)).astype(int)
	correlated_x = x_sorted[x_indices]
	
	# For y: use empirical inverse CDF  
	y_sorted = np.sort(y)
	y_indices = (u2 * (len(y) - 1)).astype(int)
	correlated_y = y_sorted[y_indices]
	
	return correlated_x, correlated_y


def sample_correlated_parameters(param1_config: Dict[str, Any], param2_config: Dict[str, Any], 
								correlation: float, n: int) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Sample two parameters with specified correlation.
	
	Args:
		param1_config: Configuration for first parameter (dist, params)
		param2_config: Configuration for second parameter (dist, params) 
		correlation: Correlation coefficient between -1 and 1
		n: Number of samples
		
	Returns:
		Tuple of (param1_samples, param2_samples)
	"""
	# Sample independently first
	if param1_config["dist"] == "PERT":
		param1_samples = sample_pert(param1_config["min"], param1_config["mode"], param1_config["max"], n)
	elif param1_config["dist"] == "Triangular":
		param1_samples = sample_triangular(param1_config["min"], param1_config["mode"], param1_config["max"], n)
	elif param1_config["dist"] == "Uniform":
		param1_samples = sample_uniform(param1_config["min"], param1_config["max"], n)
	elif param1_config["dist"] == "Lognormal (mean, sd)":
		param1_samples = sample_lognormal_mean_sd(param1_config["mean"], param1_config["sd"], n)
	elif param1_config["dist"] == "Subjective Beta (Vose)":
		param1_samples = sample_beta_subjective(param1_config["mode"], param1_config["mean"], param1_config["min"], param1_config["max"], n)
	else:
		param1_samples = np.zeros(n)
	
	if param2_config["dist"] == "PERT":
		param2_samples = sample_pert(param2_config["min"], param2_config["mode"], param2_config["max"], n)
	elif param2_config["dist"] == "Triangular":
		param2_samples = sample_triangular(param2_config["min"], param2_config["mode"], param2_config["max"], n)
	elif param2_config["dist"] == "Uniform":
		param2_samples = sample_uniform(param2_config["min"], param2_config["max"], n)
	elif param2_config["dist"] == "Lognormal (mean, sd)":
		param2_samples = sample_lognormal_mean_sd(param2_config["mean"], param2_config["sd"], n)
	elif param2_config["dist"] == "Subjective Beta (Vose)":
		param2_samples = sample_beta_subjective(param2_config["mode"], param2_config["mean"], param2_config["min"], param2_config["max"], n)
	else:
		param2_samples = np.zeros(n)
	
	# Apply correlation if not zero
	if abs(correlation) > 1e-6:
		param1_samples, param2_samples = apply_correlation(param1_samples, param2_samples, correlation)
	
	return param1_samples, param2_samples


# -----------------------------
# CO2 Density Calculation Functions
# -----------------------------

def classify_phase(T: float, p: float) -> str:
	"""Classify CO2 phase using CoolProp. Returns one of: solid, liquid, gas, supercritical, or unknown.
	
	Parameters are absolute temperature [K] and pressure [Pa].
	"""
	if not COOLPROP_AVAILABLE:
		return "unknown"
	
	try:
		phase = CP.PhaseSI("T", T, "P", p, "CO2")
	except Exception:
		return "unknown"

	s = str(phase).lower()

	# Normalize to our 4 categories
	if "supercritical" in s:
		return "supercritical"
	if "solid" in s:
		return "solid"
	if "liquid" in s:
		return "liquid"
	if "gas" in s or "vapor" in s:
		return "gas"
	return "unknown"


def compute_onshore_state(
	GL: Optional[float],
	topdepth: Optional[float],
	basedepth: Optional[float],
	avgmudline: Optional[float],
	GT_grad: float,
	a_surftemp: float,
) -> Tuple[float, float]:
	"""Compute (T[K], P[MPa]) for onshore scenario.
	
	Option a (avgmudline provided):
	  P[MPa] = 9.81 * avgmudline * 1000 / 1e6
	  T[K]   = (avgmudline/1000) * GT_grad + a_surftemp
	Option b (from depths):
	  mean_depth_msl = GL + topdepth + (basedepth - topdepth)/2
	  P[MPa] = 9.81 * mean_depth_msl * 1000 / 1e6
	  T[K]   = (mean_depth_msl/1000) * GT_grad + a_surftemp
	"""
	if avgmudline is not None:
		pressure_mpa = 9.81 * avgmudline * 1000.0 / 1_000_000.0
		temperature_k = (avgmudline / 1000.0) * GT_grad + a_surftemp
		return float(temperature_k), float(pressure_mpa)

	# Option b
	if GL is None or topdepth is None or basedepth is None:
		raise ValueError("Onshore: need either avgmudline or GL+topdepth+basedepth")
	mean_depth_msl = float(GL + topdepth + (basedepth - topdepth) / 2.0)
	pressure_mpa = 9.81 * mean_depth_msl * 1000.0 / 1_000_000.0
	temperature_k = (mean_depth_msl / 1000.0) * GT_grad + a_surftemp
	return float(temperature_k), float(pressure_mpa)


def compute_offshore_state(
	waterdepth: Optional[float],
	topdepth: Optional[float],
	basedepth: Optional[float],
	avgmudline: Optional[float],
	GT_grad: float,
	a_seabtemp: float,
) -> Tuple[float, float]:
	"""Compute (T[K], P[MPa]) for offshore scenario.
	
	Option a (avgmudline provided):
	  P[MPa] = 9.81 * (avgmudline + waterdepth) * 1000 / 1e6
	  T[K]   = (avgmudline/1000) * GT_grad + a_seabtemp
	Option b (from depths):
	  mean_depth_msl = topdepth + (basedepth - topdepth)/2
	  P[MPa] = 9.81 * mean_depth_msl * 1000 / 1e6
	  T[K]   = (mean_depth_msl/1000) * GT_grad + a_seabtemp
	"""
	# Option a
	if avgmudline is not None:
		if waterdepth is None:
			raise ValueError("Offshore: waterdepth is required when avgmudline is provided")
		pressure_mpa = 9.81 * (avgmudline + waterdepth) * 1000.0 / 1_000_000.0
		temperature_k = (avgmudline / 1000.0) * GT_grad + a_seabtemp
		return float(temperature_k), float(pressure_mpa)

	# Option b
	if topdepth is None or basedepth is None:
		raise ValueError("Offshore: need either avgmudline+waterdepth or topdepth+basedepth")
	mean_depth_msl = float(topdepth + (basedepth - topdepth) / 2.0)
	pressure_mpa = 9.81 * mean_depth_msl * 1000.0 / 1_000_000.0
	temperature_k = (mean_depth_msl / 1000.0) * GT_grad + a_seabtemp
	return float(temperature_k), float(pressure_mpa)


def calculate_co2_density_from_pt(T_k: float, P_mpa: float) -> Tuple[float, str]:
	"""Calculate CO2 density from temperature and pressure using CoolProp.
	
	Returns (density_kg_m3, phase).
	"""
	if not COOLPROP_AVAILABLE:
		return float('nan'), "unknown"
	
	try:
		P_pa = P_mpa * 1e6
		density_kg_m3 = CP.PropsSI("D", "T", T_k, "P", P_pa, "CO2")
		phase = classify_phase(T_k, P_pa)
		return float(density_kg_m3), phase
	except Exception:
		return float('nan'), "unknown"


def sample_co2_density_from_pt_distributions(
	T_samples: np.ndarray,
	P_samples: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
	"""Calculate CO2 density for arrays of temperature and pressure samples.
	
	Returns (density_samples, phase_samples).
	"""
	if not COOLPROP_AVAILABLE:
		return np.full_like(T_samples, float('nan')), np.full(len(T_samples), "unknown")
	
	density_samples = np.full_like(T_samples, float('nan'))
	phase_samples = np.full(len(T_samples), "unknown", dtype=object)
	
	for i, (T, P) in enumerate(zip(T_samples, P_samples)):
		density, phase = calculate_co2_density_from_pt(T, P)
		density_samples[i] = density
		phase_samples[i] = phase
	
	return density_samples, phase_samples


# -----------------------------
# Plotting and stats
# -----------------------------

def make_depth_area_plot(df: pd.DataFrame) -> go.Figure:
	"""
	Create a depth vs area plot showing top and base areas with colored area between them.
	Uses full precision data for calculations and displays areas in km².
	"""
	fig = go.Figure()
	
	# Add top area line with dots
	fig.add_trace(go.Scatter(
		x=df["Top area (km2)"],
		y=df["Depth"],
		mode="lines+markers",
		name="Top Area",
		line=dict(color="blue", width=2),
		marker=dict(size=8, color="blue")
	))
	
	# Add base area line with dots
	fig.add_trace(go.Scatter(
		x=df["Base area (km2)"],
		y=df["Depth"],
		mode="lines+markers",
		name="Base Area",
		line=dict(color="red", width=2),
		marker=dict(size=8, color="red")
	))
	
	# Add filled area between top and base
	fig.add_trace(go.Scatter(
		x=df["Top area (km2)"].tolist() + df["Base area (km2)"].iloc[::-1].tolist(),
		y=df["Depth"].tolist() + df["Depth"].iloc[::-1].tolist(),
		fill="toself",
		fillcolor="#FFCC99",
		line=dict(color="rgba(255, 204, 153, 0)"),
		showlegend=False,
		hoverinfo="skip"
	))
	
	# Update layout
	fig.update_layout(
		title="Depth vs Area (Full Precision)",
		xaxis_title="Area (km²)",
		yaxis_title="Depth (m)",
		yaxis=dict(autorange="reversed"),  # Depth increases downward
		legend=dict(orientation="h", y=1.02),
		margin=dict(l=40, r=40, t=60, b=40),
		plot_bgcolor="white",
		hovermode="x unified"
	)
	
	# Add grid
	fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
	fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
	
	return fig

def make_hist_cdf_figure(data: np.ndarray, title: str, xaxis_title: str, color_type: str = "calculated") -> go.Figure:
	x = np.asarray(data, dtype=float)
	x = x[~np.isnan(x)]
	if x.size == 0:
		fig = go.Figure()
		fig.update_layout(title=title)
		return fig

	x_sorted = np.sort(x)
	y_cdf = np.arange(1, x_sorted.size + 1) / x_sorted.size

	# Define colors based on type
	if color_type == "input":
		hist_color = "#87CEEB"  # Light blue for input distributions
	elif color_type == "result":
		hist_color = "#90EE90"  # Light green for result distributions
	else:  # calculated
		hist_color = "#1f77b4"  # Current blue for calculated distributions

	fig = make_subplots(specs=[[{"secondary_y": True}]])
	fig.add_trace(
		go.Histogram(
			x=x,
			name="Histogram",
			opacity=0.65,
			nbinsx=50,
			histnorm="probability",
			marker=dict(color=hist_color)
		),
		secondary_y=False,
	)
	fig.add_trace(
		go.Scatter(
			x=x_sorted,
			y=y_cdf,
			name="Cumulative",
			mode="lines",
			line=dict(color="#d62728", width=2)
		),
		secondary_y=True,
	)
	fig.update_yaxes(title_text="Probability (hist)", secondary_y=False)
	fig.update_yaxes(title_text="Cumulative", range=[0, 1], secondary_y=True)
	fig.update_xaxes(title_text=xaxis_title)
	fig.update_layout(
		title=title,
		barmode="overlay",
		legend=dict(orientation="h", y=1.02),  # Moved legend down
		margin=dict(l=40, r=40, t=60, b=40),
	)
	return fig


def summarize_array(x: np.ndarray) -> Dict[str, float]:
	x = np.asarray(x, dtype=float)
	x = x[~np.isnan(x)]
	if x.size == 0:
		return {}

	percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
	pvals = np.percentile(x, percentiles)

	mean_v = float(np.mean(x))
	min_v = float(np.min(x))
	max_v = float(np.max(x))
	std_v = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
	var_v = float(np.var(x, ddof=1)) if x.size > 1 else 0.0

	# Calculate mode (most frequent value)
	# For continuous data, we'll use histogram binning to find the most frequent bin
	if x.size > 1:
		# Use Sturges' formula for number of bins
		n_bins = int(np.ceil(1 + 3.322 * np.log10(x.size)))
		n_bins = max(min(n_bins, 50), 10)  # Limit between 10 and 50 bins
		
		hist, bin_edges = np.histogram(x, bins=n_bins)
		# Find the bin with maximum count
		max_bin_idx = np.argmax(hist)
		# Mode is the center of the most frequent bin
		mode_v = float((bin_edges[max_bin_idx] + bin_edges[max_bin_idx + 1]) / 2)
	else:
		mode_v = mean_v

	# Avoid precision-loss RuntimeWarning for nearly-constant data
	range_v = max_v - min_v
	if x.size > 2 and std_v > 0 and range_v > 1e-12 * max(1.0, abs(mean_v)):
		skew_v = float(skew(x, bias=False))
	else:
		skew_v = 0.0

	return {
		"mean": mean_v,
		"min": min_v,
		"max": max_v,
		"mode": mode_v,
		"std_dev": std_v,
		"variance": var_v,
		"skewness": skew_v,
		"P1": float(pvals[0]),
		"P5": float(pvals[1]),
		"P10": float(pvals[2]),
		"P25": float(pvals[3]),
		"P50": float(pvals[4]),
		"P75": float(pvals[5]),
		"P90": float(pvals[6]),
		"P95": float(pvals[7]),
		"P99": float(pvals[8]),
	}

def summary_table(x: np.ndarray, decimals: Optional[int] = None) -> pd.DataFrame:
	stats = summarize_array(x)
	if not stats:
		return pd.DataFrame()
	order = ["mean", "min", "max", "mode", "std_dev", "variance", "skewness",
	         "P1", "P5", "P10", "P25", "P50", "P75", "P90", "P95", "P99"]
	df = pd.DataFrame([{k: stats[k] for k in order}])
	if decimals is not None:
		# Format numeric columns to remove trailing zeros
		num_cols = df.select_dtypes(include=[np.number]).columns
		for col in num_cols:
			df[col] = df[col].apply(lambda x: f"{x:.{decimals}f}".rstrip('0').rstrip('.') if pd.notna(x) else x)
	
	# Rename columns for better display
	column_mapping = {
		"mean": "Mean",
		"min": "Min.",
		"max": "Max.",
		"mode": "Mode",
		"std_dev": "Std Dev",
		"variance": "Variance",
		"skewness": "Skewness",
		"P1": "P1",
		"P5": "P5",
		"P10": "P10",
		"P25": "P25",
		"P50": "P50",
		"P75": "P75",
		"P90": "P90",
		"P95": "P95",
		"P99": "P99"
	}
	df = df.rename(columns=column_mapping)
	
	# Center all text in the dataframe
	df_styled = df.style.set_properties(**{
		'text-align': 'center',
		'vertical-align': 'middle'
	})
	
	return df_styled


# -----------------------------
# GRV from depth/area table (Option C) — exact cumulative formula
# -----------------------------

def calculate_grv(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Input columns:
	- Depth (m)                [col A] - Use decimal point (.) not comma (,)
	- Top area (km2)           [col B] - Use decimal point (.) not comma (,)
	- Base area (km2)          [col C] - Use decimal point (.) not comma (,)

	All calculations are performed in meters with full precision:
	  Let hs_m[i]   = Depth[i] - Depth[i-1]  [m]
	  Let avg_top   = (Top[i-1] + Top[i]) / 2  [km²]
	  Let avg_base  = (Base[i-1] + Base[i]) / 2 [km²]
	  Convert areas to m², compute increment in m³:
	    inc_m3[i]   = hs_m[i] * (avg_top*1e6 - avg_base*1e6)  [m³]
	  Cumulative GRV in million m³:
	    GRV_1e6[i]  = cumsum(inc_m3) / 1e6  [×10^6 m³]

	This is algebraically equivalent to:
	  D[i] = D[i-1] + hs*(B[i-1] + 0.5*(B[i]-B[i-1])) - hs*(C[i-1] + 0.5*(C[i]-C[i-1]))
	where D is GRV (×10^6 m³), B Top area (km²), C Base area (km²), hs in m.
	
	Note: Only rows with missing Depth values are dropped. Missing area values are filled with 0.0.
	All input data is preserved for maximum precision in calculations.
	Full precision (up to 10 decimal places) is maintained throughout all calculations.
	"""
	df_work = df.copy()
	for col in ["Depth", "Top area (km2)", "Base area (km2)"]:
		df_work[col] = pd.to_numeric(df_work[col], errors="coerce")

	# Only drop rows where Depth is missing (required for calculation)
	# Missing area values will be filled with 0.0 to preserve all input data
	df_work = df_work.dropna(subset=["Depth"]).sort_values("Depth").reset_index(drop=True)

	depth_m = df_work["Depth"].astype(float)
	hs_m = depth_m.diff().fillna(0.0)

	top_km2 = df_work["Top area (km2)"].fillna(0.0).astype(float)
	base_km2 = df_work["Base area (km2)"].fillna(0.0).astype(float)

	avg_top_km2 = (top_km2.shift(1).fillna(top_km2) + top_km2) / 2.0
	avg_base_km2 = (base_km2.shift(1).fillna(base_km2) + base_km2) / 2.0

	avg_top_m2 = avg_top_km2 * 1_000_000.0
	avg_base_m2 = avg_base_km2 * 1_000_000.0

	inc_m3 = hs_m * (avg_top_m2 - avg_base_m2)
	grv_1e6_m3 = inc_m3.cumsum() / 1_000_000.0

	return pd.DataFrame({
		"Depth": depth_m,
		"Top area (km2)": top_km2,
		"Base area (km2)": base_km2,
		"GRV (1e6 m3)": grv_1e6_m3,
	})


# -----------------------------
# Depth vs Area utilities (for spill-point GRV)
# -----------------------------

def _cumulative_trapz(y: np.ndarray, step: float) -> np.ndarray:
	"""Return cumulative integral using trapezoid rule with constant depth spacing."""
	y = np.asarray(y, dtype=float)
	if y.size == 0:
		return y
	# average adjacent points then multiply by step
	avg_pairs = (y[:-1] + y[1:]) * 0.5
	volumes = np.cumsum(avg_pairs) * float(step)
	# prepend 0 to align with depths array
	return np.insert(volumes, 0, 0.0)


def compute_dgrv_top_plus_thickness(
	top_df: pd.DataFrame,
	thickness_m: float,
	step_m: float,
	extrapolate: bool,
	spill_point_m: float,
) -> Dict[str, Any]:
	"""
	Compute depth-related GRV (dGRV) from a top-area table and constant thickness.
	Areas are in km², depths in m. Returns volumes in km²·m and GRV at spill point.
	"""
	df = top_df.copy()
	df = df.dropna(subset=["Depth"]).sort_values("Depth").reset_index(drop=True)
	df["Top area (km2)"] = pd.to_numeric(df["Top area (km2)"], errors="coerce").fillna(0.0)
	depths_top = df["Depth"].astype(float).to_numpy()
	areas_top = df["Top area (km2)"].astype(float).to_numpy()

	if depths_top.size < 2:
		return {"error": "Need at least two rows in the top table"}

	max_top_depth = float(depths_top[-1])
	step_m = float(max(step_m, 1e-6))
	thickness_m = float(max(thickness_m, 0.0))

	# Optionally linear extrapolate top areas to cover required depth range
	target_max_depth = max(max_top_depth + thickness_m, float(spill_point_m))
	if extrapolate and target_max_depth > max_top_depth:
		slope = (areas_top[-1] - areas_top[-2]) / (depths_top[-1] - depths_top[-2])
		extra_depths = np.arange(max_top_depth + step_m, target_max_depth + step_m, step_m)
		extra_areas = areas_top[-1] + slope * (extra_depths - max_top_depth)
		depths_top = np.concatenate([depths_top, extra_depths])
		areas_top = np.concatenate([areas_top, extra_areas])

	# Base is a depth-shifted surface by thickness, same area contour
	depths_base = depths_top + thickness_m
	areas_base = areas_top.copy()

	# Depth grid
	z_min = float(depths_top[0])
	z_max = max(float(depths_base[-1]), float(spill_point_m))
	depth_grid = np.arange(z_min, z_max + step_m, step_m)

	# Interpolate areas onto grid
	top_interp = np.interp(depth_grid, depths_top, areas_top)
	base_interp = np.interp(depth_grid, depths_base, areas_base)

	# Cumulative volumes (km²·m)
	vol_top = _cumulative_trapz(top_interp, step_m)
	vol_base = _cumulative_trapz(base_interp, step_m)
	dgrv = vol_top - vol_base

	# GRV at spill point (km²·m)
	if spill_point_m < depth_grid[0] or spill_point_m > depth_grid[-1]:
		grv_sp_km2m = np.nan
	else:
		grv_sp_km2m = float(np.interp(spill_point_m, depth_grid, dgrv))

	return {
		"depths": depth_grid,
		"top_interp": top_interp,
		"base_interp": base_interp,
		"vol_top": vol_top,
		"vol_base": vol_base,
		"dgrv": dgrv,
		"grv_sp_km2m": grv_sp_km2m,
	}


def compute_dgrv_top_base_table(
	df_in: pd.DataFrame,
	step_m: float,
	extrapolate: bool,
	spill_point_m: float,
) -> Dict[str, Any]:
	"""
	Compute depth-related GRV (dGRV) from a Depth/Top/Base Areas table.
	Areas in km², depths in m. Supports optional linear extrapolation to spill point.
	"""
	df = df_in.copy()
	for col in ["Depth", "Top area (km2)", "Base area (km2)"]:
		df[col] = pd.to_numeric(df[col], errors="coerce")
	df = df.dropna(subset=["Depth"]).sort_values("Depth").reset_index(drop=True)

	depths = df["Depth"].astype(float).to_numpy()
	top_areas = df["Top area (km2)"].fillna(0.0).astype(float).to_numpy()
	base_areas = df["Base area (km2)"].fillna(0.0).astype(float).to_numpy()

	if depths.size < 2:
		return {"error": "Need at least two rows in the table"}

	max_depth = float(depths[-1])
	step_m = float(max(step_m, 1e-6))

	if extrapolate and float(spill_point_m) > max_depth:
		# Linear extrapolation for both top and base using last segment slopes
		dz = depths[-1] - depths[-2]
		if dz == 0:
			return {"error": "Cannot extrapolate due to zero depth interval in last rows"}
		slope_top = (top_areas[-1] - top_areas[-2]) / dz
		slope_base = (base_areas[-1] - base_areas[-2]) / dz
		extra_depths = np.arange(max_depth + step_m, float(spill_point_m) + step_m, step_m)
		extra_top = top_areas[-1] + slope_top * (extra_depths - max_depth)
		extra_base = base_areas[-1] + slope_base * (extra_depths - max_depth)
		depths = np.concatenate([depths, extra_depths])
		top_areas = np.concatenate([top_areas, extra_top])
		base_areas = np.concatenate([base_areas, extra_base])

	# Depth grid
	z_min = float(depths[0])
	z_max = max(float(depths[-1]), float(spill_point_m))
	depth_grid = np.arange(z_min, z_max + step_m, step_m)

	# Interpolate areas
	top_interp = np.interp(depth_grid, depths, top_areas)
	base_interp = np.interp(depth_grid, depths, base_areas)

	# Cumulative volumes (km²·m)
	vol_top = _cumulative_trapz(top_interp, step_m)
	vol_base = _cumulative_trapz(base_interp, step_m)
	dgrv = vol_top - vol_base

	# GRV at spill point (km²·m)
	if spill_point_m < depth_grid[0] or spill_point_m > depth_grid[-1]:
		grv_sp_km2m = np.nan
	else:
		grv_sp_km2m = float(np.interp(spill_point_m, depth_grid, dgrv))

	return {
		"depths": depth_grid,
		"top_interp": top_interp,
		"base_interp": base_interp,
		"vol_top": vol_top,
		"vol_base": vol_base,
		"dgrv": dgrv,
		"grv_sp_km2m": grv_sp_km2m,
	}


def make_area_volume_plot(
	depths: np.ndarray,
	top_interp: np.ndarray,
	base_interp: np.ndarray,
	vol_top: np.ndarray,
	vol_base: np.ndarray,
	dgrv: np.ndarray,
	spill_point_m: float,
	grv_sp_km2m: float,
) -> go.Figure:
	"""Create a two-panel plot: area vs depth and volume/dGRV vs depth."""
	fig = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=("Area vs Depth", "Volume vs Depth"))

	# Left: Areas
	fig.add_trace(go.Scatter(x=top_interp, y=depths, name="Top Area", mode="lines", line=dict(color="blue")), row=1, col=1)
	fig.add_trace(go.Scatter(x=base_interp, y=depths, name="Base Area", mode="lines", line=dict(color="red")), row=1, col=1)

	# Right: Volumes and dGRV
	fig.add_trace(go.Scatter(x=vol_top, y=depths, name="Top Volume", mode="lines", line=dict(color="#1f77b4")), row=1, col=2)
	fig.add_trace(go.Scatter(x=vol_base, y=depths, name="Base Volume", mode="lines", line=dict(color="#ff7f0e")), row=1, col=2)
	fig.add_trace(go.Scatter(x=dgrv, y=depths, name="dGRV (km²·m)", mode="lines", line=dict(color="purple")), row=1, col=2)

	# Spill point marker
	if not np.isnan(grv_sp_km2m):
		fig.add_trace(
			go.Scatter(
				x=[grv_sp_km2m], y=[spill_point_m], mode="markers",
				marker=dict(symbol="star", size=12, color="gold"),
				name="Spill Point"
			),
			row=1, col=2
		)

	fig.update_yaxes(autorange="reversed", title_text="Depth (m)")
	fig.update_xaxes(title_text="Area (km²)", row=1, col=1)
	fig.update_xaxes(title_text="Volume (km²·m)", row=1, col=2)
	fig.update_layout(margin=dict(l=40, r=40, t=60, b=40), legend=dict(orientation="h", y=1.1))
	return fig

# -----------------------------
# UI helpers
# -----------------------------

DistributionChoice = [
	"PERT",
	"Triangular",
	"Uniform",
	"Lognormal (mean, sd)",
	"Subjective Beta (Vose)",
]


def render_param(
	name_key: str,
	label: str,
	unit_hint: str,
	default_dist: str,
	default_params: Dict[str, Any],
	n_samples: int,
	plot_unit_label: Optional[str] = None,
	stats_decimals: Optional[int] = None,
	display_scale: float = 1.0,
) -> np.ndarray:
	col1, col2 = st.columns([1.3, 2.2])

	with col1:
		dist = st.selectbox(
			f"{label} distribution",
			DistributionChoice,
			index=DistributionChoice.index(default_dist),
			key=f"dist_{name_key}",
			help="Choose the distribution type for this parameter",
		)

	params: Dict[str, Any] = {}
	with col2:
		if dist == "PERT":
			min_v = st.number_input(f"{label} min ({unit_hint})", value=float(default_params.get("min", 0.0)), key=f"{name_key}_pert_min")
			mode_v = st.number_input(f"{label} mode ({unit_hint})", value=float(default_params.get("mode", min_v)), key=f"{name_key}_pert_mode")
			max_v = st.number_input(f"{label} max ({unit_hint})", value=float(default_params.get("max", mode_v)), key=f"{name_key}_pert_max")
			params = {"min": min_v, "mode": mode_v, "max": max_v}
		elif dist == "Triangular":
			min_v = st.number_input(f"{label} min ({unit_hint})", value=float(default_params.get("min", 0.0)), key=f"{name_key}_tri_min")
			mode_v = st.number_input(f"{label} mode ({unit_hint})", value=float(default_params.get("mode", min_v)), key=f"{name_key}_tri_mode")
			max_v = st.number_input(f"{label} max ({unit_hint})", value=float(default_params.get("max", mode_v)), key=f"{name_key}_tri_max")
			params = {"min": min_v, "mode": mode_v, "max": max_v}
		elif dist == "Uniform":
			min_v = st.number_input(f"{label} min ({unit_hint})", value=float(default_params.get("min", 0.0)), key=f"{name_key}_uni_min")
			max_v = st.number_input(f"{label} max ({unit_hint})", value=float(default_params.get("max", min_v)), key=f"{name_key}_uni_max")
			params = {"min": min_v, "max": max_v}
		elif dist == "Lognormal (mean, sd)":
			mean_v = st.number_input(f"{label} arithmetic mean ({unit_hint})", value=float(default_params.get("mean", 1.0)), key=f"{name_key}_ln_mean")
			sd_v = st.number_input(f"{label} arithmetic sd ({unit_hint})", value=float(default_params.get("sd", 0.1)), key=f"{name_key}_ln_sd", min_value=0.0)
			params = {"mean": mean_v, "sd": sd_v}
		elif dist == "Subjective Beta (Vose)":
			mode_v = st.number_input(f"{label} mode ({unit_hint})", value=float(default_params.get("mode", 0.0)), key=f"{name_key}_sb_mode")
			mean_v = st.number_input(f"{label} mean ({unit_hint})", value=float(default_params.get("mean", mode_v)), key=f"{name_key}_sb_mean")
			min_v = st.number_input(f"{label} min ({unit_hint})", value=float(default_params.get("min", 0.0)), key=f"{name_key}_sb_min")
			max_v = st.number_input(f"{label} max ({unit_hint})", value=float(default_params.get("max", min_v)), key=f"{name_key}_sb_max")
			params = {"mode": mode_v, "mean": mean_v, "min": min_v, "max": max_v}

	recalc = st.button(f"Recalculate {label}", key=f"recalc_{name_key}")

	ss_samples_key = f"samples_{name_key}"
	ss_conf_key = f"conf_{name_key}"
	need_init = ss_samples_key not in st.session_state
	current_conf = {"dist": dist, **params, "n": n_samples}

	if recalc or need_init or (ss_conf_key in st.session_state and st.session_state[ss_conf_key].get("n") != n_samples):
		if dist == "PERT":
			samples = sample_pert(params["min"], params["mode"], params["max"], n_samples)
		elif dist == "Triangular":
			samples = sample_triangular(params["min"], params["mode"], params["max"], n_samples)
		elif dist == "Uniform":
			samples = sample_uniform(params["min"], params["max"], n_samples)
		elif dist == "Lognormal (mean, sd)":
			samples = sample_lognormal_mean_sd(params["mean"], params["sd"], n_samples)
		elif dist == "Subjective Beta (Vose)":
			samples = sample_beta_subjective(params["mode"], params["mean"], params["min"], params["max"], n_samples)
		else:
			samples = np.zeros(n_samples)

		st.session_state[ss_samples_key] = samples
		st.session_state[ss_conf_key] = current_conf
	else:
		samples = st.session_state.get(ss_samples_key, np.zeros(n_samples))

	if ss_conf_key in st.session_state and st.session_state[ss_conf_key] != current_conf:
		st.info("Inputs changed. Press 'Recalculate' to update samples.", icon="ℹ️")

	display_samples = samples * display_scale
	unit_lbl = plot_unit_label if plot_unit_label else unit_hint
	st.plotly_chart(make_hist_cdf_figure(display_samples, f"{label} distribution", f"{label} ({unit_lbl})", "input"), use_container_width=True)
	st.dataframe(summary_table(display_samples, decimals=stats_decimals), use_container_width=True)

	return samples


def apply_correlations_to_samples(samples_dict: Dict[str, np.ndarray], correlation_values: Dict[str, float], n_samples: int) -> Dict[str, np.ndarray]:
	"""
	Apply correlations to already sampled parameters.
	
	Args:
		samples_dict: Dictionary of parameter samples
		correlation_values: Dictionary of correlation coefficients
		n_samples: Number of samples
	
	Returns:
		Updated samples dictionary with correlations applied
	"""
	updated_samples = samples_dict.copy()
	
	# Apply Temperature-Pressure correlation
	if 'temp_pressure' in correlation_values and abs(correlation_values['temp_pressure']) > 0.01:
		if 'sT' in updated_samples and 'sP' in updated_samples:
			updated_samples['sT'], updated_samples['sP'] = apply_correlation(
				updated_samples['sT'], updated_samples['sP'], correlation_values['temp_pressure']
			)
	
	# Apply Top Depth-Base Depth correlation
	if 'top_base_depth' in correlation_values and abs(correlation_values['top_base_depth']) > 0.01:
		if 'stopdepth_off' in updated_samples and 'sbasedepth_off' in updated_samples:
			updated_samples['stopdepth_off'], updated_samples['sbasedepth_off'] = apply_correlation(
				updated_samples['stopdepth_off'], updated_samples['sbasedepth_off'], correlation_values['top_base_depth']
			)
	
	# Apply Porosity-Storage Efficiency correlation
	if 'porosity_se' in correlation_values and abs(correlation_values['porosity_se']) > 0.01:
		if 'sp' in updated_samples and 'sSE' in updated_samples:
			updated_samples['sp'], updated_samples['sSE'] = apply_correlation(
				updated_samples['sp'], updated_samples['sSE'], correlation_values['porosity_se']
			)
	
	# Apply Porosity-Net-to-Gross correlation
	if 'porosity_ntg' in correlation_values and abs(correlation_values['porosity_ntg']) > 0.01:
		if 'sp' in updated_samples and 'sNtG' in updated_samples:
			updated_samples['sp'], updated_samples['sNtG'] = apply_correlation(
				updated_samples['sp'], updated_samples['sNtG'], correlation_values['porosity_ntg']
			)
	
	# Apply Thickness-GCF correlation
	if 'thickness_gcf' in correlation_values and abs(correlation_values['thickness_gcf']) > 0.01:
		if 'sh' in updated_samples and 'sGCF' in updated_samples:
			updated_samples['sh'], updated_samples['sGCF'] = apply_correlation(
				updated_samples['sh'], updated_samples['sGCF'], correlation_values['thickness_gcf']
			)
	
	# Apply Area-GCF correlation
	if 'area_gcf' in correlation_values and abs(correlation_values['area_gcf']) > 0.01:
		if 'sA' in updated_samples and 'sGCF' in updated_samples:
			updated_samples['sA'], updated_samples['sGCF'] = apply_correlation(
				updated_samples['sA'], updated_samples['sGCF'], correlation_values['area_gcf']
			)
	
	return updated_samples


# -----------------------------
# App
# -----------------------------

st.set_page_config(page_title="CO₂ Storage Capacity", layout="wide")
st.title("SCOPE-CO₂")
st.subheader("Subsurface Capacity Overview and Probability Estimator for CO₂ storage")

with st.expander("Assumptions and formulas", expanded=False):
	st.markdown(
		"- Area `A` in km²; internally converted to m².\n"
		"- Thickness `h` in meters. `GCF`, `NtG`, `p`, `SE` are fractions in [0, 1]. Density `d` in kg/m³.\n"
		"- GRV = A × GCF × h (with A converted to m²).\n"
		"- PV = GRV × NtG × p\n"
		"- SVe = PV × SE\n"
		"- SC = SVe × d\n"
		"- Lognormal(m, s): m and s are arithmetic mean and sd."
	)

with st.expander("For further explanation to methodology", expanded=False):
	st.markdown("""
	**Capture, Storage and Use of CO2 (CCUS). Evaluation of the CO2 storage potential in Denmark. Vol.1: Report & Vol 2: Appendix A and B** [Published as 2 separate volumes both with Series number 2020/46]
	
	**Authors:** Lars Hjelm, Karen Lyng Anthonsen, Knud Dideriksen, Carsten Møller Nielsen, Lars Henrik Nielsen, Anders Mathiesen
	
	**Departments:** Afdeling for Geofysik og Sedimentære Bassiner, Geologisk Datacenter, Afdeling for Geokemi, Afdeling for Geoenergi og -lagring, Afdeling for Hydrologi
	
	**Publication:** Bog/rapport › Rapport (offentligt tilgængelig)
	
	**Links:**
	- [Publication site](https://pub.geus.dk/da/publications/capture-storage-and-use-of-co2-ccus-evaluation-of-the-co2-storage)
	- [PDF download](https://data.geus.dk/pure-pdf/GEUS-R_2020_46_web.pdf)
	""")

st.sidebar.header("Simulation controls")
num_sims = st.sidebar.number_input("Number of simulations", min_value=100, max_value=2_000_000, value=10_000, step=1000, help="Samples per parameter")
st.sidebar.caption("Default = 10,000")

defaults = {
	"A": {"dist": "PERT", "min": 105.0, "mode": 210.0, "max": 273.0},  # km^2
	"GCF": {"dist": "PERT", "min": 0.6, "mode": 0.68, "max": 0.85},
	"h": {"dist": "PERT", "min": 100.0, "mode": 130.0, "max": 150.0},  # m
	"NtG": {"dist": "PERT", "min": 0.27, "mode": 0.40, "max": 0.50},
	"p": {"dist": "PERT", "min": 0.08, "mode": 0.18, "max": 0.21},
	"SE": {"dist": "PERT", "min": 0.03, "mode": 0.10, "max": 0.20},
	"d": {"dist": "PERT", "min": 584.25, "mode": 615.0, "max": 645.75},  # kg/m3
}

tabs = st.tabs(["Inputs", "Results"])

with tabs[0]:
	# ========================================
	# SECTION 1: GROSS ROCK VOLUME (GRV)
	# ========================================
	st.markdown("---")
	st.markdown("## Gross Rock Volume (GRV)")
	
	grv_option = st.radio(
		"Choose GRV input method",
		[
			"Direct input",
			"From Area, Geometry Factor and Thickness",
			"From Depth/Top/Base Areas (table)",
			"Depth vs Area vs Thickness (spill-point)",
		],
		index=1,
		horizontal=True,
	)

	if grv_option == "Direct input":
		grv_min = defaults["A"]["min"] * 1_000_000.0 * defaults["h"]["min"] * defaults["GCF"]["min"]
		grv_mode = defaults["A"]["mode"] * 1_000_000.0 * defaults["h"]["mode"] * defaults["GCF"]["mode"]
		grv_max = defaults["A"]["max"] * 1_000_000.0 * defaults["h"]["max"] * defaults["GCF"]["max"]
		defaults_grv = {"dist": "PERT", "min": grv_min, "mode": grv_mode, "max": grv_max}

		sGRV_m3 = render_param(
			"GRV",
			"Gross Rock Volume (GRV)",
			"m³",
			defaults_grv["dist"],
			defaults_grv,
			num_sims,
			plot_unit_label="×10^6 m³",
			stats_decimals=1,
			display_scale=1e-6,
		)

	elif grv_option == "From Area, Geometry Factor and Thickness":
		st.caption("Provide Area, GCF, and h; GRV is calculated and shown below.")
		sA = render_param("A", "Area A", "km²", defaults["A"]["dist"], defaults["A"], num_sims, stats_decimals=3)
		
		# GCF Distribution Method Selection
		gcf_method = st.radio(
			"GCF distribution method",
			["Direct method", "Geometric Correction Factor Calculator"],
			horizontal=True,
			help="Choose how to define the GCF distribution"
		)
		
		if gcf_method == "Direct method":
			# Original direct distribution input
			sGCF = render_param("GCF", "Geometry Correction Factor GCF", "fraction", defaults["GCF"]["dist"], defaults["GCF"], num_sims, stats_decimals=3)
		else:
			# Geometric Correction Factor Calculator
			st.markdown("### Geometric Correction Factor Calculator")
			st.markdown("Calculate GCF based on reservoir geometry using the Gehman (1970) methodology.")
			
			# Add information about the methodology
			with st.expander("Methodology and Assumptions", expanded=False):
				st.markdown("""
				### Geometric Correction Factor Methodology
				
				This calculator is based on the work of **Gehman, H.N. (1970)** and provides geometric correction factors for different trap geometries.
				
				**Reference:** Gehman, H.N. (1970). Graphs to Derive Geometric Correction Factor: Exxon Training Materials (unpublished), Houston.
				
				**Data Source:** The lookup table data has been digitized from the original graphs presented in the Gehman (1970) training materials.
				
				#### Key Concepts:
				
				1. **Geometric Shape Types:**
				   - **Type 1**: Dome, Cone, or Pyramid
				   - **Type 2**: Anticline, Prism, or Cylinder
				   - **Type 3**: Flat-top Dome
				   - **Type 4**: Flat-top Anticline
				   - **Type 5**: Block or Vertical Cylinder
				
				2. **Length/Width Ratios:**
				   - For domes and flat-top domes: L/W = 1
				   - For anticlines: L/W = 2, 5, or 10
				   - For blocks: L/W = W (width)
				
				3. **Reservoir Thickness/Closure Ratio:**
				   - Ratio of true reservoir thickness to structural relief
				   - True thickness accounts for dip angle correction
				   - As this ratio increases, GCF decreases (inverse relationship)
				
				#### Original Source Figure Description:
				
				**Input Parameters section:** Shows the reservoir thickness, height of closure, geometric shape type, and length/width ratio inputs
				
				**Closure Geometries section:** Displays the five different structural types with their visual representations
				
				**Central Graph section:** Shows the main plot of Reservoir Thickness/Height of Closure ratio vs Geometric Correction Factor curves
				
				**Result section:** Indicates how the GCF is derived from the intersection point
				
				#### Step-by-Step Process:
				1. Enter reservoir thickness and structural relief
				2. Select geometric shape type and length/width ratio
				3. Calculate Reservoir Thickness/Closure ratio
				4. Apply dip angle correction if needed
				5. Use lookup table to find corresponding GCF value
				6. Apply uncertainty multiplier if desired
				""")
			
			# Input parameters for GCF calculation
			col1, col2 = st.columns(2)
			with col1:
				reservoir_thickness = st.number_input(
					"Reservoir thickness (m)",
					min_value=0.1,
					value=100.0,
					step=0.1,
					help="True reservoir thickness in meters"
				)
				structural_relief = st.number_input(
					"Structural relief (m)",
					min_value=0.1,
					value=150.0,
					step=0.1,
					help="Height of closure (spill point to apex) in meters"
				)
			
			with col2:
				dip_angle = st.number_input(
					"Dip angle (degrees)",
					min_value=0.0,
					max_value=90.0,
					value=15.0,
					step=0.1,
					help="Average dip angle of the reservoir"
				)
			
			# Shape type selection
			shape_type = st.selectbox(
				"Geometric shape type",
				[
					(1, "Dome, Cone, or Pyramid"),
					(2, "Anticline, Prism, or Cylinder"),
					(3, "Flat-top Dome"),
					(4, "Flat-top Anticline"),
					(5, "Block or Vertical Cylinder")
				],
				format_func=lambda x: x[1],
				help="Select the geometric shape of the reservoir"
			)
			
			# Length/Width ratio selection based on shape type
			if shape_type[0] in [1, 3, 5]:  # Dome, Flat-top Dome, Block
				lw_ratio = 1
				st.info(f"Length/Width ratio set to 1 for {shape_type[1]}")
			else:  # Anticline types
				lw_ratio = st.selectbox(
					"Length/Width ratio",
					[2, 5, 10],
					help="Select the length to width ratio for the anticline"
				)
			
			# Calculate Reservoir Thickness/Closure ratio
			# Apply dip angle correction: true_thickness = measured_thickness / cos(dip_angle)
			dip_radians = math.radians(dip_angle)
			true_thickness = reservoir_thickness / math.cos(dip_radians)
			res_tk_closure_ratio = true_thickness / structural_relief
			
			# Calculate GCF using the lookup table
			gcf_calculated = interpolate_gcf(shape_type[0], lw_ratio, res_tk_closure_ratio)
			
			# Display calculation results
			st.markdown("### GCF Calculation Results")
			col1, col2, col3 = st.columns(3)
			with col1:
				st.metric("True thickness (m)", f"{true_thickness:.1f}")
			with col2:
				st.metric("Reservoir Thickness/Closure Ratio", f"{res_tk_closure_ratio:.3f}")
			with col3:
				st.metric("Calculated GCF", f"{gcf_calculated:.3f}")
			
			# Create the main plot showing GCF curves
			res_tk_closure_ratios, lookup_table = get_gcf_lookup_table()
			
			# Create the plot
			fig = go.Figure()
			
			# Define colors and line styles for different curves
			curve_configs = {
				(1, 1): {"color": "blue", "width": 3, "label": "Dome, Cone, Pyramid (L/W=1)"},
				(2, 2): {"color": "red", "width": 2, "label": "Anticline, Prism, Cylinder (L/W=2)"},
				(2, 5): {"color": "orange", "width": 2, "label": "Anticline, Prism, Cylinder (L/W=5)"},
				(2, 10): {"color": "purple", "width": 2, "label": "Anticline, Prism, Cylinder (L/W=10)"},
				(3, 1): {"color": "green", "width": 2, "label": "Flat-top Dome (L/W=1)"},
				(4, 2): {"color": "brown", "width": 2, "label": "Flat-top Anticline (L/W=2)"},
				(4, 5): {"color": "pink", "width": 2, "label": "Flat-top Anticline (L/W=5)"},
				(4, 10): {"color": "gray", "width": 2, "label": "Flat-top Anticline (L/W=10)"},
				(5, 1): {"color": "black", "width": 2, "label": "Block, Vertical Cylinder (L/W=1)"}
			}
			
			# Add curves for all shape types
			for (shape_type_plot, lw_ratio_plot), gcf_curve in lookup_table.items():
				config = curve_configs.get((shape_type_plot, lw_ratio_plot), {"color": "blue", "width": 1, "label": f"Shape {shape_type_plot} (L/W={lw_ratio_plot})"})
				
				fig.add_trace(go.Scatter(
					x=res_tk_closure_ratios,
					y=gcf_curve,
					mode='lines',
					name=config["label"],
					line=dict(width=config["width"], color=config["color"]),
					showlegend=True,
					hovertemplate='<b>%{fullData.name}</b><br>' +
								 'Reservoir Thickness/Closure Ratio: %{x:.3f}<br>' +
								 'Geometric Correction Factor: %{y:.3f}<br>' +
								 '<extra></extra>'
				))
			
			# Add point for current calculation
			fig.add_trace(go.Scatter(
				x=[res_tk_closure_ratio],
				y=[gcf_calculated],
				mode='markers',
				name='Current Point (Red Star)',
				marker=dict(size=24, color='red', symbol='star'),
				showlegend=True,
				hovertemplate='<b>Current Point</b><br>' +
							 'Reservoir Thickness/Closure Ratio: %{x:.3f}<br>' +
							 'Geometric Correction Factor: %{y:.3f}<br>' +
							 '<extra></extra>'
			))
			
			# Update layout
			fig.update_layout(
				title=dict(
					text="Geometric Correction Factor vs Reservoir Thickness/Closure Ratio",
					y=0.98  # Position title at top
				),
				xaxis_title="Reservoir Thickness/Closure Ratio",
				yaxis_title="Geometric Correction Factor",
				width=800,
				height=600,
				legend=dict(
					yanchor="top",
					y=-0.15,
					xanchor="center",
					x=0.5,
					orientation="h",
					bgcolor="rgba(255, 255, 255, 0.8)",
					bordercolor="black",
					borderwidth=1
				),
				xaxis=dict(
					range=[0, 1],
					scaleanchor="y",
					scaleratio=1
				),
				yaxis=dict(
					range=[0, 1]
				)
			)
			
			# Display the plot
			st.plotly_chart(fig, use_container_width=False)
			
			# GCF Uncertainty Multiplier (GCF_MP)
			st.markdown("### GCF Uncertainty Multiplier (GCF_MP)")
			
			# Toggle for GCF_MP
			gcf_mp_enabled = st.checkbox("Apply GCF uncertainty multiplier", value=False, help="Enable to apply uncertainty multiplier to calculated GCF value")
			
			if gcf_mp_enabled:
				gcf_mp_type = st.radio(
					"GCF_MP definition",
					["Constant value", "Probability distribution"],
					horizontal=True,
					help="Choose how to define the GCF uncertainty multiplier"
				)
				
				if gcf_mp_type == "Constant value":
					gcf_mp_constant = st.number_input(
						"GCF_MP constant value",
						value=1.0,
						min_value=0.1,
						max_value=10.0,
						step=0.01,
						help="Constant multiplier for calculated GCF value"
					)
					gcf_mp_samples = np.full(num_sims, gcf_mp_constant)
				else:
					# Use probability distribution
					gcf_mp_samples = render_param(
						"GCF_MP",
						"GCF Uncertainty Multiplier",
						"multiplier",
						"PERT",
						{"min": 0.9, "mode": 1.0, "max": 1.1},
						num_sims,
						plot_unit_label="multiplier",
						stats_decimals=3
					)
			else:
				# GCF_MP disabled - set to 1.0
				gcf_mp_samples = np.full(num_sims, 1.0)
				st.info("GCF_MP disabled - using calculated GCF value as is (multiplier = 1.0)", icon="ℹ️")
			
			# Calculate final GCF distribution
			sGCF = np.full(num_sims, gcf_calculated) * gcf_mp_samples
			
			# Display final GCF summary
			st.markdown("### Final GCF for Calculations")
			
			gcf_calc_value = float(gcf_calculated)
			gcf_mp_mean = float(np.mean(gcf_mp_samples))
			gcf_final_mean = float(np.mean(sGCF))
			
			col1, col2, col3 = st.columns(3)
			with col1:
				st.metric("Calculated GCF", f"{gcf_calc_value:.3f}")
			with col2:
				st.metric("GCF_MP (mean)", f"{gcf_mp_mean:.3f}")
			with col3:
				st.metric("Final GCF (mean)", f"{gcf_final_mean:.3f}")
			
			# Show final GCF distribution
			st.plotly_chart(make_hist_cdf_figure(sGCF, "Final GCF distribution (after GCF_MP)", "GCF", "calculated"), use_container_width=True)
			st.dataframe(summary_table(sGCF, decimals=3), use_container_width=True)
			
			# Sensitivity Analysis
			st.markdown("### GCF Sensitivity Analysis")
			st.markdown("""
			**How the Sensitivity Analysis Works:**
			
			The sensitivity analysis shows how the Geometric Correction Factor (GCF) changes when individual input parameters are varied while keeping all other parameters constant. This helps identify which parameters have the greatest impact on the final GCF value.
			
			**Methodology:**
			1. **Parameter Ranges:** Each parameter is varied over a range of ±50% around the current value (or 0-45° for dip angle)
			2. **GCF Calculation:** For each parameter value, the GCF is recalculated using the updated lookup table data from Gehman (1970)
			3. **Interpolation:** The `interpolate_gcf()` function uses the digitized data to find the appropriate GCF value for each Reservoir Thickness/Closure ratio
			4. **Visualization:** The resulting curves show the sensitivity of GCF to changes in each parameter
			
			**Plot Interpretation:**
			- **Blue lines** show how GCF changes when varying individual parameters
			- **Red stars** mark the current parameter values and resulting GCF
			- **X-axis** shows the parameter variation range
			- **Y-axis** shows the resulting GCF values
			""")
			
			# Create sensitivity analysis
			# Vary reservoir thickness
			thickness_range = np.linspace(reservoir_thickness * 0.5, reservoir_thickness * 1.5, 50)
			gcf_values_thickness = []
			for t in thickness_range:
				true_t = t / math.cos(dip_radians)
				ratio_t = true_t / structural_relief
				gcf_t = interpolate_gcf(shape_type[0], lw_ratio, ratio_t)
				gcf_values_thickness.append(gcf_t)
			
			# Vary structural relief
			relief_range = np.linspace(structural_relief * 0.5, structural_relief * 1.5, 50)
			gcf_values_relief = []
			for r in relief_range:
				ratio_r = true_thickness / r
				gcf_r = interpolate_gcf(shape_type[0], lw_ratio, ratio_r)
				gcf_values_relief.append(gcf_r)
			
			# Vary dip angle
			dip_range = np.linspace(0, 45, 50)
			gcf_values_dip = []
			for d in dip_range:
				dip_rad = math.radians(d)
				true_t_dip = reservoir_thickness / math.cos(dip_rad)
				ratio_dip = true_t_dip / structural_relief
				gcf_dip = interpolate_gcf(shape_type[0], lw_ratio, ratio_dip)
				gcf_values_dip.append(gcf_dip)
			
			# Create sensitivity plot
			fig_sens = make_subplots(rows=1, cols=3, subplot_titles=("Reservoir Thickness", "Structural Relief", "Dip Angle"))
			
			# Thickness sensitivity
			fig_sens.add_trace(
				go.Scatter(x=thickness_range, y=gcf_values_thickness, mode='lines', name='Thickness', line=dict(color='blue')),
				row=1, col=1
			)
			fig_sens.add_trace(
				go.Scatter(x=[reservoir_thickness], y=[gcf_calculated], mode='markers', name='Current', 
						  marker=dict(symbol='star', size=12, color='red')),
				row=1, col=1
			)
			
			# Relief sensitivity
			fig_sens.add_trace(
				go.Scatter(x=relief_range, y=gcf_values_relief, mode='lines', name='Relief', line=dict(color='green')),
				row=1, col=2
			)
			fig_sens.add_trace(
				go.Scatter(x=[structural_relief], y=[gcf_calculated], mode='markers', name='Current', 
						  marker=dict(symbol='star', size=12, color='red')),
				row=1, col=2
			)
			
			# Dip angle sensitivity
			fig_sens.add_trace(
				go.Scatter(x=dip_range, y=gcf_values_dip, mode='lines', name='Dip Angle', line=dict(color='orange')),
				row=1, col=3
			)
			fig_sens.add_trace(
				go.Scatter(x=[dip_angle], y=[gcf_calculated], mode='markers', name='Current', 
						  marker=dict(symbol='star', size=12, color='red')),
				row=1, col=3
			)
			
			fig_sens.update_layout(height=400, showlegend=False)
			fig_sens.update_xaxes(title_text="Reservoir Thickness (m)", row=1, col=1)
			fig_sens.update_xaxes(title_text="Structural Relief (m)", row=1, col=2)
			fig_sens.update_xaxes(title_text="Dip Angle (degrees)", row=1, col=3)
			fig_sens.update_yaxes(title_text="GCF", row=1, col=1)
			fig_sens.update_yaxes(title_text="GCF", row=1, col=2)
			fig_sens.update_yaxes(title_text="GCF", row=1, col=3)
			
			st.plotly_chart(fig_sens, use_container_width=True)
		
		sh = render_param("h", "Reservoir thickness h", "m", defaults["h"]["dist"], defaults["h"], num_sims, stats_decimals=2)

		sGRV_m3 = (sA * 1_000_000.0) * sh * sGCF

		st.plotly_chart(make_hist_cdf_figure(sGRV_m3 / 1e6, "Calculated GRV distribution", "GRV (×10^6 m³)", "calculated"), use_container_width=True)
		st.dataframe(summary_table(sGRV_m3 / 1e6, decimals=2), use_container_width=True)

	elif grv_option == "From Depth/Top/Base Areas (table)":
		st.caption("Enter depth slices and Top/Base areas. Choose depth step, extrapolation, and a Spill Point depth. GRV at Spill Point will be used in PV.")
		st.info("📝 Depth in meters; areas in km². 1 km²·m = 10⁶ m³.")
		default_table = pd.DataFrame({
			"Depth": [2040, 2050, 2060, 2070, 2080, 2090, 2100, 2110, 2120, 2130, 2140, 2150, 2160, 2170, 2180, 2190, 2200, 2210, 2220, 2230, 2240, 2250, 2260, 2270, 2280, 2290, 2300, 2310, 2320, 2330, 2340, 2350, 2360, 2370, 2380, 2390, 2400],
			"Top area (km2)": [0.0, 0.06149313, 0.743428415, 1.31427065, 1.917695765, 2.581499555, 3.357041565, 4.24747926, 5.178613535, 6.122726925, 7.07979995, 8.04906025, 9.02872446, 10.02516695, 11.0326527, 12.09376061, 13.24614827, 14.32668123, 15.41667629, 16.61170115, 17.8503389, 19.01679186, 20.11461198, 21.1684545, 22.18402345, 23.13352475, 24.06078863, 24.95179712, 25.7778064, 26.55293472, 27.3037083, 28.04968615, 28.81130504, 29.61183057, 30.47923043, 31.46059753, 32.624375],
			"Base area (km2)": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06149313, 0.743428415, 1.31427065, 1.917695765, 2.581499555, 3.357041565, 4.24747926, 5.178613535, 6.122726925, 7.07979995, 8.04906025, 9.02872446, 10.02516695, 11.0326527, 12.09376061, 13.24614827, 14.32668123, 15.41667629, 16.61170115, 17.8503389, 19.01679186, 20.11461198, 21.1684545, 22.18402345, 23.13352475, 24.06078863, 24.95179712, 25.7778064, 26.55293472, 27.3037083, 28.04968615],
		})

		if "grv_table_input" not in st.session_state:
			st.session_state["grv_table_input"] = default_table

		edited = st.data_editor(
			st.session_state["grv_table_input"],
			num_rows="dynamic",
			use_container_width=True,
			key="grv_editor",
			column_config={
				"Depth": st.column_config.NumberColumn("Depth (m)", format="%.10f"),
				"Top area (km2)": st.column_config.NumberColumn("Top area (km²)", format="%.10f"),
				"Base area (km2)": st.column_config.NumberColumn("Base area (km²)", format="%.10f"),
			}
		)
		st.session_state["grv_table_input"] = edited

		colc1, colc2, colc3 = st.columns(3)
		with colc1:
			step_size = st.number_input("Enter depth step size (m)", min_value=0.1, value=10.0)
		with colc2:
			extrap = st.checkbox("Extrapolate top/base tables beyond max depth?", value=False)
		with colc3:
			default_sp = float(pd.to_numeric(edited["Depth"], errors="coerce").dropna().max() or 0.0)
			spill_point = st.number_input("Enter Spill Point depth (m)", value=default_sp)

		res = compute_dgrv_top_base_table(edited, step_size, extrap, spill_point)
		if "error" in res:
			st.error(res["error"])
			sGRV_m3 = np.zeros(num_sims)
		else:
			fig = make_area_volume_plot(res["depths"], res["top_interp"], res["base_interp"], res["vol_top"], res["vol_base"], res["dgrv"], spill_point, res["grv_sp_km2m"])
			st.plotly_chart(fig, use_container_width=True)
			grv_sp_km2m = res["grv_sp_km2m"]
			if np.isnan(grv_sp_km2m):
				st.warning("Spill Point outside data range. Enable extrapolation or adjust spill depth.")
				grv_sp_km2m = 0.0
			st.info(f"Using GRV at Spill Point = {grv_sp_km2m:.6f} km²·m (={grv_sp_km2m:,.0f} ×10^6 m³) for PV.", icon="ℹ️")
			sGRV_m3 = np.full(num_sims, grv_sp_km2m * 1_000_000.0)
			st.plotly_chart(make_hist_cdf_figure(np.full(num_sims, grv_sp_km2m), "GRV at Spill Point", "GRV (km²·m)", "calculated"), use_container_width=True)
			st.dataframe(summary_table(np.full(num_sims, grv_sp_km2m), decimals=6), use_container_width=True)

	elif grv_option == "Depth vs Area vs Thickness (spill-point)":
		st.markdown("### Depth vs Area vs Thickness")
		st.info("Provide a Top area table (Depth vs Top Area), constant thickness, step, extrapolation, and Spill Point depth. GRV at Spill Point will be used.")

		default_top = pd.DataFrame({
			"Depth": [2040, 2050, 2060, 2070, 2080, 2090, 2100, 2110, 2120, 2130, 2140, 2150, 2160, 2170, 2180, 2190, 2200, 2210, 2220, 2230, 2240, 2250, 2260, 2270, 2280, 2290, 2300, 2310, 2320, 2330, 2340, 2350, 2360, 2370, 2380, 2390, 2400],
			"Top area (km2)": [0.0, 0.06149313, 0.743428415, 1.31427065, 1.917695765, 2.581499555, 3.357041565, 4.24747926, 5.178613535, 6.122726925, 7.07979995, 8.04906025, 9.02872446, 10.02516695, 11.0326527, 12.09376061, 13.24614827, 14.32668123, 15.41667629, 16.61170115, 17.8503389, 19.01679186, 20.11461198, 21.1684545, 22.18402345, 23.13352475, 24.06078863, 24.95179712, 25.7778064, 26.55293472, 27.3037083, 28.04968615, 28.81130504, 29.61183057, 30.47923043, 31.46059753, 32.624375],
		})
		if "top_only_table" not in st.session_state:
			st.session_state["top_only_table"] = default_top

		ed_top = st.data_editor(
			st.session_state["top_only_table"],
			num_rows="dynamic",
			use_container_width=True,
			key="top_only_editor",
			column_config={
				"Depth": st.column_config.NumberColumn("Depth (m)", format="%.10f"),
				"Top area (km2)": st.column_config.NumberColumn("Top area (km²)", format="%.10f"),
			}
		)
		st.session_state["top_only_table"] = ed_top

		colt1, colt2, colt3 = st.columns(3)
		with colt1:
			thickness = st.number_input("Enter reservoir thickness (m)", min_value=0.0, value=50.0)
		with colt2:
			step_size_d = st.number_input("Enter depth step size (m)", min_value=0.1, value=10.0, key="optD_step")
		with colt3:
			extrap_d = st.checkbox("Extrapolate top table beyond max depth?", value=True, key="optD_extrap")

		default_sp_d = float(pd.to_numeric(ed_top["Depth"], errors="coerce").dropna().max() or 0.0)
		spill_point_d = st.number_input("Enter Spill Point depth (m)", value=default_sp_d, key="optD_sp")

		resD = compute_dgrv_top_plus_thickness(ed_top, thickness, step_size_d, extrap_d, spill_point_d)
		if "error" in resD:
			st.error(resD["error"])
			sGRV_m3 = np.zeros(num_sims)
		else:
			figD = make_area_volume_plot(resD["depths"], resD["top_interp"], resD["base_interp"], resD["vol_top"], resD["vol_base"], resD["dgrv"], spill_point_d, resD["grv_sp_km2m"])
			st.plotly_chart(figD, use_container_width=True)
			grv_sp_km2m_D = resD["grv_sp_km2m"]
			if np.isnan(grv_sp_km2m_D):
				st.warning("Spill Point outside data range. Enable extrapolation or adjust spill depth.")
				grv_sp_km2m_D = 0.0
			st.info(f"Using GRV at Spill Point = {grv_sp_km2m_D:.6f} km²·m (={grv_sp_km2m_D:,.0f} ×10^6 m³) for PV.", icon="ℹ️")
			sGRV_m3 = np.full(num_sims, grv_sp_km2m_D * 1_000_000.0)
			st.plotly_chart(make_hist_cdf_figure(np.full(num_sims, grv_sp_km2m_D), "GRV at Spill Point", "GRV (km²·m)", "calculated"), use_container_width=True)
			st.dataframe(summary_table(np.full(num_sims, grv_sp_km2m_D), decimals=6), use_container_width=True)

		# Add visual separation and GRV uncertainty multiplier section
	st.markdown("---")
	st.markdown("### GRV Uncertainty Multiplier (GRV_MP)")
	
	# Toggle for GRV_MP
	grv_mp_enabled = st.checkbox("Apply GRV uncertainty multiplier", value=False, help="Enable to apply uncertainty multiplier to GRV value")
	
	if grv_mp_enabled:
		grv_mp_type = st.radio(
			"GRV_MP definition",
			["Constant value", "Probability distribution"],
			horizontal=True,
			help="Choose how to define the GRV uncertainty multiplier"
		)
		
		if grv_mp_type == "Constant value":
			grv_mp_constant = st.number_input(
				"GRV_MP constant value",
				value=1.0,
				min_value=0.1,
				max_value=10.0,
				step=0.01,
				help="Constant multiplier for GRV value"
			)
			grv_mp_samples = np.full(num_sims, grv_mp_constant)
		else:
			# Use probability distribution
			grv_mp_samples = render_param(
				"GRV_MP",
				"GRV Uncertainty Multiplier",
				"multiplier",
				"PERT",
				{"min": 0.85, "mode": 1.00, "max": 1.20},
				num_sims,
				plot_unit_label="multiplier",
				stats_decimals=3
			)
	else:
		# GRV_MP disabled - set to 1.0
		grv_mp_samples = np.full(num_sims, 1.0)
		st.info("GRV_MP disabled - using GRV value as calculated (multiplier = 1.0)", icon="ℹ️")
	
	# Apply GRV_MP to get final GRV
	sGRV_m3_final = sGRV_m3 * grv_mp_samples
	
	# Display final GRV summary
	st.markdown("---")
	st.markdown("### Final GRV for Calculations")
	
	grv_base_value = float(sGRV_m3[0] / 1_000_000.0)  # Convert back to ×10^6 m³
	grv_mp_mean = float(np.mean(grv_mp_samples))
	grv_final_mean = float(np.mean(sGRV_m3_final) / 1_000_000.0)
	
	col1, col2, col3 = st.columns(3)
	with col1:
		if grv_option.startswith("Option C") or grv_option.startswith("Option D"):
			st.metric("GRV at Spill Point (×10^6 m³)", f"{grv_base_value:.3f}")
		else:
			st.metric("Base GRV (×10^6 m³)", f"{grv_base_value:.3f}")
	with col2:
		st.metric("GRV_MP (mean)", f"{grv_mp_mean:.3f}")
	with col3:
		st.metric("Final GRV (×10^6 m³)", f"{grv_final_mean:.3f}")
	
	# Show final GRV distribution
	st.plotly_chart(make_hist_cdf_figure(sGRV_m3_final / 1e6, "Final GRV distribution (after GRV_MP)", "GRV (×10^6 m³)", "calculated"), use_container_width=True)
	st.dataframe(summary_table(sGRV_m3_final / 1e6, decimals=2), use_container_width=True)
	
	# ========================================
	# SECTION 2: NET-TO-GROSS (NtG)
	# ========================================
	st.markdown("---")
	st.markdown("## Net-to-Gross (NtG)")
	
	sNtG = render_param("NtG", "Net-to-Gross NtG", "fraction", defaults["NtG"]["dist"], defaults["NtG"], num_sims, stats_decimals=3)
	
	# ========================================
	# SECTION 3: POROSITY
	# ========================================
	st.markdown("---")
	st.markdown("## Porosity")
	
	sp = render_param("p", "Porosity p", "fraction", defaults["p"]["dist"], defaults["p"], num_sims, stats_decimals=3)
	
	# ========================================
	# SECTION 4: CO₂ DENSITY
	# ========================================
	st.markdown("---")
	st.markdown("## CO₂ Density")
	
	# CoolProp references and citations
	with st.expander("📚 CoolProp Library References and Citations", expanded=False):
		st.markdown("""
		**This tool uses the CoolProp Library** ([https://coolprop.org/fluid_properties/fluids/CarbonDioxide.html](https://coolprop.org/fluid_properties/fluids/CarbonDioxide.html))
		
		**References:**
		
		**Equation of State:**
		R. Span and W. Wagner. A New Equation of State for Carbon Dioxide Covering the Fluid Region from the Triple Point Temperature to 1100 K at Pressures up to 800 MPa. *J. Phys. Chem. Ref. Data*, 25:1509–1596, 1996. doi:10.1063/1.555991.
		
		**Thermal Conductivity:**
		M. L. Huber, E. A. Sykioti, M. J. Assael, and R. A. Perkins. Reference correlation of the thermal conductivity of carbon dioxide from the triple point to 1100 k and up to 200 mpa. *Journal of Physical and Chemical Reference Data*, 2016. doi:10.1063/1.4940892.
		
		**CoolProp Library:**
		Bell, Ian H., et al. "Pure and Pseudo-pure Fluid Thermophysical Property Evaluation and the Open-Source Thermophysical Property Library CoolProp." *Industrial & Engineering Chemistry Research*, vol. 53, no. 6, 2014, pp. 2498–2508. doi:10.1021/ie4033999.
		
		**BibTeX citation:**
		```bibtex
		@article{doi:10.1021/ie4033999,
			author = {Bell, Ian H. and Wronski, Jorrit and Quoilin, Sylvain and Lemort, Vincent},
			title = {Pure and Pseudo-pure Fluid Thermophysical Property Evaluation and
					 the Open-Source Thermophysical Property Library CoolProp},
			journal = {Industrial & Engineering Chemistry Research},
			volume = {53},
			number = {6},
			pages = {2498--2508},
			year = {2014},
			doi = {10.1021/ie4033999},
			URL = {http://pubs.acs.org/doi/abs/10.1021/ie4033999},
			eprint = {http://pubs.acs.org/doi/pdf/10.1021/ie4033999}
		}
		```
		""")
	
	co2_density_method = st.radio(
		"CO₂ density calculation method",
		["Direct distribution input", "Calculate from Pressure-Temperature"],
		horizontal=True,
		help="Choose how to calculate CO₂ density"
	)
	
	if co2_density_method == "Direct distribution input":
		# Original method - direct distribution input
		sd = render_param("d", "CO₂ density d", "kg/m³", defaults["d"]["dist"], defaults["d"], num_sims, stats_decimals=2)
		co2_phase_info = None
		
	else:  # Calculate from Pressure-Temperature
		if not COOLPROP_AVAILABLE:
			st.error("CoolProp is required for P-T based CO₂ density calculation. Install with: pip install coolprop")
			sd = render_param("d", "CO₂ density d", "kg/m³", defaults["d"]["dist"], defaults["d"], num_sims)
			co2_phase_info = None
		else:
			# P-T calculation method
			pt_calc_method = st.radio(
				"P-T calculation method",
				["Manual input", "P-T distributions", "Onshore scenario", "Offshore scenario"],
				horizontal=True,
				help="Choose how to determine pressure and temperature"
			)
			
			if pt_calc_method == "Manual input":
				# Manual P-T input
				col1, col2 = st.columns(2)
				with col1:
					T_k = st.number_input("Temperature T [K]", value=325.15, min_value=200.0, max_value=400.0, step=0.1)
				with col2:
					P_mpa = st.number_input("Pressure P [MPa]", value=27.1, min_value=0.1, max_value=1000.0, step=0.1)
				
				# Calculate single density value
				density_kg_m3, phase = calculate_co2_density_from_pt(T_k, P_mpa)
				
				if not np.isnan(density_kg_m3):
					st.success(f"CO₂ density: {density_kg_m3:.2f} kg/m³ ({phase} phase)")
					sd = np.full(num_sims, density_kg_m3)
					co2_phase_info = np.full(num_sims, phase, dtype=object)
					
					# Show density distribution plot for manual input
					st.markdown("### CO₂ Density Distribution")
					st.plotly_chart(make_hist_cdf_figure(sd, "CO₂ Density Distribution", "Density (kg/m³)", "calculated"), use_container_width=True)
					st.dataframe(summary_table(sd, decimals=2), use_container_width=True)
				else:
					st.error("Failed to calculate CO₂ density. Check P-T values.")
					sd = render_param("d", "CO₂ density d", "kg/m³", defaults["d"]["dist"], defaults["d"], num_sims)
					co2_phase_info = None
					
			elif pt_calc_method == "P-T distributions":
				# P-T distribution input
				st.info("P-T distributions: Sample from temperature and pressure distributions")
				
				col1, col2 = st.columns(2)
				with col1:
					st.markdown("**Temperature distribution:**")
					sT = render_param("T", "Temperature T", "K", "PERT", {"min": 300.0, "mode": 325.0, "max": 350.0}, num_sims)
				with col2:
					st.markdown("**Pressure distribution:**")
					sP = render_param("P", "Pressure P", "MPa", "PERT", {"min": 20.0, "mode": 27.0, "max": 35.0}, num_sims)
				
				# Calculate density for each P-T sample
				sd, co2_phase_info = sample_co2_density_from_pt_distributions(sT, sP)
				
				# Check if calculation was successful
				if np.any(np.isnan(sd)):
					nan_count = np.sum(np.isnan(sd))
					if nan_count == len(sd):
						st.error("All CO₂ density calculations failed. Check P-T ranges.")
						sd = render_param("d", "CO₂ density d", "kg/m³", defaults["d"]["dist"], defaults["d"], num_sims)
						co2_phase_info = None
					else:
						st.warning(f"CO₂ density calculation failed for {nan_count} out of {len(sd)} samples. Check P-T ranges.")
						# Replace NaN values with mean of valid values
						valid_densities = sd[~np.isnan(sd)]
						if len(valid_densities) > 0:
							mean_density = np.mean(valid_densities)
							sd = np.where(np.isnan(sd), mean_density, sd)
							st.info(f"Replaced {nan_count} failed calculations with mean density: {mean_density:.2f} kg/m³")
						else:
							st.error("No valid density calculations. Using default distribution.")
							sd = render_param("d", "CO₂ density d", "kg/m³", defaults["d"]["dist"], defaults["d"], num_sims)
							co2_phase_info = None
				else:
					st.success(f"Successfully calculated CO₂ density for all {len(sd)} samples")
				
				# Show P-T distribution plot
				fig_pt = go.Figure()
				fig_pt.add_trace(go.Scatter(
					x=sT, y=sP, mode='markers', 
					marker=dict(size=4, color=sd, colorscale='Viridis', colorbar=dict(title="Density (kg/m³)")),
					name='P-T samples'
				))
				fig_pt.update_layout(
					title="P-T Distribution with CO₂ Density",
					xaxis_title="Temperature (K)",
					yaxis_title="Pressure (MPa)",
					showlegend=False
				)
				st.plotly_chart(fig_pt, use_container_width=True)
				
				# Show density distribution plot and table
				st.markdown("### CO₂ Density Distribution")
				st.plotly_chart(make_hist_cdf_figure(sd, "CO₂ Density Distribution", "Density (kg/m³)", "calculated"), use_container_width=True)
				st.dataframe(summary_table(sd, decimals=2), use_container_width=True)
				
				# Display CO2 phase information if available
				if co2_phase_info is not None and len(co2_phase_info) > 0:
					st.markdown("---")
					st.markdown("### CO₂ Phase Information")
					
					# Count phases
					unique_phases, phase_counts = np.unique(co2_phase_info, return_counts=True)
					phase_percentages = (phase_counts / len(co2_phase_info)) * 100
					
					# Create phase distribution table
					phase_data = []
					for phase, count, percentage in zip(unique_phases, phase_counts, phase_percentages):
						phase_data.append({
							"Phase": phase,
							"Trials": int(count),
							"Percentage": f"{percentage:.1f}%"
						})
					
					phase_df = pd.DataFrame(phase_data)
					st.markdown("**Phase Distribution Table:**")
					st.dataframe(phase_df, use_container_width=True)
					
					col1, col2 = st.columns(2)
					with col1:
						st.write("**Phase distribution:**")
						for phase, count, percentage in zip(unique_phases, phase_counts, phase_percentages):
							st.write(f"- {phase}: {count} samples ({percentage:.1f}%)")
					
					with col2:
						st.write("**CO₂ density statistics:**")
						density_stats = summarize_array(sd)
						if density_stats:
							st.write(f"- Mean: {density_stats['mean']:.2f} kg/m³")
							st.write(f"- Std Dev: {density_stats['std_dev']:.2f} kg/m³")
							st.write(f"- Min: {density_stats['min']:.2f} kg/m³")
							st.write(f"- Max: {density_stats['max']:.2f} kg/m³")
					
					# Create phase distribution plot
					fig_phase = go.Figure(data=[
						go.Bar(x=unique_phases, y=phase_counts, text=[f"{p:.1f}%" for p in phase_percentages], textposition='auto')
					])
					fig_phase.update_layout(
						title="CO₂ Phase Distribution",
						xaxis_title="Phase",
						yaxis_title="Number of samples",
						showlegend=False
					)
					st.plotly_chart(fig_phase, use_container_width=True)
					
			elif pt_calc_method == "Onshore scenario":
				# Onshore P-T calculation
				st.info("Onshore scenario: Calculate P-T from depth and geothermal parameters")
				
				# Geothermal parameters with distribution inputs
				st.markdown("**Geothermal parameters:**")
				col1, col2 = st.columns(2)
				with col1:
					sGT_grad = render_param("GT_grad", "Geothermal gradient", "K/km", "PERT", {"min": 25.0, "mode": 30.0, "max": 35.0}, num_sims)
				with col2:
					sa_surftemp = render_param("a_surftemp", "Average surface temperature", "K", "PERT", {"min": 285.0, "mode": 288.0, "max": 291.0}, num_sims)
				
				# Depth input method
				onshore_option = st.radio("Depth input method", ["Measured depth (MD) to average unit depth", "Use Ground Level + Depths"], horizontal=True)
				
				if onshore_option == "Measured depth (MD) to average unit depth":
					st.markdown("**Depth parameters:**")
					savgmudline = render_param("avgmudline", "Average mudline depth", "m", "PERT", {"min": 1800.0, "mode": 2000.0, "max": 2200.0}, num_sims)
					GL = topdepth = basedepth = None
					
					# Calculate P-T for each sample
					sd = np.full(num_sims, float('nan'))
					co2_phase_info = np.full(num_sims, "unknown", dtype=object)
					
					for i in range(num_sims):
						try:
							T_k, P_mpa = compute_onshore_state(GL, topdepth, basedepth, savgmudline[i], sGT_grad[i], sa_surftemp[i])
							density_kg_m3, phase = calculate_co2_density_from_pt(T_k, P_mpa)
							if not np.isnan(density_kg_m3):
								sd[i] = density_kg_m3
								co2_phase_info[i] = phase
						except (ValueError, Exception):
							continue
					
				else:  # Use Ground Level + Depths
					st.markdown("**Depth parameters:**")
					col1, col2, col3 = st.columns(3)
					with col1:
						sGL = render_param("GL", "Ground level", "m above MSL", "PERT", {"min": -10.0, "mode": 0.0, "max": 10.0}, num_sims)
					with col2:
						stopdepth = render_param("topdepth", "Top depth", "m below MSL", "PERT", {"min": 1800.0, "mode": 2000.0, "max": 2200.0}, num_sims)
					with col3:
						sbasedepth = render_param("basedepth", "Base depth", "m below MSL", "PERT", {"min": 1850.0, "mode": 2050.0, "max": 2250.0}, num_sims)
					
					# Calculate P-T for each sample
					sd = np.full(num_sims, float('nan'))
					co2_phase_info = np.full(num_sims, "unknown", dtype=object)
					
					for i in range(num_sims):
						try:
							T_k, P_mpa = compute_onshore_state(sGL[i], stopdepth[i], sbasedepth[i], None, sGT_grad[i], sa_surftemp[i])
							density_kg_m3, phase = calculate_co2_density_from_pt(T_k, P_mpa)
							if not np.isnan(density_kg_m3):
								sd[i] = density_kg_m3
								co2_phase_info[i] = phase
						except (ValueError, Exception):
							continue
				
				# Check results and handle failures
				valid_count = np.sum(~np.isnan(sd))
				if valid_count == 0:
					st.error("All CO₂ density calculations failed. Check parameters.")
					sd = render_param("d", "CO₂ density d", "kg/m³", defaults["d"]["dist"], defaults["d"], num_sims)
					co2_phase_info = None
				elif valid_count < num_sims:
					st.warning(f"CO₂ density calculation failed for {num_sims - valid_count} out of {num_sims} samples.")
					# Replace NaN values with mean of valid values
					valid_densities = sd[~np.isnan(sd)]
					if len(valid_densities) > 0:
						mean_density = np.mean(valid_densities)
						sd = np.where(np.isnan(sd), mean_density, sd)
						st.info(f"Replaced {num_sims - valid_count} failed calculations with mean density: {mean_density:.2f} kg/m³")
				else:
					st.success(f"Successfully calculated CO₂ density for all {num_sims} samples")
				
				# Show density distribution plot and table
				st.markdown("### CO₂ Density Distribution")
				st.plotly_chart(make_hist_cdf_figure(sd, "CO₂ Density Distribution", "Density (kg/m³)"), use_container_width=True)
				st.dataframe(summary_table(sd, decimals=2), use_container_width=True)
				
				# Display CO2 phase information if available
				if co2_phase_info is not None and len(co2_phase_info) > 0:
					st.markdown("---")
					st.markdown("### CO₂ Phase Information")
					
					# Count phases
					unique_phases, phase_counts = np.unique(co2_phase_info, return_counts=True)
					phase_percentages = (phase_counts / len(co2_phase_info)) * 100
					
					# Create phase distribution table
					phase_data = []
					for phase, count, percentage in zip(unique_phases, phase_counts, phase_percentages):
						phase_data.append({
							"Phase": phase,
							"Trials": int(count),
							"Percentage": f"{percentage:.1f}%"
						})
					
					phase_df = pd.DataFrame(phase_data)
					st.markdown("**Phase Distribution Table:**")
					st.dataframe(phase_df, use_container_width=True)
					
					col1, col2 = st.columns(2)
					with col1:
						st.write("**Phase distribution:**")
						for phase, count, percentage in zip(unique_phases, phase_counts, phase_percentages):
							st.write(f"- {phase}: {count} samples ({percentage:.1f}%)")
					
					with col2:
						st.write("**CO₂ density statistics:**")
						density_stats = summarize_array(sd)
						if density_stats:
							st.write(f"- Mean: {density_stats['mean']:.2f} kg/m³")
							st.write(f"- Std Dev: {density_stats['std_dev']:.2f} kg/m³")
							st.write(f"- Min: {density_stats['min']:.2f} kg/m³")
							st.write(f"- Max: {density_stats['max']:.2f} kg/m³")
					
					# Create phase distribution plot
					fig_phase = go.Figure(data=[
						go.Bar(x=unique_phases, y=phase_counts, text=[f"{p:.1f}%" for p in phase_percentages], textposition='auto')
					])
					fig_phase.update_layout(
						title="CO₂ Phase Distribution",
						xaxis_title="Phase",
						yaxis_title="Number of samples",
						showlegend=False
					)
					st.plotly_chart(fig_phase, use_container_width=True)
					
			else:  # Offshore scenario
				# Offshore P-T calculation
				st.info("Offshore scenario: Calculate P-T from depth and geothermal parameters")
				
				# Geothermal parameters with distribution inputs
				st.markdown("**Geothermal parameters:**")
				col1, col2 = st.columns(2)
				with col1:
					sGT_grad = render_param("GT_grad_off", "Geothermal gradient", "K/km", "PERT", {"min": 25.0, "mode": 30.0, "max": 35.0}, num_sims)
				with col2:
					sa_seabtemp = render_param("a_seabtemp", "Average seabed temperature", "K", "PERT", {"min": 275.0, "mode": 278.0, "max": 281.0}, num_sims)
				
				# Depth input method
				offshore_option = st.radio("Depth input method", ["Use average Mudline + waterdepth", "Use depths only"], horizontal=True)
				
				if offshore_option == "Use average Mudline + waterdepth":
					st.markdown("**Depth parameters:**")
					col1, col2 = st.columns(2)
					with col1:
						savgmudline = render_param("avgmudline_off", "Average mudline depth", "m", "PERT", {"min": 1800.0, "mode": 2000.0, "max": 2200.0}, num_sims)
					with col2:
						swaterdepth = render_param("waterdepth", "Water depth", "m", "PERT", {"min": 80.0, "mode": 100.0, "max": 120.0}, num_sims)
					
					# Calculate P-T for each sample
					sd = np.full(num_sims, float('nan'))
					co2_phase_info = np.full(num_sims, "unknown", dtype=object)
					
					for i in range(num_sims):
						try:
							T_k, P_mpa = compute_offshore_state(swaterdepth[i], None, None, savgmudline[i], sGT_grad[i], sa_seabtemp[i])
							density_kg_m3, phase = calculate_co2_density_from_pt(T_k, P_mpa)
							if not np.isnan(density_kg_m3):
								sd[i] = density_kg_m3
								co2_phase_info[i] = phase
						except (ValueError, Exception):
							continue
					
				else:  # Use depths only
					st.markdown("**Depth parameters:**")
					col1, col2 = st.columns(2)
					with col1:
						stopdepth = render_param("topdepth_off", "Top depth", "m below MSL", "PERT", {"min": 1800.0, "mode": 2000.0, "max": 2200.0}, num_sims)
					with col2:
						sbasedepth = render_param("basedepth_off", "Base depth", "m below MSL", "PERT", {"min": 1850.0, "mode": 2050.0, "max": 2250.0}, num_sims)
					
					# Store depth samples in session state for correlation plotting
					st.session_state['stopdepth_off_samples'] = stopdepth
					st.session_state['sbasedepth_off_samples'] = sbasedepth
					
					# Calculate P-T for each sample
					sd = np.full(num_sims, float('nan'))
					co2_phase_info = np.full(num_sims, "unknown", dtype=object)
					
					for i in range(num_sims):
						try:
							T_k, P_mpa = compute_offshore_state(None, stopdepth[i], sbasedepth[i], None, sGT_grad[i], sa_seabtemp[i])
							density_kg_m3, phase = calculate_co2_density_from_pt(T_k, P_mpa)
							if not np.isnan(density_kg_m3):
								sd[i] = density_kg_m3
								co2_phase_info[i] = phase
						except (ValueError, Exception):
							continue
				
				# Check results and handle failures
				valid_count = np.sum(~np.isnan(sd))
				if valid_count == 0:
					st.error("All CO₂ density calculations failed. Check parameters.")
					sd = render_param("d", "CO₂ density d", "kg/m³", defaults["d"]["dist"], defaults["d"], num_sims)
					co2_phase_info = None
				elif valid_count < num_sims:
					st.warning(f"CO₂ density calculation failed for {num_sims - valid_count} out of {num_sims} samples.")
					# Replace NaN values with mean of valid values
					valid_densities = sd[~np.isnan(sd)]
					if len(valid_densities) > 0:
						mean_density = np.mean(valid_densities)
						sd = np.where(np.isnan(sd), mean_density, sd)
						st.info(f"Replaced {num_sims - valid_count} failed calculations with mean density: {mean_density:.2f} kg/m³")
				else:
					st.success(f"Successfully calculated CO₂ density for all {num_sims} samples")
				
				# Show density distribution plot and table
				st.markdown("### CO₂ Density Distribution")
				st.plotly_chart(make_hist_cdf_figure(sd, "CO₂ Density Distribution", "Density (kg/m³)"), use_container_width=True)
				st.dataframe(summary_table(sd, decimals=2), use_container_width=True)
				
				# Display CO2 phase information if available
				if co2_phase_info is not None and len(co2_phase_info) > 0:
					st.markdown("---")
					st.markdown("### CO₂ Phase Information")
					
					# Count phases
					unique_phases, phase_counts = np.unique(co2_phase_info, return_counts=True)
					phase_percentages = (phase_counts / len(co2_phase_info)) * 100
					
					# Create phase distribution table
					phase_data = []
					for phase, count, percentage in zip(unique_phases, phase_counts, phase_percentages):
						phase_data.append({
							"Phase": phase,
							"Trials": int(count),
							"Percentage": f"{percentage:.1f}%"
						})
					
					phase_df = pd.DataFrame(phase_data)
					st.markdown("**Phase Distribution Table:**")
					st.dataframe(phase_df, use_container_width=True)
					
					col1, col2 = st.columns(2)
					with col1:
						st.write("**Phase distribution:**")
						for phase, count, percentage in zip(unique_phases, phase_counts, phase_percentages):
							st.write(f"- {phase}: {count} samples ({percentage:.1f}%)")
					
					with col2:
						st.write("**CO₂ density statistics:**")
						density_stats = summarize_array(sd)
						if density_stats:
							st.write(f"- Mean: {density_stats['mean']:.2f} kg/m³")
							st.write(f"- Std Dev: {density_stats['std_dev']:.2f} kg/m³")
							st.write(f"- Min: {density_stats['min']:.2f} kg/m³")
							st.write(f"- Max: {density_stats['max']:.2f} kg/m³")
					
					# Create phase distribution plot
					fig_phase = go.Figure(data=[
						go.Bar(x=unique_phases, y=phase_counts, text=[f"{p:.1f}%" for p in phase_percentages], textposition='auto')
					])
					fig_phase.update_layout(
						title="CO₂ Phase Distribution",
						xaxis_title="Phase",
						yaxis_title="Number of samples",
						showlegend=False
					)
					st.plotly_chart(fig_phase, use_container_width=True)
	
	# ========================================
	# SECTION 5: STORAGE EFFICIENCY
	# ========================================
	st.markdown("---")
	st.markdown("## Storage Efficiency")
	
	sSE = render_param("SE", "Storage Efficiency SE", "fraction", defaults["SE"]["dist"], defaults["SE"], num_sims, stats_decimals=3)
	
	# ========================================
	# INPUT SUMMARY TABLE
	# ========================================
	st.markdown("---")
	st.markdown("## Input Parameters Summary")
	
	# Create summary table with the five main input parameters
	def create_input_summary_table():
		# Get statistics for each parameter
		grv_stats = summarize_array(sGRV_m3_final / 1e6)  # Convert to ×10^6 m³
		ntg_stats = summarize_array(sNtG)
		porosity_stats = summarize_array(sp)
		se_stats = summarize_array(sSE)
		density_stats = summarize_array(sd)
		
		# Create summary data
		summary_data = [
			{
				"Input Parameter": "Gross Rock Volume (GRV)",
				"Unit": "×10^6 m³",
				"Mean": f"{grv_stats.get('mean', 0):.3f}",
				"Mode": f"{grv_stats.get('mode', 0):.3f}",
				"P10": f"{grv_stats.get('P10', 0):.3f}",
				"P50": f"{grv_stats.get('P50', 0):.3f}",
				"P90": f"{grv_stats.get('P90', 0):.3f}"
			},
			{
				"Input Parameter": "Net-to-Gross (NtG)",
				"Unit": "fraction",
				"Mean": f"{ntg_stats.get('mean', 0):.3f}",
				"Mode": f"{ntg_stats.get('mode', 0):.3f}",
				"P10": f"{ntg_stats.get('P10', 0):.3f}",
				"P50": f"{ntg_stats.get('P50', 0):.3f}",
				"P90": f"{ntg_stats.get('P90', 0):.3f}"
			},
			{
				"Input Parameter": "Porosity",
				"Unit": "fraction",
				"Mean": f"{porosity_stats.get('mean', 0):.3f}",
				"Mode": f"{porosity_stats.get('mode', 0):.3f}",
				"P10": f"{porosity_stats.get('P10', 0):.3f}",
				"P50": f"{porosity_stats.get('P50', 0):.3f}",
				"P90": f"{porosity_stats.get('P90', 0):.3f}"
			},
			{
				"Input Parameter": "Storage Efficiency",
				"Unit": "fraction",
				"Mean": f"{se_stats.get('mean', 0):.3f}",
				"Mode": f"{se_stats.get('mode', 0):.3f}",
				"P10": f"{se_stats.get('P10', 0):.3f}",
				"P50": f"{se_stats.get('P50', 0):.3f}",
				"P90": f"{se_stats.get('P90', 0):.3f}"
			},
			{
				"Input Parameter": "CO₂ Density",
				"Unit": "kg/m³",
				"Mean": f"{density_stats.get('mean', 0):.1f}",
				"Mode": f"{density_stats.get('mode', 0):.1f}",
				"P10": f"{density_stats.get('P10', 0):.1f}",
				"P50": f"{density_stats.get('P50', 0):.1f}",
				"P90": f"{density_stats.get('P90', 0):.1f}"
			}
		]
		
		# Create DataFrame and style it
		summary_df = pd.DataFrame(summary_data)
		summary_styled = summary_df.style.set_properties(**{
			'text-align': 'center',
			'vertical-align': 'middle'
		})
		
		return summary_styled
	
	# Display the summary table
	st.dataframe(create_input_summary_table(), use_container_width=True)
	
	# ========================================
	# PARAMETER DEPENDENCIES/CORRELATIONS
	# ========================================
	
	# Function to create correlation scatter plot
	def create_correlation_scatter_plot(param1_values, param2_values, param1_name, param2_name, correlation_value):
		"""
		Create a scatter plot showing correlation between two parameters.
		Plots every 10th simulation point to avoid overcrowding.
		"""
		# Sample every 10th point
		sample_indices = np.arange(0, len(param1_values), 10)
		param1_sampled = param1_values[sample_indices]
		param2_sampled = param2_values[sample_indices]
		
		fig = go.Figure()
		
		# Add scatter plot
		fig.add_trace(go.Scatter(
			x=param1_sampled,
			y=param2_sampled,
			mode='markers',
			marker=dict(
				size=6,
				color='blue',
				opacity=0.6
			),
			name=f'{param1_name} vs {param2_name}',
			hovertemplate=f'{param1_name}: %{{x:.3f}}<br>{param2_name}: %{{y:.3f}}<extra></extra>'
		))
		
		# Add trend line if correlation is not zero
		if abs(correlation_value) > 0.01:
			# Calculate trend line
			z = np.polyfit(param1_sampled, param2_sampled, 1)
			p = np.poly1d(z)
			trend_x = np.linspace(param1_sampled.min(), param1_sampled.max(), 100)
			trend_y = p(trend_x)
			
			fig.add_trace(go.Scatter(
				x=trend_x,
				y=trend_y,
				mode='lines',
				line=dict(color='red', width=2, dash='dash'),
				name=f'Trend (r={correlation_value:.2f})',
				showlegend=True
			))
		
		fig.update_layout(
			title=f'{param1_name} vs {param2_name} Correlation (r={correlation_value:.2f})',
			xaxis_title=param1_name,
			yaxis_title=param2_name,
			height=400,
			showlegend=True,
			margin=dict(l=40, r=40, t=60, b=40)
		)
		
		return fig
	st.markdown("---")
	st.markdown("## Parameter Dependencies/Correlations")
	st.markdown("Configure correlations between specific parameter pairs to model realistic dependencies in your analysis.")
	
	# Main toggle for parameter dependencies
	dependencies_enabled = st.checkbox(
		"Enable parameter dependencies/correlations", 
		value=False, 
		help="Toggle to enable or disable all parameter correlations"
	)
	
	if dependencies_enabled:
		st.info("📊 Parameter correlations will be applied during Monte Carlo sampling to create more realistic parameter relationships.")
		
		# Create tabs for different dependency categories
		dependency_tabs = st.tabs([
			"CO₂ Density Dependencies", 
			"Depth Dependencies", 
			"Reservoir Properties", 
			"Geometry Dependencies"
		])
		
		# Initialize session state for correlation values
		if 'correlation_values' not in st.session_state:
			st.session_state.correlation_values = {}
		
		# CO₂ Density Dependencies Tab
		with dependency_tabs[0]:
			st.markdown("### Temperature and Pressure Correlation")
			
			# Check if CO2 density P-T distribution is selected
			pt_density_selected = (co2_density_method == "Calculate from Pressure-Temperature" and 
								  'pt_calc_method' in locals() and pt_calc_method == "P-T distributions")
			
			if pt_density_selected:
				st.success("✅ Temperature and Pressure correlation available (P-T distributions selected)")
				
				# Temperature-Pressure correlation slider
				st.session_state.correlation_values['temp_pressure'] = st.slider(
					"Temperature-Pressure correlation coefficient",
					min_value=-1.0,
					max_value=1.0,
					value=st.session_state.correlation_values.get('temp_pressure', 0.0),
					step=0.01,
					key="corr_temp_pressure",
					help="Correlation between Temperature and Pressure"
				)
				
				# Show correlation scatter plot
				if abs(st.session_state.correlation_values['temp_pressure']) > 0.01:
					st.markdown("**Temperature vs Pressure Correlation Plot**")
					# Apply correlation temporarily for visualization
					sT_temp, sP_temp = apply_correlation(sT.copy(), sP.copy(), st.session_state.correlation_values['temp_pressure'])
					scatter_fig = create_correlation_scatter_plot(
						sT_temp, sP_temp, 
						"Temperature T", "Pressure P", 
						st.session_state.correlation_values['temp_pressure']
					)
					st.plotly_chart(scatter_fig, use_container_width=True)
				
				# Note: CO2 density will be recalculated in the main calculation section with correlated P-T values
			else:
				st.warning("⚠️ Temperature-Pressure correlation requires 'Calculate from Pressure-Temperature' → 'P-T distributions' to be selected")
				st.session_state.correlation_values['temp_pressure'] = 0.0
		
		# Depth Dependencies Tab
		with dependency_tabs[1]:
			st.markdown("### Top Depth and Base Depth Correlation")
			
			# Check if depth-based GRV calculation is selected
			depth_grv_selected = (grv_option == "From Depth/Top/Base Areas (table)" or 
								 grv_option == "Depth vs Area vs Thickness (spill-point)")
			
			if depth_grv_selected:
				st.success("✅ Top Depth and Base Depth correlation available (depth-based GRV calculation selected)")
				
				# Top Depth-Base Depth correlation slider
				st.session_state.correlation_values['top_base_depth'] = st.slider(
					"Top Depth-Base Depth correlation coefficient",
					min_value=-1.0,
					max_value=1.0,
					value=st.session_state.correlation_values.get('top_base_depth', 0.0),
					step=0.01,
					key="corr_top_base_depth",
					help="Correlation between Top Depth and Base Depth"
				)
				
				# Show correlation scatter plot
				if abs(st.session_state.correlation_values['top_base_depth']) > 0.01:
					st.markdown("**Top Depth vs Base Depth Correlation Plot**")
					# Get depth samples from session state
					current_stopdepth_off = st.session_state.get('stopdepth_off_samples', np.array([]))
					current_sbasedepth_off = st.session_state.get('sbasedepth_off_samples', np.array([]))
					
					if current_stopdepth_off.size > 0 and current_sbasedepth_off.size > 0:
						# Apply correlation temporarily for visualization
						stopdepth_temp, sbasedepth_temp = apply_correlation(current_stopdepth_off.copy(), current_sbasedepth_off.copy(), st.session_state.correlation_values['top_base_depth'])
						scatter_fig = create_correlation_scatter_plot(
							stopdepth_temp, sbasedepth_temp, 
							"Top Depth", "Base Depth", 
							st.session_state.correlation_values['top_base_depth']
						)
						st.plotly_chart(scatter_fig, use_container_width=True)
					else:
						st.warning("Cannot plot Top Depth vs Base Depth correlation: samples not available. Please select a depth-based GRV calculation method and ensure inputs are valid.")
			else:
				st.warning("⚠️ Top Depth-Base Depth correlation requires depth-based GRV calculation to be selected")
				st.session_state.correlation_values['top_base_depth'] = 0.0
		
		# Reservoir Properties Tab
		with dependency_tabs[2]:
			st.markdown("### Reservoir Property Correlations")
			
			# Porosity and Storage Efficiency correlation
			st.markdown("#### Porosity and Storage Efficiency")
			
			# Porosity-Storage Efficiency correlation slider
			st.session_state.correlation_values['porosity_se'] = st.slider(
				"Porosity-Storage Efficiency correlation coefficient",
				min_value=-1.0,
				max_value=1.0,
				value=st.session_state.correlation_values.get('porosity_se', 0.0),
				step=0.01,
				key="corr_porosity_se",
				help="Correlation between Porosity and Storage Efficiency"
			)
			
						# Show correlation scatter plot
			if abs(st.session_state.correlation_values['porosity_se']) > 0.01:
				st.markdown("**Porosity vs Storage Efficiency Correlation Plot**")
				# Apply correlation temporarily for visualization
				sp_temp, sSE_temp = apply_correlation(sp.copy(), sSE.copy(), st.session_state.correlation_values['porosity_se'])
				scatter_fig = create_correlation_scatter_plot(
					sp_temp, sSE_temp, 
					"Porosity p", "Storage Efficiency SE",
					st.session_state.correlation_values['porosity_se']
				)
				st.plotly_chart(scatter_fig, use_container_width=True)
			
			# Porosity and Net-to-Gross correlation
			st.markdown("#### Porosity and Net-to-Gross")
			
			# Porosity-Net-to-Gross correlation slider
			st.session_state.correlation_values['porosity_ntg'] = st.slider(
				"Porosity-Net-to-Gross correlation coefficient",
				min_value=-1.0,
				max_value=1.0,
				value=st.session_state.correlation_values.get('porosity_ntg', 0.0),
				step=0.01,
				key="corr_porosity_ntg",
				help="Correlation between Porosity and Net-to-Gross"
			)
			
			# Show correlation scatter plot
			if abs(st.session_state.correlation_values['porosity_ntg']) > 0.01:
				st.markdown("**Porosity vs Net-to-Gross Correlation Plot**")
				# Apply correlation temporarily for visualization
				sp_temp, sNtG_temp = apply_correlation(sp.copy(), sNtG.copy(), st.session_state.correlation_values['porosity_ntg'])
				scatter_fig = create_correlation_scatter_plot(
					sp_temp, sNtG_temp, 
					"Porosity p", "Net-to-Gross NtG", 
					st.session_state.correlation_values['porosity_ntg']
				)
				st.plotly_chart(scatter_fig, use_container_width=True)
		
		# Geometry Dependencies Tab
		with dependency_tabs[3]:
			st.markdown("### Geometry Factor Correlations")
			
			# Check if GRV Area, Geometry Factor and Thickness is selected
			geometry_dependencies_available = (grv_option == "From Area, Geometry Factor and Thickness")
			
			if geometry_dependencies_available:
				st.success("✅ Geometry factor correlations available")
				
				# Thickness and GCF correlation
				st.markdown("#### Thickness and Geometry Correction Factor (GCF)")
				
				# Thickness-GCF correlation slider
				st.session_state.correlation_values['thickness_gcf'] = st.slider(
					"Thickness-GCF correlation coefficient",
					min_value=-1.0,
					max_value=1.0,
					value=st.session_state.correlation_values.get('thickness_gcf', 0.0),
					step=0.01,
					key="corr_thickness_gcf",
					help="Correlation between Thickness and Geometry Correction Factor"
				)
				
				# Show correlation scatter plot
				if abs(st.session_state.correlation_values['thickness_gcf']) > 0.01:
					st.markdown("**Thickness vs GCF Correlation Plot**")
					# Apply correlation temporarily for visualization
					sh_temp, sGCF_temp = apply_correlation(sh.copy(), sGCF.copy(), st.session_state.correlation_values['thickness_gcf'])
					scatter_fig = create_correlation_scatter_plot(
						sh_temp, sGCF_temp, 
						"Thickness h", "Geometry Correction Factor GCF", 
						st.session_state.correlation_values['thickness_gcf']
					)
					st.plotly_chart(scatter_fig, use_container_width=True)
				
				# Area and GCF correlation
				st.markdown("#### Area and Geometry Correction Factor (GCF)")
				
				# Area-GCF correlation slider
				st.session_state.correlation_values['area_gcf'] = st.slider(
					"Area-GCF correlation coefficient",
					min_value=-1.0,
					max_value=1.0,
					value=st.session_state.correlation_values.get('area_gcf', 0.0),
					step=0.01,
					key="corr_area_gcf",
					help="Correlation between Area and Geometry Correction Factor"
				)
				
				# Show correlation scatter plot
				if abs(st.session_state.correlation_values['area_gcf']) > 0.01:
					st.markdown("**Area vs GCF Correlation Plot**")
					# Apply correlation temporarily for visualization
					sA_temp, sGCF_temp = apply_correlation(sA.copy(), sGCF.copy(), st.session_state.correlation_values['area_gcf'])
					scatter_fig = create_correlation_scatter_plot(
						sA_temp, sGCF_temp, 
						"Area A", "Geometry Correction Factor GCF", 
						st.session_state.correlation_values['area_gcf']
					)
					st.plotly_chart(scatter_fig, use_container_width=True)
			else:
				st.warning("⚠️ Geometry factor correlations require 'From Area, Geometry Factor and Thickness' to be selected")
				st.session_state.correlation_values['thickness_gcf'] = 0.0
				st.session_state.correlation_values['area_gcf'] = 0.0
		
		# Summary of all correlations
		st.markdown("---")
		st.markdown("### Correlation Summary")
		
		# Create correlation summary table
		correlation_summary = []
		
		# Add available correlations to summary
		if pt_density_selected:
			correlation_summary.append({
				"Parameter Pair": "Temperature ↔ Pressure",
				"Correlation": f"{st.session_state.correlation_values.get('temp_pressure', 0.0):.2f}",
				"Status": "✅ Active"
			})
		
		if depth_grv_selected:
			correlation_summary.append({
				"Parameter Pair": "Top Depth ↔ Base Depth",
				"Correlation": f"{st.session_state.correlation_values.get('top_base_depth', 0.0):.2f}",
				"Status": "✅ Active"
			})
		
		correlation_summary.extend([
			{
				"Parameter Pair": "Porosity ↔ Storage Efficiency",
				"Correlation": f"{st.session_state.correlation_values.get('porosity_se', 0.0):.2f}",
				"Status": "✅ Active"
			},
			{
				"Parameter Pair": "Porosity ↔ Net-to-Gross",
				"Correlation": f"{st.session_state.correlation_values.get('porosity_ntg', 0.0):.2f}",
				"Status": "✅ Active"
			}
		])
		
		if geometry_dependencies_available:
			correlation_summary.extend([
				{
					"Parameter Pair": "Thickness ↔ GCF",
					"Correlation": f"{st.session_state.correlation_values.get('thickness_gcf', 0.0):.2f}",
					"Status": "✅ Active"
				},
				{
					"Parameter Pair": "Area ↔ GCF",
					"Correlation": f"{st.session_state.correlation_values.get('area_gcf', 0.0):.2f}",
					"Status": "✅ Active"
				}
			])
		
		# Display correlation summary table
		if correlation_summary:
			corr_df = pd.DataFrame(correlation_summary)
			st.dataframe(corr_df, use_container_width=True)
		else:
			st.info("No correlations are currently active. Select appropriate input methods to enable correlations.")
		
		# Success message about implementation
		st.markdown("---")
		st.success("""
		**✅ Correlation Implementation Complete:** Parameter correlations are now fully implemented in the Monte Carlo sampling. 
		When you adjust correlation coefficients, the underlying distributions are recalculated and the scatter plots update dynamically.
		""")
	
	else:
		st.info("🔒 Parameter dependencies are disabled. All parameters will be sampled independently.")

with tabs[1]:
	st.subheader("Result distributions")
	
	# Add recalculate all button
	if st.button("🔄 Recalculate All Distributions", type="primary", help="Recalculate all distributions with current input parameters"):
		# Clear session state to force recalculation
		for key in list(st.session_state.keys()):
			if key.startswith("samples_") or key.startswith("conf_") or key.startswith("corr_"):
				del st.session_state[key]
		# Keep correlation values so they are applied to new distributions
		st.rerun()

	# Apply correlations to all parameter samples before main calculations
	# Collect all available parameter samples into a dictionary
	samples_dict = {}
	
	# Add all parameter samples that might be used in correlations
	if 'sA' in locals():
		samples_dict['sA'] = sA
	if 'sGCF' in locals():
		samples_dict['sGCF'] = sGCF
	if 'sh' in locals():
		samples_dict['sh'] = sh
	if 'sNtG' in locals():
		samples_dict['sNtG'] = sNtG
	if 'sp' in locals():
		samples_dict['sp'] = sp
	if 'sSE' in locals():
		samples_dict['sSE'] = sSE
	if 'sd' in locals():
		samples_dict['sd'] = sd
	if 'sT' in locals():
		samples_dict['sT'] = sT
	if 'sP' in locals():
		samples_dict['sP'] = sP
	# Get depth samples from session state if available
	if 'stopdepth_off_samples' in st.session_state:
		samples_dict['stopdepth_off'] = st.session_state['stopdepth_off_samples']
	if 'sbasedepth_off_samples' in st.session_state:
		samples_dict['sbasedepth_off'] = st.session_state['sbasedepth_off_samples']
	if 'sGRV_m3_final' in locals():
		samples_dict['sGRV_m3_final'] = sGRV_m3_final

	# Apply correlations if correlation values exist
	if 'correlation_values' in st.session_state and st.session_state.correlation_values:
		samples_dict = apply_correlations_to_samples(samples_dict, st.session_state.correlation_values, num_sims)
		
		# Recalculate CO2 density if P-T correlations were applied
		if ('temp_pressure' in st.session_state.correlation_values and 
			abs(st.session_state.correlation_values['temp_pressure']) > 0.01 and
			'sT' in samples_dict and 'sP' in samples_dict):
			# Recalculate CO2 density with correlated P-T values
			sd_correlated, co2_phase_info_correlated = sample_co2_density_from_pt_distributions(
				samples_dict['sT'], samples_dict['sP']
			)
			
			# Check if calculation was successful
			if np.any(np.isnan(sd_correlated)):
				nan_count = np.sum(np.isnan(sd_correlated))
				if nan_count == len(sd_correlated):
					st.error("All CO₂ density calculations failed with correlated P-T. Check P-T ranges.")
				else:
					st.warning(f"CO₂ density calculation failed for {nan_count} out of {len(sd_correlated)} samples with correlated P-T. Check P-T ranges.")
					# Replace NaN values with mean of valid values
					valid_densities = sd_correlated[~np.isnan(sd_correlated)]
					if len(valid_densities) > 0:
						mean_density = np.mean(valid_densities)
						sd_correlated = np.where(np.isnan(sd_correlated), mean_density, sd_correlated)
						st.info(f"Replaced {nan_count} failed calculations with mean density: {mean_density:.2f} kg/m³")
			
			# Update the samples dictionary with the new density
			samples_dict['sd'] = sd_correlated
	
	# Check if all required variables are available for calculations
	required_vars = ['sGRV_m3_final', 'sNtG', 'sp', 'sSE', 'sd']
	missing_vars = [var for var in required_vars if var not in locals()]
	
	if missing_vars:
		st.warning(f"Some required parameters are not yet defined: {', '.join(missing_vars)}. Please complete the input sections first.")
		# Create placeholder arrays to avoid errors
		sGRV_m3_final = np.zeros(num_sims)
		sNtG = np.zeros(num_sims)
		sp = np.zeros(num_sims)
		sSE = np.zeros(num_sims)
		sd = np.zeros(num_sims)
	else:
		# Extract correlated samples for main calculations
		sGRV_m3_final = samples_dict.get('sGRV_m3_final', locals().get('sGRV_m3_final'))
		sNtG = samples_dict.get('sNtG', locals().get('sNtG'))
		sp = samples_dict.get('sp', locals().get('sp'))
		sSE = samples_dict.get('sSE', locals().get('sSE'))
		sd = samples_dict.get('sd', locals().get('sd'))

	sPV = sGRV_m3_final * sNtG * sp
	sSVe = sPV * sSE
	sSC = sSVe * sd

	PV_Mm3 = sPV / 1e6
	SVe_Mm3 = sSVe / 1e6
	SC_Mt = sSC / 1e9

	results = [
		("Pore volume PV", PV_Mm3, "×10^6 m³"),
		("Effective storage volume SVe", SVe_Mm3, "×10^6 m³"),
		("Storage capacity SC", SC_Mt, "Mt"),
	]

	for title, arr, unit in results:
		if "PV" in title:
			decimals = 2
		elif "SVe" in title:
			decimals = 2
		elif "SC" in title:
			decimals = 2
		else:
			decimals = 1
		st.plotly_chart(make_hist_cdf_figure(arr, f"{title}", f"{title} ({unit})", "result"), use_container_width=True)
		st.dataframe(summary_table(arr, decimals=decimals), use_container_width=True)

	st.markdown("**Key percentiles (SC in Mt):**")
	sc_stats = summarize_array(SC_Mt)
	if sc_stats:
		colp = st.columns(5)
		colp[0].metric("P50 SC", f"{sc_stats['P50']:.1f} Mt")
		colp[1].metric("P90 SC", f"{sc_stats['P90']:.1f} Mt")
		colp[2].metric("P10 SC", f"{sc_stats['P10']:.1f} Mt")
		colp[3].metric("Mean SC", f"{sc_stats['mean']:.1f} Mt")
		colp[4].metric("Std Dev SC", f"{sc_stats['std_dev']:.1f} Mt")
	
	# ========================================
	# TORNADO PLOT FOR SENSITIVITY ANALYSIS
	# ========================================
	st.markdown("---")
	st.markdown("## Sensitivity Analysis - Tornado Plot")
	st.markdown("This plot shows how each input parameter affects the mean storage capacity value.")
	
	# Add violin plot for storage capacity
	st.markdown("### Storage Capacity Distribution Analysis")
	
	# Create violin plot
	fig_violin = go.Figure()
	
	# Violin plot
	fig_violin.add_trace(
		go.Violin(
			y=SC_Mt,
			name="Storage Capacity",
			box_visible=True,
			line_color="blue",
			fillcolor="lightblue",
			opacity=0.6
		)
	)
	
	fig_violin.update_layout(
		title="Storage Capacity Distribution Analysis",
		yaxis_title="Storage Capacity (Mt)",
		height=500,
		showlegend=False
	)
	
	st.plotly_chart(fig_violin, use_container_width=True)
	
	# Function to create tornado plot
	def create_tornado_plot():
		# Get the mean values of all input parameters
		grv_mean = float(np.mean(sGRV_m3_final / 1e6))  # Convert to ×10^6 m³
		ntg_mean = float(np.mean(sNtG))
		porosity_mean = float(np.mean(sp))
		se_mean = float(np.mean(sSE))
		density_mean = float(np.mean(sd))
		
		# Calculate base case (using mean values)
		base_sc = (grv_mean * 1e6) * ntg_mean * porosity_mean * se_mean * density_mean / 1e9  # Convert to Mt
		
		# Calculate sensitivity for each parameter using improved approach
		# Instead of using P10/P90, we'll use the actual min/max of the distributions
		# and calculate the impact relative to the base case
		sensitivities = []
		debug_info = []
		
		# GRV sensitivity
		grv_stats = summarize_array(sGRV_m3_final / 1e6)
		if grv_stats:
			grv_min = grv_stats.get('min', grv_mean)
			grv_max = grv_stats.get('max', grv_mean)
			grv_p10 = grv_stats.get('P10', grv_mean)
			grv_p90 = grv_stats.get('P90', grv_mean)
			
			# Calculate impacts using P10 and P90 for more realistic ranges
			sc_grv_p10 = (grv_p10 * 1e6) * ntg_mean * porosity_mean * se_mean * density_mean / 1e9
			sc_grv_p90 = (grv_p90 * 1e6) * ntg_mean * porosity_mean * se_mean * density_mean / 1e9
			grv_p10_impact = sc_grv_p10 - base_sc
			grv_p90_impact = sc_grv_p90 - base_sc
			grv_range = abs(sc_grv_p90 - sc_grv_p10)
			
			debug_info.append(f"**GRV Debug:** P10={grv_p10:.6f}, P90={grv_p90:.6f}, P10_Impact={grv_p10_impact:.6f}, P90_Impact={grv_p90_impact:.6f}, Range={grv_range:.6f}")
			
			sensitivities.append({
				'Parameter': 'Gross Rock Volume (GRV)',
				'P10_Impact': grv_p10_impact,
				'P90_Impact': grv_p90_impact,
				'Range': grv_range
			})
		
		# NtG sensitivity
		ntg_stats = summarize_array(sNtG)
		if ntg_stats:
			ntg_min = ntg_stats.get('min', ntg_mean)
			ntg_max = ntg_stats.get('max', ntg_mean)
			ntg_p10 = ntg_stats.get('P10', ntg_mean)
			ntg_p90 = ntg_stats.get('P90', ntg_mean)
			
			sc_ntg_p10 = (grv_mean * 1e6) * ntg_p10 * porosity_mean * se_mean * density_mean / 1e9
			sc_ntg_p90 = (grv_mean * 1e6) * ntg_p90 * porosity_mean * se_mean * density_mean / 1e9
			ntg_p10_impact = sc_ntg_p10 - base_sc
			ntg_p90_impact = sc_ntg_p90 - base_sc
			ntg_range = abs(sc_ntg_p90 - sc_ntg_p10)
			
			debug_info.append(f"**NtG Debug:** P10={ntg_p10:.6f}, P90={ntg_p90:.6f}, P10_Impact={ntg_p10_impact:.6f}, P90_Impact={ntg_p90_impact:.6f}, Range={ntg_range:.6f}")
			
			sensitivities.append({
				'Parameter': 'Net-to-Gross (NtG)',
				'P10_Impact': ntg_p10_impact,
				'P90_Impact': ntg_p90_impact,
				'Range': ntg_range
			})
		
		# Porosity sensitivity
		porosity_stats = summarize_array(sp)
		if porosity_stats:
			porosity_min = porosity_stats.get('min', porosity_mean)
			porosity_max = porosity_stats.get('max', porosity_mean)
			porosity_p10 = porosity_stats.get('P10', porosity_mean)
			porosity_p90 = porosity_stats.get('P90', porosity_mean)
			
			sc_porosity_p10 = (grv_mean * 1e6) * ntg_mean * porosity_p10 * se_mean * density_mean / 1e9
			sc_porosity_p90 = (grv_mean * 1e6) * ntg_mean * porosity_p90 * se_mean * density_mean / 1e9
			porosity_p10_impact = sc_porosity_p10 - base_sc
			porosity_p90_impact = sc_porosity_p90 - base_sc
			porosity_range = abs(sc_porosity_p90 - sc_porosity_p10)
			
			debug_info.append(f"**Porosity Debug:** P10={porosity_p10:.6f}, P90={porosity_p90:.6f}, P10_Impact={porosity_p10_impact:.6f}, P90_Impact={porosity_p90_impact:.6f}, Range={porosity_range:.6f}")
			
			sensitivities.append({
				'Parameter': 'Porosity',
				'P10_Impact': porosity_p10_impact,
				'P90_Impact': porosity_p90_impact,
				'Range': porosity_range
			})
		
		# Storage Efficiency sensitivity
		se_stats = summarize_array(sSE)
		if se_stats:
			se_min = se_stats.get('min', se_mean)
			se_max = se_stats.get('max', se_mean)
			se_p10 = se_stats.get('P10', se_mean)
			se_p90 = se_stats.get('P90', se_mean)
			
			sc_se_p10 = (grv_mean * 1e6) * ntg_mean * porosity_mean * se_p10 * density_mean / 1e9
			sc_se_p90 = (grv_mean * 1e6) * ntg_mean * porosity_mean * se_p90 * density_mean / 1e9
			se_p10_impact = sc_se_p10 - base_sc
			se_p90_impact = sc_se_p90 - base_sc
			se_range = abs(sc_se_p90 - sc_se_p10)
			
			debug_info.append(f"**SE Debug:** P10={se_p10:.6f}, P90={se_p90:.6f}, P10_Impact={se_p10_impact:.6f}, P90_Impact={se_p90_impact:.6f}, Range={se_range:.6f}")
			
			sensitivities.append({
				'Parameter': 'Storage Efficiency',
				'P10_Impact': se_p10_impact,
				'P90_Impact': se_p90_impact,
				'Range': se_range
			})
		
		# CO₂ Density sensitivity
		density_stats = summarize_array(sd)
		if density_stats:
			density_min = density_stats.get('min', density_mean)
			density_max = density_stats.get('max', density_mean)
			density_p10 = density_stats.get('P10', density_mean)
			density_p90 = density_stats.get('P90', density_mean)
			
			sc_density_p10 = (grv_mean * 1e6) * ntg_mean * porosity_mean * se_mean * density_p10 / 1e9
			sc_density_p90 = (grv_mean * 1e6) * ntg_mean * porosity_mean * se_mean * density_p90 / 1e9
			density_p10_impact = sc_density_p10 - base_sc
			density_p90_impact = sc_density_p90 - base_sc
			density_range = abs(sc_density_p90 - sc_density_p10)
			
			debug_info.append(f"**Density Debug:** P10={density_p10:.6f}, P90={density_p90:.6f}, P10_Impact={density_p10_impact:.6f}, P90_Impact={density_p90_impact:.6f}, Range={density_range:.6f}")
			
			sensitivities.append({
				'Parameter': 'CO₂ Density',
				'P10_Impact': density_p10_impact,
				'P90_Impact': density_p90_impact,
				'Range': density_range
			})
		
		# Sort by range (impact magnitude) in descending order
		sensitivities.sort(key=lambda x: x['Range'], reverse=True)
		
		# Create tornado plot for impact analysis
		fig = go.Figure()
		
		# Add bars for each parameter (reverse order to show largest range on top)
		for i, sens in enumerate(reversed(sensitivities)):
			# P10 bar (left side)
			fig.add_trace(go.Bar(
				y=[sens['Parameter']],
				x=[sens['P10_Impact']],
				orientation='h',
				name='P10 Impact',
				marker_color='lightcoral',
				showlegend=False,
				hovertemplate=f"{sens['Parameter']}<br>P10 Impact: {sens['P10_Impact']:.2f} Mt<br>Base Case: {base_sc:.2f} Mt<extra></extra>"
			))
			
			# P90 bar (right side)
			fig.add_trace(go.Bar(
				y=[sens['Parameter']],
				x=[sens['P90_Impact']],
				orientation='h',
				name='P90 Impact',
				marker_color='lightblue',
				showlegend=False,
				hovertemplate=f"{sens['Parameter']}<br>P90 Impact: {sens['P90_Impact']:.2f} Mt<br>Base Case: {base_sc:.2f} Mt<extra></extra>"
			))
		
		# Add vertical line at base case (0)
		fig.add_vline(x=0, line_dash="dash", line_color="black", annotation_text="Base Case")
		
		# Update layout
		fig.update_layout(
			title=f"Storage Capacity Sensitivity Analysis (Impact)<br><sub>Base Case: {base_sc:.2f} Mt</sub>",
			xaxis_title="Impact on Storage Capacity (Mt)",
			yaxis_title="Input Parameters",
			barmode='relative',
			showlegend=False,
			height=400,
			margin=dict(l=40, r=40, t=80, b=40)
		)
		
		# Update x-axis to be symmetric around 0
		max_impact = max([abs(sens['P10_Impact']) for sens in sensitivities] + [abs(sens['P90_Impact']) for sens in sensitivities])
		fig.update_xaxes(range=[-max_impact*1.1, max_impact*1.1])
		
		return fig, sensitivities, base_sc, debug_info, grv_mean, ntg_mean, porosity_mean, se_mean, density_mean
	
	# Create and display tornado plot
	tornado_fig, sensitivities, base_sc, debug_info, grv_mean, ntg_mean, porosity_mean, se_mean, density_mean = create_tornado_plot()
	st.plotly_chart(tornado_fig, use_container_width=True)
	
	# Display all debug information in a single expandable box
	if debug_info:
		with st.expander("Debug Information", expanded=False):
			st.write("**Base Case Information:**")
			st.write(f"GRV mean: {grv_mean:.6f} ×10^6 m³")
			st.write(f"NtG mean: {ntg_mean:.6f}")
			st.write(f"Porosity mean: {porosity_mean:.6f}")
			st.write(f"SE mean: {se_mean:.6f}")
			st.write(f"Density mean: {density_mean:.6f} kg/m³")
			st.write(f"Base case SC: {base_sc:.6f} Mt")
			st.write("")
			st.write("**Sensitivity Analysis Details:**")
			for info in debug_info:
				st.write(info)
	
	# Alternative sensitivity analysis using full range
	st.markdown("### Alternative Sensitivity Analysis (Full Range)")
	
	def create_full_range_sensitivity():
		# Get the mean values of all input parameters
		grv_mean = float(np.mean(sGRV_m3_final / 1e6))  # Convert to ×10^6 m³
		ntg_mean = float(np.mean(sNtG))
		porosity_mean = float(np.mean(sp))
		se_mean = float(np.mean(sSE))
		density_mean = float(np.mean(sd))
		
		# Calculate base case (using mean values)
		base_sc = (grv_mean * 1e6) * ntg_mean * porosity_mean * se_mean * density_mean / 1e9  # Convert to Mt
		
		# Calculate sensitivity using full range (min/max) of distributions
		sensitivities_full = []
		
		# GRV sensitivity using full range
		grv_stats = summarize_array(sGRV_m3_final / 1e6)
		if grv_stats:
			grv_min = grv_stats.get('min', grv_mean)
			grv_max = grv_stats.get('max', grv_mean)
			
			sc_grv_min = (grv_min * 1e6) * ntg_mean * porosity_mean * se_mean * density_mean / 1e9
			sc_grv_max = (grv_max * 1e6) * ntg_mean * porosity_mean * se_mean * density_mean / 1e9
			grv_min_impact = sc_grv_min - base_sc
			grv_max_impact = sc_grv_max - base_sc
			grv_range = abs(sc_grv_max - sc_grv_min)
			
			sensitivities_full.append({
				'Parameter': 'Gross Rock Volume (GRV)',
				'Min_Impact': grv_min_impact,
				'Max_Impact': grv_max_impact,
				'Range': grv_range
			})
		
		# NtG sensitivity using full range
		ntg_stats = summarize_array(sNtG)
		if ntg_stats:
			ntg_min = ntg_stats.get('min', ntg_mean)
			ntg_max = ntg_stats.get('max', ntg_mean)
			
			sc_ntg_min = (grv_mean * 1e6) * ntg_min * porosity_mean * se_mean * density_mean / 1e9
			sc_ntg_max = (grv_mean * 1e6) * ntg_max * porosity_mean * se_mean * density_mean / 1e9
			ntg_min_impact = sc_ntg_min - base_sc
			ntg_max_impact = sc_ntg_max - base_sc
			ntg_range = abs(sc_ntg_max - sc_ntg_min)
			
			sensitivities_full.append({
				'Parameter': 'Net-to-Gross (NtG)',
				'Min_Impact': ntg_min_impact,
				'Max_Impact': ntg_max_impact,
				'Range': ntg_range
			})
		
		# Porosity sensitivity using full range
		porosity_stats = summarize_array(sp)
		if porosity_stats:
			porosity_min = porosity_stats.get('min', porosity_mean)
			porosity_max = porosity_stats.get('max', porosity_mean)
			
			sc_porosity_min = (grv_mean * 1e6) * ntg_mean * porosity_min * se_mean * density_mean / 1e9
			sc_porosity_max = (grv_mean * 1e6) * ntg_mean * porosity_max * se_mean * density_mean / 1e9
			porosity_min_impact = sc_porosity_min - base_sc
			porosity_max_impact = sc_porosity_max - base_sc
			porosity_range = abs(sc_porosity_max - sc_porosity_min)
			
			sensitivities_full.append({
				'Parameter': 'Porosity',
				'Min_Impact': porosity_min_impact,
				'Max_Impact': porosity_max_impact,
				'Range': porosity_range
			})
		
		# Storage Efficiency sensitivity using full range
		se_stats = summarize_array(sSE)
		if se_stats:
			se_min = se_stats.get('min', se_mean)
			se_max = se_stats.get('max', se_mean)
			
			sc_se_min = (grv_mean * 1e6) * ntg_mean * porosity_mean * se_min * density_mean / 1e9
			sc_se_max = (grv_mean * 1e6) * ntg_mean * porosity_mean * se_max * density_mean / 1e9
			se_min_impact = sc_se_min - base_sc
			se_max_impact = sc_se_max - base_sc
			se_range = abs(sc_se_max - sc_se_min)
			
			sensitivities_full.append({
				'Parameter': 'Storage Efficiency',
				'Min_Impact': se_min_impact,
				'Max_Impact': se_max_impact,
				'Range': se_range
			})
		
		# CO₂ Density sensitivity using full range
		density_stats = summarize_array(sd)
		if density_stats:
			density_min = density_stats.get('min', density_mean)
			density_max = density_stats.get('max', density_mean)
			
			sc_density_min = (grv_mean * 1e6) * ntg_mean * porosity_mean * se_mean * density_min / 1e9
			sc_density_max = (grv_mean * 1e6) * ntg_mean * porosity_mean * se_mean * density_max / 1e9
			density_min_impact = sc_density_min - base_sc
			density_max_impact = sc_density_max - base_sc
			density_range = abs(sc_density_max - sc_density_min)
			
			sensitivities_full.append({
				'Parameter': 'CO₂ Density',
				'Min_Impact': density_min_impact,
				'Max_Impact': density_max_impact,
				'Range': density_range
			})
		
		# Sort by range (impact magnitude) in descending order
		sensitivities_full.sort(key=lambda x: x['Range'], reverse=True)
		
		return sensitivities_full, base_sc
	
	full_range_sensitivities, base_sc_full = create_full_range_sensitivity()
	
	# Display full range sensitivity summary table
	st.markdown("#### Full Range Sensitivity Summary")
	
	# Create full range sensitivity summary table
	full_range_data = []
	for sens in full_range_sensitivities:
		full_range_data.append({
			'Parameter': sens['Parameter'],
			'Min Impact (Mt)': f"{sens['Min_Impact']:.2f}",
			'Max Impact (Mt)': f"{sens['Max_Impact']:.2f}",
			'Range (Mt)': f"{sens['Range']:.2f}",
			'Relative Impact (%)': f"{(sens['Range'] / base_sc_full * 100):.1f}%"
		})
	
	full_range_df = pd.DataFrame(full_range_data)
	full_range_styled = full_range_df.style.set_properties(**{
		'text-align': 'center',
		'vertical-align': 'middle'
	})
	
	st.dataframe(full_range_styled, use_container_width=True)
	
	# Display sensitivity summary table
	st.markdown("### Sensitivity Summary (P10/P90 Method)")
	
	# Create sensitivity summary table
	sensitivity_data = []
	for sens in sensitivities:
		sensitivity_data.append({
			'Parameter': sens['Parameter'],
			'P10 Impact (Mt)': f"{sens['P10_Impact']:.2f}",
			'P90 Impact (Mt)': f"{sens['P90_Impact']:.2f}",
			'Range (Mt)': f"{sens['Range']:.2f}",
			'Relative Impact (%)': f"{(sens['Range'] / base_sc * 100):.1f}%"
		})
	
	sensitivity_df = pd.DataFrame(sensitivity_data)
	sensitivity_styled = sensitivity_df.style.set_properties(**{
		'text-align': 'center',
		'vertical-align': 'middle'
	})
	
	st.dataframe(sensitivity_styled, use_container_width=True)
	
	# Add explanation
	st.markdown("""
	**Tornado Plot Interpretation:**
	- **Left side (red bars)**: Impact when parameter is at P10 value
	- **Right side (blue bars)**: Impact when parameter is at P90 value
	- **Bar length**: Shows the magnitude of impact on storage capacity
	- **Parameter order**: Ranked by total impact range (P90 - P10)
	- **Base case**: Calculated using mean values of all parameters
	
	**Note:** If impact values are very small (close to 0.00), it may indicate that:
	1. The parameter distributions are very narrow (low uncertainty)
	2. The P10/P90 values are very close to the mean values
	3. The sensitivity analysis method may need adjustment for your specific case
	
	The "Full Range" analysis above uses the complete min/max range of each parameter distribution, which may provide more meaningful sensitivity measures.
	""")

