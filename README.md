# SCOPE-CO₂

**Subsurface Capacity Overview and Probability Estimator for CO₂ storage**

A comprehensive Streamlit-based application for stochastic estimation of CO₂ storage capacity in subsurface reservoirs.

## Overview

SCOPE-CO₂ is a sophisticated Monte Carlo simulation tool designed for estimating CO₂ storage capacity in geological formations. The application provides multiple input methods, advanced uncertainty analysis, and comprehensive visualization capabilities for CO₂ storage projects.

## Features

### 🏗️ **Multiple GRV Input Methods**
- **Direct input**: Direct specification of Gross Rock Volume
- **Area, Geometry Factor and Thickness**: Calculate GRV from reservoir parameters
- **Depth/Top/Base Areas table**: Use depth-area relationships
- **Depth vs Area vs Thickness**: Spill-point based calculations

### 📊 **Advanced Uncertainty Modeling**
- **Multiple distribution types**: PERT, Triangular, Uniform, Lognormal, Subjective Beta
- **Parameter correlations**: Model realistic dependencies between parameters
- **Uncertainty multipliers**: GRV_MP and GCF_MP for additional uncertainty layers
- **Monte Carlo sampling**: Configurable simulation counts (100 - 2,000,000)

### 🔧 **Geometric Correction Factor Calculator**
- **Gehman (1970) methodology**: Based on digitized data from original graphs
- **Multiple shape types**: Dome, Anticline, Flat-top structures, Block
- **Length/Width ratios**: Support for various structural geometries
- **Sensitivity analysis**: Interactive parameter impact assessment
- **Visual plots**: Real-time GCF curves and current point identification

### 🌡️ **CO₂ Density Calculation**
- **CoolProp integration**: Accurate P-T based density calculations
- **Multiple scenarios**: Manual input, P-T distributions, Onshore/Offshore
- **Phase classification**: Automatic CO₂ phase determination
- **Geothermal modeling**: Temperature and pressure from depth parameters

### 📈 **Comprehensive Analysis**
- **Result distributions**: PV, SVe, and Storage Capacity with statistics
- **Sensitivity analysis**: Tornado plots and violin plots
- **Correlation visualization**: Interactive scatter plots for parameter relationships
- **Statistical summaries**: Detailed percentiles and distribution characteristics

## Installation

### Prerequisites
- Python 3.7+
- Streamlit
- NumPy
- Pandas
- Plotly
- SciPy

### Optional Dependencies
- **CoolProp**: For CO₂ density calculations from P-T
  ```bash
  pip install coolprop
  ```

### Installation Steps
1. Clone or download the repository
2. Install required packages:
   ```bash
   pip install streamlit numpy pandas plotly scipy
   ```
3. Run the application:
   ```bash
   streamlit run SCOPE-CO2.py
   ```

## Usage

### 1. **Simulation Setup**
- Set the number of Monte Carlo simulations (default: 10,000)
- Configure parameter correlations if needed

### 2. **Input Parameters**

#### **Gross Rock Volume (GRV)**
Choose from four input methods:

**Option A: Direct Input**
- Direct specification of GRV with uncertainty distribution

**Option B: Area, Geometry Factor and Thickness**
- **Area (A)**: Reservoir area in km²
- **Geometry Correction Factor (GCF)**: 
  - Direct method: Specify distribution directly
  - Geometric Correction Factor Calculator: Use Gehman (1970) methodology
    - Reservoir thickness and structural relief
    - Geometric shape type and length/width ratio
    - Dip angle correction
    - GCF uncertainty multiplier (GCF_MP)
- **Thickness (h)**: Reservoir thickness in meters

**Option C: Depth/Top/Base Areas Table**
- Input depth slices with corresponding top and base areas
- Configurable depth step and extrapolation options
- Spill point specification

**Option D: Depth vs Area vs Thickness**
- Top area table with constant thickness
- Spill point based calculations

#### **Net-to-Gross (NtG)**
- Fraction of reservoir rock that can store CO₂
- Supports all distribution types

#### **Porosity**
- Pore space fraction in the reservoir rock
- Configurable uncertainty distributions

#### **CO₂ Density**
Two calculation methods:

**Direct Distribution Input**
- Specify density distribution directly

**Calculate from Pressure-Temperature**
- **Manual input**: Single P-T values
- **P-T distributions**: Sample from temperature and pressure distributions
- **Onshore scenario**: Calculate from depth and geothermal parameters
- **Offshore scenario**: Include water depth considerations

#### **Storage Efficiency (SE)**
- Fraction of pore volume that can be filled with CO₂
- Accounts for irreducible water saturation and sweep efficiency

### 3. **Parameter Correlations**
Enable realistic parameter dependencies:
- **Temperature-Pressure correlation**: For P-T based density calculations
- **Top Depth-Base Depth correlation**: For depth-based GRV methods
- **Porosity-Storage Efficiency correlation**: Reservoir property relationships
- **Porosity-Net-to-Gross correlation**: Rock property dependencies
- **Thickness-GCF correlation**: Geometric parameter relationships
- **Area-GCF correlation**: Structural parameter dependencies

### 4. **Results Analysis**
The application provides comprehensive results including:

#### **Storage Capacity Results**
- **Pore Volume (PV)**: Total pore space in the reservoir
- **Effective Storage Volume (SVe)**: PV × Storage Efficiency
- **Storage Capacity (SC)**: Mass of CO₂ that can be stored

#### **Statistical Analysis**
- Detailed percentiles (P1, P5, P10, P25, P50, P75, P90, P95, P99)
- Mean, mode, standard deviation, variance, skewness
- Distribution plots with histograms and cumulative functions

#### **Sensitivity Analysis**
- **Tornado plots**: Parameter impact on storage capacity
- **Violin plots**: Distribution analysis
- **Correlation plots**: Parameter relationship visualization

## Methodology

### Geometric Correction Factor (GCF)
Based on **Gehman, H.N. (1970)**: "Graphs to Derive Geometric Correction Factor: Exxon Training Materials (unpublished), Houston."

The GCF calculator uses digitized data from the original graphs and supports:
- **Shape Types**: Dome/Cone/Pyramid, Anticline/Prism/Cylinder, Flat-top structures, Block
- **Length/Width Ratios**: 1, 2, 5, 10 for different structural geometries
- **Reservoir Thickness/Closure Ratio**: Accounts for true thickness and structural relief
- **Dip Angle Correction**: Converts measured thickness to true thickness

### CO₂ Density Calculation
Uses the **CoolProp library** with references to:
- **Span & Wagner (1996)**: CO₂ equation of state
- **Huber et al. (2016)**: Thermal conductivity correlations
- **Bell et al. (2014)**: CoolProp library implementation

### Uncertainty Analysis
- **Monte Carlo simulation**: Random sampling from input distributions
- **Cholesky decomposition**: For correlated parameter sampling
- **Statistical analysis**: Comprehensive distribution characterization

## File Structure

```
SCOPE-CO2.py          # Main application file
README.md             # This documentation
requirements.txt      # Python dependencies
```

## Key Formulas

### Storage Capacity Calculation
```
GRV = A × GCF × h                    # Gross Rock Volume
PV = GRV × NtG × φ                   # Pore Volume
SVe = PV × SE                        # Effective Storage Volume
SC = SVe × ρCO₂                      # Storage Capacity

Where:
- A: Area (km², converted to m²)
- GCF: Geometric Correction Factor
- h: Thickness (m)
- NtG: Net-to-Gross ratio
- φ: Porosity
- SE: Storage Efficiency
- ρCO₂: CO₂ density (kg/m³)
```

### Uncertainty Multipliers
```
Final GRV = Base GRV × GRV_MP
Final GCF = Calculated GCF × GCF_MP
```

## References

### Primary References
1. **Gehman, H.N. (1970)**: Graphs to Derive Geometric Correction Factor: Exxon Training Materials (unpublished), Houston.

2. **Span, R. and Wagner, W. (1996)**: A New Equation of State for Carbon Dioxide Covering the Fluid Region from the Triple Point Temperature to 1100 K at Pressures up to 800 MPa. *J. Phys. Chem. Ref. Data*, 25:1509–1596.

3. **Huber, M.L., et al. (2016)**: Reference correlation of the thermal conductivity of carbon dioxide from the triple point to 1100 k and up to 200 mpa. *Journal of Physical and Chemical Reference Data*.

4. **Bell, I.H., et al. (2014)**: Pure and Pseudo-pure Fluid Thermophysical Property Evaluation and the Open-Source Thermophysical Property Library CoolProp. *Industrial & Engineering Chemistry Research*, 53:2498–2508.

### Additional References
- **Capture, Storage and Use of CO2 (CCUS). Evaluation of the CO2 storage potential in Denmark. Vol.1: Report & Vol 2: Appendix A and B** (2020/46) - GEUS publication

## Contributing

This application is designed for CO₂ storage capacity estimation in geological formations. For questions or improvements, please refer to the methodology and ensure any modifications maintain the scientific integrity of the calculations.

## License

This application is provided for scientific and educational purposes. Please ensure proper attribution to the original references when using this tool for research or commercial applications.

---

**Note**: This application requires careful consideration of input parameters and geological context. Results should be validated against independent calculations and geological understanding of the specific reservoir under investigation.
