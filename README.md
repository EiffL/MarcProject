# Stellar Mass Function Analysis for COSMOS-Web

This repository contains tools for computing stellar mass functions from galaxy catalogs, specifically designed for the COSMOS-Web survey data. The analysis follows the methodology described in COSMOS2020 papers (Weaver et al. 2022, Davidzon et al. 2017).

## Overview

The **stellar mass function (SMF)** is a fundamental observable in galaxy evolution studies, describing the number density of galaxies as a function of their stellar mass. This codebase provides:

- Tools to compute stellar mass functions from FITS galaxy catalogs
- **Automatic detection** of catalog types (LePhare vs CIGALE)
- **Proper redshift binning** using photometric redshifts
- Schechter function fitting (single and double component)
- Uncertainty estimation (Poisson, cosmic variance, SED fitting errors)
- Mass completeness calculations
- Literature comparison capabilities
- Visualization and diagnostic tools

## Features

### Catalog Support
- **LePhare catalogs**: Full support with redshift information (`zpdf_med`, `mass_med`, `sfr_med`)
- **CIGALE catalogs**: Fallback support (mass functions without redshift binning)
- **Automatic detection**: Code automatically detects catalog type and uses appropriate columns

### Core Functionality
- **Mass Function Computation**: Calculate number densities in stellar mass bins
- **Volume Corrections**: Apply 1/V_max corrections for Malmquist bias
- **Completeness Limits**: Compute mass completeness following COSMOS methodology
- **Schechter Fitting**: Fit single and double Schechter functions with MCMC
- **Uncertainty Analysis**: Comprehensive error budget including cosmic variance

### Analysis Tools
- **Residual Analysis**: Diagnostic plots and statistical tests for fit quality
- **Literature Comparison**: Compare results with published measurements
- **Cosmic Stellar Mass Density**: Integrate mass functions to compute cosmic densities
- **Bootstrap Uncertainties**: Additional uncertainty estimation methods

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd stellar-mass-function
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Analysis

```python
from stellar_mass_function import StellarMassFunction

# Initialize with your catalog
smf = StellarMassFunction('data/COSMOSWeb_mastercatalog_v1_cigale.fits', area_deg2=0.54)

# Load catalog
catalog = smf.load_catalog()

# Run analysis for redshift bin z=0.2-0.5
results = smf.run_full_analysis(z_bin=(0.2, 0.5), plot=True)

# Access results
smf_data = results['smf_data']
best_fit = results['best_fit']
```

### Command Line Interface

Run the complete analysis pipeline:

```bash
python run_smf_analysis.py --catalog data/COSMOSWeb_mastercatalog_v1_cigale.fits \
                          --area 0.54 \
                          --z_min 0.2 \
                          --z_max 0.5 \
                          --save_plots \
                          --output_dir results/
```

### Command Line Options

- `--catalog`: Path to FITS catalog file
- `--area`: Survey area in square degrees (default: 0.54 for COSMOS-Web)
- `--z_min`, `--z_max`: Redshift range for analysis
- `--output_dir`: Directory for output files and plots
- `--save_plots`: Save plots to files
- `--no_plots`: Skip plot generation

## Data Requirements

The code expects a FITS catalog with the following columns:

### Required Columns
- `mass`: Stellar masses (linear M_☉ or log₁₀(M_☉))

### Optional Columns (for enhanced analysis)
- `z_phot` or `redshift`: Photometric redshifts
- `sfr`: Star formation rates
- Magnitude columns for completeness calculations

**Note**: The current implementation works with CIGALE SED fitting results. For full functionality with photometric redshifts, ensure your catalog includes redshift information.

## Methodology

### Stellar Mass Function Calculation

1. **Galaxy Selection**: Apply redshift and quality cuts
2. **Mass Completeness**: Apply mass limits based on survey depth
3. **Volume Calculation**: Compute comoving volumes for redshift bins
4. **Number Density**: Count galaxies in mass bins, apply volume corrections
5. **Uncertainty Estimation**: Combine Poisson, cosmic variance, and SED errors

### Schechter Function Fitting

The code fits both single and double Schechter functions:

**Single Schechter:**
```
Φ(M) = ln(10) × Φ* × exp(-M/M*) × (M/M*)^(α+1)
```

**Double Schechter:**
```
Φ(M) = ln(10) × exp(-M/M*) × [Φ₁*(M/M*)^(α₁+1) + Φ₂*(M/M*)^(α₂+1)]
```

Where:
- Φ*: Normalization parameter(s)
- M*: Characteristic stellar mass
- α: Faint-end slope(s)

### Uncertainty Budget

Following COSMOS2020 methodology:

1. **Poisson Uncertainties**: σ_N = √N for galaxy counts
2. **Cosmic Variance**: Following Moster et al. (2011) / Steinhardt et al. (2021)
3. **SED Fitting Errors**: Mass-dependent uncertainties from template fitting
4. **Total**: σ_total = √(σ_N² + σ_CV² + σ_SED²)

## Output

### Numerical Results
- `output/smf_results_z{z_min}-{z_max}.txt`: Tabulated mass function
- Best-fit Schechter parameters with uncertainties
- Cosmic stellar mass density

### Plots
- Stellar mass function with Schechter fit
- Literature comparison plots
- Residual analysis plots
- Uncertainty breakdowns

### Analysis Summary
```
==================================================
ANALYSIS RESULTS
==================================================
Redshift range: 0.2 < z ≤ 0.5
Number of galaxies: 123456
Comoving volume: 1.23e+06 Mpc³

Stellar Mass Function:
Mass [log M/M☉]   Φ [Mpc⁻³ dex⁻¹]    σ_Φ
--------------------------------------------------
    8.12          1.234e-02         5.67e-04
    8.37          8.901e-03         3.45e-04
    ...

Best Schechter Function Fit:
Function type: single_schechter
Reduced χ²: 1.23
log(Φ*) = -2.876 ± 0.045
log(M*) = 10.785 ± 0.032
α = -1.234 ± 0.089

Cosmic stellar mass density: 1.23e+08 M☉ Mpc⁻³
```

## Advanced Usage

### Custom Uncertainty Analysis

```python
from smf_utils import UncertaintyCalculator

uncertainties = UncertaintyCalculator()

# Calculate cosmic variance
cosmic_var = uncertainties.cosmic_variance(
    stellar_masses, redshift=1.0, area_deg2=0.54
)

# SED fitting uncertainties
sed_errors = uncertainties.sed_fitting_uncertainty(
    stellar_masses, redshift=1.0
)
```

### Mass Completeness Calculations

```python
from smf_utils import MassCompletenessCalculator

# COSMOS-Web depth
mass_limit = MassCompletenessCalculator.cosmos_mass_completeness(
    redshift=1.0, survey_depth='cosmos_web'
)

# Pozzetti method
mass_limit = MassCompletenessCalculator.pozzetti_method(
    catalog_masses, catalog_magnitudes, magnitude_limit=25.0
)
```

### Literature Comparison

```python
from smf_utils import LiteratureComparison

lit_comp = LiteratureComparison()
reference_data = [lit_comp.load_cosmos2020_reference(z_bin)]

lit_comp.plot_comparison(smf_data, reference_data)
```

## Scientific Background

### Stellar Mass Functions in Galaxy Evolution

The stellar mass function is a cornerstone observable in galaxy evolution studies because:

1. **Galaxy Assembly**: Traces the buildup of stellar mass over cosmic time
2. **Quenching Mechanisms**: Different slopes and normalizations reveal star formation cessation processes
3. **Dark Matter Connection**: Links observed galaxies to underlying dark matter halos
4. **Cosmological Tests**: Provides constraints on galaxy formation models

### COSMOS-Web Survey

COSMOS-Web is a 255-hour JWST Treasury program covering ~0.54 deg² with:
- Deep NIRCam imaging (F115W, F150W, F277W, F444W)
- MIRI parallel observations (F770W)
- Unprecedented depth and area combination
- >700,000 galaxies across cosmic time

## Validation

The code has been validated against:
- COSMOS2020 stellar mass function measurements
- Literature Schechter function parameters
- Expected cosmic stellar mass density evolution
- Statistical tests for fit quality

## Known Limitations

1. **Redshift Information**: Current CIGALE catalog lacks photometric redshifts
2. **Selection Functions**: Simplified V_max corrections (full implementation requires magnitude limits)
3. **Eddington Bias**: Basic kernel convolution (could be enhanced with more sophisticated methods)
4. **Cosmic Variance**: Approximate empirical relations (field-specific calculations would be ideal)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## References

- Weaver et al. 2022, "COSMOS2020: The galaxy stellar mass function"
- Davidzon et al. 2017, "COSMOS2015 stellar mass functions" 
- Schechter 1976, "An analytic expression for the luminosity function"
- Pozzetti et al. 2010, "Mass completeness method"
- Casey et al. 2023, "COSMOS-Web: An Overview"

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or issues, please open a GitHub issue or contact the maintainers.
