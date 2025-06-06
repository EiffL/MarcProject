#!/usr/bin/env python3
"""
Utility functions for stellar mass function analysis

This module provides additional tools for uncertainty estimation, cosmic variance,
and validation of stellar mass function results.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate, stats
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from typing import Tuple, Dict, List, Optional
import warnings

# Standard cosmology
COSMO = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Om0=0.3)

class UncertaintyCalculator:
    """
    Class for calculating various uncertainties in stellar mass function measurements.
    
    Following the methodology from COSMOS2020 (Weaver et al. 2022)
    """
    
    @staticmethod
    def poisson_uncertainty(n_galaxies: np.ndarray) -> np.ndarray:
        """
        Calculate Poisson uncertainty for galaxy counts.
        
        Parameters:
        -----------
        n_galaxies : array
            Number of galaxies in each mass bin
            
        Returns:
        --------
        array : Poisson uncertainties (sqrt(N))
        """
        return np.sqrt(np.maximum(n_galaxies, 1))  # Avoid sqrt(0)
    
    @staticmethod
    def cosmic_variance(stellar_mass: np.ndarray, redshift: float, 
                       area_deg2: float) -> np.ndarray:
        """
        Estimate cosmic variance following Moster et al. (2011) and 
        Steinhardt et al. (2021) prescriptions.
        
        Parameters:
        -----------
        stellar_mass : array
            Stellar masses in log10(M/M_sun)
        redshift : float
            Redshift
        area_deg2 : float
            Survey area in square degrees
            
        Returns:
        --------
        array : Cosmic variance uncertainties
        """
        # Convert area to steradians
        area_ster = area_deg2 * (np.pi/180)**2
        
        # Empirical fits from Moster et al. (2011) extended by Steinhardt et al. (2021)
        # This is a simplified approximation
        
        # Mass dependence
        log_mass = stellar_mass
        mass_factor = np.exp(-0.5 * ((log_mass - 10.5) / 1.0)**2)
        
        # Redshift dependence  
        z_factor = (1 + redshift)**1.5
        
        # Area dependence
        area_factor = (area_ster / (1.0 * (np.pi/180)**2))**(-0.5)
        
        # Combine factors (this is a rough approximation)
        cosmic_variance = 0.1 * mass_factor * z_factor * area_factor
        
        return np.maximum(cosmic_variance, 0.02)  # Minimum 2% uncertainty
    
    @staticmethod
    def sed_fitting_uncertainty(stellar_mass: np.ndarray, redshift: float) -> np.ndarray:
        """
        Estimate SED fitting uncertainties on stellar mass.
        
        Based on typical uncertainties reported in COSMOS catalogs.
        
        Parameters:
        -----------
        stellar_mass : array
            Stellar masses in log10(M/M_sun)
        redshift : float
            Redshift
            
        Returns:
        --------
        array : SED fitting uncertainties in dex
        """
        # Base uncertainty that increases with redshift and decreases with mass
        base_uncertainty = 0.1 + 0.05 * redshift
        
        # Mass dependence (higher uncertainty for lower mass galaxies)
        mass_factor = np.exp(-0.5 * ((stellar_mass - 10.0) / 2.0)**2) + 0.5
        
        sed_uncertainty = base_uncertainty * mass_factor
        
        return np.clip(sed_uncertainty, 0.05, 0.5)  # 0.05 - 0.5 dex range
    
    @staticmethod
    def total_uncertainty(poisson_err: np.ndarray, cosmic_var_err: np.ndarray,
                         sed_err: np.ndarray) -> np.ndarray:
        """
        Combine uncertainties in quadrature.
        
        Parameters:
        -----------
        poisson_err : array
            Poisson uncertainties
        cosmic_var_err : array
            Cosmic variance uncertainties  
        sed_err : array
            SED fitting uncertainties
            
        Returns:
        --------
        array : Total combined uncertainties
        """
        return np.sqrt(poisson_err**2 + cosmic_var_err**2 + sed_err**2)

class MassCompletenessCalculator:
    """
    Class for calculating mass completeness limits following COSMOS methodology.
    """
    
    @staticmethod
    def cosmos_mass_completeness(redshift: float, survey_depth: str = 'cosmos_web') -> float:
        """
        Calculate mass completeness limits using COSMOS2020 prescriptions.
        
        Parameters:
        -----------
        redshift : float
            Redshift
        survey_depth : str
            Survey type ('cosmos_web', 'cosmos2020', 'deep')
            
        Returns:
        --------
        float : Mass completeness limit in log10(M/M_sun)
        """
        if survey_depth == 'cosmos_web':
            # COSMOS-Web is deeper than COSMOS2020
            # Approximate improvement of ~0.5 dex
            log_mass_limit = 8.0 + 0.4 * redshift + 0.05 * redshift**2
        elif survey_depth == 'cosmos2020':
            # From COSMOS2020 paper (Equations 3-5)
            log_mass_limit = 8.5 + 0.5 * redshift + 0.1 * redshift**2
        else:
            # Generic deep survey
            log_mass_limit = 9.0 + 0.6 * redshift + 0.1 * redshift**2
            
        return log_mass_limit
    
    @staticmethod
    def pozzetti_method(catalog_masses: np.ndarray, catalog_magnitudes: np.ndarray,
                       magnitude_limit: float, percentile: float = 95) -> float:
        """
        Calculate mass completeness using Pozzetti et al. (2010) method.
        
        Parameters:
        -----------
        catalog_masses : array
            Stellar masses of galaxies
        catalog_magnitudes : array
            Magnitudes in detection band
        magnitude_limit : float
            Survey magnitude limit
        percentile : float
            Percentile to use (default: 95%)
            
        Returns:
        --------
        float : Mass completeness limit
        """
        # Select 30% faintest galaxies
        mag_threshold = np.percentile(catalog_magnitudes, 70)
        faint_mask = catalog_magnitudes >= mag_threshold
        
        # Rescale masses to magnitude limit
        delta_mag = magnitude_limit - catalog_magnitudes[faint_mask]
        rescaled_masses = catalog_masses[faint_mask] + 0.4 * delta_mag
        
        # Take 95th percentile as mass limit
        mass_limit = np.percentile(rescaled_masses, percentile)
        
        return mass_limit

class SchechterFitDiagnostics:
    """
    Class for diagnosing and validating Schechter function fits.
    """
    
    @staticmethod
    def calculate_integrated_density(log_phi_star: float, log_m_star: float, 
                                   alpha: float, mass_range: Tuple[float, float]) -> float:
        """
        Calculate integrated number density from Schechter parameters.
        
        Parameters:
        -----------
        log_phi_star : float
            Normalization parameter
        log_m_star : float
            Characteristic mass
        alpha : float
            Faint-end slope
        mass_range : tuple
            Integration range in log10(M/M_sun)
            
        Returns:
        --------
        float : Integrated number density in Mpc^-3
        """
        from scipy.special import gamma, gammainc
        
        phi_star = 10**log_phi_star
        m_star = 10**log_m_star
        
        # Convert mass range to linear units
        m_min, m_max = 10**mass_range[0], 10**mass_range[1]
        
        # Incomplete gamma function integration
        # This is an approximation for the integral
        x_min = m_min / m_star
        x_max = m_max / m_star
        
        integral = phi_star * m_star * (
            gamma(alpha + 2) * (gammainc(alpha + 2, x_max) - gammainc(alpha + 2, x_min))
        )
        
        return integral
    
    @staticmethod
    def residual_analysis(observed: np.ndarray, fitted: np.ndarray, 
                         uncertainties: np.ndarray) -> Dict:
        """
        Analyze residuals between observed and fitted mass function.
        
        Parameters:
        -----------
        observed : array
            Observed number densities
        fitted : array
            Fitted values
        uncertainties : array
            Uncertainties on observed values
            
        Returns:
        --------
        dict : Residual analysis results
        """
        # Calculate residuals
        residuals = observed - fitted
        
        # Mask out points with zero uncertainties or invalid values
        valid_mask = (uncertainties > 0) & np.isfinite(uncertainties) & np.isfinite(residuals)
        
        if np.sum(valid_mask) < 3:
            # Not enough valid points for analysis
            return {
                'residuals': residuals,
                'normalized_residuals': np.full_like(residuals, np.nan),
                'mean_residual': np.nan,
                'std_residual': np.nan,
                'chi2': np.nan,
                'ks_statistic': np.nan,
                'ks_pvalue': np.nan
            }
        
        # Calculate normalized residuals only for valid points
        normalized_residuals = np.full_like(residuals, np.nan)
        normalized_residuals[valid_mask] = residuals[valid_mask] / uncertainties[valid_mask]
        
        # Statistics on valid points only
        valid_norm_residuals = normalized_residuals[valid_mask]
        mean_residual = np.mean(valid_norm_residuals)
        std_residual = np.std(valid_norm_residuals)
        chi2 = np.sum(valid_norm_residuals**2)
        
        # Kolmogorov-Smirnov test for normality
        try:
            ks_stat, ks_pvalue = stats.kstest(valid_norm_residuals, 'norm')
        except:
            ks_stat, ks_pvalue = np.nan, np.nan
        
        return {
            'residuals': residuals,
            'normalized_residuals': normalized_residuals,
            'mean_residual': mean_residual,
            'std_residual': std_residual,
            'chi2': chi2,
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pvalue
        }
    
    @staticmethod
    def plot_residuals(mass_centers: np.ndarray, residuals: np.ndarray,
                      uncertainties: np.ndarray, title: str = "",
                      save_path: str = None):
        """
        Create residual plots for Schechter function fits.
        
        Parameters:
        -----------
        mass_centers : array
            Mass bin centers
        residuals : array
            Residuals (observed - fitted)
        uncertainties : array
            Uncertainties on observations
        title : str
            Plot title
        save_path : str, optional
            Path to save plot
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), 
                                      gridspec_kw={'height_ratios': [2, 1]})
        
        # Calculate normalized residuals, handling zero uncertainties
        valid_mask = (uncertainties > 0) & np.isfinite(uncertainties) & np.isfinite(residuals)
        normalized_residuals = np.full_like(residuals, np.nan)
        normalized_residuals[valid_mask] = residuals[valid_mask] / uncertainties[valid_mask]
        
        # Plot only valid points
        valid_points = valid_mask & np.isfinite(normalized_residuals)
        ax1.errorbar(mass_centers[valid_points], normalized_residuals[valid_points], yerr=1, 
                    fmt='o', capsize=3, markersize=6)
        ax1.axhline(0, color='red', linestyle='--', alpha=0.7)
        ax1.axhline(1, color='gray', linestyle=':', alpha=0.5)
        ax1.axhline(-1, color='gray', linestyle=':', alpha=0.5)
        ax1.set_ylabel('Normalized Residuals', fontsize=12)
        ax1.set_title(title, fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Histogram of normalized residuals (only finite values)
        finite_residuals = normalized_residuals[np.isfinite(normalized_residuals)]
        if len(finite_residuals) > 0:
            ax2.hist(finite_residuals, bins=10, alpha=0.7, density=True)
        x_norm = np.linspace(-3, 3, 100)
        ax2.plot(x_norm, stats.norm.pdf(x_norm), 'r-', 
                label='Standard Normal', linewidth=2)
        ax2.set_xlabel('Normalized Residuals', fontsize=12)
        ax2.set_ylabel('Density', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Residual plot saved to {save_path}")
        
        plt.show()

class LiteratureComparison:
    """
    Class for comparing results with literature measurements.
    """
    
    @staticmethod
    def load_cosmos2020_reference(redshift_range: Tuple[float, float]) -> Dict:
        """
        Load reference COSMOS2020 stellar mass function for comparison.
        
        This would normally load actual literature data.
        For now, returns typical values from the literature.
        
        Parameters:
        -----------
        redshift_range : tuple
            Redshift range
            
        Returns:
        --------
        dict : Reference mass function data
        """
        # Typical values from COSMOS2020 at z~0.5
        # These are approximate values for demonstration
        mass_centers = np.array([8.25, 8.75, 9.25, 9.75, 10.25, 10.75, 11.25, 11.75])
        
        if redshift_range[1] <= 1.0:
            # Low redshift values
            number_density = np.array([1e-2, 5e-3, 2e-3, 8e-4, 3e-4, 1e-4, 2e-5, 3e-6])
        elif redshift_range[1] <= 3.0:
            # Intermediate redshift values  
            number_density = np.array([5e-3, 2e-3, 8e-4, 3e-4, 1e-4, 3e-5, 5e-6, 5e-7])
        else:
            # High redshift values
            number_density = np.array([1e-3, 5e-4, 2e-4, 8e-5, 2e-5, 5e-6, 8e-7, 1e-7])
        
        # Typical uncertainties (20-50%)
        uncertainties = number_density * np.array([0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6])
        
        return {
            'mass_centers': mass_centers,
            'number_density': number_density,
            'uncertainties': uncertainties,
            'redshift_range': redshift_range,
            'reference': 'COSMOS2020 (approximate)'
        }
    
    @staticmethod
    def plot_comparison(our_data: Dict, literature_data: List[Dict],
                       title: str = "", save_path: str = None):
        """
        Plot comparison with literature measurements.
        
        Parameters:
        -----------
        our_data : dict
            Our stellar mass function data
        literature_data : list
            List of literature datasets
        title : str
            Plot title
        save_path : str, optional
            Path to save plot
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Plot our data
        ax.errorbar(our_data['mass_centers'], our_data['number_density'],
                   yerr=our_data['number_density_err'],
                   fmt='o', capsize=3, markersize=8, linewidth=2,
                   label='This work (COSMOS-Web)', color='red')
        
        # Plot literature data
        colors = ['blue', 'green', 'orange', 'purple', 'brown']
        for i, lit_data in enumerate(literature_data):
            color = colors[i % len(colors)]
            
            ax.errorbar(lit_data['mass_centers'], lit_data['number_density'],
                       yerr=lit_data['uncertainties'],
                       fmt='s', capsize=2, markersize=6, alpha=0.7,
                       label=lit_data['reference'], color=color)
        
        ax.set_xlabel('log₁₀(M*/M☉)', fontsize=14)
        ax.set_ylabel('Φ [Mpc⁻³ dex⁻¹]', fontsize=14)
        ax.set_yscale('log')
        ax.set_ylim(1e-7, 1e-1)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        
        if title:
            ax.set_title(title, fontsize=16)
        else:
            z_range = our_data.get('z_range', (0, 1))
            ax.set_title(f'Stellar Mass Function Comparison\nz = {z_range[0]:.1f} - {z_range[1]:.1f}', 
                        fontsize=16)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to {save_path}")
        
        plt.show()

def bootstrap_uncertainty(data: np.ndarray, n_bootstrap: int = 1000) -> Tuple[float, float]:
    """
    Calculate bootstrap uncertainties on data.
    
    Parameters:
    -----------
    data : array
        Input data
    n_bootstrap : int
        Number of bootstrap samples
        
    Returns:
    --------
    tuple : (median, std) of bootstrap distribution
    """
    bootstrap_samples = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        resampled = np.random.choice(data, size=len(data), replace=True)
        bootstrap_samples.append(np.median(resampled))
    
    return np.median(bootstrap_samples), np.std(bootstrap_samples)

def compute_cosmic_stellar_mass_density(mass_centers: np.ndarray,
                                      number_density: np.ndarray,
                                      mass_bin_width: float = 0.25) -> float:
    """
    Compute the cosmic stellar mass density by integrating the mass function.
    
    Parameters:
    -----------
    mass_centers : array
        Mass bin centers in log10(M/M_sun)
    number_density : array
        Number density per mass bin
    mass_bin_width : float
        Width of mass bins in dex
        
    Returns:
    --------
    float : Cosmic stellar mass density in M_sun Mpc^-3
    """
    # Convert to linear masses
    masses = 10**mass_centers
    
    # Integrate: rho = sum(Phi * M * dM)
    mass_density = np.sum(number_density * masses * mass_bin_width * np.log(10) * masses)
    
    return mass_density 