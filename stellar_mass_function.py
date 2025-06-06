#!/usr/bin/env python3
"""
Stellar Mass Function Computation for COSMOS-Web Data

This module provides tools to compute stellar mass functions from galaxy catalogs,
following the methodology described in COSMOS2020 papers (Weaver et al. 2022).
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
import pandas as pd
from scipy import optimize, integrate
from scipy.interpolate import interp1d
import warnings
from typing import Tuple, Dict, Optional, List

# Standard cosmology (following COSMOS papers)
COSMO = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Om0=0.3)

class StellarMassFunction:
    """
    Class for computing and analyzing stellar mass functions from galaxy catalogs.
    
    This implementation follows the methodology from:
    - Weaver et al. 2022 (COSMOS2020)
    - Davidzon et al. 2017 
    - Ilbert et al. 2013
    """
    
    def __init__(self, catalog_path: str, area_deg2: float = 0.54):
        """
        Initialize the stellar mass function calculator.
        
        Parameters:
        -----------
        catalog_path : str
            Path to the FITS catalog file
        area_deg2 : float
            Survey area in square degrees (default: 0.54 for COSMOS-Web)
        """
        self.catalog_path = catalog_path
        self.area_deg2 = area_deg2
        self.area_rad2 = area_deg2 * (np.pi/180)**2  # Convert to steradians
        self.catalog = None
        
        # Redshift bins (following COSMOS2020)
        self.z_bins = np.array([
            [0.2, 0.5], [0.5, 0.8], [0.8, 1.1], [1.1, 1.5], [1.5, 2.0],
            [2.0, 2.5], [2.5, 3.0], [3.0, 3.5], [3.5, 4.0], [4.0, 4.5],
            [4.5, 5.5], [5.5, 6.5], [6.5, 7.5]
        ])
        
        # Mass bins (log10 solar masses)
        self.mass_bin_width = 0.25
        self.mass_bins = np.arange(8.0, 12.5 + self.mass_bin_width, self.mass_bin_width)
        
    def load_catalog(self) -> Table:
        """Load and prepare the galaxy catalog."""
        try:
            with fits.open(self.catalog_path) as hdul:
                data = hdul[1].data
                
                # Convert to astropy Table for easier handling
                self.catalog = Table(data)
                
                # Detect catalog type and set appropriate column names
                if 'mass_med' in self.catalog.colnames:
                    # LePhare catalog
                    print("Detected LePhare catalog")
                    mass_col = 'mass_med'
                    z_col = 'zpdf_med'
                    sfr_col = 'sfr_med'
                    # Filter out bad/null values (typically -999 or -99 in LePhare)
                    good_mass_mask = (self.catalog[mass_col] > -10) & np.isfinite(self.catalog[mass_col])
                    good_z_mask = (self.catalog[z_col] >= 0) & np.isfinite(self.catalog[z_col])
                elif 'mass' in self.catalog.colnames:
                    # CIGALE catalog  
                    print("Detected CIGALE catalog")
                    mass_col = 'mass'
                    z_col = None  # CIGALE doesn't have redshift
                    sfr_col = 'sfr_inst' if 'sfr_inst' in self.catalog.colnames else None
                    good_mass_mask = np.isfinite(self.catalog[mass_col])
                    good_z_mask = np.ones(len(self.catalog), dtype=bool)  # No redshift filtering
                else:
                    raise ValueError("Could not identify catalog type - no recognizable mass column found")
                
                print(f"Found {np.sum(good_mass_mask)} sources with good masses out of {len(self.catalog)} total")
                if z_col:
                    print(f"Found {np.sum(good_z_mask)} sources with good redshifts out of {len(self.catalog)} total")
                
                # Apply quality masks
                quality_mask = good_mass_mask & good_z_mask
                self.catalog = self.catalog[quality_mask]
                
                # Set up mass column
                mass_values = self.catalog[mass_col]
                if np.median(mass_values) > 1e6:
                    # Assume linear solar masses, convert to log
                    mass_values = np.maximum(mass_values, 1e-10)  # Minimum mass floor
                    self.catalog['log_mass'] = np.log10(mass_values)
                    print("Converted linear stellar masses to log10(M/M_sun)")
                else:
                    # Assume already in log units
                    self.catalog['log_mass'] = mass_values
                    print("Using stellar masses as log10(M/M_sun)")
                
                # Set up redshift column if available
                if z_col:
                    self.catalog['redshift'] = self.catalog[z_col]
                    print(f"Using {z_col} as redshift column")
                    print(f"Redshift range: {np.min(self.catalog['redshift']):.3f} - {np.max(self.catalog['redshift']):.3f}")
                else:
                    self.catalog['redshift'] = np.full(len(self.catalog), np.nan)
                    print("No redshift information available")
                
                # Set up SFR column if available
                if sfr_col and sfr_col in self.catalog.colnames:
                    # Filter out bad SFR values
                    good_sfr_mask = (self.catalog[sfr_col] > -10) & np.isfinite(self.catalog[sfr_col])
                    self.catalog['sfr'] = np.where(good_sfr_mask, self.catalog[sfr_col], np.nan)
                    print(f"Loaded SFR data from {sfr_col}")
                else:
                    self.catalog['sfr'] = np.full(len(self.catalog), np.nan)
                    print("No SFR information available")
                
                # Further filter for reasonable mass values (above 1 M_sun, below 10^15 M_sun)
                reasonable_mass_mask = (self.catalog['log_mass'] > 0) & (self.catalog['log_mass'] < 15)
                if np.sum(~reasonable_mass_mask) > 0:
                    print(f"Filtering out {np.sum(~reasonable_mass_mask)} sources with unreasonable masses")
                    self.catalog = self.catalog[reasonable_mass_mask]
                
                print(f"Final catalog with {len(self.catalog)} sources")
                if len(self.catalog) > 0:
                    print(f"Mass range: {np.min(self.catalog['log_mass']):.2f} - "
                          f"{np.max(self.catalog['log_mass']):.2f} log(M/M_sun)")
                
                return self.catalog
                
        except Exception as e:
            print(f"Error loading catalog: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def compute_mass_completeness_limit(self, z_bin: Tuple[float, float], 
                                      completeness_level: float = 0.7) -> float:
        """
        Compute mass completeness limit for a given redshift bin.
        
        Following Pozzetti et al. (2010) method used in COSMOS2020.
        
        Parameters:
        -----------
        z_bin : tuple
            Redshift range (z_min, z_max)
        completeness_level : float
            Completeness fraction (default: 0.7 for 70% completeness)
            
        Returns:
        --------
        float : Mass completeness limit in log10(M/M_sun)
        """
        z_min, z_max = z_bin
        z_center = (z_min + z_max) / 2
        
        # Empirical mass completeness from COSMOS2020 (Equations 3-5)
        # Total sample completeness
        log_mass_limit = (8.5 + 0.5 * z_center + 0.1 * z_center**2)
        
        return log_mass_limit
    
    def compute_comoving_volume(self, z_bin: Tuple[float, float]) -> float:
        """
        Compute comoving volume for a redshift bin.
        
        Parameters:
        -----------
        z_bin : tuple
            Redshift range (z_min, z_max)
            
        Returns:
        --------
        float : Comoving volume in Mpc^3
        """
        z_min, z_max = z_bin
        
        # Compute comoving distance
        d_min = COSMO.comoving_distance(z_min).value  # Mpc
        d_max = COSMO.comoving_distance(z_max).value  # Mpc
        
        # Volume element
        volume = (4 * np.pi / 3) * (d_max**3 - d_min**3) * (self.area_rad2 / (4 * np.pi))
        
        return volume
    
    def apply_vmax_correction(self, masses: np.ndarray, redshifts: np.ndarray,
                            z_bin: Tuple[float, float]) -> np.ndarray:
        """
        Apply 1/Vmax correction to account for Malmquist bias.
        
        Following Schmidt (1968) method.
        
        Parameters:
        -----------
        masses : array
            Stellar masses in log10(M/M_sun)
        redshifts : array  
            Photometric redshifts
        z_bin : tuple
            Redshift bin range
            
        Returns:
        --------
        array : 1/Vmax weights
        """
        z_min, z_max = z_bin
        weights = np.ones(len(masses))
        
        # For each galaxy, compute maximum redshift it could be observed
        # This is simplified - in practice would use SED fitting
        for i in range(len(masses)):
            # Assume we can detect galaxies of this mass out to z_max
            # In practice, this depends on magnitude limits and SED properties
            z_max_observable = z_max
            
            # Compute accessible volume
            d_min = COSMO.comoving_distance(max(z_min, 0.01)).value
            d_max_obs = COSMO.comoving_distance(min(z_max_observable, z_max)).value
            
            v_max = (4 * np.pi / 3) * (d_max_obs**3 - d_min**3) * (self.area_rad2 / (4 * np.pi))
            v_total = self.compute_comoving_volume(z_bin)
            
            weights[i] = v_total / v_max if v_max > 0 else 1.0
            
        return weights
    
    def compute_stellar_mass_function(self, z_bin: Tuple[float, float], 
                                    apply_completeness: bool = True,
                                    apply_vmax: bool = True) -> Dict:
        """
        Compute the stellar mass function for a given redshift bin.
        
        Parameters:
        -----------
        z_bin : tuple
            Redshift range (z_min, z_max)
        apply_completeness : bool
            Whether to apply mass completeness cuts
        apply_vmax : bool
            Whether to apply Vmax corrections
            
        Returns:
        --------
        dict : Dictionary containing mass bins, number densities, and uncertainties
        """
        if self.catalog is None:
            print("Error: No catalog loaded. Call load_catalog() first.")
            return {}
            
        z_min, z_max = z_bin
        
        # Check if we have redshift information
        if 'redshift' in self.catalog.colnames and not np.all(np.isnan(self.catalog['redshift'])):
            # Select galaxies in the redshift bin
            z_mask = (self.catalog['redshift'] >= z_min) & (self.catalog['redshift'] < z_max)
            mass_mask = np.isfinite(self.catalog['log_mass'])
            combined_mask = z_mask & mass_mask
            
            selected_masses = self.catalog['log_mass'][combined_mask]
            selected_redshifts = self.catalog['redshift'][combined_mask]
            
            print(f"Selected {len(selected_masses)} galaxies in redshift range {z_min:.1f} < z < {z_max:.1f}")
            if len(selected_masses) > 0:
                print(f"Actual redshift range: {np.min(selected_redshifts):.3f} - {np.max(selected_redshifts):.3f}")
        else:
            # Fallback to using all sources (for CIGALE or catalogs without redshift)
            print(f"Warning: No redshift information available.")
            print(f"Computing SMF for all sources (assuming single redshift bin)")
            
            mask = np.isfinite(self.catalog['log_mass'])
            selected_masses = self.catalog['log_mass'][mask]
            selected_redshifts = np.full(len(selected_masses), (z_min + z_max) / 2)
        
        # Apply mass completeness if requested
        if apply_completeness:
            mass_limit = self.compute_mass_completeness_limit(z_bin)
            completeness_mask = selected_masses >= mass_limit
            selected_masses = selected_masses[completeness_mask]
            selected_redshifts = selected_redshifts[completeness_mask]
            print(f"Applied mass completeness cut at log(M) = {mass_limit:.2f}")
            print(f"Kept {len(selected_masses)} galaxies above mass limit")
        
        # Compute volume
        volume = self.compute_comoving_volume(z_bin)
        print(f"Comoving volume: {volume:.2e} Mpc^3")
        
        # Create mass function
        mass_centers = self.mass_bins[:-1] + self.mass_bin_width / 2
        number_density = np.zeros(len(mass_centers))
        number_density_err = np.zeros(len(mass_centers))
        
        for i, (m_low, m_high) in enumerate(zip(self.mass_bins[:-1], self.mass_bins[1:])):
            bin_mask = (selected_masses >= m_low) & (selected_masses < m_high)
            n_gal = np.sum(bin_mask)
            
            if apply_vmax and n_gal > 0:
                weights = self.apply_vmax_correction(
                    selected_masses[bin_mask], selected_redshifts[bin_mask], z_bin
                )
                n_gal_weighted = np.sum(weights)
            else:
                n_gal_weighted = n_gal
            
            # Number density per Mpc^3 per dex
            number_density[i] = n_gal_weighted / (volume * self.mass_bin_width)
            
            # Poisson uncertainty
            number_density_err[i] = np.sqrt(n_gal) / (volume * self.mass_bin_width)
        
        return {
            'mass_centers': mass_centers,
            'number_density': number_density,
            'number_density_err': number_density_err,
            'volume': volume,
            'n_galaxies': len(selected_masses),
            'z_range': z_bin
        }
    
    def fit_schechter_function(self, mass_centers: np.ndarray, 
                             number_density: np.ndarray,
                             number_density_err: np.ndarray,
                             single_schechter: bool = True) -> Dict:
        """
        Fit Schechter function to stellar mass function data.
        
        Single Schechter: Φ(M) = ln(10) * Φ* * exp(-M/M*) * (M/M*)^(α+1)
        Double Schechter: sum of two Schechter functions
        
        Parameters:
        -----------
        mass_centers : array
            Mass bin centers in log10(M/M_sun)
        number_density : array
            Number density values
        number_density_err : array
            Uncertainties on number density
        single_schechter : bool
            Whether to fit single (True) or double (False) Schechter function
            
        Returns:
        --------
        dict : Best-fit parameters and uncertainties
        """
        # Mask out zero or negative densities
        valid_mask = (number_density > 0) & (number_density_err > 0)
        x = mass_centers[valid_mask]
        y = number_density[valid_mask]
        yerr = number_density_err[valid_mask]
        
        if len(x) < 3:
            return {'success': False, 'message': 'Insufficient data points for fitting'}
        
        def schechter_function(log_mass, log_phi_star, log_m_star, alpha):
            """Single Schechter function"""
            phi_star = 10**log_phi_star
            m_star = 10**log_m_star
            mass = 10**log_mass
            
            return (np.log(10) * phi_star * 
                   np.exp(-mass/m_star) * 
                   (mass/m_star)**(alpha + 1))
        
        def double_schechter_function(log_mass, log_phi_star1, log_phi_star2, 
                                    log_m_star, alpha1, alpha2):
            """Double Schechter function"""
            phi_star1 = 10**log_phi_star1
            phi_star2 = 10**log_phi_star2
            m_star = 10**log_m_star
            mass = 10**log_mass
            
            component1 = (np.log(10) * phi_star1 * 
                         np.exp(-mass/m_star) * 
                         (mass/m_star)**(alpha1 + 1))
            
            component2 = (np.log(10) * phi_star2 * 
                         np.exp(-mass/m_star) * 
                         (mass/m_star)**(alpha2 + 1))
            
            return component1 + component2
        
        try:
            if single_schechter:
                # Initial guesses for single Schechter
                p0 = [-3.0, 10.8, -1.2]  # log_phi_star, log_m_star, alpha
                bounds = ([-6, 9, -3], [0, 12, 0])
                
                popt, pcov = optimize.curve_fit(
                    schechter_function, x, y, p0=p0, sigma=yerr, 
                    bounds=bounds, maxfev=5000
                )
                
                # Calculate goodness of fit
                y_fit = schechter_function(x, *popt)
                chi2 = np.sum(((y - y_fit) / yerr)**2)
                dof = len(x) - len(popt)
                
                result = {
                    'success': True,
                    'function_type': 'single_schechter',
                    'log_phi_star': popt[0],
                    'log_m_star': popt[1], 
                    'alpha': popt[2],
                    'errors': np.sqrt(np.diag(pcov)),
                    'chi2': chi2,
                    'dof': dof,
                    'chi2_reduced': chi2/dof if dof > 0 else np.inf
                }
                
            else:
                # Initial guesses for double Schechter
                p0 = [-3.5, -2.5, 10.8, -1.5, -0.5]  # log_phi_star1, log_phi_star2, log_m_star, alpha1, alpha2
                bounds = ([-6, -6, 9, -3, -3], [0, 0, 12, 0, 2])
                
                popt, pcov = optimize.curve_fit(
                    double_schechter_function, x, y, p0=p0, sigma=yerr,
                    bounds=bounds, maxfev=5000
                )
                
                # Calculate goodness of fit
                y_fit = double_schechter_function(x, *popt)
                chi2 = np.sum(((y - y_fit) / yerr)**2)
                dof = len(x) - len(popt)
                
                result = {
                    'success': True,
                    'function_type': 'double_schechter',
                    'log_phi_star1': popt[0],
                    'log_phi_star2': popt[1],
                    'log_m_star': popt[2],
                    'alpha1': popt[3],
                    'alpha2': popt[4],
                    'errors': np.sqrt(np.diag(pcov)),
                    'chi2': chi2,
                    'dof': dof,
                    'chi2_reduced': chi2/dof if dof > 0 else np.inf
                }
                
        except Exception as e:
            result = {'success': False, 'message': f'Fitting failed: {str(e)}'}
            
        return result
    
    def plot_stellar_mass_function(self, smf_data: Dict, fit_result: Dict = None,
                                 title: str = "", save_path: str = None):
        """
        Plot the stellar mass function with optional Schechter fit.
        
        Parameters:
        -----------
        smf_data : dict
            Output from compute_stellar_mass_function()
        fit_result : dict, optional
            Output from fit_schechter_function()
        title : str
            Plot title
        save_path : str, optional
            Path to save the plot
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        mass_centers = smf_data['mass_centers']
        number_density = smf_data['number_density']
        number_density_err = smf_data['number_density_err']
        
        # Plot data points
        valid_mask = number_density > 0
        ax.errorbar(mass_centers[valid_mask], number_density[valid_mask], 
                   yerr=number_density_err[valid_mask],
                   fmt='o', capsize=3, label='Data', markersize=6)
        
        # Plot Schechter fit if provided
        if fit_result and fit_result.get('success'):
            mass_fine = np.linspace(mass_centers.min() - 0.5, mass_centers.max() + 0.5, 100)
            
            if fit_result['function_type'] == 'single_schechter':
                def schechter_function(log_mass, log_phi_star, log_m_star, alpha):
                    phi_star = 10**log_phi_star
                    m_star = 10**log_m_star
                    mass = 10**log_mass
                    return (np.log(10) * phi_star * 
                           np.exp(-mass/m_star) * 
                           (mass/m_star)**(alpha + 1))
                
                y_fit = schechter_function(mass_fine, 
                                         fit_result['log_phi_star'],
                                         fit_result['log_m_star'],
                                         fit_result['alpha'])
                
                label = (f"Schechter fit\n"
                        f"log(Φ*) = {fit_result['log_phi_star']:.2f}\n"
                        f"log(M*) = {fit_result['log_m_star']:.2f}\n"
                        f"α = {fit_result['alpha']:.2f}\n"
                        f"χ²/ν = {fit_result['chi2_reduced']:.2f}")
                
            else:  # double_schechter
                def double_schechter_function(log_mass, log_phi_star1, log_phi_star2, 
                                            log_m_star, alpha1, alpha2):
                    phi_star1 = 10**log_phi_star1
                    phi_star2 = 10**log_phi_star2
                    m_star = 10**log_m_star
                    mass = 10**log_mass
                    
                    component1 = (np.log(10) * phi_star1 * 
                                 np.exp(-mass/m_star) * 
                                 (mass/m_star)**(alpha1 + 1))
                    
                    component2 = (np.log(10) * phi_star2 * 
                                 np.exp(-mass/m_star) * 
                                 (mass/m_star)**(alpha2 + 1))
                    
                    return component1 + component2
                
                y_fit = double_schechter_function(mass_fine,
                                                fit_result['log_phi_star1'],
                                                fit_result['log_phi_star2'],
                                                fit_result['log_m_star'],
                                                fit_result['alpha1'],
                                                fit_result['alpha2'])
                
                label = (f"Double Schechter fit\n"
                        f"log(M*) = {fit_result['log_m_star']:.2f}\n"
                        f"χ²/ν = {fit_result['chi2_reduced']:.2f}")
            
            ax.plot(mass_fine, y_fit, 'r-', linewidth=2, label=label)
        
        ax.set_xlabel('log₁₀(M*/M☉)', fontsize=14)
        ax.set_ylabel('Φ [Mpc⁻³ dex⁻¹]', fontsize=14)
        ax.set_yscale('log')
        ax.set_ylim(1e-7, 1e-1)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        if title:
            ax.set_title(title, fontsize=16)
        else:
            z_range = smf_data.get('z_range', (0, 1))
            ax.set_title(f'Stellar Mass Function (z = {z_range[0]:.1f} - {z_range[1]:.1f})', 
                        fontsize=16)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
        
    def run_full_analysis(self, z_bin: Tuple[float, float] = (0.2, 0.5),
                         plot: bool = True, save_plots: bool = False) -> Dict:
        """
        Run complete stellar mass function analysis for a redshift bin.
        
        Parameters:
        -----------
        z_bin : tuple
            Redshift range to analyze
        plot : bool
            Whether to create plots
        save_plots : bool
            Whether to save plots to files
            
        Returns:
        --------
        dict : Complete analysis results
        """
        print(f"\n=== Stellar Mass Function Analysis ===")
        print(f"Redshift range: {z_bin[0]} < z ≤ {z_bin[1]}")
        print(f"Survey area: {self.area_deg2} deg²")
        
        # Load catalog if not already loaded
        if self.catalog is None:
            self.load_catalog()
        
        # Compute stellar mass function
        smf_data = self.compute_stellar_mass_function(z_bin)
        
        print(f"\nStellar Mass Function Results:")
        print(f"Number of galaxies: {smf_data['n_galaxies']}")
        print(f"Comoving volume: {smf_data['volume']:.2e} Mpc³")
        
        # Fit Schechter functions
        print(f"\nFitting Schechter functions...")
        
        single_fit = self.fit_schechter_function(
            smf_data['mass_centers'], 
            smf_data['number_density'],
            smf_data['number_density_err'],
            single_schechter=True
        )
        
        double_fit = self.fit_schechter_function(
            smf_data['mass_centers'], 
            smf_data['number_density'],
            smf_data['number_density_err'],
            single_schechter=False
        )
        
        # Choose best fit based on chi2
        best_fit = single_fit
        if (double_fit.get('success') and single_fit.get('success')):
            if double_fit['chi2_reduced'] < single_fit['chi2_reduced']:
                best_fit = double_fit
                print("Double Schechter provides better fit")
            else:
                print("Single Schechter provides better fit")
        
        if best_fit.get('success'):
            print(f"Best fit: {best_fit['function_type']}")
            print(f"Reduced χ²: {best_fit['chi2_reduced']:.2f}")
        
        # Create plots
        if plot:
            title = f"COSMOS-Web Stellar Mass Function\nz = {z_bin[0]:.1f} - {z_bin[1]:.1f}"
            save_path = f"stellar_mass_function_z{z_bin[0]:.1f}-{z_bin[1]:.1f}.png" if save_plots else None
            
            self.plot_stellar_mass_function(smf_data, best_fit, title, save_path)
        
        return {
            'smf_data': smf_data,
            'single_fit': single_fit,
            'double_fit': double_fit,
            'best_fit': best_fit
        } 