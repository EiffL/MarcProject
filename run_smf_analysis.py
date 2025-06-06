#!/usr/bin/env python3
"""
Main script to run stellar mass function analysis on COSMOS-Web data

This script demonstrates how to use the stellar mass function tools
to analyze galaxy catalogs and compute mass functions.
"""

import numpy as np
import matplotlib.pyplot as plt
from stellar_mass_function import StellarMassFunction
from smf_utils import (UncertaintyCalculator, MassCompletenessCalculator, 
                      SchechterFitDiagnostics, LiteratureComparison,
                      compute_cosmic_stellar_mass_density)
import argparse
import sys
import os

def main():
    """Run the stellar mass function analysis."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Compute stellar mass functions from COSMOS-Web data')
    parser.add_argument('--catalog', default='data/COSMOSWeb_mastercatalog_v1_lephare.fits',
                       help='Path to FITS catalog file')
    parser.add_argument('--area', type=float, default=0.54,
                       help='Survey area in square degrees')
    parser.add_argument('--z_min', type=float, default=0.2,
                       help='Minimum redshift')
    parser.add_argument('--z_max', type=float, default=0.5,
                       help='Maximum redshift')
    parser.add_argument('--output_dir', default='output',
                       help='Output directory for plots and results')
    parser.add_argument('--no_plots', action='store_true',
                       help='Skip creating plots')
    parser.add_argument('--save_plots', action='store_true',
                       help='Save plots to files')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*60)
    print("COSMOS-Web Stellar Mass Function Analysis")
    print("="*60)
    print(f"Catalog: {args.catalog}")
    print(f"Survey area: {args.area} deg²")
    print(f"Redshift range: {args.z_min} < z ≤ {args.z_max}")
    print(f"Output directory: {args.output_dir}")
    
    # Check if catalog exists
    if not os.path.exists(args.catalog):
        print(f"Error: Catalog file not found: {args.catalog}")
        sys.exit(1)
    
    try:
        # Initialize stellar mass function calculator
        smf = StellarMassFunction(args.catalog, area_deg2=args.area)
        
        # Load catalog
        print(f"\nLoading catalog...")
        catalog = smf.load_catalog()
        if catalog is None:
            print("Error: Failed to load catalog")
            sys.exit(1)
        
        # Define redshift bin
        z_bin = (args.z_min, args.z_max)
        
        # Run analysis
        print(f"\nRunning stellar mass function analysis...")
        results = smf.run_full_analysis(
            z_bin=z_bin,
            plot=(not args.no_plots),
            save_plots=args.save_plots
        )
        
        # Extract results
        smf_data = results['smf_data']
        best_fit = results['best_fit']
        
        # Print detailed results
        print(f"\n" + "="*50)
        print("ANALYSIS RESULTS")
        print("="*50)
        
        print(f"Redshift range: {z_bin[0]} < z ≤ {z_bin[1]}")
        print(f"Number of galaxies: {smf_data['n_galaxies']}")
        print(f"Comoving volume: {smf_data['volume']:.2e} Mpc³")
        
        # Mass function table
        print(f"\nStellar Mass Function:")
        print(f"{'Mass [log M/M☉]':<15} {'Φ [Mpc⁻³ dex⁻¹]':<18} {'σ_Φ':<12}")
        print("-" * 50)
        
        for i in range(len(smf_data['mass_centers'])):
            mass = smf_data['mass_centers'][i]
            phi = smf_data['number_density'][i]
            err = smf_data['number_density_err'][i]
            
            if phi > 0:
                print(f"{mass:8.2f}          {phi:.3e}         {err:.3e}")
            else:
                print(f"{mass:8.2f}          {'<' + f'{err:.2e}':>11}         {err:.3e}")
        
        # Schechter fit results
        if best_fit.get('success'):
            print(f"\nBest Schechter Function Fit:")
            print(f"Function type: {best_fit['function_type']}")
            print(f"Reduced χ²: {best_fit['chi2_reduced']:.2f}")
            
            if best_fit['function_type'] == 'single_schechter':
                print(f"log(Φ*) = {best_fit['log_phi_star']:.3f} ± {best_fit['errors'][0]:.3f}")
                print(f"log(M*) = {best_fit['log_m_star']:.3f} ± {best_fit['errors'][1]:.3f}")
                print(f"α = {best_fit['alpha']:.3f} ± {best_fit['errors'][2]:.3f}")
            else:
                print(f"log(Φ₁*) = {best_fit['log_phi_star1']:.3f} ± {best_fit['errors'][0]:.3f}")
                print(f"log(Φ₂*) = {best_fit['log_phi_star2']:.3f} ± {best_fit['errors'][1]:.3f}")
                print(f"log(M*) = {best_fit['log_m_star']:.3f} ± {best_fit['errors'][2]:.3f}")
                print(f"α₁ = {best_fit['alpha1']:.3f} ± {best_fit['errors'][3]:.3f}")
                print(f"α₂ = {best_fit['alpha2']:.3f} ± {best_fit['errors'][4]:.3f}")
        
        # Cosmic stellar mass density
        cosmic_smd = compute_cosmic_stellar_mass_density(
            smf_data['mass_centers'],
            smf_data['number_density']
        )
        print(f"\nCosmic stellar mass density: {cosmic_smd:.2e} M☉ Mpc⁻³")
        
        # Additional analysis
        print(f"\n" + "="*50)
        print("ADDITIONAL ANALYSIS")
        print("="*50)
        
        # Mass completeness
        z_center = (z_bin[0] + z_bin[1]) / 2
        mass_completeness = MassCompletenessCalculator.cosmos_mass_completeness(
            z_center, survey_depth='cosmos_web'
        )
        print(f"Mass completeness limit: {mass_completeness:.2f} log(M/M☉)")
        
        # Uncertainty analysis
        print(f"\nUncertainty breakdown (approximate):")
        uncertainties = UncertaintyCalculator()
        
        # Example for one mass bin
        example_mass = 10.0
        example_idx = np.argmin(np.abs(smf_data['mass_centers'] - example_mass))
        
        if example_idx < len(smf_data['mass_centers']):
            n_gal = smf_data['number_density'][example_idx] * smf_data['volume'] * 0.25
            poisson_err = uncertainties.poisson_uncertainty(np.array([n_gal]))[0]
            cosmic_var = uncertainties.cosmic_variance(
                np.array([example_mass]), z_center, args.area
            )[0]
            sed_err = uncertainties.sed_fitting_uncertainty(
                np.array([example_mass]), z_center
            )[0]
            
            print(f"At log(M) = {example_mass:.1f}:")
            print(f"  Poisson uncertainty: {poisson_err:.1%}")
            print(f"  Cosmic variance: {cosmic_var:.1%}")
            print(f"  SED fitting uncertainty: {sed_err:.1%}")
        
        # Literature comparison (if plotting enabled)
        if not args.no_plots:
            print(f"\nCreating literature comparison plot...")
            
            # Load reference data
            lit_comp = LiteratureComparison()
            reference_data = [lit_comp.load_cosmos2020_reference(z_bin)]
            
            # Create comparison plot
            save_path = None
            if args.save_plots:
                save_path = os.path.join(args.output_dir, 
                                       f"smf_comparison_z{z_bin[0]:.1f}-{z_bin[1]:.1f}.png")
            
            lit_comp.plot_comparison(
                smf_data, reference_data,
                title="COSMOS-Web vs Literature Comparison",
                save_path=save_path
            )
        
        # Residual analysis for fit
        if best_fit.get('success') and not args.no_plots:
            print(f"\nPerforming residual analysis...")
            
            # Calculate fitted values
            mass_fine = smf_data['mass_centers']
            
            if best_fit['function_type'] == 'single_schechter':
                def schechter_function(log_mass, log_phi_star, log_m_star, alpha):
                    phi_star = 10**log_phi_star
                    m_star = 10**log_m_star
                    mass = 10**log_mass
                    return (np.log(10) * phi_star * 
                           np.exp(-mass/m_star) * 
                           (mass/m_star)**(alpha + 1))
                
                fitted_values = schechter_function(
                    mass_fine,
                    best_fit['log_phi_star'],
                    best_fit['log_m_star'],
                    best_fit['alpha']
                )
            else:
                # Double Schechter case would go here
                fitted_values = smf_data['number_density']  # Placeholder
            
            # Residual analysis
            diagnostics = SchechterFitDiagnostics()
            residual_results = diagnostics.residual_analysis(
                smf_data['number_density'],
                fitted_values,
                smf_data['number_density_err']
            )
            
            print(f"Residual analysis:")
            print(f"  Mean normalized residual: {residual_results['mean_residual']:.3f}")
            print(f"  Std normalized residual: {residual_results['std_residual']:.3f}")
            print(f"  K-S test p-value: {residual_results['ks_pvalue']:.3f}")
            
            # Plot residuals
            save_path = None
            if args.save_plots:
                save_path = os.path.join(args.output_dir, 
                                       f"residuals_z{z_bin[0]:.1f}-{z_bin[1]:.1f}.png")
            
            diagnostics.plot_residuals(
                mass_fine,
                residual_results['residuals'],
                smf_data['number_density_err'],
                title=f"Residual Analysis (z = {z_bin[0]:.1f} - {z_bin[1]:.1f})",
                save_path=save_path
            )
        
        # Save numerical results
        output_file = os.path.join(args.output_dir, 
                                 f"smf_results_z{z_bin[0]:.1f}-{z_bin[1]:.1f}.txt")
        
        with open(output_file, 'w') as f:
            f.write("# COSMOS-Web Stellar Mass Function Results\n")
            f.write(f"# Redshift range: {z_bin[0]} < z ≤ {z_bin[1]}\n")
            f.write(f"# Survey area: {args.area} deg²\n")
            f.write(f"# Number of galaxies: {smf_data['n_galaxies']}\n")
            f.write(f"# Comoving volume: {smf_data['volume']:.2e} Mpc³\n")
            f.write("#\n")
            f.write("# Columns: log_mass  number_density  uncertainty\n")
            f.write("# Units: [log M/M_sun]  [Mpc^-3 dex^-1]  [Mpc^-3 dex^-1]\n")
            
            for i in range(len(smf_data['mass_centers'])):
                f.write(f"{smf_data['mass_centers'][i]:.3f}  "
                       f"{smf_data['number_density'][i]:.6e}  "
                       f"{smf_data['number_density_err'][i]:.6e}\n")
        
        print(f"\nResults saved to: {output_file}")
        
        print(f"\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 