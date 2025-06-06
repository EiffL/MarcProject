#!/usr/bin/env python3
"""
Script to explore the structure of the COSMOS-Web LePhare catalog
"""

import astropy.io.fits as fits
import numpy as np

def explore_lephare_catalog():
    """Explore the structure of the COSMOS-Web LePhare FITS catalog"""
    
    try:
        with fits.open('data/COSMOSWeb_mastercatalog_v1_lephare.fits') as hdul:
            print('FITS file structure:')
            hdul.info()
            
            print('\nColumn names and types:')
            for i, (name, format) in enumerate(zip(hdul[1].columns.names, hdul[1].columns.formats)):
                if i < 50:  # Show first 50 columns
                    print(f'{name}: {format}')
            
            if len(hdul[1].columns.names) > 50:
                print(f'... and {len(hdul[1].columns.names) - 50} more columns')
            print(f'\nTotal number of sources: {len(hdul[1].data)}')
            
            # Look for key columns we need for stellar mass function
            data = hdul[1].data
            column_names = hdul[1].columns.names
            
            print('\nLooking for key columns:')
            key_columns = ['ID', 'RA', 'DEC', 'z_phot', 'z_best', 'redshift', 'mass', 'mstar', 'sfr']
            for key in key_columns:
                matches = [col for col in column_names if key.lower() in col.lower()]
                if matches:
                    print(f'{key}: {matches}')
                else:
                    print(f'{key}: NOT FOUND')
            
            # Check for magnitude columns
            print('\nMagnitude/flux columns:')
            mag_cols = [col for col in column_names if ('mag' in col.lower() or 'flux' in col.lower())]
            print(f'Found {len(mag_cols)} magnitude/flux columns')
            if len(mag_cols) > 0:
                print(f'Examples: {mag_cols[:10]}')
            
            # Check data ranges for key columns
            print('\nChecking data ranges for key columns:')
            
            # Redshift columns
            z_cols = [col for col in column_names if ('z_' in col.lower() or 'redshift' in col.lower())]
            for col in z_cols[:5]:  # Check first 5 redshift columns
                if col in data.dtype.names:
                    values = data[col]
                    finite_values = values[np.isfinite(values)]
                    if len(finite_values) > 0:
                        print(f'{col}: min={np.min(finite_values):.3f}, max={np.max(finite_values):.3f}, median={np.median(finite_values):.3f}')
                        print(f'  {len(finite_values)} finite values out of {len(values)} total')
            
            # Mass columns
            mass_cols = [col for col in column_names if ('mass' in col.lower() or 'mstar' in col.lower())]
            for col in mass_cols[:5]:  # Check first 5 mass columns
                if col in data.dtype.names:
                    values = data[col]
                    finite_values = values[np.isfinite(values)]
                    if len(finite_values) > 0:
                        print(f'{col}: min={np.min(finite_values):.2e}, max={np.max(finite_values):.2e}, median={np.median(finite_values):.2e}')
                        print(f'  {len(finite_values)} finite values out of {len(values)} total')
            
            # SFR columns
            sfr_cols = [col for col in column_names if 'sfr' in col.lower()]
            for col in sfr_cols[:3]:  # Check first 3 SFR columns
                if col in data.dtype.names:
                    values = data[col]
                    finite_values = values[np.isfinite(values)]
                    if len(finite_values) > 0:
                        print(f'{col}: min={np.min(finite_values):.2e}, max={np.max(finite_values):.2e}, median={np.median(finite_values):.2e}')
                        print(f'  {len(finite_values)} finite values out of {len(values)} total')
                    
    except Exception as e:
        print(f"Error reading FITS file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    explore_lephare_catalog() 