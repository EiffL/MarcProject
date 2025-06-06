#!/usr/bin/env python3
"""
Script to explore the structure of the COSMOS-Web catalog
"""

import astropy.io.fits as fits
import numpy as np

def explore_cosmos_catalog():
    """Explore the structure of the COSMOS-Web FITS catalog"""
    
    try:
        with fits.open('data/COSMOSWeb_mastercatalog_v1_cigale.fits') as hdul:
            print('FITS file structure:')
            hdul.info()
            
            print('\nColumn names and types:')
            for i, (name, format) in enumerate(zip(hdul[1].columns.names, hdul[1].columns.formats)):
                if i < 30:  # Show first 30 columns
                    print(f'{name}: {format}')
            
            print(f'... and {len(hdul[1].columns.names) - 30} more columns')
            print(f'\nTotal number of sources: {len(hdul[1].data)}')
            
            # Look for key columns we need for stellar mass function
            data = hdul[1].data
            column_names = hdul[1].columns.names
            
            print('\nLooking for key columns:')
            key_columns = ['ID', 'RA', 'DEC', 'z_phot', 'mass', 'sfr', 'redshift']
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
                print(f'Examples: {mag_cols[:5]}')
            
            # Check data ranges for potential stellar mass column
            print('\nChecking data ranges for potential stellar mass columns:')
            mass_cols = [col for col in column_names if 'mass' in col.lower() or 'mstar' in col.lower()]
            for col in mass_cols[:5]:  # Check first 5 mass columns
                values = data[col]
                finite_values = values[np.isfinite(values)]
                if len(finite_values) > 0:
                    print(f'{col}: min={np.min(finite_values):.2f}, max={np.max(finite_values):.2f}, median={np.median(finite_values):.2f}')
                    
    except Exception as e:
        print(f"Error reading FITS file: {e}")

if __name__ == "__main__":
    explore_cosmos_catalog() 