#!/usr/bin/env python3
"""
Debug script to understand FITS file structure and loading issues
"""

import astropy.io.fits as fits
import numpy as np

def debug_fits_file():
    """Debug the FITS file structure in detail"""
    
    try:
        with fits.open('data/COSMOSWeb_mastercatalog_v1_cigale.fits') as hdul:
            print("FITS file info:")
            hdul.info()
            
            print(f"\nPrimary HDU:")
            print(f"Shape: {hdul[0].data}")
            print(f"Header keys: {list(hdul[0].header.keys())}")
            
            print(f"\nSecond HDU (data table):")
            data_hdu = hdul[1]
            print(f"Type: {type(data_hdu)}")
            print(f"Data shape: {data_hdu.data.shape if data_hdu.data is not None else 'None'}")
            print(f"Number of columns: {len(data_hdu.columns)}")
            print(f"Number of rows: {len(data_hdu.data) if data_hdu.data is not None else 'None'}")
            
            print(f"\nColumn details:")
            for i, col in enumerate(data_hdu.columns):
                print(f"{i:2d}: {col.name:<25} {col.format:<8} {col.unit if col.unit else ''}")
                
            print(f"\nSample data from first few columns:")
            try:
                data = data_hdu.data
                if data is not None and len(data) > 0:
                    print(f"First 5 rows:")
                    for i in range(min(5, len(data))):
                        row_data = []
                        for j in range(min(5, len(data_hdu.columns))):
                            col_name = data_hdu.columns[j].name
                            value = data[col_name][i]
                            row_data.append(f"{value}")
                        print(f"Row {i}: {', '.join(row_data)}")
                        
                    # Check specific columns we need
                    print(f"\nChecking key columns:")
                    for col_name in ['mass', 'age_form', 'sfr_inst']:
                        if col_name in data.dtype.names:
                            values = data[col_name]
                            finite_values = values[np.isfinite(values)]
                            print(f"{col_name}: {len(finite_values)} finite values out of {len(values)}")
                            if len(finite_values) > 0:
                                print(f"  Range: {np.min(finite_values):.3e} to {np.max(finite_values):.3e}")
                                print(f"  Median: {np.median(finite_values):.3e}")
                else:
                    print("No data found in HDU")
                    
            except Exception as e:
                print(f"Error reading data: {e}")
                
    except Exception as e:
        print(f"Error opening FITS file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_fits_file() 