import pandas as pd

print('='*70)
print('VERIFICATION: All 3 CSV files have IDENTICAL columns')
print('='*70)

# Load just the headers
kepler_df = pd.read_csv(r'c:\Exoplanet-AI-Hunter\data\processed\kepler_standardized.csv', nrows=0)
k2_df = pd.read_csv(r'c:\Exoplanet-AI-Hunter\data\processed\k2_standardized.csv', nrows=0)
tess_df = pd.read_csv(r'c:\Exoplanet-AI-Hunter\data\processed\tess_standardized.csv', nrows=0)

kepler_cols = list(kepler_df.columns)
k2_cols = list(k2_df.columns)
tess_cols = list(tess_df.columns)

print('\nKEPLER columns:', kepler_cols)
print('\nK2 columns:', k2_cols)
print('\nTESS columns:', tess_cols)

# Check if all are identical
all_same = (kepler_cols == k2_cols == tess_cols)

print('\n' + '='*70)
print(f'All columns identical? {all_same}')
print('='*70)

if all_same:
    print('\n✨ SUCCESS! All 3 datasets have the same standardized structure!')
    print(f'\nTotal columns: {len(kepler_cols)}')
    print('\nColumn list:')
    for i, col in enumerate(kepler_cols, 1):
        print(f'  {i:2d}. {col}')
