"""
Count total California weather events available across all CSV files.
"""
import pandas as pd
from pathlib import Path

def count_california_events():
    """Count all California events in the downloaded CSV files."""
    data_dir = Path('data/unzipped')
    csv_files = sorted(data_dir.glob('StormEvents_details-ftp_v1.0_d*.csv'))
    
    if not csv_files:
        print("No CSV files found in data/unzipped/")
        print("Please run: python src/get_data.py first")
        return
    
    total_count = 0
    year_counts = {}
    
    print(f"Counting California events across {len(csv_files)} CSV files...\n")
    
    for csv_file in csv_files:
        try:
            # Extract year from filename (format: StormEvents_details-ftp_v1.0_d2014_c20250520.csv)
            # Try to get year from the 'd' part
            parts = csv_file.name.split('_d')
            if len(parts) > 1:
                year = parts[1][:4]  # First 4 digits after '_d'
            else:
                year = "Unknown"
            
            print(f"Processing {csv_file.name}...", end=' ')
            
            # Read CSV and filter for California
            df = pd.read_csv(csv_file, low_memory=False)
            ca_events = df[df['STATE'].str.upper() == 'CALIFORNIA']
            count = len(ca_events)
            
            # Get year from the data itself (use the most common year in the file, or first row)
            if count > 0:
                year = int(ca_events['YEAR'].iloc[0]) if 'YEAR' in ca_events.columns else "Unknown"
            else:
                year = "Unknown"
            
            if year not in year_counts:
                year_counts[year] = 0
            year_counts[year] += count
            total_count += count
            
            print(f"{count} California events")
            
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nTotal California events available: {total_count:,}")
    print(f"\nEvents by year:")
    for year in sorted(year_counts.keys()):
        print(f"  {year}: {year_counts[year]:,} events")
    
    print(f"\nNote: Currently extracted {min(1000, total_count):,} events to the database.")
    print(f"You can extract all events by modifying extract_california_events.py")
    
    return total_count, year_counts

if __name__ == '__main__':
    count_california_events()
