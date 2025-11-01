"""
Extract California weather events from CSV files and store in SQLite database.
"""
import os
import sqlite3
import pandas as pd
from pathlib import Path

def parse_damage_value(damage_str):
    """Parse damage strings like '1.5M', '500K', '0.00K' to float."""
    if pd.isna(damage_str) or damage_str == '' or damage_str == '0.00K':
        return 0.0
    
    damage_str = str(damage_str).strip().upper()
    if 'M' in damage_str:
        return float(damage_str.replace('M', '')) * 1_000_000
    elif 'K' in damage_str:
        return float(damage_str.replace('K', '')) * 1_000
    else:
        try:
            return float(damage_str)
        except:
            return 0.0

def create_database(db_path='data/california_events.db'):
    """Create SQLite database and events table."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create events table with all attributes
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS events (
            event_id TEXT PRIMARY KEY,
            begin_yearmonth INTEGER,
            begin_day INTEGER,
            begin_time INTEGER,
            end_yearmonth INTEGER,
            end_day INTEGER,
            end_time INTEGER,
            episode_id TEXT,
            state TEXT,
            state_fips INTEGER,
            year INTEGER,
            month_name TEXT,
            event_type TEXT,
            cz_type TEXT,
            cz_fips INTEGER,
            cz_name TEXT,
            wfo TEXT,
            begin_date_time TEXT,
            cz_timezone TEXT,
            end_date_time TEXT,
            injuries_direct INTEGER,
            injuries_indirect INTEGER,
            deaths_direct INTEGER,
            deaths_indirect INTEGER,
            damage_property REAL,
            damage_crops REAL,
            source TEXT,
            magnitude REAL,
            magnitude_type TEXT,
            flood_cause TEXT,
            category INTEGER,
            tor_f_scale TEXT,
            tor_length REAL,
            tor_width REAL,
            begin_lat REAL,
            begin_lon REAL,
            end_lat REAL,
            end_lon REAL,
            episode_narrative TEXT,
            event_narrative TEXT,
            data_source TEXT
        )
    ''')
    
    conn.commit()
    return conn

def load_california_events(limit=None):
    """Load California events from CSV files and return as DataFrame.
    
    Args:
        limit: Maximum number of events to load. If None, loads all events.
    """
    data_dir = Path('data/unzipped')
    all_events = []
    
    # Get all CSV files
    csv_files = sorted(data_dir.glob('StormEvents_details-ftp_v1.0_d*.csv'))
    
    print(f"Found {len(csv_files)} CSV files to process...")
    
    for csv_file in csv_files:
        print(f"Processing {csv_file.name}...")
        try:
            df = pd.read_csv(csv_file, low_memory=False)
            
            # Filter for California
            ca_events = df[df['STATE'].str.upper() == 'CALIFORNIA'].copy()
            
            if len(ca_events) > 0:
                print(f"  Found {len(ca_events)} California events")
                all_events.append(ca_events)
            
            # Stop early if we have a limit and have enough events
            if limit is not None:
                total_so_far = sum(len(df) for df in all_events)
                if total_so_far >= limit:
                    print(f"  Reached limit of {limit} events, stopping...")
                    break
                
        except Exception as e:
            print(f"  Error processing {csv_file.name}: {e}")
            continue
    
    if not all_events:
        raise ValueError("No California events found in CSV files!")
    
    # Combine all events
    combined_df = pd.concat(all_events, ignore_index=True)
    
    # Apply limit if specified
    if limit is not None:
        combined_df = combined_df.head(limit)
    
    # Clean and convert data
    combined_df['INJURIES_DIRECT'] = pd.to_numeric(combined_df['INJURIES_DIRECT'], errors='coerce').fillna(0).astype(int)
    combined_df['INJURIES_INDIRECT'] = pd.to_numeric(combined_df['INJURIES_INDIRECT'], errors='coerce').fillna(0).astype(int)
    combined_df['DEATHS_DIRECT'] = pd.to_numeric(combined_df['DEATHS_DIRECT'], errors='coerce').fillna(0).astype(int)
    combined_df['DEATHS_INDIRECT'] = pd.to_numeric(combined_df['DEATHS_INDIRECT'], errors='coerce').fillna(0).astype(int)
    
    # Parse damage values
    combined_df['DAMAGE_PROPERTY'] = combined_df['DAMAGE_PROPERTY'].apply(parse_damage_value)
    combined_df['DAMAGE_CROPS'] = combined_df['DAMAGE_CROPS'].apply(parse_damage_value)
    
    # Convert coordinates
    for col in ['BEGIN_LAT', 'BEGIN_LON', 'END_LAT', 'END_LON']:
        combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
    
    # Convert magnitude
    combined_df['MAGNITUDE'] = pd.to_numeric(combined_df['MAGNITUDE'], errors='coerce')
    
    print(f"\nTotal California events extracted: {len(combined_df)}")
    return combined_df

def save_to_database(df, db_path='data/california_events.db'):
    """Save events DataFrame to SQLite database."""
    conn = sqlite3.connect(db_path)
    
    # Map column names (some might have spaces or special chars)
    column_mapping = {
        'EVENT_ID': 'event_id',
        'BEGIN_YEARMONTH': 'begin_yearmonth',
        'BEGIN_DAY': 'begin_day',
        'BEGIN_TIME': 'begin_time',
        'END_YEARMONTH': 'end_yearmonth',
        'END_DAY': 'end_day',
        'END_TIME': 'end_time',
        'EPISODE_ID': 'episode_id',
        'STATE': 'state',
        'STATE_FIPS': 'state_fips',
        'YEAR': 'year',
        'MONTH_NAME': 'month_name',
        'EVENT_TYPE': 'event_type',
        'CZ_TYPE': 'cz_type',
        'CZ_FIPS': 'cz_fips',
        'CZ_NAME': 'cz_name',
        'WFO': 'wfo',
        'BEGIN_DATE_TIME': 'begin_date_time',
        'CZ_TIMEZONE': 'cz_timezone',
        'END_DATE_TIME': 'end_date_time',
        'INJURIES_DIRECT': 'injuries_direct',
        'INJURIES_INDIRECT': 'injuries_indirect',
        'DEATHS_DIRECT': 'deaths_direct',
        'DEATHS_INDIRECT': 'deaths_indirect',
        'DAMAGE_PROPERTY': 'damage_property',
        'DAMAGE_CROPS': 'damage_crops',
        'SOURCE': 'source',
        'MAGNITUDE': 'magnitude',
        'MAGNITUDE_TYPE': 'magnitude_type',
        'FLOOD_CAUSE': 'flood_cause',
        'CATEGORY': 'category',
        'TOR_F_SCALE': 'tor_f_scale',
        'TOR_LENGTH': 'tor_length',
        'TOR_WIDTH': 'tor_width',
        'BEGIN_LAT': 'begin_lat',
        'BEGIN_LON': 'begin_lon',
        'END_LAT': 'end_lat',
        'END_LON': 'end_lon',
        'EPISODE_NARRATIVE': 'episode_narrative',
        'EVENT_NARRATIVE': 'event_narrative',
        'DATA_SOURCE': 'data_source'
    }
    
    # Create a new dataframe with renamed columns
    db_df = df.copy()
    db_df = db_df.rename(columns=column_mapping)
    
    # Select only columns that exist
    available_cols = [col for col in column_mapping.values() if col in db_df.columns]
    db_df = db_df[available_cols]
    
    # Insert data
    db_df.to_sql('events', conn, if_exists='replace', index=False)
    
    conn.close()
    print(f"Saved {len(db_df)} events to {db_path}")

if __name__ == '__main__':
    import sys
    
    # Check if limit argument provided
    limit = None
    if len(sys.argv) > 1:
        try:
            limit = int(sys.argv[1])
            print(f"Extracting up to {limit} California weather events...")
        except ValueError:
            print("Usage: python extract_california_events.py [limit]")
            print("  If no limit provided, extracts all events.")
            sys.exit(1)
    else:
        print("Extracting ALL California weather events...")
    
    df = load_california_events(limit=limit)
    save_to_database(df)
    
    count = len(df)
    print(f"\nDone! Database created at data/california_events.db with {count:,} events")
