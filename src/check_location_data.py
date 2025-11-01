"""Check what location information is available in the dataset."""
import sqlite3

conn = sqlite3.connect('data/california_events.db')
cursor = conn.cursor()

# Count events with coordinates
cursor.execute('SELECT COUNT(*) FROM events WHERE begin_lat IS NOT NULL AND begin_lon IS NOT NULL')
events_with_coords = cursor.fetchone()[0]

cursor.execute('SELECT COUNT(*) FROM events')
total_events = cursor.fetchone()[0]

print(f"Total events: {total_events:,}")
print(f"Events with latitude/longitude coordinates: {events_with_coords:,} ({events_with_coords/total_events*100:.1f}%)")

# Check for county information
cursor.execute('SELECT COUNT(*) FROM events WHERE cz_name IS NOT NULL AND cz_name != ""')
events_with_county = cursor.fetchone()[0]

print(f"Events with county (cz_name): {events_with_county:,}")

# Get sample events with different location data types
print("\n" + "="*80)
print("Sample events with LATITUDE/LONGITUDE coordinates:")
print("="*80)
cursor.execute('''
    SELECT event_id, event_type, cz_name, begin_lat, begin_lon, 
           end_lat, end_lon
    FROM events 
    WHERE begin_lat IS NOT NULL 
    LIMIT 5
''')
rows = cursor.fetchall()
for row in rows:
    print(f"\nEvent ID: {row[0]}")
    print(f"  Type: {row[1]}")
    print(f"  County: {row[2]}")
    print(f"  Begin: Lat {row[3]}, Lon {row[4]}")
    if row[5] and row[6]:
        print(f"  End: Lat {row[5]}, Lon {row[6]}")

print("\n" + "="*80)
print("Sample events with ONLY COUNTY information (no coordinates):")
print("="*80)
cursor.execute('''
    SELECT event_id, event_type, cz_name, begin_date_time, episode_narrative
    FROM events 
    WHERE begin_lat IS NULL
    LIMIT 5
''')
rows = cursor.fetchall()
for row in rows:
    narrative_preview = row[4][:100] + "..." if row[4] and len(str(row[4])) > 100 else (row[4] or "No narrative")
    print(f"\nEvent ID: {row[0]}")
    print(f"  Type: {row[1]}")
    print(f"  County: {row[2]}")
    print(f"  Date: {row[3]}")
    print(f"  Narrative preview: {narrative_preview}")

conn.close()
