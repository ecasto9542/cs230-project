"""
Remove impact scores and impacting_delivery columns from route_events.db.

This ensures these columns exist ONLY in routes_scores.db.
"""
import sqlite3

def remove_scores_from_route_events(db_path='data/route_events.db'):
    """Remove impact_score and impacting_delivery columns from routes table."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("Removing impact_score and impacting_delivery from route_events.db...")
    print("These columns will exist ONLY in routes_scores.db\n")
    
    # SQLite doesn't support DROP COLUMN directly, so we need to recreate the table
    # First, check if columns exist
    cursor.execute("PRAGMA table_info(routes)")
    columns = cursor.fetchall()
    column_names = [col[1] for col in columns]
    
    has_impact_score = 'impact_score' in column_names
    has_impacting = 'impacting_delivery' in column_names
    has_event_count = 'event_count' in column_names
    
    if not (has_impact_score or has_impacting or has_event_count):
        print("Columns not found - they may have already been removed.")
        conn.close()
        return
    
    print("Found columns to remove:")
    if has_impact_score:
        print("  - impact_score")
    if has_impacting:
        print("  - impacting_delivery")
    if has_event_count:
        print("  - event_count")
    
    # Get all column names except the ones to remove
    columns_to_keep = [
        col[1] for col in columns 
        if col[1] not in ['impact_score', 'impacting_delivery', 'event_count']
    ]
    
    # Create new table without those columns
    print("\nCreating new routes table without score columns...")
    cursor.execute('''
        CREATE TABLE routes_new (
            route_id INTEGER PRIMARY KEY,
            county_count INTEGER NOT NULL,
            counties TEXT NOT NULL,
            created_at TIMESTAMP
        )
    ''')
    
    # Copy data (excluding score columns)
    columns_str = ', '.join(columns_to_keep)
    cursor.execute(f'''
        INSERT INTO routes_new ({columns_str})
        SELECT {columns_str}
        FROM routes
    ''')
    
    # Drop old table and rename new one
    cursor.execute('DROP TABLE routes')
    cursor.execute('ALTER TABLE routes_new RENAME TO routes')
    
    # Recreate indexes
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_counties ON routes(counties)')
    
    conn.commit()
    
    # Verify
    cursor.execute("PRAGMA table_info(routes)")
    remaining_columns = [col[1] for col in cursor.fetchall()]
    
    print("\nRemaining columns in routes table:")
    for col in remaining_columns:
        print(f"  - {col}")
    
    if 'impact_score' not in remaining_columns and 'impacting_delivery' not in remaining_columns:
        print("\n✓ Successfully removed score columns from route_events.db")
        print("  Impact scores now exist ONLY in routes_scores.db")
    else:
        print("\n✗ Error: Columns still present")
    
    conn.close()

if __name__ == '__main__':
    remove_scores_from_route_events()
