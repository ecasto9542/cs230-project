"""
Create a joined database that links delivery routes with weather events.

Architecture:
- Normalized structure with routes table and route_events junction table
- Proper indexes for efficient algorithm queries
- Allows fast lookups of: events by route, routes by event, events by county sequence
"""
import sqlite3
import os
from typing import List, Tuple

def create_joined_database(db_path='data/route_events.db'):
    """Create the joined database with optimized schema."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Routes table - copy route structure
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS routes (
            route_id INTEGER PRIMARY KEY,
            county_count INTEGER NOT NULL,
            counties TEXT NOT NULL,
            created_at TIMESTAMP
        )
    ''')
    
    # Events table - copy event structure (denormalized for fast access)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS events (
            event_id TEXT PRIMARY KEY,
            cz_name TEXT NOT NULL,
            event_type TEXT,
            year INTEGER,
            month_name TEXT,
            begin_date_time TEXT,
            end_date_time TEXT,
            injuries_direct INTEGER,
            injuries_indirect INTEGER,
            deaths_direct INTEGER,
            deaths_indirect INTEGER,
            damage_property REAL,
            damage_crops REAL,
            magnitude REAL,
            magnitude_type TEXT,
            tor_f_scale TEXT,
            begin_lat REAL,
            begin_lon REAL,
            end_lat REAL,
            end_lon REAL,
            episode_narrative TEXT,
            event_narrative TEXT
        )
    ''')
    
    # Junction table: route_events
    # Links routes to events, also stores which county matched and its sequence
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS route_events (
            route_id INTEGER NOT NULL,
            event_id TEXT NOT NULL,
            county TEXT NOT NULL,
            county_sequence INTEGER NOT NULL,
            PRIMARY KEY (route_id, event_id, county),
            FOREIGN KEY (route_id) REFERENCES routes(route_id),
            FOREIGN KEY (event_id) REFERENCES events(event_id)
        )
    ''')
    
    # Create indexes for efficient algorithm queries
    print("Creating indexes for efficient queries...")
    
    # Index for: Get all events for a specific route
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_route_events_route 
        ON route_events(route_id)
    ''')
    
    # Index for: Get all routes affected by a specific event
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_route_events_event 
        ON route_events(event_id)
    ''')
    
    # Index for: Get events by county in route (for sequential processing)
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_route_events_county_seq 
        ON route_events(route_id, county_sequence)
    ''')
    
    # Index for: Get events by event type (for filtering)
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_events_type 
        ON events(event_type)
    ''')
    
    # Index for: Get events by county
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_events_county 
        ON events(cz_name)
    ''')
    
    # Index for: Get events by year (for time-based analysis)
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_events_year 
        ON events(year)
    ''')
    
    # Index for: Composite index for route queries
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_route_events_composite 
        ON route_events(route_id, event_id)
    ''')
    
    conn.commit()
    conn.close()
    print(f"Created joined database schema at {db_path}")

def copy_routes(source_db='data/delivery_routes.db', target_db='data/route_events.db'):
    """Copy routes from delivery_routes.db to route_events.db."""
    source_conn = sqlite3.connect(source_db)
    target_conn = sqlite3.connect(target_db)
    
    source_cursor = source_conn.cursor()
    target_cursor = target_conn.cursor()
    
    print("Copying routes...")
    source_cursor.execute('SELECT route_id, county_count, counties, created_at FROM routes')
    routes = source_cursor.fetchall()
    
    target_cursor.executemany('''
        INSERT OR REPLACE INTO routes (route_id, county_count, counties, created_at)
        VALUES (?, ?, ?, ?)
    ''', routes)
    
    target_conn.commit()
    print(f"Copied {len(routes):,} routes")
    
    source_conn.close()
    target_conn.close()

def copy_events(source_db='data/california_events.db', target_db='data/route_events.db'):
    """Copy events from california_events.db to route_events.db (only needed fields)."""
    source_conn = sqlite3.connect(source_db)
    target_conn = sqlite3.connect(target_db)
    
    source_cursor = source_conn.cursor()
    target_cursor = target_conn.cursor()
    
    print("Copying events...")
    source_cursor.execute('''
        SELECT 
            event_id, cz_name, event_type, year, month_name,
            begin_date_time, end_date_time,
            injuries_direct, injuries_indirect, deaths_direct, deaths_indirect,
            damage_property, damage_crops,
            magnitude, magnitude_type, tor_f_scale,
            begin_lat, begin_lon, end_lat, end_lon,
            episode_narrative, event_narrative
        FROM events
    ''')
    
    events = source_cursor.fetchall()
    
    batch_size = 1000
    for i in range(0, len(events), batch_size):
        batch = events[i:i+batch_size]
        target_cursor.executemany('''
            INSERT OR REPLACE INTO events (
                event_id, cz_name, event_type, year, month_name,
                begin_date_time, end_date_time,
                injuries_direct, injuries_indirect, deaths_direct, deaths_indirect,
                damage_property, damage_crops,
                magnitude, magnitude_type, tor_f_scale,
                begin_lat, begin_lon, end_lat, end_lon,
                episode_narrative, event_narrative
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', batch)
        target_conn.commit()
        
        if (i + batch_size) % 5000 == 0:
            print(f"  Copied {min(i + batch_size, len(events)):,} events...")
    
    print(f"Copied {len(events):,} events")
    
    source_conn.close()
    target_conn.close()

def create_route_event_links(target_db='data/route_events.db'):
    """Create links between routes and events based on county matching."""
    conn = sqlite3.connect(target_db)
    cursor = conn.cursor()
    
    print("Creating route-event links...")
    print("  This may take a few minutes for 20,000 routes...")
    
    # Get all routes
    cursor.execute('SELECT route_id, counties FROM routes')
    routes = cursor.fetchall()
    
    total_links = 0
    batch_links = []
    batch_size = 5000
    
    for route_idx, (route_id, counties_str) in enumerate(routes):
        # Parse counties
        route_counties = [c.strip() for c in counties_str.split(',')]
        
        # For each county in route, find matching events
        for seq, county in enumerate(route_counties, start=1):
            cursor.execute('''
                SELECT event_id FROM events WHERE cz_name = ?
            ''', (county,))
            matching_events = cursor.fetchall()
            
            # Create links
            for (event_id,) in matching_events:
                batch_links.append((route_id, event_id, county, seq))
                total_links += 1
        
        # Insert in batches
        if len(batch_links) >= batch_size:
            cursor.executemany('''
                INSERT OR IGNORE INTO route_events (route_id, event_id, county, county_sequence)
                VALUES (?, ?, ?, ?)
            ''', batch_links)
            conn.commit()
            batch_links = []
        
        # Progress update
        if (route_idx + 1) % 1000 == 0:
            print(f"  Processed {route_idx + 1:,} routes, created {total_links:,} links so far...")
    
    # Insert remaining links
    if batch_links:
        cursor.executemany('''
            INSERT OR IGNORE INTO route_events (route_id, event_id, county, county_sequence)
            VALUES (?, ?, ?, ?)
        ''', batch_links)
        conn.commit()
    
    print(f"Created {total_links:,} route-event links")
    
    conn.close()

def verify_database(db_path='data/route_events.db'):
    """Verify the joined database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('SELECT COUNT(*) FROM routes')
    route_count = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM events')
    event_count = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM route_events')
    link_count = cursor.fetchone()[0]
    
    # Get sample route with events
    cursor.execute('''
        SELECT r.route_id, r.county_count, r.counties,
               COUNT(re.event_id) as event_count
        FROM routes r
        LEFT JOIN route_events re ON r.route_id = re.route_id
        GROUP BY r.route_id
        ORDER BY event_count DESC
        LIMIT 5
    ''')
    top_routes = cursor.fetchall()
    
    # Get average events per route
    cursor.execute('''
        SELECT AVG(event_count) FROM (
            SELECT COUNT(event_id) as event_count
            FROM route_events
            GROUP BY route_id
        )
    ''')
    avg_events = cursor.fetchone()[0]
    
    print("\n" + "="*80)
    print("ROUTE-EVENTS DATABASE VERIFICATION")
    print("="*80)
    print(f"\nRoutes: {route_count:,}")
    print(f"Events: {event_count:,}")
    print(f"Route-Event Links: {link_count:,}")
    print(f"Average events per route: {avg_events:.1f}")
    
    print("\nTop 5 routes by event count:")
    for route_id, county_count, counties, event_count in top_routes:
        counties_list = counties.split(',')[:3]
        counties_display = ', '.join(counties_list)
        if len(counties.split(',')) > 3:
            counties_display += "..."
        print(f"\n  Route {route_id}: {event_count} events")
        print(f"    Counties ({county_count}): {counties_display}")
    
    conn.close()

if __name__ == '__main__':
    print("Creating joined route-events database...")
    print("="*80)
    
    # Create database schema
    create_joined_database()
    
    # Copy routes
    copy_routes()
    
    # Copy events
    copy_events()
    
    # Create links
    create_route_event_links()
    
    # Verify
    verify_database()
    
    print("\n" + "="*80)
    print("Done! Joined database created successfully.")
    print("="*80)
