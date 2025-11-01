"""
Create a database of delivery routes across California counties.

Generates 20,000 delivery routes, each with 3-10 counties in sequential order.
"""
import sqlite3
import random
import os

def get_california_counties(db_path='data/california_events.db'):
    """Get all unique California counties from the events database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT DISTINCT cz_name 
        FROM events 
        WHERE cz_name IS NOT NULL AND cz_name != ''
        ORDER BY cz_name
    ''')
    
    counties = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    print(f"Found {len(counties)} unique counties/zones in California")
    return counties

def create_routes_database(db_path='data/delivery_routes.db'):
    """Create the delivery routes database."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create routes table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS routes (
            route_id INTEGER PRIMARY KEY AUTOINCREMENT,
            county_count INTEGER,
            counties TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create index for faster queries
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_counties ON routes(counties)
    ''')
    
    conn.commit()
    conn.close()
    print(f"Created routes database at {db_path}")

def generate_routes(counties, num_routes=20000):
    """Generate random delivery routes."""
    routes = []
    
    print(f"Generating {num_routes:,} delivery routes...")
    
    for route_id in range(1, num_routes + 1):
        # Random number of counties between 3 and 10
        num_counties = random.randint(3, 10)
        
        # Randomly select counties (can repeat within a route, but we'll shuffle to make it sequential)
        # Actually, let's make sure counties are distinct within a route
        route_counties = random.sample(counties, min(num_counties, len(counties)))
        
        # Join counties as a comma-separated string
        counties_str = ','.join(route_counties)
        
        routes.append({
            'county_count': num_counties,
            'counties': counties_str
        })
        
        if route_id % 2000 == 0:
            print(f"  Generated {route_id:,} routes...")
    
    print(f"Generated {len(routes):,} routes")
    return routes

def save_routes_to_database(routes, db_path='data/delivery_routes.db'):
    """Save routes to the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Clear existing routes if any
    cursor.execute('DELETE FROM routes')
    
    # Insert routes in batches
    batch_size = 1000
    for i in range(0, len(routes), batch_size):
        batch = routes[i:i+batch_size]
        cursor.executemany('''
            INSERT INTO routes (county_count, counties)
            VALUES (?, ?)
        ''', [(r['county_count'], r['counties']) for r in batch])
        
        if (i + batch_size) % 5000 == 0:
            print(f"  Saved {min(i + batch_size, len(routes)):,} routes to database...")
        conn.commit()
    
    conn.close()
    print(f"Saved {len(routes):,} routes to {db_path}")

def verify_routes(db_path='data/delivery_routes.db'):
    """Verify the routes database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Count total routes
    cursor.execute('SELECT COUNT(*) FROM routes')
    total = cursor.fetchone()[0]
    
    # Count by county count
    cursor.execute('''
        SELECT county_count, COUNT(*) 
        FROM routes 
        GROUP BY county_count 
        ORDER BY county_count
    ''')
    distribution = cursor.fetchall()
    
    # Get sample routes
    cursor.execute('SELECT route_id, county_count, counties FROM routes LIMIT 5')
    samples = cursor.fetchall()
    
    print("\n" + "="*80)
    print("ROUTES DATABASE VERIFICATION")
    print("="*80)
    print(f"\nTotal routes: {total:,}")
    print("\nDistribution by number of counties:")
    for county_count, count in distribution:
        print(f"  {county_count} counties: {count:,} routes ({count/total*100:.1f}%)")
    
    print("\nSample routes:")
    for route_id, county_count, counties in samples:
        counties_list = counties.split(',')
        print(f"\nRoute ID: {route_id}")
        print(f"  Counties ({county_count}): {', '.join(counties_list)}")
    
    conn.close()

if __name__ == '__main__':
    print("Creating delivery routes database...")
    print("="*80)
    
    # Get California counties
    counties = get_california_counties()
    
    # Create database
    create_routes_database()
    
    # Generate routes
    routes = generate_routes(counties, num_routes=20000)
    
    # Save to database
    save_routes_to_database(routes)
    
    # Verify
    verify_routes()
    
    print("\n" + "="*80)
    print("Done! Delivery routes database created successfully.")
    print("="*80)
