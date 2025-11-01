"""
Create a new relational database that only holds delivery routes with impact scores.

This is a standalone database separate from route_events.db that contains:
- Route ID
- Counties list
- Weighted impact score
- Impacting delivery boolean (threshold >= 0.3)
"""
import sqlite3
import os

def create_routes_scores_database(source_db='data/route_events.db', 
                                  target_db='data/routes_scores.db'):
    """
    Create a new database with only route information and impact scores.
    
    Args:
        source_db: Source database (route_events.db) to read from
        target_db: Target database to create (routes_scores.db)
    """
    print("="*80)
    print("CREATING ROUTES SCORES DATABASE")
    print("="*80)
    
    # Remove existing database if it exists (fresh start)
    if os.path.exists(target_db):
        os.remove(target_db)
        print(f"Removed existing {target_db}")
    
    # Create new database
    os.makedirs(os.path.dirname(target_db), exist_ok=True)
    
    target_conn = sqlite3.connect(target_db)
    target_cursor = target_conn.cursor()
    
    # Create routes table with only necessary columns
    target_cursor.execute('''
        CREATE TABLE routes (
            route_id INTEGER PRIMARY KEY,
            counties TEXT NOT NULL,
            impact_score REAL NOT NULL,
            impacting_delivery INTEGER NOT NULL
        )
    ''')
    
    # Create indexes for efficient queries
    target_cursor.execute('''
        CREATE INDEX idx_impact_score ON routes(impact_score)
    ''')
    
    target_cursor.execute('''
        CREATE INDEX idx_impacting ON routes(impacting_delivery)
    ''')
    
    target_conn.commit()
    print(f"\nCreated new database schema at {target_db}")
    
    # Read routes from source database
    source_conn = sqlite3.connect(source_db)
    source_cursor = source_conn.cursor()
    
    source_cursor.execute('''
        SELECT route_id, counties, impact_score
        FROM routes
        WHERE impact_score IS NOT NULL
        ORDER BY route_id
    ''')
    
    routes = source_cursor.fetchall()
    print(f"\nFound {len(routes):,} routes with impact scores in source database")
    
    # Process and insert routes with new threshold (>= 0.3)
    NEW_THRESHOLD = 0.3
    processed_routes = []
    
    print(f"\nApplying new threshold: impacting_delivery = 1 if score >= {NEW_THRESHOLD}")
    print("Copying routes to new database...\n")
    
    for route_id, counties, impact_score in routes:
        # Determine impacting_delivery with new threshold
        impacting = 1 if impact_score >= NEW_THRESHOLD else 0
        
        processed_routes.append((route_id, counties, impact_score, impacting))
    
    # Insert in batches
    batch_size = 1000
    for i in range(0, len(processed_routes), batch_size):
        batch = processed_routes[i:i+batch_size]
        target_cursor.executemany('''
            INSERT INTO routes (route_id, counties, impact_score, impacting_delivery)
            VALUES (?, ?, ?, ?)
        ''', batch)
        target_conn.commit()
        
        if (i + batch_size) % 5000 == 0:
            print(f"  Copied {min(i + batch_size, len(processed_routes)):,} routes...")
    
    source_conn.close()
    target_conn.close()
    
    print(f"Copied {len(processed_routes):,} routes to {target_db}")
    
    # Verify the new database
    verify_database(target_db, NEW_THRESHOLD)

def verify_database(db_path='data/routes_scores.db', threshold=0.3):
    """Verify the routes scores database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Count routes
    cursor.execute('SELECT COUNT(*) FROM routes')
    total = cursor.fetchone()[0]
    
    # Count impacting routes
    cursor.execute('SELECT COUNT(*) FROM routes WHERE impacting_delivery = 1')
    impacting = cursor.fetchone()[0]
    
    # Score statistics
    cursor.execute('''
        SELECT 
            AVG(impact_score) as avg_score,
            MIN(impact_score) as min_score,
            MAX(impact_score) as max_score
        FROM routes
    ''')
    avg_score, min_score, max_score = cursor.fetchone()
    
    # Sample routes
    cursor.execute('''
        SELECT route_id, impact_score, impacting_delivery
        FROM routes
        ORDER BY impact_score DESC
        LIMIT 5
    ''')
    top_routes = cursor.fetchall()
    
    cursor.execute('''
        SELECT route_id, impact_score, impacting_delivery
        FROM routes
        WHERE impacting_delivery = 1
        ORDER BY impact_score DESC
        LIMIT 5
    ''')
    impacting_routes = cursor.fetchall()
    
    print("\n" + "="*80)
    print("DATABASE VERIFICATION")
    print("="*80)
    print(f"\nTotal routes: {total:,}")
    print(f"Routes impacting delivery (score >= {threshold}): {impacting:,} ({impacting/total*100:.1f}%)")
    print(f"Routes NOT impacting delivery: {total - impacting:,} ({(total-impacting)/total*100:.1f}%)")
    print(f"\nScore statistics:")
    print(f"  Average: {avg_score:.4f}")
    print(f"  Minimum: {min_score:.4f}")
    print(f"  Maximum: {max_score:.4f}")
    
    print(f"\nTop 5 routes by score:")
    for route_id, score, impacting in top_routes:
        impacting_str = "YES" if impacting else "NO"
        print(f"  Route {route_id}: Score {score:.4f}, Impacting: {impacting_str}")
    
    if impacting_routes:
        print(f"\nTop 5 IMPACTING routes (score >= {threshold}):")
        for route_id, score, impacting in impacting_routes:
            print(f"  Route {route_id}: Score {score:.4f}")
    else:
        print(f"\nNo routes with score >= {threshold}")
    
    # Verify schema
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='routes'")
    schema = cursor.fetchone()[0]
    print(f"\nDatabase schema:")
    print(f"  {schema}")
    
    conn.close()

if __name__ == '__main__':
    create_routes_scores_database()
    
    print("\n" + "="*80)
    print("Done! Routes scores database created successfully.")
    print("="*80)
    print(f"\nThis database is separate from route_events.db and contains only:")
    print("  - route_id")
    print("  - counties")
    print("  - impact_score")
    print("  - impacting_delivery (threshold >= 0.3)")
