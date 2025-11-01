"""
List events from the California events database for easy reference.
"""
import sqlite3
import sys

def list_events(limit=20, filter_severe=False):
    """List events from the database."""
    db_path = 'data/california_events.db'
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    if filter_severe:
        query = """
            SELECT 
                event_id, 
                event_type, 
                year,
                month_name,
                cz_name as county,
                (injuries_direct + injuries_indirect) as total_injuries,
                (deaths_direct + deaths_indirect) as total_deaths,
                (damage_property + damage_crops) as total_damage
            FROM events 
            WHERE (injuries_direct + injuries_indirect > 0 
                   OR deaths_direct + deaths_indirect > 0 
                   OR damage_property + damage_crops > 50000)
            ORDER BY total_damage DESC, total_deaths DESC, total_injuries DESC
            LIMIT ?
        """
    else:
        query = """
            SELECT 
                event_id, 
                event_type, 
                year,
                month_name,
                cz_name as county,
                (injuries_direct + injuries_indirect) as total_injuries,
                (deaths_direct + deaths_indirect) as total_deaths,
                (damage_property + damage_crops) as total_damage
            FROM events 
            LIMIT ?
        """
    
    cursor.execute(query, (limit,))
    rows = cursor.fetchall()
    
    print(f"\n{'Event ID':<12} {'Type':<25} {'Year':<6} {'County':<20} {'Injuries':<10} {'Deaths':<8} {'Damage ($)':<15}")
    print("-" * 120)
    
    for row in rows:
        damage = row['total_damage']
        if damage >= 1_000_000:
            damage_str = f"${damage/1_000_000:.1f}M"
        elif damage >= 1_000:
            damage_str = f"${damage/1_000:.1f}K"
        else:
            damage_str = f"${damage:.0f}"
        
        print(f"{row['event_id']:<12} {row['event_type'][:24]:<25} {row['year']:<6} {str(row['county'])[:19]:<20} {row['total_injuries']:<10} {row['total_deaths']:<8} {damage_str:<15}")
    
    print(f"\nTotal events shown: {len(rows)}")
    print("\nTo assess a route with specific events:")
    print("  python src/route_assessment_cli.py <event_id1> <event_id2> <event_id3>")
    
    conn.close()

if __name__ == '__main__':
    filter_severe = '--severe' in sys.argv
    limit = 20
    
    if '--limit' in sys.argv:
        idx = sys.argv.index('--limit')
        if idx + 1 < len(sys.argv):
            limit = int(sys.argv[idx + 1])
    
    list_events(limit=limit, filter_severe=filter_severe)
