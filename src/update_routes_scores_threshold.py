"""
Update the impacting_delivery boolean in routes_scores.db with new threshold (0.25).
"""
import sqlite3

def update_threshold(db_path='data/routes_scores.db', new_threshold=0.25):
    """Update impacting_delivery boolean with new threshold."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("="*80)
    print(f"UPDATING THRESHOLD TO {new_threshold}")
    print("="*80)
    
    # Count current impacting routes
    cursor.execute('SELECT COUNT(*) FROM routes WHERE impacting_delivery = 1')
    old_impacting = cursor.fetchone()[0]
    
    # Count routes that will be impacting with new threshold
    cursor.execute(f'SELECT COUNT(*) FROM routes WHERE impact_score >= {new_threshold}')
    new_impacting = cursor.fetchone()[0]
    
    print(f"\nCurrent threshold: 0.3")
    print(f"  Current impacting routes: {old_impacting}")
    print(f"\nNew threshold: {new_threshold}")
    print(f"  Routes that will be impacting: {new_impacting}")
    print(f"  Change: +{new_impacting - old_impacting} routes")
    
    # Update all routes with new threshold
    cursor.execute(f'''
        UPDATE routes
        SET impacting_delivery = CASE 
            WHEN impact_score >= {new_threshold} THEN 1
            ELSE 0
        END
    ''')
    
    conn.commit()
    
    # Verify
    cursor.execute('SELECT COUNT(*) FROM routes WHERE impacting_delivery = 1')
    updated_impacting = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM routes WHERE impact_score >= ? AND impact_score < ?', 
                   (new_threshold, 0.3))
    newly_impacting = cursor.fetchone()[0]
    
    print(f"\n✓ Update complete!")
    print(f"  Total impacting routes: {updated_impacting:,}")
    print(f"  Routes in range [{new_threshold}, 0.3): {newly_impacting:,}")
    
    # Get score distribution
    cursor.execute('''
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN impact_score >= 0.25 AND impact_score < 0.3 THEN 1 ELSE 0 END) as in_range_25_30,
            SUM(CASE WHEN impact_score >= 0.3 THEN 1 ELSE 0 END) as above_30
        FROM routes
    ''')
    total, in_range, above = cursor.fetchone()
    
    print(f"\nScore Distribution:")
    print(f"  Below {new_threshold}: {total - updated_impacting:,} routes ({100*(total-updated_impacting)/total:.1f}%)")
    print(f"  {new_threshold} - 0.30: {in_range:,} routes ({100*in_range/total:.1f}%)")
    print(f"  Above 0.30: {above:,} routes ({100*above/total:.1f}%)")
    
    conn.close()
    print("="*80)

if __name__ == '__main__':
    update_threshold(new_threshold=0.25)
