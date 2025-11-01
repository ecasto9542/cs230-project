"""
Batch process all delivery routes and assign impact scores.

Uses the updated weighting algorithm with temporal weighting to assess
all 20,000 routes in the route_events database.
"""
import sqlite3
from assess_delivery_impact import DeliveryRouteImpactAssessor
from query_route_events import RouteEventsQuery
from typing import List, Dict
import time

class RouteEventsImpactAssessor(DeliveryRouteImpactAssessor):
    """Extended assessor that works with route_events database."""
    
    def __init__(self, db_path='data/route_events.db'):
        """Initialize with route_events database."""
        # Override parent init to use route_events.db
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        
        # Also initialize query helper
        self.query = RouteEventsQuery(db_path)
    
    def assess_route_from_db(self, route_id: int) -> Dict:
        """
        Assess a route directly from the route_events database.
        
        Args:
            route_id: Route identifier
        
        Returns:
            Dictionary with route assessment results
        """
        # Get all events for this route
        events = self.query.get_events_for_route(route_id)
        
        if not events:
            # Route has no associated events
            return {
                'route_id': route_id,
                'event_count': 0,
                'overall_route_score': 0.0,
                'route_impacting_delivery': False,
                'impacting_count': 0
            }
        
        # Assess each event
        event_impacts = []
        for event in events:
            impact = self.assess_event(event)
            event_impacts.append(impact)
        
        # Calculate overall route score
        # Weighted average of individual event scores, weighted by severity
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for impact in event_impacts:
            # Weight by event severity + temporal recency
            weight = impact.severity_score + impact.weighted_attributes.get('temporal_score', 0.5) + 0.1
            total_weighted_score += impact.total_score * weight
            total_weight += weight
        
        overall_route_score = total_weighted_score / total_weight if total_weight > 0 else 0.0
        overall_route_score = min(overall_route_score, 1.0)  # Cap at 1.0
        
        # Count impacting events
        impacting_count = sum(1 for impact in event_impacts if impact.impacting_delivery)
        route_impacting = overall_route_score >= self.IMPACT_THRESHOLD
        
        return {
            'route_id': route_id,
            'event_count': len(event_impacts),
            'overall_route_score': overall_route_score,
            'route_impacting_delivery': route_impacting,
            'impacting_count': impacting_count
        }
    
    def batch_assess_routes(self, route_ids: List[int], batch_size: int = 1000) -> List[Dict]:
        """
        Assess multiple routes in batches.
        
        Args:
            route_ids: List of route IDs to assess
            batch_size: Number of routes to process before committing to DB
        
        Returns:
            List of assessment dictionaries
        """
        results = []
        total = len(route_ids)
        
        print(f"Assessing {total:,} routes...")
        
        for i, route_id in enumerate(route_ids, 1):
            try:
                result = self.assess_route_from_db(route_id)
                results.append(result)
                
                if i % batch_size == 0:
                    print(f"  Processed {i:,}/{total:,} routes ({i/total*100:.1f}%)")
            except Exception as e:
                print(f"  Error assessing route {route_id}: {e}")
                results.append({
                    'route_id': route_id,
                    'event_count': 0,
                    'overall_route_score': 0.0,
                    'route_impacting_delivery': False,
                    'impacting_count': 0,
                    'error': str(e)
                })
        
        return results
    
    def save_scores_to_database(self, assessments: List[Dict], db_path='data/route_events.db'):
        """
        Save route impact scores to the database.
        
        Adds/updates scores in the routes table.
        """
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Add score column if it doesn't exist
        try:
            cursor.execute('ALTER TABLE routes ADD COLUMN impact_score REAL')
            cursor.execute('ALTER TABLE routes ADD COLUMN impacting_delivery INTEGER')
            cursor.execute('ALTER TABLE routes ADD COLUMN event_count INTEGER')
            conn.commit()
            print("Added score columns to routes table")
        except sqlite3.OperationalError:
            # Columns already exist
            pass
        
        # Update routes with scores
        cursor.executemany('''
            UPDATE routes 
            SET impact_score = ?,
                impacting_delivery = ?,
                event_count = ?
            WHERE route_id = ?
        ''', [
            (
                a['overall_route_score'],
                1 if a['route_impacting_delivery'] else 0,
                a['event_count'],
                a['route_id']
            )
            for a in assessments
        ])
        
        conn.commit()
        
        # Create index for fast score queries
        try:
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_impact_score ON routes(impact_score)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_impacting ON routes(impacting_delivery)')
            conn.commit()
        except:
            pass
        
        conn.close()
        print(f"Saved scores for {len(assessments):,} routes to database")

def assess_all_routes(db_path='data/route_events.db', limit=None):
    """
    Assess all routes in the database and save scores.
    
    Args:
        db_path: Path to route_events database
        limit: Optional limit on number of routes to process (for testing)
    """
    print("="*80)
    print("BATCH ROUTE IMPACT ASSESSMENT")
    print("="*80)
    
    start_time = time.time()
    
    # Initialize assessor
    assessor = RouteEventsImpactAssessor(db_path)
    
    # Get all route IDs
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    if limit:
        cursor.execute('SELECT route_id FROM routes ORDER BY route_id LIMIT ?', (limit,))
    else:
        cursor.execute('SELECT route_id FROM routes ORDER BY route_id')
    
    route_ids = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    total_routes = len(route_ids)
    print(f"\nFound {total_routes:,} routes to assess")
    print(f"Using temporal weighting: recent events (2025) = 1.0, old events (2014) = 0.3")
    print("\nStarting assessment...\n")
    
    # Assess all routes in batches
    assessments = assessor.batch_assess_routes(route_ids, batch_size=500)
    
    # Save to database
    assessor.save_scores_to_database(assessments, db_path)
    
    # Calculate statistics
    impacting_routes = sum(1 for a in assessments if a.get('route_impacting_delivery', False))
    avg_score = sum(a['overall_route_score'] for a in assessments) / len(assessments)
    avg_events = sum(a['event_count'] for a in assessments) / len(assessments)
    
    # Score distribution
    score_ranges = {
        '0.0-0.2': 0,
        '0.2-0.4': 0,
        '0.4-0.6': 0,
        '0.6-0.8': 0,
        '0.8-1.0': 0
    }
    
    for a in assessments:
        score = a['overall_route_score']
        if score < 0.2:
            score_ranges['0.0-0.2'] += 1
        elif score < 0.4:
            score_ranges['0.2-0.4'] += 1
        elif score < 0.6:
            score_ranges['0.4-0.6'] += 1
        elif score < 0.8:
            score_ranges['0.6-0.8'] += 1
        else:
            score_ranges['0.8-1.0'] += 1
    
    elapsed_time = time.time() - start_time
    
    # Print results
    print("\n" + "="*80)
    print("ASSESSMENT RESULTS")
    print("="*80)
    print(f"\nTotal routes assessed: {total_routes:,}")
    print(f"Routes impacting delivery: {impacting_routes:,} ({impacting_routes/total_routes*100:.1f}%)")
    print(f"Average impact score: {avg_score:.4f}")
    print(f"Average events per route: {avg_events:.1f}")
    print(f"\nScore distribution:")
    for range_name, count in score_ranges.items():
        print(f"  {range_name}: {count:,} routes ({count/total_routes*100:.1f}%)")
    
    print(f"\nProcessing time: {elapsed_time:.1f} seconds")
    print(f"Average time per route: {elapsed_time/total_routes*1000:.1f} ms")
    print("="*80)
    
    assessor.close()
    assessor.query.close()

if __name__ == '__main__':
    import sys
    
    # Allow limiting for testing
    limit = None
    if len(sys.argv) > 1:
        try:
            limit = int(sys.argv[1])
            print(f"LIMITED MODE: Processing only first {limit} routes for testing")
        except ValueError:
            print("Usage: python assess_all_routes.py [limit]")
            sys.exit(1)
    
    assess_all_routes(limit=limit)
