"""
Helper functions for efficiently querying the route-events joined database.

Provides optimized query methods for common algorithm use cases.
"""
import sqlite3
from typing import List, Dict, Optional

class RouteEventsQuery:
    """Efficient query interface for route-events database."""
    
    def __init__(self, db_path='data/route_events.db'):
        """Initialize with database connection."""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
    
    def get_route_info(self, route_id: int) -> Optional[Dict]:
        """Get route information by route_id."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM routes WHERE route_id = ?', (route_id,))
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None
    
    def get_events_for_route(self, route_id: int, 
                             event_type: Optional[str] = None,
                             year: Optional[int] = None) -> List[Dict]:
        """
        Get all events associated with a route.
        
        Args:
            route_id: Route identifier
            event_type: Optional filter by event type
            year: Optional filter by year
        
        Returns:
            List of event dictionaries with county_sequence included
        """
        cursor = self.conn.cursor()
        
        query = '''
            SELECT e.*, re.county, re.county_sequence
            FROM route_events re
            JOIN events e ON re.event_id = e.event_id
            WHERE re.route_id = ?
        '''
        params = [route_id]
        
        if event_type:
            query += ' AND e.event_type = ?'
            params.append(event_type)
        
        if year:
            query += ' AND e.year = ?'
            params.append(year)
        
        query += ' ORDER BY re.county_sequence, e.begin_date_time'
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    
    def get_routes_for_event(self, event_id: str) -> List[Dict]:
        """
        Get all routes that include a specific event.
        
        Args:
            event_id: Event identifier
        
        Returns:
            List of route dictionaries
        """
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT DISTINCT r.*, re.county, re.county_sequence
            FROM routes r
            JOIN route_events re ON r.route_id = re.route_id
            WHERE re.event_id = ?
            ORDER BY r.route_id
        ''', (event_id,))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    
    def get_events_by_county_sequence(self, route_id: int, 
                                     sequence_start: int = None,
                                     sequence_end: int = None) -> List[Dict]:
        """
        Get events for a route filtered by county sequence position.
        Useful for analyzing impact at different stages of delivery.
        
        Args:
            route_id: Route identifier
            sequence_start: Starting sequence position (1-based)
            sequence_end: Ending sequence position (1-based)
        
        Returns:
            List of event dictionaries
        """
        cursor = self.conn.cursor()
        
        query = '''
            SELECT e.*, re.county, re.county_sequence
            FROM route_events re
            JOIN events e ON re.event_id = e.event_id
            WHERE re.route_id = ?
        '''
        params = [route_id]
        
        if sequence_start is not None:
            query += ' AND re.county_sequence >= ?'
            params.append(sequence_start)
        
        if sequence_end is not None:
            query += ' AND re.county_sequence <= ?'
            params.append(sequence_end)
        
        query += ' ORDER BY re.county_sequence, e.begin_date_time'
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    
    def get_route_statistics(self, route_id: int) -> Dict:
        """
        Get comprehensive statistics for a route.
        
        Returns:
            Dictionary with route stats including event counts by type, year, etc.
        """
        cursor = self.conn.cursor()
        
        stats = {}
        
        # Total events
        cursor.execute('''
            SELECT COUNT(*) FROM route_events WHERE route_id = ?
        ''', (route_id,))
        stats['total_events'] = cursor.fetchone()[0]
        
        # Events by type
        cursor.execute('''
            SELECT e.event_type, COUNT(*) as count
            FROM route_events re
            JOIN events e ON re.event_id = e.event_id
            WHERE re.route_id = ?
            GROUP BY e.event_type
            ORDER BY count DESC
        ''', (route_id,))
        stats['events_by_type'] = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Events by year
        cursor.execute('''
            SELECT e.year, COUNT(*) as count
            FROM route_events re
            JOIN events e ON re.event_id = e.event_id
            WHERE re.route_id = ?
            GROUP BY e.year
            ORDER BY e.year
        ''', (route_id,))
        stats['events_by_year'] = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Events by county (in route)
        cursor.execute('''
            SELECT re.county, re.county_sequence, COUNT(*) as count
            FROM route_events re
            WHERE re.route_id = ?
            GROUP BY re.county, re.county_sequence
            ORDER BY re.county_sequence
        ''', (route_id,))
        stats['events_by_county'] = [
            {'county': row[0], 'sequence': row[1], 'event_count': row[2]}
            for row in cursor.fetchall()
        ]
        
        # Total damage
        cursor.execute('''
            SELECT 
                SUM(e.damage_property + e.damage_crops) as total_damage,
                SUM(e.injuries_direct + e.injuries_indirect) as total_injuries,
                SUM(e.deaths_direct + e.deaths_indirect) as total_deaths
            FROM route_events re
            JOIN events e ON re.event_id = e.event_id
            WHERE re.route_id = ?
        ''', (route_id,))
        row = cursor.fetchone()
        stats['total_damage'] = row[0] or 0
        stats['total_injuries'] = row[1] or 0
        stats['total_deaths'] = row[2] or 0
        
        return stats
    
    def get_events_for_counties(self, counties: List[str]) -> List[Dict]:
        """
        Get all events for a list of counties.
        Useful for finding events affecting multiple routes.
        
        Args:
            counties: List of county names
        
        Returns:
            List of unique event dictionaries
        """
        cursor = self.conn.cursor()
        placeholders = ','.join(['?'] * len(counties))
        cursor.execute(f'''
            SELECT DISTINCT e.*
            FROM events e
            WHERE e.cz_name IN ({placeholders})
            ORDER BY e.begin_date_time
        ''', counties)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    
    def batch_get_routes_info(self, route_ids: List[int]) -> List[Dict]:
        """
        Efficiently get information for multiple routes.
        
        Args:
            route_ids: List of route IDs
        
        Returns:
            List of route dictionaries
        """
        cursor = self.conn.cursor()
        placeholders = ','.join(['?'] * len(route_ids))
        cursor.execute(f'''
            SELECT * FROM routes 
            WHERE route_id IN ({placeholders})
            ORDER BY route_id
        ''', route_ids)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    
    def close(self):
        """Close database connection."""
        self.conn.close()

# Example usage
if __name__ == '__main__':
    query = RouteEventsQuery()
    
    # Get route info
    route = query.get_route_info(1)
    print(f"Route 1: {route['counties'][:100]}...")
    
    # Get events for route
    events = query.get_events_for_route(1)
    print(f"\nEvents for Route 1 (showing first 5):")
    for event in events[:5]:
        print(f"  - {event['event_type']} in {event['county']} (sequence {event['county_sequence']})")
    
    # Get statistics
    stats = query.get_route_statistics(1)
    print(f"\nRoute 1 Statistics:")
    print(f"  Total events: {stats['total_events']}")
    print(f"  Total damage: ${stats['total_damage']:,.0f}")
    print(f"  Top event types: {list(stats['events_by_type'].items())[:3]}")
    
    query.close()
