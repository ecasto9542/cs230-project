"""
Command-line interface for delivery route impact assessment.
Usage: python route_assessment_cli.py event_id1 event_id2 event_id3 ...
"""
import sys
from assess_delivery_impact import DeliveryRouteImpactAssessor

def main():
    if len(sys.argv) < 2:
        print("Usage: python route_assessment_cli.py <event_id1> [event_id2] [event_id3] ...")
        print("\nExample: python route_assessment_cli.py 545981 545982 535048")
        print("\nTo see available events, you can query the database:")
        print("  sqlite3 data/california_events.db 'SELECT event_id, event_type FROM events LIMIT 10'")
        sys.exit(1)
    
    event_ids = sys.argv[1:]
    print(f"Assessing delivery route with {len(event_ids)} events: {event_ids}")
    
    assessor = DeliveryRouteImpactAssessor()
    
    try:
        result = assessor.assess_route(event_ids, verbose=True)
        
        # Return exit code based on impact
        sys.exit(1 if result['route_impacting_delivery'] else 0)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        assessor.close()

if __name__ == '__main__':
    main()
