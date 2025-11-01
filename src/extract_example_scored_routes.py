"""
Extract example scored routes from routes_scores.db and save as JSON.
Includes both routes above and below the 0.25 threshold.
"""
import sqlite3
import json
import os

def extract_example_scored_routes(num_above=10, num_below=10, threshold=0.25):
    """Extract example routes above and below threshold."""
    conn = sqlite3.connect('data/routes_scores.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    examples_dir = 'examples/event-examples'
    os.makedirs(examples_dir, exist_ok=True)
    
    # Get routes above threshold
    cursor.execute('''
        SELECT route_id, counties, impact_score, impacting_delivery
        FROM routes
        WHERE impact_score >= ?
        ORDER BY impact_score DESC
        LIMIT ?
    ''', (threshold, num_above))
    
    routes_above = []
    for row in cursor.fetchall():
        route_dict = dict(row)
        route_dict['counties_list'] = route_dict['counties'].split(',')
        route_dict['threshold'] = threshold
        route_dict['above_threshold'] = True
        routes_above.append(route_dict)
    
    # Get routes below threshold
    cursor.execute('''
        SELECT route_id, counties, impact_score, impacting_delivery
        FROM routes
        WHERE impact_score < ?
        ORDER BY impact_score DESC
        LIMIT ?
    ''', (threshold, num_below))
    
    routes_below = []
    for row in cursor.fetchall():
        route_dict = dict(row)
        route_dict['counties_list'] = route_dict['counties'].split(',')
        route_dict['threshold'] = threshold
        route_dict['above_threshold'] = False
        routes_below.append(route_dict)
    
    conn.close()
    
    # Combine all examples
    all_routes = {
        'threshold': threshold,
        'routes_above_threshold': routes_above,
        'routes_below_threshold': routes_below,
        'summary': {
            'total_examples': len(routes_above) + len(routes_below),
            'above_threshold_count': len(routes_above),
            'below_threshold_count': len(routes_below)
        }
    }
    
    # Save combined examples
    output_file = os.path.join(examples_dir, 'example_scored_routes.json')
    with open(output_file, 'w') as f:
        json.dump(all_routes, f, indent=2)
    
    # Save single example above threshold
    if routes_above:
        single_above = routes_above[0].copy()
        single_above.pop('counties_list', None)  # Remove redundant list
        single_file_above = os.path.join(examples_dir, 'example_route_above_threshold.json')
        with open(single_file_above, 'w') as f:
            json.dump(single_above, f, indent=2)
    
    # Save single example below threshold
    if routes_below:
        single_below = routes_below[0].copy()
        single_below.pop('counties_list', None)  # Remove redundant list
        single_file_below = os.path.join(examples_dir, 'example_route_below_threshold.json')
        with open(single_file_below, 'w') as f:
            json.dump(single_below, f, indent=2)
    
    print(f"Extracted {len(routes_above)} routes above threshold ({threshold})")
    print(f"Extracted {len(routes_below)} routes below threshold ({threshold})")
    print(f"\nSaved to:")
    print(f"  - {output_file} (all examples)")
    if routes_above:
        print(f"  - {examples_dir}/example_route_above_threshold.json (single example above)")
    if routes_below:
        print(f"  - {examples_dir}/example_route_below_threshold.json (single example below)")
    
    return all_routes

if __name__ == '__main__':
    extract_example_scored_routes(num_above=10, num_below=10, threshold=0.25)
