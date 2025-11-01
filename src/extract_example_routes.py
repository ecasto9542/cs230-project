"""
Extract example delivery routes and save as JSON files.
"""
import sqlite3
import json
import os

def extract_example_routes(num_examples=10):
    """Extract example routes from the database and save as JSON."""
    conn = sqlite3.connect('data/delivery_routes.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    examples_dir = 'examples/event-examples'
    os.makedirs(examples_dir, exist_ok=True)
    
    # Get routes with different county counts
    cursor.execute('''
        SELECT route_id, county_count, counties
        FROM routes
        ORDER BY route_id
        LIMIT ?
    ''', (num_examples,))
    
    routes = []
    for row in cursor.fetchall():
        route_dict = dict(row)
        # Parse counties string into list
        route_dict['counties_list'] = route_dict['counties'].split(',')
        routes.append(route_dict)
    
    conn.close()
    
    # Save to JSON
    output_file = os.path.join(examples_dir, 'example_routes.json')
    with open(output_file, 'w') as f:
        json.dump(routes, f, indent=2)
    
    # Also save a single example
    if routes:
        single_example = routes[0]
        # Remove counties_list for single example (redundant with counties string)
        single_example.pop('counties_list', None)
        single_file = os.path.join(examples_dir, 'example_route_single.json')
        with open(single_file, 'w') as f:
            json.dump(single_example, f, indent=2)
    
    print(f"Extracted {len(routes)} example routes")
    print(f"Saved to:")
    print(f"  - {output_file} (all examples)")
    print(f"  - {examples_dir}/example_route_single.json (single example)")
    
    return routes

if __name__ == '__main__':
    extract_example_routes(num_examples=10)
