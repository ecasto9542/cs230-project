"""
Extract example events from the database and save as JSON files.
"""
import sqlite3
import json
import os

def extract_example_events():
    """Extract a few diverse example events and save as JSON."""
    conn = sqlite3.connect('data/california_events.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    examples_dir = 'examples'
    os.makedirs(examples_dir, exist_ok=True)
    
    # Create event-examples subdirectory for scored routes
    event_examples_dir = 'examples/event-examples'
    os.makedirs(event_examples_dir, exist_ok=True)
    
    # Get different types of events for variety
    queries = [
        ("Typical event - Drought", "SELECT * FROM events WHERE event_type = 'Drought' LIMIT 1"),
        ("Event with injuries and damage - Wildfire", "SELECT * FROM events WHERE injuries_direct + injuries_indirect > 0 AND damage_property > 0 LIMIT 1"),
        ("Event with location data", "SELECT * FROM events WHERE begin_lat IS NOT NULL AND begin_lon IS NOT NULL LIMIT 1"),
        ("Event with deaths", "SELECT * FROM events WHERE deaths_direct + deaths_indirect > 0 LIMIT 1"),
        ("High damage event", "SELECT * FROM events WHERE damage_property + damage_crops > 1000000 ORDER BY (damage_property + damage_crops) DESC LIMIT 1"),
    ]
    
    events_extracted = []
    
    for description, query in queries:
        cursor.execute(query)
        row = cursor.fetchone()
        if row:
            event_dict = dict(row)
            # Convert None values to null for JSON
            event_dict = {k: (None if v is None else v) for k, v in event_dict.items()}
            events_extracted.append((description, event_dict))
    
    # Save all examples to a single file
    output_file = os.path.join(examples_dir, 'example_events.json')
    with open(output_file, 'w') as f:
        json.dump(events_extracted, f, indent=2, default=str)
    
    # Also save a single comprehensive example
    if events_extracted:
        single_example = events_extracted[1][1] if len(events_extracted) > 1 else events_extracted[0][1]
        single_file = os.path.join(examples_dir, 'example_event_single.json')
        with open(single_file, 'w') as f:
            json.dump(single_example, f, indent=2, default=str)
    
    conn.close()
    
    print(f"Extracted {len(events_extracted)} example events")
    print(f"Saved to:")
    print(f"  - {output_file} (all examples)")
    print(f"  - {examples_dir}/example_event_single.json (single comprehensive example)")
    
    return events_extracted

if __name__ == '__main__':
    extract_example_events()
