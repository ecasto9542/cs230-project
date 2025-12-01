# cs230-project

## Weather Event Database and Delivery Route Impact Assessment

This project extracts weather events from NOAA's storm events database, stores them in SQLite relational databases (SQL-based, table-structured), and applies a deterministic weighting algorithm to assess and label delivery routes based on their historical weather event impact.

**Note on Database Type**: This project uses **SQLite relational databases** (table-based SQL databases). Each weather event and route is stored as a row in a table with structured columns. While the data represents "objects" (events, routes), the storage is relational (SQL-based) rather than object-oriented databases (like MongoDB document stores).

### Overview

The project follows a data pipeline:
1. **Data Collection**: Download weather events from NOAA (2014-2025)
2. **Data Extraction**: Filter and extract California weather events
3. **Route Generation**: Create random delivery routes across California counties
4. **Data Integration**: Join routes with weather events via county matching
5. **Impact Scoring**: Apply deterministic algorithm to calculate impact scores
6. **Labeling**: Assign binary labels (impacting/not impacting) based on threshold

All results are stored in **SQLite relational databases** (SQL-based, table-structured) optimized for query efficiency. Each event and route is stored as a structured row in normalized tables with defined schemas.

### Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`

#### Recreating Database Structures

**What's Already Included:**
After cloning, you'll have 3 databases in `data/`:
- ✅ `california_events.db` - All California weather events (24,039 events)
- ✅ `delivery_routes.db` - All delivery routes (20,000 routes)
- ✅ `routes_scores.db` - Labeled dataset with impact scores

**To Recreate the Missing Database:**

The `route_events.db` file (1.4GB) is too large for GitHub and is excluded. To recreate it locally, run:

```bash
python src/create_route_events_database.py
```

This script requires:
- `data/california_events.db` ✅ (already included)
- `data/delivery_routes.db` ✅ (already included)

It will create `data/route_events.db` by:
1. Copying routes from `delivery_routes.db`
2. Copying events from `california_events.db`
3. Creating route-event links based on county matching

**Note**: If you want to regenerate `routes_scores.db` with fresh impact scores, you would also need to run:
1. `python src/assess_all_routes.py` (calculates scores in `route_events.db`)
2. `python src/create_routes_scores_database.py` (extracts scores to `routes_scores.db`)

However, `routes_scores.db` is already included in the repository, so this is only needed if you want to recalculate scores.

### Database Architecture & Contents

The project maintains four distinct **SQLite relational databases** (SQL-based, table-structured), each serving a specific purpose. Each database stores data in structured tables with defined schemas:

- **Relational Structure**: Data stored in tables with rows and columns (not document/object stores)
- **SQL-based**: Standard SQL queries used for all operations
- **Normalized Design**: Tables related via foreign keys and junction tables

#### 1. California Weather Events Database (`data/california_events.db`)

**Purpose**: Source of truth for all California weather events from NOAA.

**Contents**:
- **24,039 weather events** from 2014-2025
- Complete event attributes: event type, location (county, coordinates), dates, injuries, deaths, property/crop damage, magnitude, narratives
- **Schema**: Single `events` table with all event attributes
- **Use Case**: Source data for impact scoring and route-event matching

**Key Fields**:
- `event_id`: Unique event identifier
- `cz_name`: County/zone name (used for route matching)
- `event_type`: Type of weather event
- `year`: Event year (used for temporal weighting)
- `injuries_direct`, `injuries_indirect`, `deaths_direct`, `deaths_indirect`: Human impact metrics
- `damage_property`, `damage_crops`: Economic impact metrics
- `begin_lat`, `begin_lon`: Coordinates (available for 31.4% of events)

**Example Events:** See `examples/example_event_single.json` and `examples/example_events.json` for sample event data showing all available attributes. These JSON files show the structure and types of data stored for each weather event.

#### 2. Delivery Routes Database (`data/delivery_routes.db`)

**Purpose**: Base delivery routes without weather event associations.

**Contents**:
- **20,000 delivery routes** randomly generated
- Each route: 3-10 counties visited sequentially
- Routes generated from 312 unique California counties/zones
- **Schema**: Single `routes` table
- **Use Case**: Source data for route generation and structure

**Key Fields**:
- `route_id`: Unique route identifier
- `county_count`: Number of counties (3-10)
- `counties`: Comma-separated list of counties in visit order

**Route Structure:**
- Each route has a unique `route_id`
- Routes contain 3-10 counties (randomly distributed)
- Counties are listed in the order the driver visits them
- Routes are randomly generated from 312 unique California counties/zones

**Example Routes:** See `examples/example_route_single.json` and `examples/example_routes.json` for sample route data.

**Example Scored Routes:** See `examples/event-examples/example_scored_routes.json`, `examples/event-examples/example_route_above_threshold.json`, and `examples/event-examples/example_route_below_threshold.json` for examples of routes with impact scores.

#### 3. Routes Scores Database (`data/routes_scores.db`) ⭐ **LABELED DATASET**

**Purpose**: **Final labeled dataset** for machine learning/analysis. Contains only routes with their impact scores and binary labels.

**Contents**:
- **20,000 labeled routes** with impact scores and binary classifications
- **Schema**: Single `routes` table with minimal attributes
- **Labels**: Binary classification (`impacting_delivery`) based on 0.25 threshold
- **Use Case**: Primary dataset for model training, analysis, and deployment

**Database Structure:**
```sql
routes (
    route_id INTEGER PRIMARY KEY,
    counties TEXT NOT NULL,              -- Comma-separated county list
    impact_score REAL NOT NULL,          -- Calculated impact score (0.0-1.0)
    impacting_delivery INTEGER NOT NULL   -- Binary label (1 if score >= 0.25, 0 otherwise)
)
```

**Labeling Logic Applied:**
- Each route's `impact_score` calculated using the deterministic weighting algorithm
- Algorithm considers all historical weather events (2014-2025) in route counties
- `impacting_delivery` assigned: `1` if score >= 0.25, `0` otherwise
- Threshold of 0.25 selected to capture routes with moderate+ weather impact

**Statistics:**
- Total routes: 20,000
- Routes labeled as impacting (`impacting_delivery = 1`): 4,699 (23.5%)
- Routes labeled as not impacting (`impacting_delivery = 0`): 15,301 (76.5%)
- Average impact score: 0.2294
- Score range: 0.1329 - 0.3201

**Important**: This database is **completely separate** from `route_events.db`. The impact scores and labels exist ONLY in this database.

#### 4. Route-Events Joined Database (`data/route_events.db`)

**Purpose**: Normalized relational database for efficient algorithm execution and detailed analysis. Used to calculate impact scores but does NOT store the final labels.

**Contents**:
- **Routes**: 20,000 routes (structure only, no scores)
- **Events**: 24,039 weather events (key attributes for scoring)
- **Route-Event Links**: 10+ million junction records matching routes to events via county

**Database Architecture (3-table normalized design):**
```
routes (
    route_id PRIMARY KEY,
    county_count,
    counties,
    created_at
)

events (
    event_id PRIMARY KEY,
    cz_name,                    -- County name (for matching)
    event_type, year,           -- Used in scoring algorithm
    injuries_*, deaths_*,        -- Used in scoring algorithm
    damage_property, damage_crops,  -- Used in scoring algorithm
    magnitude, magnitude_type,  -- Used in scoring algorithm
    begin_lat, begin_lon        -- Location data
)

route_events (
    route_id, event_id, county, county_sequence  -- Junction table
    PRIMARY KEY (route_id, event_id, county)
)
```

**How It's Used for Labeling:**
1. Query all events for a route via `route_events` junction table
2. Score each event using the weighting algorithm
3. Aggregate event scores to route-level score
4. Store final score and label in `routes_scores.db` (NOT in this database)

**Key Features:**
- **Normalized design**: Eliminates data duplication
- **Heavily indexed**: 7 indexes for fast queries (routes, events, county sequences, event types, years)
- **Optimized for algorithms**: Fast retrieval of events by route, county, or sequence
- **Average**: ~500 events per route

**Query Interface:**
Use `src/query_route_events.py` for efficient database queries:
- `get_events_for_route(route_id)` - Get all events for a route
- `get_routes_for_event(event_id)` - Get all routes affected by an event
- `get_events_by_county_sequence(route_id, start, end)` - Get events for route segments
- `get_route_statistics(route_id)` - Get comprehensive route statistics

**Note**: This database does NOT contain `impact_score` or `impacting_delivery` columns. Those exist ONLY in `routes_scores.db`.

**Note:** If you need to rebuild the database:
- Download weather data: `python src/get_data.py` (downloads CSV files temporarily)
- Extract California events: `python src/extract_california_events.py` (extracts all events) or `python src/extract_california_events.py 1000` (extracts first 1000)
- CSV files can be deleted after database creation to save space (~787MB freed)

### Viewing Databases

SQLite databases are binary files and cannot be viewed directly in text editors. Use a GUI database browser or command-line tools:

#### GUI Database Browser (Recommended)

**Install DB Browser for SQLite** (free, cross-platform):
- **macOS**: `brew install --cask db-browser-for-sqlite` or download from [sqlitebrowser.org](https://sqlitebrowser.org/)
- **Windows/Linux**: Download from [sqlitebrowser.org](https://sqlitebrowser.org/)

**Usage:**
1. Open DB Browser for SQLite
2. Click "Open Database"
3. Navigate to `data/routes_scores.db` (or any `.db` file)
4. Browse tables, view data, run queries visually

#### VS Code Extension

**SQLite Viewer Extension for VS Code:**
1. Install "SQLite Viewer" extension in VS Code
2. Right-click any `.db` file → "Open Database"
3. Browse tables and run queries in the sidebar

#### SQLite Command Line

**Quick queries from terminal:**
```bash
# View table schema
sqlite3 data/routes_scores.db ".schema routes"

# Query data
sqlite3 data/routes_scores.db "SELECT * FROM routes LIMIT 10;"

# Export to CSV
sqlite3 -header -csv data/routes_scores.db "SELECT * FROM routes LIMIT 100;" > routes_sample.csv

# Export to JSON (macOS/Linux)
sqlite3 data/routes_scores.db ".mode json" "SELECT * FROM routes LIMIT 10;"
```

### Quick Start & Usage

#### Viewing Labeled Data

**Query the labeled dataset directly:**
Use SQL queries on `routes_scores.db` or a database browser tool (see "Viewing Databases" section above).

**Example labeled routes:**
- Above threshold: `examples/event-examples/example_route_above_threshold.json`
- Below threshold: `examples/event-examples/example_route_below_threshold.json`
- Both: `examples/event-examples/example_scored_routes.json`

#### Database Queries

**Query routes_scores.db (labeled dataset):**
```sql
-- All routes with impact scores
SELECT route_id, counties, impact_score, impacting_delivery 
FROM routes 
ORDER BY impact_score DESC;

-- Routes labeled as impacting
SELECT * FROM routes WHERE impacting_delivery = 1;

-- Score distribution
SELECT 
    CASE 
        WHEN impact_score < 0.2 THEN 'Low'
        WHEN impact_score < 0.25 THEN 'Medium-Low'
        WHEN impact_score < 0.3 THEN 'Medium'
        ELSE 'High'
    END as category,
    COUNT(*) as count
FROM routes
GROUP BY category;
```

#### Re-assessing Routes (if algorithm changes)

If you modify the scoring algorithm and need to re-label:
1. Update algorithm in `src/assess_delivery_impact.py`
2. Run: `python src/assess_all_routes.py` (processes all 20,000 routes)
3. Update `routes_scores.db`: `python src/create_routes_scores_database.py`
4. Update threshold if needed: `python src/update_routes_scores_threshold.py`

### Impact Scoring Algorithm & Labeling Logic

The project uses a **deterministic multi-factor weighting algorithm** to assess weather event impact on delivery routes. The algorithm evaluates each weather event individually, then aggregates scores to produce a route-level impact assessment.

#### Algorithm Components

**1. Event-Level Scoring (per weather event):**

Each weather event is scored across 6 dimensions:

- **Event Type Severity (22% weight)**: Different event types have different baseline severities
  - Tornado: 1.0 (highest risk)
  - Flood/Flash Flood: 0.9
  - Wildfire: 0.85
  - Debris Flow: 0.75
  - Landslide: 0.7
  - Sneakerwave: 0.65
  - Thunderstorm Wind: 0.6
  - High Surf/High Wind: 0.55
  - Hail/Strong Wind: 0.5
  - Excessive Heat/Extreme Cold: 0.4
  - Heavy Rain/Snow: 0.3
  - Drought/Dense Fog: 0.2 (lowest risk)

- **Injuries (18% weight)**: Human injury impact
  - 0 injuries: 0.0
  - 1-5 injuries: 0.2
  - 6-20 injuries: 0.5
  - 21-50 injuries: 0.8
  - 50+ injuries: 1.0

- **Deaths (22% weight)**: Human fatality impact
  - 0 deaths: 0.0
  - 1 death: 0.5
  - 2-5 deaths: 0.8
  - 6+ deaths: 1.0

- **Property/Crop Damage (18% weight)**: Economic impact
  - $0: 0.0
  - <$10K: 0.1
  - <$100K: 0.3
  - <$1M: 0.6
  - <$10M: 0.9
  - <$50M: 0.95
  - $50M+: 1.0

- **Magnitude/Scale (10% weight)**: Event intensity
  - Tornadoes: F-scale mapping (F0=0.3, F1=0.5, F2=0.7, F3=0.85, F4=0.95, F5=1.0)
  - Other events: Normalized magnitude/100 (capped at 1.0)

- **Temporal Recency (10% weight)**: How recent the event occurred
  - Events from 2025 (newest): 1.0
  - Events from 2014 (oldest): 0.3
  - Linear interpolation: More recent events get higher weights

**Event Score Formula:**
```
event_score = (event_type × 0.22) + (injuries × 0.18) + (deaths × 0.22) + 
              (damage × 0.18) + (magnitude × 0.10) + (temporal × 0.10)
```

**2. Route-Level Aggregation:**

For each delivery route, all associated weather events are aggregated:

- Individual event scores are weighted by their severity + temporal score
- Route impact score = Weighted average of all event scores
- Score range: 0.0 (no impact) to 1.0 (maximum impact)

**3. Binary Labeling:**

Routes are labeled as impacting delivery based on threshold:
- `impacting_delivery = 1` if `impact_score >= 0.25`
- `impacting_delivery = 0` if `impact_score < 0.25`

**Rationale**: The 0.25 threshold captures routes with moderate to high weather event activity, balancing sensitivity (catching potentially impacted routes) with specificity (avoiding false positives from minimal events).

#### Dataset Labeling Process

1. **Route-Event Matching**: Each route is matched to all historical weather events occurring in the counties along its path (2014-2025)
2. **Event Scoring**: Each matched event is scored using the 6-factor algorithm
3. **Route Scoring**: Route-level score calculated as weighted average of all event scores
4. **Threshold Application**: Binary label assigned based on 0.25 threshold
5. **Database Storage**: Scores and labels stored in `routes_scores.db`

**Labeling Statistics:**
- Total routes labeled: 20,000
- Routes with `impacting_delivery = 1`: 4,699 (23.5%)
- Routes with `impacting_delivery = 0`: 15,301 (76.5%)
- Average impact score: 0.2294
- Score distribution: 0.1329 - 0.3201

### Project Structure

**Databases:**
- `data/california_events.db` - SQLite database with 24,039 California weather events (2014-2025)
- `data/delivery_routes.db` - SQLite database with 20,000 delivery routes (3-10 counties each)
- `data/route_events.db` - Normalized joined database linking routes to events (optimized for algorithm queries)
- `data/routes_scores.db` - Standalone database with routes and impact scores ONLY (separate from other databases)

**Scripts:**
- `src/extract_california_events.py` - Extracts California events to SQLite database
- `src/create_delivery_routes.py` - Creates delivery routes database
- `src/create_route_events_database.py` - Creates joined route-events database
- `src/create_routes_scores_database.py` - Creates standalone routes scores database
- `src/query_route_events.py` - Efficient query interface for route-events database
- `src/assess_delivery_impact.py` - Core weighting algorithm with temporal weighting
- `src/assess_all_routes.py` - Batch process all routes and assign impact scores
- `src/route_assessment_cli.py` - Command-line interface for route assessment
- `src/list_events.py` - List and browse events in the database
- `src/count_california_events.py` - Count total California events available
- `src/get_data.py` - Downloads NOAA storm event CSV files (optional, only needed to rebuild database)
  
**Notebooks:**
- `notebooks/logreg_baseline.ipynb` - Logistic Regression baseline model includes preprocessing, model training, evaluation (AUPRC, Brier score), and feature importance.
- `notebooks/random_forest.ipynb` - Random Forest model includes preprocessing, model training, evaluation (AUPRC, Brier score), and feature importance.
- `notebooks/deep_model_mlp.ipynb` - Implements a TensorFlow MLP to predict weather-impacted delivery routes and includes an ablation study with and without impact_score.