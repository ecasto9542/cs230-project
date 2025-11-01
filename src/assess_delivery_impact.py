"""
Deterministic weighting algorithm to assess if weather events impact a delivery route.

The algorithm evaluates events based on:
1. Event type severity (tornadoes, floods > thunderstorms, heat)
2. Injuries and deaths (direct and indirect)
3. Property and crop damage
4. Event magnitude/scale
5. Spatial and temporal factors

Returns a score between 0 and 1 indicating delivery route impact.
"""
import sqlite3
from typing import List, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime

@dataclass
class EventImpact:
    """Structured impact assessment for a single event."""
    event_id: str
    event_type: str
    severity_score: float
    injury_score: float
    death_score: float
    damage_score: float
    magnitude_score: float
    total_score: float
    weighted_attributes: Dict[str, float]
    impacting_delivery: bool

class DeliveryRouteImpactAssessor:
    """
    Deterministic algorithm to assess delivery route impact from weather events.
    
    Scoring Rules:
    - Event Type Weights (severity):
      * Tornado: 1.0
      * Flood: 0.9
      * Wildfire: 0.85
      * Hurricane/Typhoon: 0.9
      * Thunderstorm Wind: 0.6
      * Hail: 0.5
      * Excessive Heat/Cold: 0.4
      * Heavy Rain/Snow: 0.3
      * Other: 0.2
    
    - Injury Thresholds (per event):
      * 0 injuries: 0.0
      * 1-5 injuries: 0.2
      * 6-20 injuries: 0.5
      * 21-50 injuries: 0.8
      * 50+ injuries: 1.0
    
    - Death Thresholds (per event):
      * 0 deaths: 0.0
      * 1 death: 0.5
      * 2-5 deaths: 0.8
      * 6+ deaths: 1.0
    
    - Property Damage Thresholds (per event):
      * $0: 0.0
      * $1-$10K: 0.1
      * $10K-$100K: 0.3
      * $100K-$1M: 0.6
      * $1M-$10M: 0.8
      * $10M+: 1.0
    
    - Magnitude/Scale Scoring:
      * For tornadoes (F-scale): F0=0.3, F1=0.5, F2=0.7, F3=0.85, F4=0.95, F5=1.0
      * For other events: normalized magnitude/100 capped at 1.0
    
    Final score = weighted combination of all factors.
    Impact threshold: score >= 0.5 indicates delivery impact.
    """
    
    # Event type severity weights
    EVENT_TYPE_WEIGHTS = {
        'TORNADO': 1.0,
        'FLOOD': 0.9,
        'FLASH FLOOD': 0.9,
        'WILDFIRE': 0.85,
        'HURRICANE': 0.9,
        'TYPHOON': 0.9,
        'THUNDERSTORM WIND': 0.6,
        'HAIL': 0.5,
        'EXCESSIVE HEAT': 0.4,
        'EXTREME COLD': 0.4,
        'HEAVY RAIN': 0.3,
        'HEAVY SNOW': 0.3,
        'DROUGHT': 0.2,
        'DENSE FOG': 0.2,
        'HIGH SURF': 0.55,
        'SNEAKERWAVE': 0.65,
        'STRONG WIND': 0.5,
        'HIGH WIND': 0.55,
        'DEBRIS FLOW': 0.75,
        'LANDSLIDE': 0.7,
        'DUST STORM': 0.5,
        'WINTER STORM': 0.4,
        'ICE STORM': 0.5,
    }
    
    # Component weights for final score
    EVENT_TYPE_WEIGHT = 0.22
    INJURY_WEIGHT = 0.18
    DEATH_WEIGHT = 0.22
    DAMAGE_WEIGHT = 0.18
    MAGNITUDE_WEIGHT = 0.10
    TEMPORAL_WEIGHT = 0.10  # New: temporal recency weight
    
    # Impact threshold
    IMPACT_THRESHOLD = 0.5
    
    # Temporal scoring parameters
    # Most recent year gets weight 1.0, oldest gets weight 0.3
    # Linear decay based on years since most recent event
    OLDEST_YEAR = 2014
    NEWEST_YEAR = 2025
    
    def __init__(self, db_path='data/california_events.db'):
        """Initialize with path to events database."""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
    
    def get_event(self, event_id: str) -> Dict:
        """Fetch a single event from database."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM events WHERE event_id = ?', (str(event_id),))
        row = cursor.fetchone()
        if row:
            return dict(row)
        raise ValueError(f"Event {event_id} not found in database")
    
    def get_events(self, event_ids: List[str]) -> List[Dict]:
        """Fetch multiple events from database."""
        events = []
        for event_id in event_ids:
            try:
                events.append(self.get_event(event_id))
            except ValueError as e:
                print(f"Warning: {e}")
        return events
    
    def score_event_type(self, event_type: str) -> float:
        """Score based on event type severity."""
        event_type_upper = str(event_type).upper().strip()
        return self.EVENT_TYPE_WEIGHTS.get(event_type_upper, 0.2)
    
    def score_injuries(self, injuries_direct: int, injuries_indirect: int) -> float:
        """Score based on total injuries."""
        total_injuries = injuries_direct + injuries_indirect
        
        if total_injuries == 0:
            return 0.0
        elif total_injuries <= 5:
            return 0.2
        elif total_injuries <= 20:
            return 0.5
        elif total_injuries <= 50:
            return 0.8
        else:
            return 1.0
    
    def score_deaths(self, deaths_direct: int, deaths_indirect: int) -> float:
        """Score based on total deaths."""
        total_deaths = deaths_direct + deaths_indirect
        
        if total_deaths == 0:
            return 0.0
        elif total_deaths == 1:
            return 0.5
        elif total_deaths <= 5:
            return 0.8
        else:
            return 1.0
    
    def score_damage(self, property_damage: float, crop_damage: float) -> float:
        """Score based on property and crop damage."""
        total_damage = property_damage + crop_damage
        
        if total_damage == 0:
            return 0.0
        elif total_damage < 10_000:
            return 0.1
        elif total_damage < 100_000:
            return 0.3
        elif total_damage < 1_000_000:
            return 0.6
        elif total_damage < 10_000_000:
            return 0.9
        elif total_damage < 50_000_000:
            return 0.95
        else:
            return 1.0
    
    def score_magnitude(self, event_type: str, magnitude: float, tor_f_scale: str) -> float:
        """Score based on event magnitude or scale."""
        event_type_upper = str(event_type).upper().strip()
        
        # Special handling for tornadoes (F-scale)
        if 'TORNADO' in event_type_upper or tor_f_scale:
            f_scale_map = {
                'EF0': 0.3, 'F0': 0.3,
                'EF1': 0.5, 'F1': 0.5,
                'EF2': 0.7, 'F2': 0.7,
                'EF3': 0.85, 'F3': 0.85,
                'EF4': 0.95, 'F4': 0.95,
                'EF5': 1.0, 'F5': 1.0,
            }
            if tor_f_scale:
                return f_scale_map.get(str(tor_f_scale).upper().strip(), 0.5)
            else:
                return 0.5  # Tornado without scale
        
        # For other events, normalize magnitude
        if magnitude and magnitude > 0:
            normalized = min(magnitude / 100.0, 1.0)
            return normalized
        
        return 0.3  # Default for events without magnitude
    
    def score_temporal(self, year: int) -> float:
        """
        Score based on how recent the event occurred.
        More recent events get higher scores.
        
        Args:
            year: Event year (2014-2025)
        
        Returns:
            Temporal score between 0.3 (oldest) and 1.0 (most recent)
        """
        if year is None:
            return 0.5  # Default for unknown year
        
        # Normalize year to 0-1 range, then scale to 0.3-1.0
        year_range = self.NEWEST_YEAR - self.OLDEST_YEAR
        if year_range == 0:
            return 1.0
        
        # Normalize: 0 = oldest, 1 = newest
        normalized = (year - self.OLDEST_YEAR) / year_range
        
        # Scale to 0.3-1.0 range (oldest gets 0.3, newest gets 1.0)
        temporal_score = 0.3 + (normalized * 0.7)
        
        return temporal_score
    
    def assess_event(self, event: Dict) -> EventImpact:
        """Assess impact of a single event."""
        # Extract event attributes
        event_id = str(event['event_id'])
        event_type = str(event.get('event_type', 'UNKNOWN'))
        injuries_direct = int(event.get('injuries_direct', 0) or 0)
        injuries_indirect = int(event.get('injuries_indirect', 0) or 0)
        deaths_direct = int(event.get('deaths_direct', 0) or 0)
        deaths_indirect = int(event.get('deaths_indirect', 0) or 0)
        property_damage = float(event.get('damage_property', 0) or 0)
        crop_damage = float(event.get('damage_crops', 0) or 0)
        magnitude = event.get('magnitude')
        if magnitude is not None:
            magnitude = float(magnitude) if str(magnitude) != 'nan' else None
        tor_f_scale = event.get('tor_f_scale')
        year = event.get('year')
        if year is not None:
            try:
                year = int(year)
            except (ValueError, TypeError):
                year = None
        
        # Calculate component scores
        event_type_score = self.score_event_type(event_type)
        injury_score = self.score_injuries(injuries_direct, injuries_indirect)
        death_score = self.score_deaths(deaths_direct, deaths_indirect)
        damage_score = self.score_damage(property_damage, crop_damage)
        magnitude_score = self.score_magnitude(event_type, magnitude, tor_f_scale)
        temporal_score = self.score_temporal(year)
        
        # Calculate weighted total score
        total_score = (
            event_type_score * self.EVENT_TYPE_WEIGHT +
            injury_score * self.INJURY_WEIGHT +
            death_score * self.DEATH_WEIGHT +
            damage_score * self.DAMAGE_WEIGHT +
            magnitude_score * self.MAGNITUDE_WEIGHT +
            temporal_score * self.TEMPORAL_WEIGHT
        )
        
        # Determine if impacting delivery
        impacting_delivery = total_score >= self.IMPACT_THRESHOLD
        
        # Create weighted attributes dictionary
        weighted_attributes = {
            'event_type_score': event_type_score,
            'injury_score': injury_score,
            'death_score': death_score,
            'damage_score': damage_score,
            'magnitude_score': magnitude_score,
            'temporal_score': temporal_score,
            'year': year,
            'total_injuries': injuries_direct + injuries_indirect,
            'total_deaths': deaths_direct + deaths_indirect,
            'total_damage': property_damage + crop_damage,
        }
        
        return EventImpact(
            event_id=event_id,
            event_type=event_type,
            severity_score=event_type_score,
            injury_score=injury_score,
            death_score=death_score,
            damage_score=damage_score,
            magnitude_score=magnitude_score,
            total_score=total_score,
            weighted_attributes=weighted_attributes,
            impacting_delivery=impacting_delivery
        )
    
    def assess_events(self, event_ids: List[str]) -> Tuple[List[EventImpact], float]:
        """
        Assess multiple events and calculate overall route impact score.
        
        Args:
            event_ids: List of event IDs to assess
        
        Returns:
            Tuple of (list of EventImpact objects, overall_route_score)
        """
        events = self.get_events(event_ids)
        
        if not events:
            raise ValueError("No valid events found")
        
        # Assess each event
        event_impacts = []
        for event in events:
            impact = self.assess_event(event)
            event_impacts.append(impact)
        
        # Calculate overall route score
        # Weighted average of individual event scores
        # More severe events contribute more to overall score
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for impact in event_impacts:
            # Weight by event severity
            weight = impact.severity_score + 0.1  # Minimum weight
            total_weighted_score += impact.total_score * weight
            total_weight += weight
        
        overall_route_score = total_weighted_score / total_weight if total_weight > 0 else 0.0
        
        # Cap at 1.0
        overall_route_score = min(overall_route_score, 1.0)
        
        return event_impacts, overall_route_score
    
    def assess_route(self, event_ids: List[str], verbose: bool = True) -> Dict:
        """
        Main method to assess delivery route impact.
        
        Args:
            event_ids: List of event IDs (e.g., ['1174463', '1195301', '1234567'])
            verbose: Whether to print detailed results
        
        Returns:
            Dictionary with assessment results
        """
        event_impacts, overall_score = self.assess_events(event_ids)
        
        impacting_count = sum(1 for impact in event_impacts if impact.impacting_delivery)
        
        result = {
            'event_count': len(event_impacts),
            'impacting_count': impacting_count,
            'overall_route_score': overall_score,
            'route_impacting_delivery': overall_score >= self.IMPACT_THRESHOLD,
            'events': [
                {
                    'event_id': impact.event_id,
                    'event_type': impact.event_type,
                    'total_score': impact.total_score,
                    'impacting_delivery': impact.impacting_delivery,
                    'weighted_attributes': impact.weighted_attributes
                }
                for impact in event_impacts
            ]
        }
        
        if verbose:
            self._print_results(result)
        
        return result
    
    def _print_results(self, result: Dict):
        """Print formatted assessment results."""
        print("\n" + "="*80)
        print("DELIVERY ROUTE IMPACT ASSESSMENT")
        print("="*80)
        print(f"\nEvents Assessed: {result['event_count']}")
        print(f"Events Impacting Delivery: {result['impacting_count']}")
        print(f"\nOverall Route Score: {result['overall_route_score']:.3f}")
        print(f"Route Impacting Delivery: {'YES' if result['route_impacting_delivery'] else 'NO'}")
        
        print("\n" + "-"*80)
        print("Individual Event Assessments:")
        print("-"*80)
        
        for event in result['events']:
            print(f"\nEvent ID: {event['event_id']}")
            print(f"  Type: {event['event_type']}")
            print(f"  Total Score: {event['total_score']:.3f}")
            print(f"  Impacting Delivery: {'YES' if event['impacting_delivery'] else 'NO'}")
            print(f"  Weighted Attributes:")
            for attr, value in event['weighted_attributes'].items():
                if isinstance(value, float):
                    print(f"    - {attr}: {value:.3f}")
                else:
                    print(f"    - {attr}: {value}")
        
        print("\n" + "="*80)
    
    def close(self):
        """Close database connection."""
        self.conn.close()

if __name__ == '__main__':
    # Example usage
    assessor = DeliveryRouteImpactAssessor()
    
    # Get some event IDs from the database
    cursor = assessor.conn.cursor()
    cursor.execute('SELECT event_id FROM events LIMIT 5')
    sample_ids = [row[0] for row in cursor.fetchall()]
    
    print(f"Assessing route with events: {sample_ids}")
    result = assessor.assess_route(sample_ids, verbose=True)
    
    assessor.close()
