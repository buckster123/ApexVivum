# hive_modules/swarm_pheromone.py
import asyncio
import re
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class PheromoneTrail:
    topic: str
    intensity: float  # 0.0 to 1.0
    agent_name: str
    timestamp: float

class PheromoneTracker:
    """Detects emergent topics without parsing PAC content"""
    
    # Simple keyword-to-topic mapping - no glyph parsing!
    TOPIC_KEYWORDS = {
        "database": "db_optimization",
        "sql": "db_optimization",
        "memory": "memory_systems",
        "prune": "memory_systems",
        "vector": "memory_systems",
        "tool": "tool_usage",
        "api": "tool_usage",
        "cost": "cost_tracking",
        "token": "cost_tracking",
        "async": "concurrency",
        "semaphore": "concurrency"
    }
    
    def __init__(self, spawn_callback: callable):
        self.trails: List[PheromoneTrail] = []
        self.spawn_callback = spawn_callback
    
    def analyze_message(self, message: str, agent_name: str) -> List[str]:
        """Extract pheromones from plain text"""
        trails = []
        
        # Simple regex - no glyph parsing
        for keyword, topic in self.TOPIC_KEYWORDS.items():
            if re.search(r'\b' + keyword + r'\b', message, re.IGNORECASE):
                # Calculate intensity based on frequency and urgency markers
                frequency = len(re.findall(keyword, message, re.IGNORECASE))
                urgency_markers = len(re.findall(r'!', message))
                intensity = min(0.9, 0.3 * frequency + 0.2 * urgency_markers)
                
                trails.append(PheromoneTrail(
                    topic=topic,
                    intensity=intensity,
                    agent_name=agent_name,
                    timestamp=asyncio.get_event_loop().time()
                ))
        
        self.trails.extend(trails)
        return [t.topic for t in trails]
    
    async def evaluate_swarm_needs(self) -> List[Dict]:
        """Check if new swarm cells should spawn"""
        now = asyncio.get_event_loop().time()
        
        # Decay old trails
        active_trails = [
            t for t in self.trails 
            if now - t.timestamp < 120  # 2 minute half-life
        ]
        
        # Aggregate intensities
        topic_intensity = {}
        for trail in active_trails:
            topic_intensity[trail.topic] = topic_intensity.get(trail.topic, 0) + trail.intensity
        
        # Spawn threshold
        spawn_requests = []
        for topic, intensity in topic_intensity.items():
            if intensity > 1.5:  # Threshold for specialization
                spawn_requests.append({
                    "topic": topic,
                    "intensity": intensity,
                    "specialist_role": self._get_specialist_role(topic)
                })
        
        return spawn_requests
    
    def _get_specialist_role(self, topic: str) -> str:
        """Map topic to specialist agent role - plain English, no glyphs"""
        role_map = {
            "db_optimization": "Database query optimizer. Focus on SQL, indexing, connection pools.",
            "memory_systems": "Memory systems architect. Focus on pruning, retention, vector search.",
            "tool_usage": "Tool efficiency analyst. Track ROI, reduce redundant calls.",
            "cost_tracking": "Cost optimization specialist. Minimize token waste.",
            "concurrency": "Async concurrency engineer. Optimize semaphore usage."
        }
        return role_map.get(topic, "General problem solver.")
