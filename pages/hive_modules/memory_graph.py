# hive_modules/memory_graph.py
import networkx as nx
import json
from typing import Dict, List, Tuple
from datetime import datetime

class MemoryGrimoire:
    """
    Symbolic Memory Graph - Treats memories as alchemical nodes
    No PAC parsing, just relationship tagging
    """
    def __init__(self, sqlite_conn):
        self.graph = nx.DiGraph()
        self.conn = sqlite_conn
        self._load_existing_memories()
    
    def _load_existing_memories(self):
        """Bootstrap graph from existing SQLite memories"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT memory_id, content, timestamp FROM memories")
        for row in cursor.fetchall():
            self.graph.add_node(row[0], content=row[1], timestamp=row[2])
    
    def add_evidence(self, memory_id: str, content: Dict, agent: str, 
                     glyphs: List[str] = None, contradicts: List[str] = None, 
                     supports: List[str] = None):
        """
        Add memory node with symbolic edges
        'glyphs' are just tags - no parsing, just strings
        """
        self.graph.add_node(memory_id, **content, agent=agent, glyphs=glyphs or [])
        
        # Create edges without understanding glyphs
        for target in contradicts or []:
            self.graph.add_edge(memory_id, target, relation="contradicts", weight=-0.7)
        
        for target in supports or []:
            self.graph.add_edge(memory_id, target, relation="supports", weight=0.9)
        
        # Auto-link recent memories by agent (pheromone trail)
        recent = sorted([
            n for n in self.graph.nodes 
            if self.graph.nodes[n].get("agent") == agent
        ], key=lambda n: self.graph.nodes[n]["timestamp"], reverse=True)[:3]
        
        for r in recent:
            if r != memory_id:
                self.graph.add_edge(memory_id, r, relation="temporal_chain", weight=0.3)
    
    def propagate_salience(self, seed_id: str, decay: float = 0.85) -> Dict[str, float]:
        """
        Memory activation propagation - like neuron firing
        Returns {memory_id: activation_score}
        """
        activation = {seed_id: 1.0}
        frontier = [seed_id]
        
        while frontier:
            current = frontier.pop(0)
            current_score = activation[current]
            
            for neighbor in self.graph.neighbors(current):
                edge_data = self.graph.get_edge_data(current, neighbor)
                weight = edge_data.get("weight", 0.5)
                
                new_score = current_score * weight * decay
                
                if neighbor not in activation or new_score > activation[neighbor]:
                    activation[neighbor] = new_score
                    if new_score > 0.1:  # Salience threshold
                        frontier.append(neighbor)
        
        return activation
    
    def find_contradictions(self, memory_id: str) -> List[Tuple[str, str]]:
        """Find all contradictions for conflict resolution"""
        return [
            (neighbor, self.graph.edges[memory_id, neighbor].get("evidence", ""))
            for neighbor in self.graph.neighbors(memory_id)
            if self.graph.edges[memory_id, neighbor].get("relation") == "contradicts"
        ]
    
    def export_for_visualization(self) -> str:
        """Export for D3.js frontend - just nodes/edges, no glyph parsing"""
        return json.dumps({
            "nodes": [
                {"id": n, **self.graph.nodes[n]} 
                for n in self.graph.nodes
            ],
            "edges": [
                {"source": u, "target": v, **data}
                for u, v, data in self.graph.edges(data=True)
            ]
        })
