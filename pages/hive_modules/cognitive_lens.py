# hive_modules/cognitive_lens.py
import plotly.graph_objects as go
import streamlit.components.v1 as components
from typing import List,Dict

class TokenFlowVisualizer:
    """Real-time token flow Sankey diagram"""
    
    def __init__(self, ledger):
        self.ledger = ledger
        self.flow_log = []  # [(from, to, tokens, timestamp)]
    
    def log_flow(self, source: str, target: str, tokens: int):
        """Log token transfer (agent->tool, tool->agent, etc.)"""
        self.flow_log.append({
            "source": source,
            "target": target,
            "tokens": tokens,
            "timestamp": datetime.now().isoformat()
        })
    
    def render_sankey(self) -> str:
        """Generate Plotly Sankey diagram HTML"""
        # Aggregate flows
        flow_map = {}
        for flow in self.flow_log[-50:]:  # Last 50 flows
            key = (flow["source"], flow["target"])
            flow_map[key] = flow_map.get(key, 0) + flow["tokens"]
        
        # Create nodes and links
        nodes = list(set([s for s,_ in flow_map.keys()] + [t for _,t in flow_map.keys()]))
        node_indices = {n: i for i, n in enumerate(nodes)}
        
        fig = go.Figure(data=[go.Sankey(
            node = dict(label = nodes),
            link = dict(
                source = [node_indices[s] for s,_ in flow_map.keys()],
                target = [node_indices[t] for _,t in flow_map.keys()],
                value = list(flow_map.values())
            )
        )])
        
        return fig.to_html()

class AgentStateLens:
    """Sidebar inspector for agent cognitive state"""
    
    @staticmethod
    def inspect(agent_name: str, history: List[Dict]) -> str:
        """Generate human-readable state summary without parsing PAC"""
        agent_msgs = [m for m in history if m["name"] == agent_name][-10:]
        
        # Simple heuristics - no glyph parsing
        tool_usage = len([m for m in agent_msgs if "🜛 **Tool Results:**" in m["content"]])
        avg_length = sum(len(m["content"]) for m in agent_msgs) / len(agent_msgs) if agent_msgs else 0
        
        # Cost via ledger (you'd pass this in)
        # cost = ledger.usage.get(agent_name, {}).get("cost", 0)
        
        return f"""
        **Agent: {agent_name}**
        - Recent messages: {len(agent_msgs)}
        - Tool calls: {tool_usage}
        - Avg message length: {avg_length:.0f} chars
        - Focus area: {AgentStateLens._classify_focus(agent_msgs)}
        """
    
    @staticmethod
    def _classify_focus(msgs: List[Dict]) -> str:
        """Crude topic classification - no parsing"""
        text = " ".join([m["content"] for m in msgs])
        if "tool" in text.lower() and "cost" in text.lower():
            return "Cost optimization"
        if "memory" in text.lower():
            return "Memory systems"
        return "General reasoning"
