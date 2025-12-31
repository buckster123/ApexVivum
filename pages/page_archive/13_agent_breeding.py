# pages/08_agent_breeding.py
# Agent Breeding: Evolve agents from successful hive runs

import streamlit as st
import json
import os
import sys
from pathlib import Path
import random
import uuid
from typing import List, Dict, Any

# === Import Diagnostics ===
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from main import state, memory_query, memory_insert, container
except ImportError as e:
    st.error(f"⚠️ Import error: {e}")
    st.stop()

# === Breeding Engine ===
class AgentBreeder:
    """Evolve agents by mutating successful configurations"""
    
    def __init__(self):
        self.sandbox_dir = Path(state.sandbox_dir) / "bred_agents"
        self.sandbox_dir.mkdir(exist_ok=True)
    
    def load_hive_history(self, convo_uuid: str) -> List[Dict]:
        """Load agent performance from hive run"""
        try:
            # Query memory for agent results and tool usage
            agent_data = memory_query(limit=100, convo_uuid=convo_uuid)
            if agent_data == "Key not found." or not agent_data:
                return []
            
            history = []
            if isinstance(agent_data, dict):
                for key, data in agent_data.items():
                    if key.startswith("agent_") and isinstance(data, dict):
                        history.append({
                            "agent_id": data.get("agent_id", "unknown"),
                            "performance": self._score_agent(data),
                            "config": {
                                "model": data.get("model", "kimi-k2-thinking"),
                                "temperature": data.get("temperature", 0.7),
                                "tools_used": data.get("tools_used", [])
                            }
                        })
            return sorted(history, key=lambda x: x["performance"], reverse=True)
        except Exception as e:
            st.error(f"Failed to load hive history: {e}")
            return []
    
    def _score_agent(self, agent_data: dict) -> float:
        """Calculate agent performance score"""
        score = 0.0
        # Tool usage efficiency
        tools_used = len(agent_data.get("tools_used", []))
        score += tools_used * 3
        
        # Response length (proxy for thoroughness)
        response = agent_data.get("response", "")
        score += min(len(response) / 1000, 5)  # Cap at 5 points
        
        # Success rate (if available)
        if "success" in agent_data:
            score += 10 if agent_data["success"] else -5
        
        return score
    
    def mutate_config(self, parent_config: dict, mutation_rate: float = 0.3) -> dict:
        """Create child config by mutating parent"""
        child = parent_config.copy()
        
        # Mutate temperature
        if random.random() < mutation_rate:
            delta = random.uniform(-0.3, 0.3)
            child["temperature"] = max(0.1, min(1.5, child.get("temperature", 0.7) + delta))
        
        # Mutate model (with lower probability)
        if random.random() < mutation_rate / 2:
            models = ["kimi-k2-thinking", "moonshot-v1-32k", "moonshot-v1-128k"]
            child["model"] = random.choice(models)
        
        # Add custom mutations
        if random.random() < mutation_rate:
            child["mutation_id"] = uuid.uuid4().hex[:8]
        
        return child
    
    def breed_agents(self, top_agents: List[Dict], offspring_count: int = 3) -> List[Dict]:
        """Create new agents from top performers"""
        if len(top_agents) < 2:
            st.warning("Need at least 2 agents to breed")
            return []
        
        offspring = []
        for i in range(offspring_count):
            # Select two parents (weighted by performance)
            parents = random.choices(top_agents[:3], k=2, weights=[a["performance"] for a in top_agents[:3]])
            
            # Crossover: mix configs
            child_config = {
                "model": parents[0]["config"]["model"] if random.random() < 0.5 else parents[1]["config"]["model"],
                "temperature": (parents[0]["config"]["temperature"] + parents[1]["config"]["temperature"]) / 2,
                "parent1": parents[0]["agent_id"],
                "parent2": parents[1]["agent_id"],
                "generation": 1
            }
            
            # Mutate
            child_config = self.mutate_config(child_config)
            
            offspring.append({
                "agent_id": f"bred_{uuid.uuid4().hex[:8]}",
                "config": child_config,
                "lineage": f"{parents[0]['agent_id']} x {parents[1]['agent_id']}"
            })
        
        return offspring
    
    def persist_bred_agent(self, agent_data: dict):
        """Save bred agent to disk and memory"""
        agent_file = self.sandbox_dir / f"{agent_data['agent_id']}.json"
        with open(agent_file, "w") as f:
            json.dump(agent_data, f, indent=2)
        
        # Save to memory
        memory_insert(
            f"breed_agent_{agent_data['agent_id']}",
            {
                "agent_id": agent_data["agent_id"],
                "config": agent_data["config"],
                "lineage": agent_data["lineage"],
                "created_at": datetime.now().isoformat(),
                "status": "bred"
            },
            convo_uuid=str(uuid.uuid4())
        )

breeder = AgentBreeder()

# === UI ===
st.title("🧬 Agent Breeding Lab")
st.markdown("*Evolve agents from successful hive runs*")

# Hive run selector
st.subheader("🐝 Select Parent Hive Run")
convo_uuids = []
try:
    with state.conn:
        state.cursor.execute(
            "SELECT DISTINCT uuid FROM memory WHERE mem_key LIKE 'agent_%' ORDER BY timestamp DESC LIMIT 20"
        )
        convo_uuids = [row[0] for row in state.cursor.fetchall() if row[0]]
except:
    pass

if not convo_uuids:
    st.info("No hive runs found. Run the hive first!")
    st.stop()

selected_uuid = st.selectbox("Choose a hive run:", convo_uuids, key="breed_selector")

# Load and display parent agents
if st.button("🔍 Analyze Hive Run"):
    with st.spinner("Loading agent performance data..."):
        agents = breeder.load_hive_history(selected_uuid)
        st.session_state["parent_agents"] = agents
    
    if agents:
        st.success(f"Found {len(agents)} agents")
        
        # Display top performers
        st.subheader("🏆 Top Performers")
        for i, agent in enumerate(agents[:5], 1):
            with st.expander(f"{i}. {agent['agent_id']} (Score: {agent['performance']:.1f})"):
                st.json(agent["config"])
    else:
        st.warning("No agent data found in this run")

# Breeding controls
if "parent_agents" in st.session_state and st.session_state["parent_agents"]:
    st.divider()
    st.subheader("🧬 Breeding Configuration")
    
    top_n = st.slider("Select Top N Parents", 2, min(5, len(st.session_state["parent_agents"])), 3)
    offspring_count = st.number_input("Number of Offspring", 1, 10, 3)
    mutation_rate = st.slider("Mutation Rate", 0.0, 1.0, 0.3, help="Higher = more variation")
    
    if st.button("🚀 Breed Agents", use_container_width=True):
        with st.spinner("Breeding in progress..."):
            top_agents = st.session_state["parent_agents"][:top_n]
            offspring = breeder.breed_agents(top_agents, offspring_count)
            
            if offspring:
                st.success(f"Created {len(offspring)} new agents!")
                
                # Display and persist offspring
                st.subheader("🌱 New Generation")
                for child in offspring:
                    with st.expander(f"{child['agent_id']}", expanded=True):
                        st.json({
                            "config": child["config"],
                            "lineage": child["lineage"],
                            "predicted_strength": "HIGH" if child["config"]["temperature"] > 0.6 else "BALANCED"
                        })
                        breeder.persist_bred_agent(child)
                
                # Export button
                st.download_button(
                    "📥 Export Offspring Configs",
                    json.dumps(offspring, indent=2),
                    file_name=f"bred_agents_{selected_uuid}.json",
                    mime="application/json"
                )

# Bred agents gallery
st.divider()
st.subheader("🎨 Bred Agents Gallery")

bred_files = list(breeder.sandbox_dir.glob("*.json"))
if bred_files:
    for agent_file in bred_files[:10]:  # Show first 10
        with open(agent_file, "r") as f:
            agent_data = json.load(f)
        
        with st.expander(f"🧬 {agent_data['agent_id']}"):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.json(agent_data["config"])
            with col2:
                st.caption(f"Lineage: {agent_data['lineage']}")
                if st.button("🗑️ Delete", key=f"del_{agent_data['agent_id']}"):
                    agent_file.unlink()
                    st.rerun()
else:
    st.info("No bred agents yet. Start breeding!")

# === Integration: Export to Hive ===
st.divider()
if "offspring" in locals() and offspring:
    if st.button("🐝 Send Offspring to Hive"):
        # This would integrate with your hive page
        st.session_state["bred_agents_for_hive"] = offspring
        st.success("Agents queued for hive deployment!")
        st.info("Go to the Hive page to spawn these bred agents")
