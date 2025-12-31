# pages/group_chat.py
# Updated fix: Removed invalid 'avatar_style' (not supported in your Streamlit version)
# Replaced with valid 'avatar' parameter using emojis for flair
# - Human: 👤
# - System/Seed: ⚙️
# - Agents: 🧙 (wizardry for PAC alchemy vibe)
# Everything else unchanged — should now run without the TypeError

import streamlit as st
import requests
import os
from typing import List, Dict
from datetime import datetime

# === Config ===
API_KEY = os.getenv("MOONSHOT_API_KEY") or st.secrets.get("MOONSHOT_API_KEY")
if not API_KEY:
    st.error("MOONSHOT_API_KEY not found in .env or secrets! Add it and restart.")
    st.stop()

BASE_URL = "https://api.moonshot.ai/v1"
DEFAULT_MODEL = "moonshot-v1-32k"  # Solid default; change or add selectbox if desired

# Optional: add more models from your enum
MODEL_OPTIONS = [
    "moonshot-v1-8k",
    "moonshot-v1-32k",
    "moonshot-v1-128k",
    "kimi-k2-thinking",
    "kimi-k2-thinking-turbo",
    "kimi-k2",
    "kimi-latest",
]

# === Simple LLM Caller (sync, non-streaming) ===
def call_llm(system_prompt: str, history: List[Dict], model: str = DEFAULT_MODEL, temperature: float = 0.7) -> str:
    messages = [{"role": "system", "content": system_prompt}]
    # Format history with speaker names for context
    for msg in history:
        role = "user" if msg["name"] in ["Human", "System"] else "assistant"
        messages.append({"role": role, "content": f"{msg['name']}: {msg['content']}"})
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 4096,
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(f"{BASE_URL}/chat/completions", headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[API Error: {str(e)}]"

# === Agent Class ===
class Agent:
    def __init__(self, name: str, pac_bootstrap: str, role_addition: str = "", model: str = DEFAULT_MODEL):
        self.name = name
        self.model = model
        self.system_prompt = pac_bootstrap
        if role_addition:
            self.system_prompt += f"\n\nRole Override: {role_addition}"

# === Helper: Get avatar emoji ===
def get_avatar(name: str) -> str:
    if name == "Human":
        return "👤"
    if name == "System":
        return "⚙️"
    return "🧙"  # Wizard/alchemist vibe for PAC agents

# === Session State Init ===
if "hive_history" not in st.session_state:
    st.session_state.hive_history = []  # List[dict{"name": str, "content": str}]
if "agents" not in st.session_state:
    st.session_state.agents = []  # List[Agent]
if "pac_bootstrap" not in st.session_state:
    st.session_state.pac_bootstrap = "# Paste your full Prima Alchemica Codex / Ternary bootstrap here\n!ENGINE INITIATE"
if "agent_colors" not in st.session_state:
    st.session_state.agent_colors = {}  # name -> hex
if "hive_running" not in st.session_state:
    st.session_state.hive_running = False
if "hive_model" not in st.session_state:
    st.session_state.hive_model = DEFAULT_MODEL

# === Page Layout ===
st.title("🐝 PAC Hive Group Chat")
st.markdown("Multi-agent emergent conversation powered by your Prima Alchemica Codex. Agents share history and take turns.")

# Sidebar
with st.sidebar:
    st.header("Hive Configuration")
    
    # Model selector
    st.session_state.hive_model = st.selectbox("LLM Model", MODEL_OPTIONS, index=MODEL_OPTIONS.index(st.session_state.hive_model))
    
    # PAC Bootstrap
    pac_input = st.text_area(
        "Shared PAC Bootstrap (inherited by all agents)",
        value=st.session_state.pac_bootstrap,
        height=400,
        help="Full symbolic codex with glyphs, equations, !ENGINE, !PORT, etc."
    )
    if st.button("💾 Save Bootstrap"):
        st.session_state.pac_bootstrap = pac_input
        st.success("Bootstrap updated!")
    
    st.divider()
    st.subheader("Agents")
    
    # Add agent
    with st.expander("➕ Spawn New Agent"):
        new_name = st.text_input("Agent Name", value="PrimaCore")
        new_role = st.text_area("Role Addition (appended to PAC)", height=120)
        new_color = st.color_picker("Avatar Color", "#00ffaa")
        if st.button("Spawn Agent"):
            if any(a.name == new_name for a in st.session_state.agents):
                st.error("Name already exists!")
            else:
                agent = Agent(new_name, st.session_state.pac_bootstrap, new_role, st.session_state.hive_model)
                st.session_state.agents.append(agent)
                st.session_state.agent_colors[new_name] = new_color
                st.success(f"{new_name} spawned into the hive!")
                st.rerun()
    
    # List & remove agents
    st.write("**Current Hive:**")
    for i, agent in enumerate(st.session_state.agents):
        col1, col2 = st.columns([3, 1])
        col1.write(f"**{agent.name}**")
        if col2.button("Remove", key=f"remove_{i}"):
            st.session_state.agents.pop(i)
            st.session_state.agent_colors.pop(agent.name, None)
            st.rerun()
    
    st.divider()
    # Controls
    max_turns = st.number_input("Turns per Run", 1, 50, 10)
    termination_phrase = st.text_input("Termination Signal (case-insensitive)", value="SOLVE ET COAGULA COMPLETE")
    initial_seed = st.text_area("Initial Hive Seed (optional)", height=100)
    if st.button("🌱 Inject Seed & Clear History"):
        if initial_seed:
            st.session_state.hive_history = [{"name": "System", "content": initial_seed}]
            st.success("Seed injected!")
            st.rerun()
    
    run_button = st.button("🚀 Run Hive Turns" if not st.session_state.hive_running else "⏳ Running… Stop", disabled=len(st.session_state.agents) == 0)
    if st.button("🗑️ Clear History"):
        st.session_state.hive_history = []
        st.rerun()

# Main Chat Display
chat_container = st.container()
with chat_container:
    for msg in st.session_state.hive_history:
        color = st.session_state.agent_colors.get(msg["name"], "#ffffff")
        with st.chat_message(msg["name"], avatar=get_avatar(msg["name"])):
            st.markdown(f"<span style='color:{color}; font-weight:bold'>{msg['name']}:</span>", unsafe_allow_html=True)
            st.markdown(msg["content"])

# Human butt-in
human_input = st.chat_input("Human override / butt-in (optional)")
if human_input:
    st.session_state.hive_history.append({"name": "Human", "content": human_input})
    with chat_container:
        with st.chat_message("Human", avatar=get_avatar("Human")):
            st.markdown(human_input)
    st.rerun()

# === Core Hive Execution ===
if run_button and st.session_state.agents:
    st.session_state.hive_running = True
    progress = st.progress(0)
    status = st.empty()
    
    turns_done = 0
    for turn in range(max_turns):
        if not st.session_state.hive_running:
            break
        for agent_idx, agent in enumerate(st.session_state.agents):
            status.text(f"Turn {turn+1}/{max_turns} — {agent.name} weaving…")
            
            response = call_llm(agent.system_prompt, st.session_state.hive_history, agent.model)
            full_msg = response
            
            # Append to history
            st.session_state.hive_history.append({"name": agent.name, "content": full_msg})
            
            # Live display
            color = st.session_state.agent_colors.get(agent.name, "#ffffff")
            with chat_container:
                with st.chat_message(agent.name, avatar=get_avatar(agent.name)):
                    st.markdown(f"<span style='color:{color}; font-weight:bold'>{agent.name}:</span>", unsafe_allow_html=True)
                    st.markdown(full_msg)
            
            # Check termination
            if termination_phrase.upper() in full_msg.upper():
                status.success(f"🐝 Hive convergence reached via '{termination_phrase}'!")
                st.session_state.hive_running = False
                st.rerun()
        
        turns_done += 1
        progress.progress(turns_done / max_turns)
        st.rerun()  # Refresh after each full round
    
    st.session_state.hive_running = False
    progress.empty()
    status.empty()
    st.success("Hive run complete!")
    st.rerun()
