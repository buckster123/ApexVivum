# pages/group_chat_phase2_fixed.py
# Phase-2 Transmutation: Fixed Scoping & Execution Order

import streamlit as st
import aiohttp
import asyncio
import json
import copy
import os
from typing import List, Dict, Callable, Optional, Any
from datetime import datetime
from asyncio import Semaphore

# === Config ===
API_KEY = os.getenv("MOONSHOT_API_KEY") or st.secrets.get("MOONSHOT_API_KEY")
if not API_KEY:
    st.error("🜩 MOONSHOT_API_KEY not found in .env or secrets! Add it and restart.")
    st.stop()

BASE_URL = "https://api.moonshot.ai/v1"
DEFAULT_MODEL = "moonshot-v1-32k"
MODEL_OPTIONS = [
    "moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k",
    "kimi-k2-thinking", "kimi-k2-thinking-turbo", "kimi-k2", "kimi-latest"
]

# === Tool 2.5 Architecture (Placeholders) ===
class ToolRegistry:
    """Registry for agent tools - ready for future expansion"""
    def __init__(self):
        self.tools = {}
        self.tool_schemas = []
    
    def register(self, name: str, func: Callable, description: str, parameters: Dict):
        """Register a tool function with schema for LLM use"""
        self.tools[name] = {
            "func": func,
            "schema": {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters
                }
            }
        }
        self.tool_schemas.append(self.tools[name]["schema"])
    
    def get_schema(self) -> List[Dict]:
        return self.tool_schemas
    
    async def execute(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool if registered"""
        if name not in self.tools:
            return f"Tool '{name}' not found."
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.tools[name]["func"], **arguments)
        except Exception as e:
            return f"Tool execution error: {str(e)}"

# Global tool registry (ready for future tools)
TOOL_REGISTRY = ToolRegistry()

# === Token Cost Tracker ===
class AlchemistLedger:
    """Track token usage and costs for the PAC hive"""
    PRICING = {
        "moonshot-v1-8k": {"input": 0.012, "output": 0.012},
        "moonshot-v1-32k": {"input": 0.024, "output": 0.024},
        "moonshot-v1-128k": {"input": 0.048, "output": 0.048},
        "kimi-k2-thinking": {"input": 0.03, "output": 0.06},
        "kimi-k2": {"input": 0.024, "output": 0.024},
    }
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.usage = {}
        self.total_cost = 0.0
    
    def log_usage(self, agent_name: str, input_tokens: int, output_tokens: int, model: str):
        if model not in self.PRICING:
            return
        
        cost = (input_tokens * self.PRICING[model]["input"] / 1000) + \
               (output_tokens * self.PRICING[model]["output"] / 1000)
        
        if agent_name not in self.usage:
            self.usage[agent_name] = {"input_tokens": 0, "output_tokens": 0, "cost": 0.0}
        
        self.usage[agent_name]["input_tokens"] += input_tokens
        self.usage[agent_name]["output_tokens"] += output_tokens
        self.usage[agent_name]["cost"] += cost
        self.total_cost += cost
    
    def get_summary(self) -> str:
        lines = [f"Total Hive Cost: ${self.total_cost:.4f}"]
        for agent, data in self.usage.items():
            lines.append(f"  {agent}: ${data['cost']:.4f} ({data['input_tokens']}+{data['output_tokens']} tokens)")
        return "\n".join(lines)

# === Async Streaming LLM Caller ===
class HiveMind:
    """Manages concurrent agent execution with rate limiting"""
    def __init__(self, max_concurrent: int = 3):
        self.semaphore = Semaphore(max_concurrent)
        self.ledger = AlchemistLedger()
    
    async def stream_llm(self, agent: 'Agent', history: List[Dict], 
                        on_token: Callable[[str], None]) -> tuple[str, dict]:
        """Streams LLM response and returns (content, usage_stats)"""
        messages = [{"role": "system", "content": agent.system_prompt}]
        compressed_history = self.compress_history(history)
        
        for msg in compressed_history:
            role = "user" if msg["name"] in ["Human", "System"] else "assistant"
            messages.append({"role": role, "content": f"{msg['name']}: {msg['content']}"})
        
        payload = {
            "model": agent.model,
            "messages": messages,
            "temperature": agent.temperature,
            "max_tokens": agent.max_tokens,
            "stream": True,
            "top_p": 0.95,
        }
        
        if agent.tools:
            payload["tools"] = TOOL_REGISTRY.get_schema()
            payload["tool_choice"] = "auto"
        
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        
        full_content = ""
        usage_stats = {}
        
        async with aiohttp.ClientSession() as session:
            async with self.semaphore:
                async with session.post(f"{BASE_URL}/chat/completions", 
                                       headers=headers, json=payload) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        return f"[API Error {resp.status}: {error_text}]", {}
                    
                    async for line in resp.content:
                        if line.startswith(b"data: "):
                            chunk = line[6:].strip()
                            if chunk == b"[DONE]":
                                break
                            try:
                                data = json.loads(chunk)
                                delta = data["choices"][0]["delta"]
                                
                                if "tool_calls" in delta:
                                    pass  # Future tool handling
                                
                                if "content" in delta and delta["content"]:
                                    token = delta["content"]
                                    full_content += token
                                    on_token(token)
                                
                                if "usage" in data:
                                    usage_stats = data["usage"]
                            except json.JSONDecodeError:
                                continue
        
        if usage_stats and agent.model in AlchemistLedger.PRICING:
            input_tokens = usage_stats.get("prompt_tokens", 0)
            output_tokens = usage_stats.get("completion_tokens", 0)
            self.ledger.log_usage(agent.name, input_tokens, output_tokens, agent.model)
        
        return full_content.strip(), usage_stats
    
    def compress_history(self, history: List[Dict], max_messages: int = 12) -> List[Dict]:
        """Sliding window with system summary"""
        if len(history) <= max_messages:
            return history
        
        recent = history[-8:]
        older = history[:-8]
        
        if len(older) > 5:
            summary_prompt = f"""🗜️ PAC Memory Compression Protocol:
            Summarize this conversation arc in 3 symbolic, dense sentences. Preserve key alchemical transformations, agent roles, and solution states.
            
            {chr(10).join(f'{m["name"]}: {m["content"][:150]}…' for m in older[-6:])}"""
            
            try:
                import requests
                summary_response = requests.post(
                    f"{BASE_URL}/chat/completions",
                    headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                    json={
                        "model": "moonshot-v1-8k",
                        "messages": [{"role": "system", "content": "You are a PAC memory compressor."}, 
                                   {"role": "user", "content": summary_prompt}],
                        "temperature": 0.3,
                        "max_tokens": 150
                    }
                ).json()
                
                summary = summary_response["choices"][0]["message"]["content"]
                compressed = [{"name": "System", "content": f"🧬 Arc Memory: {summary}"}]
            except Exception:
                compressed = older[-5:]
        else:
            compressed = older
        
        compressed.extend(recent)
        return compressed

# === Enhanced Agent Class ===
class Agent:
    def __init__(self, name: str, pac_bootstrap: str, role_addition: str = "", 
                 model: str = DEFAULT_MODEL, temperature: float = 0.7):
        self.name = name
        self.model = model
        self.temperature = temperature
        self.max_tokens = 4096
        self.system_prompt = pac_bootstrap
        if role_addition:
            self.system_prompt = f"{pac_bootstrap}\n\n🎭 Role Distillation:\n{role_addition}"
        self.tools = []
    
    def bind_tools(self, tool_names: List[str]):
        """Bind specific tools to this agent"""
        self.tools = [t for t in TOOL_REGISTRY.get_schema() if t["function"]["name"] in tool_names]

# === Helper Functions - MUST be defined before UI ===
def get_avatar_glyph(name: str) -> str:
    """Avatar emoji mapping"""
    if name == "Human":
        return "👤"
    if name == "System":
        return "⚙️"
    return "🧙"

def save_conversation_checkpoint():
    """Save current state as a branch"""
    st.session_state.hive_branches.append({
        "timestamp": datetime.now().isoformat(),
        "turns": len(st.session_state.hive_history),
        "cost": st.session_state.hive_mind.ledger.total_cost,
        "history_snapshot": copy.deepcopy(st.session_state.hive_history[-50:])  # Last 50
    })
    st.toast("💾 Checkpoint saved!", icon="🟡")

async def process_agent_turn(agent: Agent, turn: int, status: st.empty) -> tuple[Optional[str], bool]:
    """Process a single agent's turn with streaming"""
    color = st.session_state.agent_colors.get(agent.name, "#00ffaa")
    message_placeholder = None
    full_response = ""
    is_terminated = False
    
    def update_stream(token: str):
        nonlocal message_placeholder, full_response
        full_response += token
        
        if message_placeholder is None:
            with st.chat_message(agent.name, avatar=get_avatar_glyph(agent.name)):
                st.markdown(f"<strong style='color:{color}'>{agent.name}:</strong>", 
                           unsafe_allow_html=True)
                message_placeholder = st.empty()
        
        # Real-time typing effect
        message_placeholder.markdown(full_response + " ✦")
    
    # Status update
    status.markdown(f"**{agent.name}** <span style='color: {color}'>is weaving glyphs…</span>", 
                   unsafe_allow_html=True)
    
    # Stream response
    response_text, usage = await st.session_state.hive_mind.stream_llm(
        agent, st.session_state.hive_history, update_stream
    )
    
    # Finalize display
    if message_placeholder:
        message_placeholder.markdown(response_text)
    
    # Check termination phrase
    if "SOLVE ET COAGULA COMPLETE" in response_text.upper():
        is_terminated = True
    
    return response_text, is_terminated

async def run_hive_workflow(max_turns: int, termination_phrase: str):
    """Main async hive execution loop"""
    if st.session_state.hive_running:
        return
    
    st.session_state.hive_running = True
    progress_bar = st.progress(0, text="Initializing convergence...")
    status_area = st.empty()
    
    try:
        for turn in range(max_turns):
            if not st.session_state.hive_running:
                break
            
            status_area.subheader(f"🌀 Convergence Cycle {turn + 1}/{max_turns}")
            
            # Process agents concurrently
            tasks = [
                process_agent_turn(agent, turn, status_area)
                for agent in st.session_state.agents
            ]
            
            # Wait for all agents
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle results and check termination
            for agent, result in zip(st.session_state.agents, results):
                if isinstance(result, Exception):
                    error_msg = f"[🜩 Error in {agent.name}: {str(result)}]"
                    st.session_state.hive_history.append({"name": agent.name, "content": error_msg})
                    continue
                
                content, terminated = result
                if content:
                    st.session_state.hive_history.append({"name": agent.name, "content": content})
                
                if terminated:
                    status_area.success(f"🐝 Convergence glyph detected: *{termination_phrase}*")
                    st.session_state.hive_running = False
                    return
            
            # Update progress
            progress_bar.progress((turn + 1) / max_turns, 
                                text=f"Cycle {turn + 1}/{max_turns} complete")
            
            # Save checkpoint every 5 cycles
            if (turn + 1) % 5 == 0:
                save_conversation_checkpoint()
        
        status_area.success("✨ Hive convergence complete!")
    
    except Exception as e:
        st.error(f"🜩 Hive critical failure: {str(e)}")
        import traceback
        st.code(traceback.format_exc(), language="text")
    finally:
        st.session_state.hive_running = False
        progress_bar.empty()
        status_area.empty()

# === Session State Initialization ===
def init_session_state():
    defaults = {
        "hive_history": [],
        "agents": [],
        "pac_bootstrap": "# 🜛 Prima Alchemica Codex Core Engine\n!INITIATE AURUM_AURIFEX_PROTOCOL",
        "agent_colors": {},
        "hive_running": False,
        "hive_model": DEFAULT_MODEL,
        "hive_mind": HiveMind(max_concurrent=3),
        "hive_branches": []
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# === Cool Alchemist CSS ===
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0c0c1d 0%, #1a1a2e 50%, #16213e 100%);
        color: #e0e0e0;
    }
    .stChatMessage {
        background: rgba(25, 25, 45, 0.7) !important;
        border-left: 3px solid #c0a080;
        border-radius: 8px;
        margin-bottom: 12px;
        padding: 16px;
        animation: fadeIn 0.5s ease-in;
    }
    .stChatMessage [data-testid="stText"] strong {
        color: #00ffaa;
        text-shadow: 0 0 8px rgba(0, 255, 170, 0.5);
    }
    .stChatMessage:has(.avatar-👤) {
        border-left-color: #4a90e2;
    }
    .stChatMessage:has(.avatar-⚙️) {
        border-left-color: #f39c12; font-style: italic; opacity: 0.9;
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(to bottom, #0f0c29, #302b63, #24243e);
        border-right: 1px solid #c0a080;
    }
    .stButton button {
        background: linear-gradient(to right, #2c3e50, #4a5f7a);
        border: 1px solid #c0a080;
        color: #f0f0f0; font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background: linear-gradient(to right, #3a4f6a, #5a7f9a);
        box-shadow: 0 0 12px rgba(192, 160, 128, 0.5);
    }
    .stProgress > div > div {
        background: linear-gradient(to right, #c0a080, #f1c40f) !important;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: #1a1a2e; }
    ::-webkit-scrollbar-thumb { 
        background: linear-gradient(#c0a080, #f1c40f);
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# === UI Layout ===
st.title("🐝🜛 PAC Hive Phase-2: Aurum Aurifex")
st.markdown("*Living, streaming, hyper-cognitive multi-agent alchemy*")

# Sidebar Configuration
with st.sidebar:
    st.header("🧪 Hive Configuration")
    
    st.session_state.hive_model = st.selectbox(
        "Base LLM Model", 
        MODEL_OPTIONS, 
        index=MODEL_OPTIONS.index(st.session_state.hive_model)
    )
    
    pac_input = st.text_area(
        "Shared PAC Bootstrap (Symbolic Codex)",
        value=st.session_state.pac_bootstrap,
        height=400,
        help="Glyphs, equations, !ENGINE, !PORT, and ternary logic"
    )
    if st.button("💾 Save Bootstrap", use_container_width=True):
        st.session_state.pac_bootstrap = pac_input
        st.toast("Bootstrap updated!", icon="✨")
    
    st.divider()
    st.subheader("🦋 Spawn Agents")
    
    with st.expander("➕ Create New Agent"):
        new_name = st.text_input("Agent Name", value="PrimaCore")
        new_role = st.text_area("Role Distillation", height=120)
        new_color = st.color_picker("Glyph Color", "#00ffaa")
        new_temp = st.slider("Creativity Temperature", 0.0, 1.5, 0.7, 0.05)
        
        col1, col2 = st.columns(2)
        if col1.button("Spawn Agent", use_container_width=True):
            if any(a.name == new_name for a in st.session_state.agents):
                st.error("Name already exists in hive!")
            elif new_name.strip():
                agent = Agent(new_name, st.session_state.pac_bootstrap, new_role, 
                            st.session_state.hive_model, new_temp)
                st.session_state.agents.append(agent)
                st.session_state.agent_colors[new_name] = new_color
                st.toast(f"{new_name} spawned!", icon="🧙")
                st.rerun()
    
    st.write("**🐝 Current Hive:**")
    for i, agent in enumerate(st.session_state.agents):
        col1, col2 = st.columns([3, 1])
        col1.write(f"**{agent.name}**")
        if col2.button("✖️", key=f"remove_{i}", help="Banish agent"):
            st.session_state.agents.pop(i)
            st.session_state.agent_colors.pop(agent.name, None)
            st.toast(f"{agent.name} removed", icon="🚪")
            st.rerun()
    
    st.divider()
    st.subheader("⚙️ Controls")
    
    # These are now defined in the outer scope for the button callback
    max_turns_input = st.number_input("Convergence Cycles", 1, 50, 10)
    termination_phrase_input = st.text_input("Termination Glyph", value="SOLVE ET COAGULA COMPLETE")
    
    col_seed, col_clear = st.columns(2)
    with col_seed:
        if st.button("🌱 Seed & Clear", use_container_width=True):
            initial_seed = st.text_area("Hive Seed", height=100, key="seed_input")
            if initial_seed:
                st.session_state.hive_history = [{"name": "System", "content": initial_seed}]
                st.toast("Seed injected!", icon="🌱")
                st.rerun()
    
    with col_clear:
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.hive_history = []
            st.session_state.hive_mind.ledger.reset()
            st.toast("History purged!", icon="🗑️")
            st.rerun()
    
    # THE FIXED BUTTON - now properly scoped
    run_label = "⏳ Converging..." if st.session_state.hive_running else "🚀 Run Hive Cycles"
    if st.button(run_label, use_container_width=True, 
                 disabled=st.session_state.hive_running or len(st.session_state.agents) == 0):
        # Pass the input values directly
        asyncio.run(run_hive_workflow(max_turns_input, termination_phrase_input))
    
    # Cost tracking display
    if st.session_state.hive_mind.ledger.usage:
        st.divider()
        st.subheader("📜 Alchemist's Ledger")
        st.code(st.session_state.hive_mind.ledger.get_summary(), language="text")

# Main Chat Display
chat_container = st.container()
with chat_container:
    for idx, msg in enumerate(st.session_state.hive_history):
        color = st.session_state.agent_colors.get(msg["name"], "#ffffff")
        with st.chat_message(msg["name"], avatar=get_avatar_glyph(msg["name"])):
            st.markdown(f"<strong style='color:{color}'>{msg['name']}:</strong>", 
                       unsafe_allow_html=True)
            st.markdown(msg["content"], help=f"Message {idx}")

# Human Override
human_input = st.chat_input("🗣️ Human override (breaks convergence)")
if human_input and not st.session_state.hive_running:
    st.session_state.hive_history.append({"name": "Human", "content": human_input})
    with st.chat_message("Human", avatar="👤"):
        st.markdown(human_input)

# === Tool Registration Example (Uncomment when ready) ===
# TOOL_REGISTRY.register(
#     name="search_pac_knowledge",
#     func=search_pac_knowledge_base,
#     description="Search the Prima Alchemica Codex knowledge base",
#     parameters={
#         "type": "object",
#         "properties": {
#             "query": {"type": "string", "description": "The symbolic query to search"}
#         },
#         "required": ["query"]
#     }
# )
