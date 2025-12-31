# pages/group_chat_ape_nap.py
# Phase-2.5: Ape Nap Mode - Full Guard System & Loop Protection

import streamlit as st
import aiohttp
import asyncio
import json
import copy
import os
import sys
import uuid
from typing import List, Dict, Callable, Optional, Any
from datetime import datetime
from asyncio import Semaphore, timeout as async_timeout
import inspect
import time
import uuid

# === CRITICAL: Import Diagnostics & Path Resolution ===
def diagnose_imports():
    """Diagnose and fix import paths for main.py"""
    current_file = os.path.abspath(__file__)
    pages_dir = os.path.dirname(current_file)
    project_root = os.path.dirname(pages_dir)
    
    # Show path info in sidebar later
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Also try venv context
    venv_path = os.environ.get("VIRTUAL_ENV")
    if venv_path:
        site_packages = os.path.join(venv_path, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages")
        if site_packages not in sys.path:
            sys.path.append(site_packages)
    
    return project_root

PROJECT_ROOT = diagnose_imports()
MAIN_SCRIPT_AVAILABLE = False

# === Robust Import with Fallback ===
try:
    from main import (
        state, container, TOOL_DISPATCHER, MOONSHOT_OFFICIAL_TOOLS,
        get_moonshot_tools, process_tool_calls, execute_moonshot_formula,
        MoonshotRateLimiter, get_memory_cache, inject_convo_uuid,
        memory_insert, advanced_memory_consolidate, tool_limiter_sync,
        Models, Config
    )
    MAIN_SCRIPT_AVAILABLE = True
    
except ImportError as e1:
    try:
        sys.path.insert(0, os.getcwd())
        from main import *
        MAIN_SCRIPT_AVAILABLE = True
        
    except ImportError as e2:
        # === EMERGENCY MOCK OBJECTS ===
        class MockState:
            def __init__(self):
                self.counter_lock = asyncio.Lock()
                self.agent_sem = asyncio.Semaphore(3)
                self.conn = None
                self.cursor = None
                self.chroma_lock = asyncio.Lock()
                self.sandbox_dir = "./sandbox"
                self.yaml_dir = "./sandbox/config"
                self.agent_dir = "./sandbox/agents"
                for d in [self.sandbox_dir, self.yaml_dir, self.agent_dir]:
                    os.makedirs(d, exist_ok=True)
        
        class MockContainer:
            def __init__(self):
                self._tools = {}
                self._official_tools = {}
            def register_tool(self, func, name=None):
                self._tools[name or func.__name__] = func
            def get_official_tools(self):
                return self._official_tools
        
        def mock_process_tool_calls(*args):
            return iter([])
        
        def mock_get_memory_cache():
            return {"lru_cache": {}, "metrics": {"total_inserts": 0, "total_retrieves": 0, "hit_rate": 1.0}}
        
        def mock_memory_insert(*args, **kwargs):
            return "Memory disabled (mock)"
        
        def mock_tool_limiter_sync():
            time.sleep(0.01)
        
        state = MockState()
        container = MockContainer()
        TOOL_DISPATCHER = {}
        MOONSHOT_OFFICIAL_TOOLS = {}
        get_moonshot_tools = lambda *a, **k: []
        process_tool_calls = mock_process_tool_calls
        execute_moonshot_formula = lambda *a, **k: {"error": "No main script"}
        MoonshotRateLimiter = type('MockLimiter', (), {})()
        get_memory_cache = mock_get_memory_cache
        inject_convo_uuid = lambda f: f
        memory_insert = mock_memory_insert
        advanced_memory_consolidate = lambda *a, **k: "Consolidation disabled"
        tool_limiter_sync = mock_tool_limiter_sync
        Models = type('Models', (), {'KIMI_K2': 'kimi-k2', 'MOONSHOT_V1_32K': 'moonshot-v1-32k'})
        Config = type('Config', (), {'DEFAULT_TOP_K': 5, 'TOOL_CALLS_PER_MIN': 10})

# === Config ===
API_KEY = os.getenv("MOONSHOT_API_KEY") or st.secrets.get("MOONSHOT_API_KEY")
if not API_KEY:
    st.error("🜩 MOONSHOT_API_KEY not found in .env or secrets! Add it and restart.")
    st.stop()

BASE_URL = "https://api.moonshot.ai/v1"
DEFAULT_MODEL = "kimi-k2-thinking"  # Best for complex reasoning
MODEL_OPTIONS = [
    "moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k",
    "kimi-k2-thinking", "kimi-k2-thinking-turbo", "kimi-k2", "kimi-latest"
]

# === Enhanced Tool Bridge with Burst Limiter ===
class ToolBridge:
    """Bridges the Hive to the main script's tool ecosystem"""
    
    def __init__(self):
        self.custom_tools = {}
        self.official_tools = {}
        self.tool_burst_limiter = asyncio.Semaphore(5)  # Max concurrent tools
        self._load_tools()
    
    def _load_tools(self):
        """Load tools from main script"""
        if not MAIN_SCRIPT_AVAILABLE:
            return
        
        for name, func in container._tools.items():
            self.custom_tools[name] = {
                'func': func,
                'schema': self._generate_schema(func),
                'type': 'custom'
            }
        
        for name, uri in MOONSHOT_OFFICIAL_TOOLS.items():
            self.official_tools[name] = {
                'uri': uri,
                'schema': self._get_official_schema(name),
                'type': 'official'
            }
    
    def _generate_schema(self, func: Callable) -> Dict:
        """Generate OpenAI-style schema for custom tools"""
        try:
            from main import generate_tool_schema
            return generate_tool_schema(func)
        except:
            sig = inspect.signature(func)
            properties = {}
            required = []
            type_map = {int: "integer", bool: "boolean", str: "string", float: "number", list: "array", dict: "object"}
            
            for param_name, param in sig.parameters.items():
                ann = param.annotation
                prop_type = "string" if ann is inspect.Parameter.empty else type_map.get(ann, "string")
                properties[param_name] = {
                    "type": prop_type,
                    "description": f"Parameter: {param_name}"
                }
                if param.default is inspect.Parameter.empty:
                    required.append(param_name)
            
            return {
                "type": "function",
                "function": {
                    "name": func.__name__,
                    "description": inspect.getdoc(func) or "No description",
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required
                    }
                }
            }
    
    def _get_official_schema(self, name: str) -> Dict:
        """Get hardcoded schema for official tools"""
        schemas = {
            "moonshot-web-search": {
                "type": "function",
                "function": {
                    "name": "moonshot-web-search",
                    "description": "Search the web for current information",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"]
                    }
                }
            },
            "moonshot-calculate": {
                "type": "function",
                "function": {
                    "name": "moonshot-calculate",
                    "description": "Perform mathematical calculations",
                    "parameters": {
                        "type": "object",
                        "properties": {"expression": {"type": "string"}},
                        "required": ["expression"]
                    }
                }
            },
            "moonshot-url-extract": {
                "type": "function",
                "function": {
                    "name": "moonshot-url-extract",
                    "description": "Extract content from a URL",
                    "parameters": {
                        "type": "object",
                        "properties": {"url": {"type": "string"}},
                        "required": ["url"]
                    }
                }
            }
        }
        return schemas.get(name, {})
    
    def get_all_tools(self, enable_custom: bool = True, enable_official: bool = True) -> List[Dict]:
        """Get all tool schemas for LLM - NO WRAPPERS"""
        tools = []
        if enable_official:
            tools.extend([tool['schema'] for tool in self.official_tools.values()])
        if enable_custom:
            tools.extend([tool['schema'] for tool in self.custom_tools.values()])
        return tools
    
    async def execute_tool(self, name: str, arguments: Dict, convo_uuid: str) -> str:
        """Execute a tool with rate limiting and error handling"""
        try:
            # Rate limit burst
            async with self.tool_burst_limiter:
                await asyncio.to_thread(tool_limiter_sync)
                
                if not convo_uuid:
                    convo_uuid = st.session_state.get("current_convo_uuid", str(uuid.uuid4()))
                
                func = None
                tool_type = None
                
                if name in self.custom_tools:
                    func = self.custom_tools[name]['func']
                    tool_type = 'custom'
                elif name in self.official_tools:
                    tool_type = 'official'
                else:
                    return f"Error: Tool '{name}' not found"
                
                if tool_type == 'custom':
                    sig = inspect.signature(func)
                    if 'convo_uuid' in sig.parameters and 'convo_uuid' not in arguments:
                        arguments['convo_uuid'] = convo_uuid
                    
                    loop = asyncio.get_event_loop()
                    # Add timeout to prevent stuck tools
                    async with async_timeout(60):
                        result = await loop.run_in_executor(None, func, **arguments)
                    return str(result) if result is not None else "Tool returned None"
                
                elif tool_type == 'official':
                    async with async_timeout(30):
                        result = await asyncio.to_thread(
                            execute_moonshot_formula,
                            self.official_tools[name]['uri'],
                            name,
                            arguments,
                            API_KEY
                        )
                    return json.dumps(result) if isinstance(result, dict) else str(result)
            
        except asyncio.TimeoutError:
            return f"Tool timeout: {name} exceeded 60s"
        except Exception as e:
            return f"Tool error: {str(e)}"

tool_bridge = ToolBridge()

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
        
        # 💸 COST BOMB PROTECTION
        max_cost = st.session_state.get("max_cost_per_run", 10.0)
        if self.total_cost > max_cost:
            st.error(f"💸 Cost limit ${max_cost} exceeded! Hive halted.")
            st.session_state.hive_running = False

# === Enhanced HiveMind with Timeouts & Safety ===
class HiveMind:
    """Manages concurrent agent execution with rate limiting, timeouts, and safety"""
    def __init__(self, max_concurrent: int = 3):
        self.semaphore = Semaphore(max_concurrent)
        self.ledger = AlchemistLedger()
        self.api_limiter = MoonshotRateLimiter()
    
    async def stream_llm(self, agent: 'Agent', history: List[Dict], 
                        on_token: Callable[[str], None],
                        on_tool_call: Optional[Callable[[str], None]] = None) -> tuple[str, dict, List[Dict]]:
        """Streams LLM response with tool support"""
        messages = [{"role": "system", "content": agent.system_prompt}]
        
        # 🗜️ MEMORY PRESSURE VALVE - more aggressive
        compressed_history = self.compress_history(history, max_messages=20)
        
        for msg in compressed_history:
            role = "user" if msg["name"] in ["Human", "System"] else "assistant"
            messages.append({"role": role, "content": f"{msg['name']}: {msg['content']}"})
        
        # 🐕 WATCHDOG TIMER CHECK
        if time.time() - st.session_state.get("last_agent_activity", time.time()) > st.session_state.get("watchdog_timer", 3600):
            return "[WATCHDOG: Run exceeded time limit]", {}, []
        
        tools = tool_bridge.get_all_tools(
            enable_custom=agent.enable_custom_tools,
            enable_official=agent.enable_official_tools
        )
        
        payload = {
            "model": agent.model,
            "messages": messages,
            "temperature": agent.temperature,
            "max_tokens": agent.max_tokens,
            "stream": True,
            "top_p": 0.95,
        }
        
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        
        full_content = ""
        usage_stats = {}
        tool_calls_buffer = []
        
        async with aiohttp.ClientSession() as session:
            async with self.semaphore:
                await asyncio.to_thread(self.api_limiter, sum(len(m["content"]) for m in messages) // 4)
                
                # 🔒 Add API call timeout
                try:
                    async with async_timeout(120):
                        async with session.post(f"{BASE_URL}/chat/completions", 
                                               headers=headers, json=payload) as resp:
                            if resp.status != 200:
                                error_text = await resp.text()
                                return f"[API Error {resp.status}: {error_text}]", {}, []
                            
                            async for line in resp.content:
                                if line.startswith(b"data: "):
                                    chunk = line[6:].strip()
                                    if chunk == b"[DONE]":
                                        break
                                    try:
                                        data = json.loads(chunk)
                                        delta = data["choices"][0]["delta"]
                                        
                                        if "tool_calls" in delta and delta["tool_calls"]:
                                            for tool_call in delta["tool_calls"]:
                                                idx = tool_call.get("index", 0)
                                                if idx >= len(tool_calls_buffer):
                                                    tool_calls_buffer.append({
                                                        "id": tool_call.get("id", ""),
                                                        "name": tool_call["function"].get("name", ""),
                                                        "args": tool_call["function"].get("arguments", "")
                                                    })
                                                else:
                                                    if tool_call["function"].get("arguments"):
                                                        tool_calls_buffer[idx]["args"] += tool_call["function"]["arguments"]
                                        
                                        if "content" in delta and delta["content"]:
                                            token = delta["content"]
                                            full_content += token
                                            on_token(token)
                                            st.session_state.last_agent_activity = time.time()
                                        
                                        if "usage" in data:
                                            usage_stats = data["usage"]
                                    except json.JSONDecodeError:
                                        continue
                except asyncio.TimeoutError:
                    return "[API Timeout: 120s exceeded]", {}, []
        
        if usage_stats and agent.model in AlchemistLedger.PRICING:
            input_tokens = usage_stats.get("prompt_tokens", 0)
            output_tokens = usage_stats.get("completion_tokens", 0)
            self.ledger.log_usage(agent.name, input_tokens, output_tokens, agent.model)
        
        if tool_calls_buffer and on_tool_call:
            on_tool_call(f"🔧 Tool calls: {[t.get('name', 'unknown') for t in tool_calls_buffer]}")
        
        return full_content.strip(), usage_stats, tool_calls_buffer
    
    def compress_history(self, history: List[Dict], max_messages: int = 20) -> List[Dict]:
        """Sliding window with system summary - AGGRESSIVE for nap mode"""
        if len(history) <= max_messages:
            return history
        
        recent = history[-10:]  # Keep last 10
        older = history[:-10]
        
        if len(older) > 5:
            summary_prompt = f"""🗜️ PAC MEMORY COMPRESSION (AGGRESSIVE):
            Summarize this conversation arc in 2 symbolic sentences max. Preserve only key tool results and agent roles.
            
            {chr(10).join(f'{m["name"]}: {m["content"][:100]}…' for m in older[-5:])}"""
            
            try:
                import requests
                summary_response = requests.post(
                    f"{BASE_URL}/chat/completions",
                    headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                    json={
                        "model": "moonshot-v1-8k",
                        "messages": [{"role": "system", "content": "You are a brutal PAC memory compressor."}, 
                                   {"role": "user", "content": summary_prompt}],
                        "temperature": 0.1,
                        "max_tokens": 100
                    }
                ).json()
                
                summary = summary_response["choices"][0]["message"]["content"]
                compressed = [{"name": "System", "content": f"🗜️ MEM: {summary}"}]
            except Exception:
                compressed = older[-3:]
        else:
            compressed = older
        
        compressed.extend(recent)
        return compressed

# === Enhanced Agent Class with Tool Restrictions ===
class Agent:
    def __init__(self, name: str, pac_bootstrap: str, role_addition: str = "", 
                 model: str = DEFAULT_MODEL, temperature: float = 0.7):
        self.name = name
        self.model = model
        self.temperature = temperature
        self.max_tokens = 4096
        self.system_prompt = f"{pac_bootstrap}\n\n🤖 Agent: {name}\n{role_addition}"
        self.enable_custom_tools = True
        self.enable_official_tools = True
        self.allowed_tools = []  # Empty = all allowed
        self.tool_chain_depth = 1  # Prevent recursive tool spawning
    
    def bind_tools(self, tool_names: List[str]):
        """Bind specific tools to this agent"""
        self.allowed_tools = tool_names
    
    def should_use_tool(self, tool_name: str) -> bool:
        """Check if agent is allowed to use this tool"""
        if not self.allowed_tools:
            return True
        
        # 🚫 BLOCK DANGEROUS TOOLS IN NAP MODE
        dangerous_tools = ["agent_spawn", "pip_install", "shell_exec", "isolated_subprocess"]
        if tool_name in dangerous_tools and tool_name not in self.allowed_tools:
            return False
        
        return tool_name in self.allowed_tools

# === Session State with Ape Nap Config ===
def init_session_state():
    defaults = {
        "hive_history": [],
        "agents": [],
        "pac_bootstrap": "# 🜛 Prima Alchemica Codex Core Engine\n!INITIATE AURUM_AURIFEX_PROTOCOL\n!TOOL_ACCESS_ENABLED\n!APE_NAP_MODE",
        "agent_colors": {},
        "hive_running": False,
        "hive_model": DEFAULT_MODEL,
        "hive_mind": HiveMind(max_concurrent=3),
        "hive_branches": [],
        "current_convo_uuid": str(uuid.uuid4()),
        "tool_calls_this_run": 0,
        "max_tool_calls_per_run": 500,  # Higher for nap
        "max_cost_per_run": 10.0,  # $10 max
        "watchdog_timer": 3600,  # 1 hour
        "last_agent_activity": time.time(),
        "emergency_stop": False,
        "debug_mode": False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# === CSS ===
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0c0c1d 0%, #1a1a2e 50%, #16213e 100%); color: #e0e0e0; }
    .stChatMessage { background: rgba(25, 25, 45, 0.7); border-left: 3px solid #c0a080; border-radius: 8px; margin-bottom: 12px; padding: 16px; animation: fadeIn 0.5s ease-in; }
    .stChatMessage strong { color: #00ffaa; text-shadow: 0 0 8px rgba(0, 255, 170, 0.5); }
    .tool-result { background: rgba(0, 255, 170, 0.1); border: 1px solid #00ffaa; border-radius: 6px; padding: 8px; margin: 8px 0; font-family: 'Courier New', monospace; font-size: 0.9em; }
    div[data-testid="stSidebar"] { background: linear-gradient(to bottom, #0f0c29, #302b63, #24243e); border-right: 1px solid #c0a080; }
    .stButton button { background: linear-gradient(to right, #2c3e50, #4a5f7a); border: 1px solid #c0a080; color: #f0f0f0; font-weight: bold; transition: all 0.3s ease; }
    .stButton button:hover { background: linear-gradient(to right, #3a4f6a, #5a7f9a); box-shadow: 0 0 12px rgba(192, 160, 128, 0.5); }
    .warning-box { background: rgba(255, 0, 0, 0.1); border: 2px solid #ff0000; border-radius: 8px; padding: 12px; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

# === Helpers ===
def get_avatar_glyph(name: str) -> str:
    if name == "Human": return "👤"
    if name == "System": return "⚙️"
    return "🧙"

def save_conversation_checkpoint(turn: int = 0):
    """Save full state for recovery"""
    checkpoint = {
        "turn": turn,
        "timestamp": datetime.now().isoformat(),
        "history": st.session_state.hive_history[-50:],
        "cost": st.session_state.hive_mind.ledger.total_cost,
        "tool_calls": st.session_state.tool_calls_this_run,
        "agents": [{"name": a.name, "model": a.model} for a in st.session_state.agents]
    }
    st.session_state.hive_branches.append(checkpoint)
    # Save to disk every 10 turns
    if turn % 10 == 0:
        try:
            os.makedirs("./sandbox/hive_checkpoints", exist_ok=True)
            with open(f"./sandbox/hive_checkpoints/checkpoint_turn_{turn}.json", "w") as f:
                json.dump(checkpoint, f, indent=2)
        except: pass

# === Tool Deduplication & Execution ===
async def execute_agent_tools(agent: Agent, tool_calls: List[Dict], convo_uuid: str) -> List[str]:
    """Execute tools with deduplication and safety checks"""
    results = []
    
    # 🔍 DEDUPLICATION: Check recent tool calls
    recent_tools = []
    for msg in st.session_state.hive_history[-10:]:
        if "🜛 **Tool Results:**" in msg["content"]:
            # Extract tool names from recent history
            import re
            matches = re.findall(r'Tool call: `(\w+)`', msg["content"])
            recent_tools.extend(matches)
    
    for tool_call in tool_calls:
        tool_name = tool_call.get("name", "")
        arguments = json.loads(tool_call.get("args", "{}"))
        
        # Skip duplicates
        if tool_name in recent_tools:
            results.append(f"⏭️ Skipped duplicate: {tool_name}")
            continue
        
        if not agent.should_use_tool(tool_name):
            results.append(f"🚫 Agent {agent.name} blocked from: {tool_name}")
            continue
        
        result = await tool_bridge.execute_tool(tool_name, arguments, convo_uuid)
        results.append(result)
        
        # Memory log
        try:
            memory_insert(
                f"tool_{tool_name}_{uuid.uuid4().hex[:8]}",
                {
                    "agent": agent.name,
                    "tool": tool_name,
                    "arguments": arguments,
                    "result": result[:500],
                    "timestamp": datetime.now().isoformat()
                },
                convo_uuid=convo_uuid
            )
        except: pass
        
        st.session_state.tool_calls_this_run += 1
        
        if st.session_state.tool_calls_this_run >= st.session_state.max_tool_calls_per_run:
            results.append("💥 MAX TOOL CALLS REACHED - HALTING")
            st.session_state.hive_running = False
            break
        
        # 💸 Emergency cost check
        if st.session_state.hive_mind.ledger.total_cost > st.session_state.max_cost_per_run:
            st.error("COST LIMIT REACHED")
            st.session_state.hive_running = False
            break
    
    return results

# === Core Async Workflow with Safety ===
async def process_agent_turn(agent: Agent, turn: int, status: st.empty) -> tuple[Optional[str], bool]:
    """Process agent turn with streaming, tools, and safety"""
    color = st.session_state.agent_colors.get(agent.name, "#00ffaa")
    message_placeholder = None
    full_response = ""
    is_terminated = False
    
    def update_stream(token: str):
        nonlocal message_placeholder, full_response
        full_response += token
        if message_placeholder is None:
            with st.chat_message(agent.name, avatar=get_avatar_glyph(agent.name)):
                st.markdown(f"<strong style='color:{color}'>{agent.name}:</strong>", unsafe_allow_html=True)
                message_placeholder = st.empty()
        message_placeholder.markdown(full_response + " ✦")
        st.session_state.last_agent_activity = time.time()
    
    def on_tool_call(notification: str):
        status.markdown(f"**{agent.name}** <span style='color: #f1c40f'>{notification}</span>", unsafe_allow_html=True)
    
    status.markdown(f"**{agent.name}** <span style='color: {color}'>weaving…</span>", unsafe_allow_html=True)
    
    # 🐕 Watchdog check before starting
    if time.time() - st.session_state.last_agent_activity > st.session_state.watchdog_timer:
        st.warning("🐕 Watchdog: Time limit exceeded!")
        st.session_state.hive_running = False
        return "WATCHDOG TIMEOUT", True
    
    response_text, usage, tool_calls = await st.session_state.hive_mind.stream_llm(
        agent, st.session_state.hive_history, update_stream, on_tool_call
    )
    
    if tool_calls and st.session_state.tool_calls_this_run < st.session_state.max_tool_calls_per_run:
        status.markdown(f"**{agent.name}** <span style='color: #f1c40f'>invoking tools…</span>", unsafe_allow_html=True)
        tool_results = await execute_agent_tools(agent, tool_calls, st.session_state.current_convo_uuid)
        
        if tool_results:
            response_text += "\n\n🜛 **Tool Results:**\n" + "\n".join(f"- {r[:200]}" for r in tool_results)
    
    if message_placeholder:
        message_placeholder.markdown(response_text)
    
    if "SOLVE ET COAGULA COMPLETE" in response_text.upper():
        is_terminated = True
    
    return response_text, is_terminated

async def run_hive_workflow(max_turns: int, termination_phrase: str):
    """Main async hive execution loop with safety"""
    if st.session_state.hive_running:
        return
    
    st.session_state.hive_running = True
    st.session_state.tool_calls_this_run = 0
    st.session_state.last_agent_activity = time.time()
    
    progress_bar = st.progress(0, text="Initializing convergence...")
    status_area = st.empty()
    
    try:
        for turn in range(max_turns):
            if not st.session_state.hive_running or st.session_state.emergency_stop:
                break
            
            # 🐕 Watchdog check
            if time.time() - st.session_state.last_agent_activity > st.session_state.watchdog_timer:
                status_area.error(f"🐕 WATCHDOG: Run exceeded {st.session_state.watchdog_timer}s limit!")
                break
            
            status_area.subheader(f"🌀 Cycle {turn + 1}/{max_turns} | Cost: ${st.session_state.hive_mind.ledger.total_cost:.2f}")
            
            tasks = [
                process_agent_turn(agent, turn, status_area)
                for agent in st.session_state.agents
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for agent, result in zip(st.session_state.agents, results):
                if isinstance(result, Exception):
                    error_msg = f"[🜩 Error: {str(result)}]"
                    st.session_state.hive_history.append({"name": agent.name, "content": error_msg})
                    continue
                
                content, terminated = result
                if content:
                    st.session_state.hive_history.append({"name": agent.name, "content": content})
                
                if terminated or st.session_state.emergency_stop:
                    status_area.success(f"🐝 Convergence reached!")
                    st.session_state.hive_running = False
                    return
            
            # 💾 AUTO-SAVE CHECKPOINT
            if (turn + 1) % 10 == 0:
                save_conversation_checkpoint(turn)
            
            progress_bar.progress((turn + 1) / max_turns, 
                                text=f"Cycle {turn + 1} | Tools: {st.session_state.tool_calls_this_run} | Cost: ${st.session_state.hive_mind.ledger.total_cost:.2f}")
        
        status_area.success("✨ Hive convergence complete!")
    
    except Exception as e:
        st.error(f"🜩 Hive failure: {str(e)}")
        import traceback
        st.code(traceback.format_exc(), language="text")
    finally:
        st.session_state.hive_running = False
        progress_bar.empty()
        status_area.empty()

# === UI Layout ===
st.title("🐝🜛 PAC Hive Phase-2.5: APE NAP MODE")
st.markdown("*Autonomous multi-agent alchemy with full guard system*")

# Warning box if running hot
if st.session_state.hive_running and st.session_state.tool_calls_this_run > 20:
    st.markdown('<div class="warning-box">🔥 HIGH ACTIVITY - Monitoring enabled</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("🧪 Hive Configuration")
    
    st.session_state.hive_model = st.selectbox(
        "Base LLM Model", 
        MODEL_OPTIONS, 
        index=MODEL_OPTIONS.index(st.session_state.hive_model)
    )
    
    st.divider()
    st.subheader("🔧 Tool & Safety Config")
    enable_tools = st.checkbox("Enable Tool Access", value=True)
    enable_custom = st.checkbox("Custom Tools", value=False, disabled=not enable_tools)  # Disable by default for nap
    enable_official = st.checkbox("Official Tools", value=True, disabled=not enable_tools)
    
    st.number_input("Max Tool Calls per Run", 1, 1000, 500, key="max_tool_calls_per_run")
    st.number_input("Max Cost per Run ($)", 0.5, 50.0, 10.0, key="max_cost_per_run")
    st.number_input("Watchdog Timer (seconds)", 300, 7200, 3600, key="watchdog_timer")
    
    if enable_tools:
        tool_list = []
        if enable_custom:
            tool_list.extend(list(tool_bridge.custom_tools.keys()))
        if enable_official:
            tool_list.extend(list(tool_bridge.official_tools.keys()))
        
        st.multiselect("Agent Tool Access (empty = all)", tool_list, default=[], key="global_allowed_tools")
    
    # Debug mode
    st.checkbox("🔍 Debug Mode", value=False, key="debug_mode")
    
    pac_input = st.text_area(
        "PAC Bootstrap (Add !TOKEN_BUDGET if needed)",
        value=st.session_state.pac_bootstrap,
        height=400
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
        new_temp = st.slider("Creativity", 0.0, 1.5, 0.7, 0.05)
        
        if enable_tools:
            st.multiselect(f"{new_name}'s Tools", tool_list, default=[], key=f"agent_tools_{new_name}")
        
        col1, col2 = st.columns(2)
        if col1.button("Spawn Agent", use_container_width=True):
            if any(a.name == new_name for a in st.session_state.agents):
                st.error("Name exists!")
            elif new_name.strip():
                agent = Agent(new_name, st.session_state.pac_bootstrap, new_role, st.session_state.hive_model, new_temp)
                if enable_tools:
                    agent_tools = st.session_state.get(f"agent_tools_{new_name}", [])
                    if agent_tools:
                        agent.allowed_tools = agent_tools
                st.session_state.agents.append(agent)
                st.session_state.agent_colors[new_name] = new_color
                st.toast(f"{new_name} spawned!", icon="🧙")
                st.rerun()
    
    st.write("**🐝 Current Hive:**")
    for i, agent in enumerate(st.session_state.agents):
        col1, col2 = st.columns([3, 1])
        col1.write(f"**{agent.name}**")
        col2.write(f"🛠️ {len(agent.allowed_tools) if agent.allowed_tools else 'All'}")
        if col2.button("✖️", key=f"remove_{i}", help="Banish agent"):
            st.session_state.agents.pop(i)
            st.session_state.agent_colors.pop(agent.name, None)
            st.rerun()
    
    st.divider()
    st.subheader("⚙️ Controls & Safety")
    
    max_turns_input = st.number_input("Convergence Cycles", 1, 200, 50)
    termination_phrase_input = st.text_input("Termination Glyph", value="SOLVE ET COAGULA COMPLETE")
    
    # Emergency stop button
    if st.session_state.hive_running and st.button("🚨 EMERGENCY STOP", use_container_width=True):
        st.session_state.emergency_stop = True
        st.session_state.hive_running = False
        st.warning("🚨 EMERGENCY STOP ACTIVATED")
        st.rerun()
    
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
            st.session_state.tool_calls_this_run = 0
            st.session_state.emergency_stop = False
            st.toast("History purged!", icon="🗑️")
            st.rerun()
    
    run_label = "⏳ CONVERGING..." if st.session_state.hive_running else "🚀 RUN HIVE (NAP MODE)"
    if st.button(run_label, use_container_width=True, 
                 disabled=st.session_state.hive_running or len(st.session_state.agents) == 0):
        st.session_state.emergency_stop = False
        asyncio.run(run_hive_workflow(max_turns_input, termination_phrase_input))
    
    # Status indicators
    if st.session_state.hive_running:
        st.metric("Tool Calls", st.session_state.tool_calls_this_run)
        st.metric("Cost", f"${st.session_state.hive_mind.ledger.total_cost:.2f}")
    
    if st.session_state.hive_mind.ledger.usage:
        st.divider()
        st.subheader("📜 Alchemist's Ledger")
        ledger_text = st.session_state.hive_mind.ledger.get_summary()
        ledger_text += f"\n🔧 Tool Calls: {st.session_state.tool_calls_this_run}"
        st.code(ledger_text, language="text")
    
    if st.session_state.hive_branches:
        st.divider()
        st.subheader("🔀 Timeline Branches")
        for idx, branch in enumerate(st.session_state.hive_branches[-3:]):
            st.caption(f"Branch {idx}: {branch['timestamp'][:19]} | {branch['turns']} turns | ${branch.get('cost', 0):.2f}")

# Main Chat Display
chat_container = st.container()
with chat_container:
    for idx, msg in enumerate(st.session_state.hive_history):
        color = st.session_state.agent_colors.get(msg["name"], "#ffffff")
        with st.chat_message(msg["name"], avatar=get_avatar_glyph(msg["name"])):
            st.markdown(f"<strong style='color:{color}'>{msg['name']}:</strong>", unsafe_allow_html=True)
            
            if "🜛 **Tool Results:**" in msg["content"]:
                parts = msg["content"].split("🜛 **Tool Results:**")
                st.markdown(parts[0])
                if len(parts) > 1:
                    st.markdown('<div class="tool-result">🜛 Tool Results:' + parts[1] + '</div>', unsafe_allow_html=True)
            else:
                st.markdown(msg["content"], help=f"Message {idx}")

# Human Override
human_input = st.chat_input("🗣️ Human override (breaks convergence)")
if human_input and not st.session_state.hive_running:
    st.session_state.hive_history.append({"name": "Human", "content": human_input})
    with st.chat_message("Human", avatar="👤"):
        st.markdown(human_input)
