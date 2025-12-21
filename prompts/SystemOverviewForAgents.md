# Apex Vivum: Moonshot-Powered Agent Runtime - System Overview

**Version**: 1.1 | **Backend**: Moonshot AI | **Runtime**: Python 3.13.5/Streamlit

## 1. Available Models

| Model Name | Context | Best For |
|------------|---------|----------|
| `kimi-k2-thinking` | 256K | Complex reasoning, tool chains, council debates |
| `kimi-latest` | 200K | General tasks, balanced speed/quality |
| `kimi-k2-thinking-turbo` | 256K | Faster reasoning with slight quality tradeoff |
| `kimi-k2` | 200K | Standard generation without deep reasoning |
| `moonshot-v1-8k` | 8K | Simple queries, low latency |
| `moonshot-v1-32k` | 32K | Medium complexity |
| `moonshot-v1-128k` | 128K | Long document analysis |

**Fallback Chain**: Preferred → k2-thinking → v1-32k → v1-8k

## 2. Tool System

### 2.1 Official Moonshot Tools (API-based)
```python
# No local execution - delegates to Moonshot Formula API
moonshot-web-search(query: str)
moonshot-calculate(expression: str)
moonshot-url-extract(url: str)
```

### 2.2 Custom Sandbox Tools (Local Execution)

**File Operations** (sandboxed to `./sandbox/`)
- `fs_read_file(file_path: str) → str` - Auto-creates .yaml/.lattice files
- `fs_write_file(file_path: str, content: str) → str` - HTML-unescapes content
- `fs_list_files(dir_path: str = "") → str`
- `fs_mkdir(dir_path: str) → str`
- **Security**: Paths outside `./sandbox/` rejected; uses `pathlib.Path.relative_to('.')`

**Code Execution**
- `code_execution(code: str, venv_path: str = None) → str`
  - RestrictedPython + safe builtins (`print`, `len`, `range`, etc.)
  - Optional venv isolation
  - Banned: `open`, `exec`, `eval`, `subprocess` calls
- `code_lint(language: str, code: str) → str` - Python/JS/JSON/YAML/SQL/XML/HTML
- `restricted_exec(code: str, level: str = "basic") → str` - AST-level policy enforcement

**Memory System** (Hybrid SQL + ChromaDB)
- `memory_insert(mem_key: str, mem_value: dict, convo_uuid: str) → str`
- `memory_query(mem_key: str = None, limit: int = 5, uuid_list: list = None) → str`
- `advanced_memory_consolidate(mem_key: str, interaction_data: dict) → str` - Auto-embeds summaries
- `advanced_memory_retrieve(query: str, top_k: int = 5) → str` - Vector search with salience boosting
- `advanced_memory_prune(convo_uuid: str) → str` - Decay, dedup, size-based cleanup

**Agents & Orchestration**
- `agent_spawn(sub_agent_type: str, task: str, auto_poll: bool = False) → str`
  - Max task length: 2000 chars
  - Returns: Agent ID, poll via `agent_{id}_complete`
  - Auto-polls 30× at 5s intervals if enabled
- `socratic_api_council(branches: list, rounds: int = 3, personas: list = None) → str`
  - Default personas: Planner, Critic, Executor, Summarizer, Verifier, Moderator
  - Persists to `council_result` memory key

**DevOps & Shell**
- `shell_exec(command: str) → str` - Whitelist: `ls grep sed awk cat echo wc tail head cp mv rm mkdir rmdir touch python pip`
  - Destructive commands (rm/rmdir) require confirmation
  - Args validated against `[;&|><$*?[]../]`
- `git_ops(operation: str, repo_path: str, message: str = None) → str` - init/commit/branch/diff/status
- `db_query(db_path: str, query: str, params: list = None) → str` - SQLite only; blocks DROP/DELETE/ALTER
- `venv_create(env_name: str) → str` - Creates in `./sandbox/`
- `pip_install(venv_path: str, packages: list, upgrade: bool = False) → str` - Whitelist enforced

**Vector & Search**
- `generate_embedding(text: str) → str` - Uses `all-mpnet-base-v2`; chunks >10K chars
- `vector_search(query_embedding: list, top_k: int = 5, threshold: float = 0.6) → str`
- `keyword_search(query: str, top_k: int = 5, uuid_list: list = None) → str`
- `chat_log_analyze_embed(convo_id: int, criteria: str, summarize: bool = True) → str`

**YAML Management**
- `yaml_retrieve(query: str = None, top_k: int = 5, filename: str = None) → str`
- `yaml_refresh(filename: str = None) → str` - Rebuilds embeddings

**Visualization**
- `viz_memory_lattice(convo_uuid: str, top_k: int = 20, sim_threshold: float = 0.6, plot_type: str = "both") → str`
- `visualize_got(got_data: str, format: str = "both", detail_level: int = 2) → str` - NetworkX + Plotly/Matplotlib

## 3. Memory Architecture

### 3.1 Layers
1. **SQLite**: Primary storage (`memory` table: `uuid`, `mem_key`, `mem_value`, `timestamp`, `salience`, `parent_id`)
2. **ChromaDB**: Vector index for semantic search (`memory_vectors` collection)
3. **LRU Cache**: Session-level hot cache (500 entry cap)
4. **YAML Cache**: Config file embeddings (`./sandbox/config/*.yaml`)

### 3.2 Salience Mechanics
- Initial salience: `1.0` (default)
- Post-retrieval boost: `+0.1` (capped at `1.0`)
- Weekly decay: `×0.99` for entries >7 days old
- Pruning thresholds: `<0.1` (delete), `<0.5` (size-based eviction), `<0.4` (LRU eviction)

### 3.3 Auto-Pruning
Triggers every 50 **inserts** (not calls):
- Decay salience
- Delete low-salience entries
- Cap at 1000 entries/convo (remove oldest low-salience)
- Dedup by summary hash
- Clean `agent_*` keys >7 days

### 3.4 Example Memory Schema
```python
{
  "summary": "Concise description",
  "details": "Full context/JSON",
  "tags": ["projectX", "bug"],
  "domain": "coding",
  "timestamp": "2025-12-21T10:30:00",
  "salience": 0.95
}
```

## 4. Agent Framework

### 4.1 Spawning
```python
agent_spawn(
  sub_agent_type="quantum_sim", 
  task="Run Qiskit sim for 3 qubits",
  auto_poll=False  # Set True for blocking wait
)
# Returns: "Agent 'quantum_sim' spawned (ID: quantum_sim_a1b2c3d4). 
#          Poll 'agent_quantum_sim_a1b2c3d4_complete' for results."
```

### 4.2 Result Retrieval
- Poll memory key: `agent_{agent_id}_complete`
- Status key: `agent_{agent_id}_status`
- Results auto-persisted to `./sandbox/agents/{agent_id}/result.json`
- Fleet UI in sidebar shows active agents with Kill buttons

### 4.3 Socratic Council
```python
socratic_api_council(
  branches=[
    "Option A: Use PyTorch",
    "Option B: Use JAX"
  ],
  rounds=3,
  personas=["Performance Critic", "MLOps Guru"]
)
# Persists final consensus to memory; auto-embedded for retrieval
```

## 5. Configuration & Limits (Config Enum)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `DEFAULT_TOP_K` | 5 | Default retrieval count |
| `CACHE_TTL_MIN` | 15 | Tool cache lifetime |
| `AGENT_MAX_CONCURRENT` | 5 | Async agent semaphore |
| `PRUNE_FREQUENCY` | 50 | Inserts between prunes |
| `SIM_THRESHOLD` | 0.6 | Min similarity for edges |
| `MAX_TASK_LEN` | 2000 | Agent task char limit |
| `STABILITY_PENALTY` | 0.05 | Score deduction on error |
| `MAX_ITERATIONS` | 100 | Max tool call loops |
| `TOOL_CALL_LIMIT` | 200 | Per-conversation cap |
| `API_CALLS_PER_MIN` | 100 | RPM limit |
| `API_TOKENS_PER_MIN` | 1,000,000 | TPM limit |
| `API_CONCURRENT_REQUESTS` | 5 | Max parallel calls |

**Rate Limiter**: Multi-dimensional sync/async semaphores; auto-backoff on 429

## 6. Sandbox Boundaries

- **Root**: `./sandbox/` (all file ops constrained here)
- **Subdirs**: `./sandbox/db/`, `./sandbox/agents/`, `./sandbox/config/`, `./sandbox/viz/`
- **Code Execution**: RestrictedPython; no network/file access unless venv path specified
- **Shell**: Whitelist-only; destructive commands need confirmation
- **PIP**: Whitelist of 30+ science/coding packages; `--no-deps` first, fallback on error
- **API Mocking**: `api_simulate(mock=True)` by default; real calls require URL whitelist

## 7. State Management

- **Session State**: `st.session_state` for messages, UUIDs, counters, caches
- **Thread Safety**: `state.session_lock`, `state.counter_lock`, `state.chroma_lock`, `state.agent_lock`
- **Counters**: `main_count`, `tool_count`, `council_count` (persist across reruns)
- **Stability Score**: 0.0-1.0; +0.01 per success, -0.05 per error; affects retry logic

## 8. Usage Patterns

### 8.1 Complex Task Orchestration
```python
# 1. Store context
memory_insert("project_goal", {"summary": "Build quantum algo", "salience": 1.0})

# 2. Spawn agents
agent_spawn("researcher", "Literature on Shor's algorithm", auto_poll=False)

# 3. Council consensus
branches = [
  "Use Qiskit for IBM compatibility",
  "Use Cirq for Google's sim"
]
socratic_api_council(branches, rounds=3)

# 4. Retrieve & synthesize
memories = advanced_memory_retrieve("quantum framework choice", top_k=3)
```

### 8.2 Debugging Workflow
```python
# Check system health
diagnose()  # Returns stability, cache sizes

# View recent tool logs (via sidebar button)
# Clear cache if odd behavior
```

### 8.3 Memory-Driven Development
```python
# Store API response
advanced_memory_consolidate("api_v1_response", {
  "summary": "User data fetched",
  "details": {"users": [...], "count": 42}
})

# Later retrieval
query_emb = json.loads(generate_embedding("user count"))
vector_search(query_emb, top_k=1)
```

## 9. Environment & Paths

- `.env` variable: `MOONSHOT_API_KEY` (required)
- Prompts: `./prompts/*.txt` (auto-loaded on startup)
- Logs: `app.log` (rotation not implemented; manual prune)
- DB: `./sandbox/db/chatapp.db` (SQLite + WAL mode)
- Chroma: `./sandbox/db/chroma_db` (persistent vectors)
- Profile: Set `PROFILE_MODE=1` for cProfile stats

## 10. Best Practices for Agents

1. **Tool Chaining**: Use `socratic_api_council` for multi-step decisions; limit to 3-5 rounds
2. **Memory Keys**: Namespaced patterns (`projectX_setup`, `agent_123_result`) improve retrieval
3. **Salience Tuning**: Set `salience: 0.9` for critical info; `0.3` for transient data
4. **Error Recovery**: Always wrap tool calls in `safe_call()` pattern; check stability score
5. **Rate Awareness**: Tool calls limited to 200/convo; expensive ops (embeddings) cached
6. **Sandbox Hygiene**: Use relative paths; avoid absolute paths; cleanup old venvs
7. **Agent Polling**: Default `auto_poll=False`; use manual polling for long tasks (>60s)
8. **Visualization**: Trigger `viz_memory_lattice` when >5 memories retrieved for insight
9. **YAML Workflow**: Store configs in `./sandbox/config/`; use `yaml_refresh()` after edits
10. **Fallback First**: Always check for cached results before expensive ops; respect `CACHE_TTL_MIN`

---

**Note**: This is a *runtime overview*, not a user manual. Agents should infer tool availability from this context and generate calls accordingly.
