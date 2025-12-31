# pages/02_Code_Lab_Pro_v2.py
import streamlit as st
from streamlit_ace import st_ace
import pathlib
import os
import json
import sys
import io
import subprocess
import uuid
import threading
from typing import Optional, Dict, List, Tuple, Any
import sqlite3
import asyncio
import numpy
import builtins
import time
import re
import difflib
import ast
from datetime import datetime, timedelta
from functools import lru_cache, wraps
import logging
from logging.handlers import RotatingFileHandler
from dataclasses import dataclass, asdict
import shutil

# ============================================================================
# LOGGING SETUP
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        RotatingFileHandler("/tmp/codelab.log", maxBytes=10*1024*1024, backupCount=3)
    ]
)
logger = logging.getLogger("CodeLabProV2")

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
def _init_session_defaults():
    """Initialize all session state defaults"""
    defaults = {
        "code_editor_tabs": {},
        "code_active_file": None,
        "show_agent_panel": False,
        "code_debug_session": None,
        "code_agent_suggestions": [],
        "expanded_folders": set(),
        "auto_save_last": time.time(),
        "restored": False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ============================================================================
# STANDALONE MODE FALLBACKS
# ============================================================================

def _get_sandbox_dir() -> str:
    """Get sandbox dir from session or default"""
    if "app_state" in st.session_state and hasattr(st.session_state.app_state, 'sandbox_dir'):
        return st.session_state.app_state.sandbox_dir
    # Fallback to a sandbox within pages directory
    return os.path.join(os.path.dirname(__file__), "codelab_sandbox")

def _get_state(key: str, default: Any = None) -> Any:
    """Safe session state getter with fallback"""
    return st.session_state.get(key, default)

def _safe_tool_call(func_name: str, **kwargs) -> Optional[Any]:
    """
    Safely call a tool if dispatcher exists, else return None.
    Converts pathlib objects to strings for JSON serialization.
    """
    if "TOOL_DISPATCHER" not in st.session_state:
        logger.warning(f"Tool dispatcher not available, cannot call {func_name}")
        return None
    
    dispatcher = st.session_state.TOOL_DISPATCHER
    if func_name not in dispatcher:
        logger.warning(f"Tool {func_name} not found in dispatcher")
        return None
    
    # Convert pathlib objects to strings
    clean_kwargs = {}
    for k, v in kwargs.items():
        if isinstance(v, pathlib.Path):
            clean_kwargs[k] = str(v)
        else:
            clean_kwargs[k] = v
    
    try:
        result = [None]
        def _call():
            result[0] = dispatcher[func_name](**clean_kwargs)
        thread = threading.Thread(target=_call)
        thread.start()
        thread.join(timeout=30)
        return result[0]
    except Exception as e:
        logger.error(f"Tool error calling {func_name}: {e}")
        return None

# ============================================================================
# ASYNC EXECUTION ENGINE
# ============================================================================

class AsyncExecutor:
    """Async subprocess executor with resource limits"""
    
    @staticmethod
    async def execute_code(
        code: str, 
        timeout: int = 30, 
        memory_mb: int = 256,
        cwd: Optional[str] = None
    ) -> Tuple[str, str, int]:
        """Execute code in isolated subprocess"""
        def set_limits():
            try:
                import resource
                resource.setrlimit(resource.RLIMIT_AS, (memory_mb * 1024 * 1024,) * 2)
            except (ImportError, AttributeError):
                pass
        
        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable, "-c", code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                preexec_fn=set_limits,
                limit=1024 * 512  # 512KB pipe buffer
            )
            
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
            return stdout.decode('utf-8', errors='replace'), \
                   stderr.decode('utf-8', errors='replace'), \
                   process.returncode
                   
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return "", "Execution timed out", -1
        except Exception as e:
            logger.error(f"Execution error: {e}")
            return "", str(e), -1

# ============================================================================
# STATE MANAGEMENT WITH PERSISTENCE
# ============================================================================

@dataclass
class TabState:
    """Represents a single editor tab"""
    path: str
    content: str
    is_dirty: bool = False
    last_saved: float = 0.0
    undo_stack: List[str] = None
    redo_stack: List[str] = None
    
    def __post_init__(self):
        if self.undo_stack is None:
            self.undo_stack = []
        if self.redo_stack is None:
            self.redo_stack = []

class CodeLabStateV2:
    """Enhanced state manager with SQLite persistence"""
    
    def __init__(self):
        self.sandbox_dir = _get_sandbox_dir()
        self._abs_sandbox = pathlib.Path(self.sandbox_dir).resolve()
        self._ensure_sandbox()
        self._init_db()
        self._restore_session()
        
    def _ensure_sandbox(self):
        """Create sandbox directory if missing"""
        self._abs_sandbox.mkdir(parents=True, exist_ok=True)
        
    def _init_db(self):
        """Initialize SQLite for state persistence"""
        db_path = self._abs_sandbox / ".codelab" / "session.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use check_same_thread=False for Streamlit's threading model
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS tabs (
                path TEXT PRIMARY KEY,
                content TEXT,
                is_dirty INTEGER DEFAULT 0,
                last_saved REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        
        # Try FTS5, fall back to regular table
        try:
            self.conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS search_index 
                USING fts5(path UNINDEXED, content, tokenize='trigram')
            """)
        except sqlite3.OperationalError:
            logger.info("FTS5 not available, using basic search")
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS search_index (
                    path TEXT PRIMARY KEY,
                    content TEXT
                )
            """)
        self.conn.commit()
        
    def _restore_session(self):
        """Restore saved session state on first load"""
        if "restored" in st.session_state and st.session_state.restored:
            return
        
        logger.info("Restoring previous session...")
        
        # Load ALL tabs (including dirty ones)
        tabs = {}
        cursor = self.conn.execute("SELECT path, content, is_dirty FROM tabs")
        for path, content, is_dirty in cursor:
            tabs[path] = TabState(path=path, content=content, is_dirty=bool(is_dirty))
            
        st.session_state.code_editor_tabs = tabs
        st.session_state.restored = True
        
        if tabs:
            # Set first tab as active
            first_tab = list(tabs.keys())[0]
            st.session_state.code_active_file = first_tab
            
        logger.info(f"Restored {len(tabs)} tabs")

    def persist_tab(self, tab: TabState):
        """Persist tab to SQLite"""
        self.conn.execute(
            "INSERT OR REPLACE INTO tabs (path, content, is_dirty, last_saved) VALUES (?, ?, ?, ?)",
            (tab.path, tab.content, int(tab.is_dirty), tab.last_saved)
        )
        self.conn.commit()
        
    def _validate_path(self, path: pathlib.Path) -> bool:
        """Strict sandbox validation with symlink protection"""
        try:
            # Resolve to absolute path and check it's within sandbox
            full_path = (self._abs_sandbox / path).resolve()
            
            # Additional check for symlink attacks
            if full_path.is_symlink():
                logger.warning(f"Symlink path rejected: {path}")
                return False
                
            return str(full_path).startswith(str(self._abs_sandbox))
        except Exception as e:
            logger.error(f"Path validation error for {path}: {e}")
            return False
            
    @lru_cache(maxsize=128)
    def list_files(self, path: str = "") -> List[str]:
        """List files with caching"""
        try:
            target = pathlib.Path(self.sandbox_dir) / path
            if not target.exists():
                return []
            
            files = []
            for item in target.iterdir():
                try:
                    rel = item.relative_to(self.sandbox_dir)
                    if self._validate_path(rel):
                        files.append(str(rel) + ("/" if item.is_dir() else ""))
                except ValueError:
                    continue
                    
            return sorted(files, key=lambda p: (not p.endswith('/'), p.lower()))
        except Exception as e:
            logger.error(f"File list error: {e}")
            return []
    
    def read_file(self, file_path: str) -> Optional[str]:
        """Read file with validation and indexing"""
        if not self._validate_path(pathlib.Path(file_path)):
            logger.warning(f"Path validation failed for read: {file_path}")
            return None
            
        # Prefer tool system (pass relative path)
        if result := _safe_tool_call("fs_read_file", file_path=file_path):
            if "Error" not in result:
                self._index_file(file_path, result)
                return result
        
        # Fallback
        try:
            safe_path = self._abs_sandbox / file_path
            with open(safe_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
                self._index_file(file_path, content)
                return content
        except Exception as e:
            logger.error(f"Read error: {e}")
            return None
    
    def write_file(self, file_path: str, content: str) -> bool:
        """Write file with validation"""
        if not self._validate_path(pathlib.Path(file_path)):
            logger.warning(f"Path validation failed for write: {file_path}")
            return False
            
        # Prefer tool system (pass relative path)
        if result := _safe_tool_call("fs_write_file", file_path=file_path, content=content):
            if "successfully" in result:
                self._index_file(file_path, content)
                return True
        
        # Fallback
        try:
            safe_path = self._abs_sandbox / file_path
            safe_path.parent.mkdir(parents=True, exist_ok=True)
            with open(safe_path, 'w', encoding='utf-8') as f:
                f.write(content)
            self._index_file(file_path, content)
            return True
        except Exception as e:
            logger.error(f"Write error: {e}")
            return False
            
    def _index_file(self, path: str, content: str):
        """Update search index"""
        try:
            self.conn.execute(
                "INSERT OR REPLACE INTO search_index (path, content) VALUES (?, ?)",
                (path, content)
            )
            self.conn.commit()
        except sqlite3.OperationalError:
            # Fallback for non-FTS5
            self.conn.execute(
                "INSERT OR REPLACE INTO search_index (path, content) VALUES (?, ?)",
                (path, content)
            )
            self.conn.commit()
        
    def search_files(self, query: str, limit: int = 50) -> List[Tuple[str, str]]:
        """Search files with FTS fallback"""
        try:
            cursor = self.conn.execute(
                "SELECT path, snippet(search_index, 2, '<mark>', '</mark>', '...', 10) "
                "FROM search_index WHERE content MATCH ? LIMIT ?",
                (query, limit)
            )
            return cursor.fetchall()
        except sqlite3.OperationalError:
            # Fallback for non-FTS5
            cursor = self.conn.execute(
                "SELECT path, content FROM search_index WHERE content LIKE ? LIMIT ?",
                (f"%{query}%", limit)
            )
            results = []
            for path, content in cursor:
                idx = content.find(query)
                snippet = content[max(0, idx-20):idx+20] if idx >= 0 else ""
                results.append((path, snippet))
            return results
    
    def delete_file(self, file_path: str) -> bool:
        """Delete file with recycle bin"""
        if not self._validate_path(pathlib.Path(file_path)):
            return False
            
        try:
            safe_path = self._abs_sandbox / file_path
            if safe_path.exists():
                # Move to recycle bin instead of permanent deletion
                trash_dir = self._abs_sandbox / ".codelab" / "recycle"
                trash_dir.mkdir(parents=True, exist_ok=True)
                trash_path = trash_dir / f"{file_path.replace('/', '_')}_{int(time.time())}"
                trash_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(safe_path), str(trash_path))
                
                # Remove from index
                self.conn.execute("DELETE FROM search_index WHERE path = ?", (file_path,))
                self.conn.commit()
                return True
        except Exception as e:
            logger.error(f"Delete error: {e}")
        return False

    def get_repl_namespace(self) -> dict:
        """Get REPL namespace with module imports"""
        if "repl_namespace" in st.session_state:
            return st.session_state.repl_namespace
        
        # SAFE BUILTINS
        safe_builtins = {
            b: getattr(builtins, b)
            for b in [
                "print", "len", "range", "str", "int", "float", "list", "dict",
                "set", "tuple", "abs", "round", "max", "min", "sum", "sorted",
                "enumerate", "zip", "map", "filter", "any", "all", "bool",
                "type", "isinstance", "hasattr", "getattr", "pow", "chr", "ord"
            ]
        }
        
        namespace = {"__builtins__": safe_builtins}
        
        # DYNAMIC MODULE LOADING (Pi 5 optimized)
        modules = [
            ("numpy", "np"), ("sympy", "sympy"), ("mpmath", "mpmath"),
            ("networkx", "nx"), ("chess", "chess"), ("pygame", "pygame"),
            ("qutip", "qutip"), ("qiskit", "qiskit"), ("torch", "torch"),
            ("scipy", "scipy"), ("pandas", "pd"), ("sklearn", "sklearn"),
            ("matplotlib", "matplotlib")
        ]
        
        for module_name, alias in modules:
            try:
                module = __import__(module_name)
                namespace[alias] = module
                if alias != module_name:
                    namespace[module_name] = module
            except ImportError:
                pass
        
        try:
            import matplotlib.pyplot as plt
            namespace["plt"] = plt
        except ImportError:
            pass
        
        st.session_state.repl_namespace = namespace
        return namespace

# ============================================================================
# AST ANALYZER
# ============================================================================

class CodeAnalyzer(ast.NodeVisitor):
    """AST-based code analyzer"""
    
    def __init__(self):
        self.metrics = {
            "functions": 0,
            "classes": 0,
            "imports": [],
            "complexity": 0,
            "lines": 0,
            "comments": 0
        }
        
    def visit_FunctionDef(self, node):
        self.metrics["functions"] += 1
        # Cyclomatic complexity
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.ExceptHandler)):
                self.metrics["complexity"] += 1
        self.generic_visit(node)
        
    def visit_ClassDef(self, node):
        self.metrics["classes"] += 1
        self.generic_visit(node)
        
    def visit_Import(self, node):
        self.metrics["imports"].extend(alias.name for alias in node.names)
        self.generic_visit(node)
        
    def visit_ImportFrom(self, node):
        if node.module:
            self.metrics["imports"].extend(f"{node.module}.{alias.name}" for alias in node.names)
        self.generic_visit(node)
        
    def visit_Expr(self, node):
        # Count comments (approximate)
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            self.metrics["comments"] += 1
        self.generic_visit(node)

# ============================================================================
# ENHANCED DEBUGGER
# ============================================================================

class EnhancedDebugger:
    """AST-based debugger with watch expressions"""
    
    def __init__(self, code: str, namespace: dict):
        self.code = code
        self.namespace = namespace.copy()
        self.tree = ast.parse(code)
        self.instructions = list(self._flatten_ast(self.tree))
        self.current_inst = 0
        self.breakpoints = set()
        self.watch_expressions: Dict[str, str] = {}
        self.output = ""
        self.paused = False
        
    def _flatten_ast(self, tree: ast.Module) -> List[ast.AST]:
        """Flatten to executable instructions"""
        instructions = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Expr, ast.Assign, ast.AnnAssign, 
                                ast.AugAssign, ast.If, ast.For, ast.While,
                                ast.FunctionDef, ast.ClassDef, ast.Return)):
                instructions.append(node)
        return sorted(instructions, key=lambda x: getattr(x, 'lineno', 0))
        
    def toggle_breakpoint(self, line: int):
        """Toggle breakpoint on line"""
        if line in self.breakpoints:
            self.breakpoints.remove(line)
        else:
            self.breakpoints.add(line)
    
    def add_watch(self, expression: str):
        """Add watch expression"""
        self.watch_expressions[expression] = ""
        
    def remove_watch(self, expression: str):
        """Remove watch expression"""
        self.watch_expressions.pop(expression, None)
        
    def eval_watch(self, expression: str) -> str:
        """Evaluate watch expression"""
        try:
            result = eval(expression, self.namespace)
            return str(result)[:200]
        except Exception as e:
            return f"Error: {e}"
        
    def step(self) -> Tuple[bool, Optional[Dict]]:
        """Execute one instruction"""
        if self.current_inst >= len(self.instructions):
            return False, {}
            
        inst = self.instructions[self.current_inst]
        
        if not hasattr(inst, 'lineno'):
            self.current_inst += 1
            return self.current_inst < len(self.instructions), {}
            
        line_no = inst.lineno
        
        # Check breakpoint
        if line_no in self.breakpoints:
            self.paused = True
            return True, self._get_namespace_snapshot()
        
        # Execute
        try:
            module = ast.Module([inst], type_ignores=[])
            code_obj = compile(module, '<debugger>', 'exec')
            
            old_stdout = sys.stdout
            captured = io.StringIO()
            sys.stdout = captured
            
            exec(code_obj, self.namespace)
            
            sys.stdout = old_stdout
            self.output = captured.getvalue()
            self.current_inst += 1
            
            # Check if next is breakpoint
            should_pause = self.current_inst < len(self.instructions) and \
                          hasattr(self.instructions[self.current_inst], 'lineno') and \
                          self.instructions[self.current_inst].lineno in self.breakpoints
                          
            return should_pause, self._get_namespace_snapshot()
            
        except Exception as e:
            sys.stdout = old_stdout
            self.output = f"Error at line {line_no}: {e}"
            self.paused = True
            return True, {"__error__": str(e)}
            
    def run_to_completion(self) -> Tuple[str, Dict]:
        """Run all code"""
        old_stdout = sys.stdout
        captured = io.StringIO()
        sys.stdout = captured
        
        try:
            for i in range(len(self.instructions)):
                self.current_inst = i
                paused, _ = self.step()
                if paused:
                    break
                    
            sys.stdout = old_stdout
            return captured.getvalue() + self.output, self._get_namespace_snapshot()
            
        except Exception as e:
            sys.stdout = old_stdout
            return f"Fatal error: {e}", self._get_namespace_snapshot()
            
    def _get_namespace_snapshot(self) -> Dict:
        """Get safe namespace snapshot"""
        return {
            k: v for k, v in self.namespace.items() 
            if not k.startswith('_') and k not in ['__builtins__'] and len(str(v)) < 1000
        }

# ============================================================================
# AGENT INTEGRATION
# ============================================================================

class AgentIntegration:
    """Enhanced agent integration"""
    
    def __init__(self):
        self.conversation_history: List[Dict] = []
        
    def spawn_agent(self, task: str, file_path: str, code: str, context: Optional[Dict] = None) -> Optional[str]:
        """Spawn agent with context"""
        if not self._is_agent_available():
            return None
            
        convo_uuid = _get_state("current_convo_uuid", str(uuid.uuid4()))
        
        analysis = self._analyze_code_context(code)
        
        agent_task = f"""
        You are a pair-programming AI assistant. Analyze the code and provide actionable suggestions.
        
        FILE: {file_path}
        CODE CONTEXT:
        ```python
        {code[:2000]}
        ```
        
        CODE METRICS:
        - Lines: {analysis['lines']}
        - Functions: {analysis['functions']}
        - Complexity: {analysis['complexity']}
        - Imports: {', '.join(analysis['imports'][:5])}
        
        TASK: {task}
        
        FORMAT:
        - [LINE X] [REPLACE|INSERT] code here
        - Explanation: ...
        """
        
        result = _safe_tool_call(
            "agent_spawn",
            sub_agent_type="coder",
            task=agent_task,
            convo_uuid=convo_uuid,
            model="kimi-k2-thinking"
        )
        
        if result:
            self.conversation_history.append({
                "task": task,
                "file": file_path,
                "result": result,
                "timestamp": time.time()
            })
            
        return result
    
    def _is_agent_available(self) -> bool:
        """Check if agent tool is available"""
        dispatcher = _get_state("TOOL_DISPATCHER", {})
        if "agent_spawn" not in dispatcher:
            logger.warning("Agent tool not available in dispatcher")
            return False
        return True
        
    def _analyze_code_context(self, code: str) -> Dict:
        """Analyze code for context"""
        try:
            tree = ast.parse(code)
            analyzer = CodeAnalyzer()
            analyzer.visit(tree)
            return analyzer.metrics
        except SyntaxError as e:
            return {"error": str(e)}
            
    @staticmethod
    def parse_suggestions(response: str) -> List[Dict]:
        """Parse agent suggestions"""
        suggestions = []
        lines = response.split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i]
            if "[LINE" in line and "]" in line:
                try:
                    match = re.search(r'\[LINE\s*(\d+)\]', line)
                    if match:
                        line_num = int(match.group(1))
                        
                        action = "INSERT"
                        if "[REPLACE" in line:
                            action = "REPLACE"
                            
                        code_start = line.find("]") + 1
                        suggestion_code = line[code_start:].strip()
                        
                        explanation = ""
                        if i + 1 < len(lines) and "Explanation:" in lines[i+1]:
                            explanation = lines[i+1].split(":", 1)[1].strip()
                            i += 1
                            
                        suggestions.append({
                            "line": line_num,
                            "action": action,
                            "code": suggestion_code,
                            "explanation": explanation,
                            "accepted": False,
                            "id": str(uuid.uuid4())[:8]
                        })
                except Exception as e:
                    logger.error(f"Failed to parse suggestion: {e}")
            i += 1
            
        return suggestions
        
    @staticmethod
    def apply_suggestion(original: str, suggestion: Dict) -> str:
        """Apply suggestion with syntax validation"""
        lines = original.splitlines()
        line_num = suggestion["line"] - 1
        
        if suggestion["action"] == "REPLACE" and line_num < len(lines):
            lines[line_num] = suggestion["code"]
        elif suggestion["action"] == "INSERT" and line_num <= len(lines):
            lines.insert(line_num, suggestion["code"])
            
        new_code = "\n".join(lines)
        
        try:
            ast.parse(new_code)
            return new_code
        except SyntaxError as e:
            raise ValueError(f"Suggestion would break syntax: {e}")

# ============================================================================
# UI COMPONENTS
# ============================================================================

class FileBrowser:
    """Enhanced file browser with search"""
    
    def __init__(self, state: CodeLabStateV2):
        self.state = state
        self.expanded_folders = _get_state("expanded_folders", set())
        
    def render(self):
        """Render file browser sidebar"""
        st.sidebar.header("📂 Sandbox Explorer")
        
        # Search with debounce
        search_query = st.sidebar.text_input(
            "🔍 Search files", 
            placeholder="Type to search..."
        )
        if search_query and len(search_query) > 2:
            results = self.state.search_files(search_query)
            if results:
                st.sidebar.markdown("### Search Results")
                for path, snippet in results[:10]:
                    # Clean snippet for display
                    clean_snippet = re.sub(r'<[^>]*>', '', snippet)
                    if st.sidebar.button(
                        f"📄 {path}\n`{clean_snippet[:50]}...`",
                        key=f"search_{path}",
                        help="Click to open file"
                    ):
                        st.session_state["workspace_selected_file"] = path
                        st.rerun()
                st.sidebar.markdown("---")
        
        # Directory tree
        try:
            self._render_directory(pathlib.Path(self.state.sandbox_dir), level=0)
        except Exception as e:
            logger.error(f"Directory render error: {e}")
            st.sidebar.error("Failed to load directory")
        
        # File operations
        self._render_file_operations()
        
    def _render_directory(self, directory: pathlib.Path, level: int = 0):
        """Render directory recursively"""
        indent = "    " * level
        
        items = self.state.list_files(str(directory.relative_to(self.state.sandbox_dir)))
        if not items:
            return
            
        folders = [i for i in items if i.endswith('/')]
        files = [i for i in items if not i.endswith('/')]
        
        # Render folders
        for folder in folders:
            folder_name = folder.strip('/').split('/')[-1]
            folder_key = f"folder_{folder}_{level}"
            
            is_expanded = folder_key in self.expanded_folders
            if st.sidebar.checkbox(
                f"{indent}📁 {folder_name}/",
                value=is_expanded,
                key=folder_key,
                help=f"Toggle {folder_name}"
            ):
                self.expanded_folders.add(folder_key)
                self._render_directory(
                    pathlib.Path(self.state.sandbox_dir) / folder.strip('/'),
                    level + 1
                )
            elif folder_key in self.expanded_folders:
                self.expanded_folders.remove(folder_key)
        
        # Render files
        for file in files:
            file_name = file.split('/')[-1]
            if st.sidebar.button(
                f"{indent}📄 {file_name}",
                key=f"file_{file}_{level}",
                help=f"Open {file}"
            ):
                st.session_state["workspace_selected_file"] = file
                st.rerun()
                    
    def _render_file_operations(self):
        """Render file operation controls"""
        st.sidebar.markdown("---")
        
        # Create new file
        new_name = st.sidebar.text_input(
            "New file name", 
            placeholder="e.g., script.py",
            key="new_file_input"
        )
        if st.sidebar.button("📄 Create", key="create_file_btn") and new_name.strip():
            new_path = new_name.strip()
            if self.state._validate_path(pathlib.Path(new_path)):
                full_path = self.state._abs_sandbox / new_path
                try:
                    full_path.parent.mkdir(parents=True, exist_ok=True)
                    full_path.touch()
                    st.session_state["workspace_selected_file"] = new_path
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"Failed to create file: {e}")
            else:
                st.sidebar.error("Path outside sandbox!")
        
        # Refresh button
        if st.sidebar.button("🔄 Refresh", key="refresh_files_btn"):
            # Clear cache
            self.list_files.cache_clear()
            st.rerun()

class Editor:
    """Enhanced code editor"""
    
    def __init__(self, state: CodeLabStateV2):
        self.state = state
        self.agent = AgentIntegration()
        self.auto_save_timer = _get_state("auto_save_last", time.time())
        
    def render(self):
        """Render main editor area"""
        st.header("🎨 Code Editor Pro")
        
        # Auto-save every 30 seconds
        if time.time() - self.auto_save_timer > 30:
            self._auto_save()
            st.session_state.auto_save_last = time.time()
            
        # Load selected file
        self._load_selected_file()
        
        # Show tabs or empty state
        tab_count = self.state.conn.execute("SELECT COUNT(*) FROM tabs").fetchone()[0]
        if tab_count == 0:
            st.info("📂 Select a file from the browser to start editing")
            return
            
        # Render tabs
        self._render_tabs()
        
    def _load_selected_file(self):
        """Auto-load selected file"""
        selected = st.session_state.get("workspace_selected_file")
        if not selected:
            return
            
        # Check if already open
        if self.state.conn.execute("SELECT 1 FROM tabs WHERE path = ?", (selected,)).fetchone():
            st.session_state.code_active_file = selected
            return
            
        # Load content
        content = self.state.read_file(selected)
        if content is not None:
            self.state.conn.execute(
                "INSERT INTO tabs (path, content, is_dirty, last_saved) VALUES (?, ?, 0, ?)",
                (selected, content, time.time())
            )
            self.state.conn.commit()
            st.session_state.code_active_file = selected
            st.rerun()
        else:
            st.error(f"Could not load file: {selected}")
            st.session_state.workspace_selected_file = None
            
    def _render_tabs(self):
        """Render tabbed interface"""
        tabs_data = self.state.conn.execute("SELECT path, content FROM tabs ORDER BY path").fetchall()
        if not tabs_data:
            return
            
        tab_paths = [path for path, _ in tabs_data]
        tabs = st.tabs(tab_paths + ["+"])
        
        for i, (path, content) in enumerate(tabs_data):
            with tabs[i]:
                self._render_editor_tab(path, content)
                
        # Handle new tab click
        if len(tabs) > len(tab_paths) and tabs[len(tab_paths)].button("New File", key="new_tab_btn"):
            st.session_state["workspace_selected_file"] = None
            st.rerun()
            
    def _render_editor_tab(self, file_path: str, content: str):
        """Render individual tab content"""
        col_actions, col_status = st.columns([4, 1])
        
        with col_actions:
            self._render_action_buttons(file_path)
            
        with col_status:
            is_dirty = self.state.conn.execute(
                "SELECT is_dirty FROM tabs WHERE path = ?", (file_path,)
            ).fetchone()[0]
            status = "🔴 Unsaved" if is_dirty else "🟢 Saved"
            st.markdown(f"**{status}**")
                
        # Editor
        self._render_ace_editor(file_path, content)
        
    def _render_action_buttons(self, file_path: str):
        """Render action buttons"""
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            if st.button("💾 Save", key=f"save_{file_path}"):
                self._save_file(file_path)
                
        with col2:
            if file_path.endswith('.py'):
                if st.button("▶️ Run", key=f"run_{file_path}"):
                    self._execute_file(file_path)
                    
        with col3:
            if st.button("🐛 Debug", key=f"debug_{file_path}"):
                self._debug_file(file_path)
                
        with col4:
            if st.button("🤖 Agent", key=f"agent_{file_path}"):
                st.session_state.show_agent_panel = True
                
        with col5:
            if st.button("❌ Close", key=f"close_{file_path}"):
                self._close_tab(file_path)
                
    def _render_ace_editor(self, file_path: str, content: str):
        """Render ACE editor"""
        ext = pathlib.Path(file_path).suffix.lower()
        lang_map = {
            ".py": "python", ".js": "javascript", ".ts": "typescript",
            ".html": "html", ".css": "css", ".json": "json",
            ".md": "markdown", ".yaml": "yaml", ".yml": "yaml",
            ".txt": "text", ".sql": "sql", ".xml": "xml",
            ".sh": "sh", ".bash": "sh", ".zsh": "sh"
        }
        language = lang_map.get(ext, "text")
        
        current_content = self.state.conn.execute(
            "SELECT content FROM tabs WHERE path = ?", (file_path,)
        ).fetchone()[0]
        
        editor_content = st_ace(
            value=current_content,
            language=language,
            theme="monokai",
            font_size=14,
            tab_size=4,
            show_gutter=True,
            wrap=True,
            auto_update=False,
            height=500,
            key=f"ace_{file_path}"
        )
        
        if editor_content != current_content:
            self.state.conn.execute(
                "UPDATE tabs SET content = ?, is_dirty = 1 WHERE path = ?",
                (editor_content, file_path)
            )
            self.state.conn.commit()
            st.session_state.code_active_file = file_path
            
    def _save_file(self, file_path: str):
        """Save file"""
        content = self.state.conn.execute(
            "SELECT content FROM tabs WHERE path = ?", (file_path,)
        ).fetchone()[0]
        
        if self.state.write_file(file_path, content):
            self.state.conn.execute(
                "UPDATE tabs SET is_dirty = 0, last_saved = ? WHERE path = ?",
                (time.time(), file_path)
            )
            self.state.conn.commit()
            st.success("✅ Saved!", icon="✅")
            st.rerun()
        else:
            st.error("❌ Save failed", icon="❌")
            
    def _execute_file(self, file_path: str):
        """Execute file"""
        content = self.state.conn.execute(
            "SELECT content FROM tabs WHERE path = ?", (file_path,)
        ).fetchone()[0]
        
        async def run():
            return await AsyncExecutor.execute_code(
                content,
                cwd=str(self.state._abs_sandbox),
                timeout=60
            )
        
        stdout, stderr, code = asyncio.run(run())
        
        output = f"Exit code: {code}\n"
        if stdout:
            output += f"STDOUT:\n{stdout}\n"
        if stderr:
            output += f"STDERR:\n{stderr}\n"
            
        st.session_state.code_debug_session = {
            "file": file_path,
            "output": output,
            "type": "execution"
        }
        st.rerun()
        
    def _debug_file(self, file_path: str):
        """Debug file"""
        content = self.state.conn.execute(
            "SELECT content FROM tabs WHERE path = ?", (file_path,)
        ).fetchone()[0]
        
        debugger = EnhancedDebugger(content, self.state.get_repl_namespace())
        st.session_state.code_debug_session = {
            "file": file_path,
            "debugger": debugger,
            "type": "debug"
        }
        st.rerun()
        
    def _close_tab(self, file_path: str):
        """Close tab with unsaved check"""
        is_dirty = self.state.conn.execute(
            "SELECT is_dirty FROM tabs WHERE path = ?", (file_path,)
        ).fetchone()[0]
        
        if is_dirty:
            # Show warning in sidebar
            with st.sidebar:
                st.warning("Unsaved changes will be lost!")
                if not st.checkbox("Confirm close", key=f"confirm_close_{file_path}"):
                    return
                
        self.state.conn.execute("DELETE FROM tabs WHERE path = ?", (file_path,))
        self.state.conn.commit()
        
        # Clear active file if it was this file
        if st.session_state.code_active_file == file_path:
            st.session_state.code_active_file = None
            
        st.rerun()
        
    def _auto_save(self):
        """Auto-save dirty tabs"""
        cursor = self.state.conn.execute(
            "SELECT path, content FROM tabs WHERE is_dirty = 1"
        )
        saved_count = 0
        for path, content in cursor:
            if self.state.write_file(path, content):
                self.state.conn.execute(
                    "UPDATE tabs SET is_dirty = 0, last_saved = ? WHERE path = ?",
                    (time.time(), path)
                )
                saved_count += 1
        self.state.conn.commit()
        if saved_count > 0:
            logger.info(f"Auto-saved {saved_count} files")

class DebuggerPanel:
    """Enhanced debugger panel"""
    
    def render(self):
        """Render debugger panel"""
        if not st.session_state.code_debug_session:
            return
            
        st.markdown("---")
        st.subheader("🐛 Debugger")
        
        session = st.session_state.code_debug_session
        
        if session["type"] == "execution":
            self._render_execution_output(session)
        elif session["type"] == "debug" and "debugger" in session:
            self._render_debug_controls(session["debugger"])
            
    def _render_execution_output(self, session: Dict):
        """Render execution output"""
        st.code(session["output"], language="text")
        
    def _render_debug_controls(self, debugger: EnhancedDebugger):
        """Render debug controls"""
        if debugger.current_inst < len(debugger.instructions):
            current_line = debugger.instructions[debugger.current_inst].lineno
            st.info(f"Current line: {current_line}")
        
        self._render_code_with_breakpoints(debugger)
        self._render_control_buttons(debugger)
        self._render_watch_expressions(debugger)
        self._render_variables(debugger)
        
    def _render_code_with_breakpoints(self, debugger: EnhancedDebugger):
        """Render code with breakpoints"""
        source_lines = debugger.code.splitlines()
        
        for i, line in enumerate(source_lines, 1):
            col1, col2 = st.columns([1, 20])
            
            with col1:
                is_bp = i in debugger.breakpoints
                if st.checkbox(
                    "⚫" if is_bp else "⚪",
                    value=is_bp,
                    key=f"bp_{i}",
                    label_visibility="collapsed"
                ):
                    debugger.toggle_breakpoint(i)
                    st.rerun()
                    
            with col2:
                if debugger.current_inst < len(debugger.instructions) and \
                   debugger.instructions[debugger.current_inst].lineno == i:
                    st.markdown(f"**→ {i:3} | {line}**")
                else:
                    st.code(f"{i:3} | {line}", language=None)
                    
    def _render_control_buttons(self, debugger: EnhancedDebugger):
        """Render control buttons"""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Step Over", key="step_btn"):
                debugger.step()
                st.rerun()
                
        with col2:
            if st.button("Continue", key="continue_btn"):
                with st.spinner("Running..."):
                    while debugger.current_inst < len(debugger.instructions):
                        should_pause, _ = debugger.step()
                        if should_pause:
                            break
                st.rerun()
                
        with col3:
            if st.button("Stop", key="stop_debug_btn"):
                st.session_state.code_debug_session = None
                st.rerun()
                
    def _render_watch_expressions(self, debugger: EnhancedDebugger):
        """Render watch expressions"""
        st.markdown("---")
        st.subheader("👁️ Watch Expressions")
        
        new_watch = st.text_input("Add watch", placeholder="variable_name", key="new_watch_input")
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("Add", key="add_watch_btn") and new_watch:
                debugger.add_watch(new_watch)
                st.rerun()
        with col2:
            if st.button("Clear All", key="clear_watch_btn"):
                debugger.watch_expressions.clear()
                st.rerun()
            
        # Display watches
        for expr in list(debugger.watch_expressions.keys()):
            col1, col2 = st.columns([3, 1])
            with col1:
                value = debugger.eval_watch(expr)
                st.text(f"{expr}: {value}")
            with col2:
                if st.button("Remove", key=f"remove_watch_{expr}"):
                    debugger.remove_watch(expr)
                    st.rerun()
                    
    def _render_variables(self, debugger: EnhancedDebugger):
        """Render local variables"""
        st.markdown("---")
        st.subheader("📊 Variables")
        
        snapshot = debugger._get_namespace_snapshot()
        if snapshot:
            for var, val in snapshot.items():
                if var != "__error__":
                    st.text(f"{var}: {val}")
        else:
            st.info("No local variables")

class AgentPanel:
    """Enhanced agent panel"""
    
    def __init__(self, state: CodeLabStateV2):
        self.state = state
        self.agent = AgentIntegration()
        
    def render(self):
        """Render agent panel"""
        if not _get_state("show_agent_panel", False):
            return
            
        # Check if agent is available
        if not self.agent._is_agent_available():
            st.sidebar.warning("🤖 Agent not available in this session")
            return
            
        st.sidebar.markdown("---")
        st.sidebar.header("🤖 Agent Pair-Programmer")
        
        task = st.sidebar.text_area(
            "What do you want the agent to help with?",
            "Review this code for bugs and suggest improvements",
            height=100,
            key="agent_task_input"
        )
        
        with st.sidebar.expander("Advanced Options"):
            focus = st.selectbox(
                "Focus area",
                ["General Review", "Bug Detection", "Performance", "Security", "Testing"]
            )
            depth = st.slider("Analysis depth", 1, 5, 3)
            
        if st.sidebar.button("Spawn Agent", type="primary", key="spawn_agent_btn"):
            self._spawn_agent(task, focus, depth)
            
        suggestions = _get_state("code_agent_suggestions", [])
        if suggestions:
            self._render_suggestions(suggestions)
            
    def _spawn_agent(self, task: str, focus: str, depth: int):
        """Spawn agent"""
        active_file = st.session_state.code_active_file
        if not active_file:
            st.sidebar.error("No active file")
            return
            
        # Fetch with null safety
        result = self.state.conn.execute(
            "SELECT content FROM tabs WHERE path = ?", (active_file,)
        ).fetchone()
        if result is None:
            st.sidebar.error("File content not available")
            return
            
        content = result[0]
        
        context = {"focus": focus, "depth": depth}
        
        with st.spinner("Agent analyzing..."):
            result = self.agent.spawn_agent(task, active_file, content, context)
            
        if result:
            suggestions = self.agent.parse_suggestions(result)
            st.session_state.code_agent_suggestions = suggestions
            st.sidebar.success(f"✅ {len(suggestions)} suggestions ready!")
        else:
            st.sidebar.error("Agent failed to respond - tool not available")
            
    def _render_suggestions(self, suggestions: List[Dict]):
        """Render suggestions"""
        st.sidebar.markdown("---")
        st.sidebar.subheader("💡 Suggestions")
        
        for sugg in suggestions:
            with st.sidebar.expander(
                f"Line {sugg['line']}: {sugg['code'][:40]}...",
                expanded=False
            ):
                st.code(sugg['code'], language="python")
                if sugg['explanation']:
                    st.text(f"💡 {sugg['explanation']}")
                    
                col1, col2 = st.sidebar.columns(2)
                
                with col1:
                    if st.button("✅ Accept", key=f"accept_{sugg['id']}"):
                        try:
                            active_file = st.session_state.code_active_file
                            # Null-safe fetch
                            result = self.state.conn.execute(
                                "SELECT content FROM tabs WHERE path = ?", (active_file,)
                            ).fetchone()
                            if result is None:
                                st.sidebar.error("File content not found")
                                return
                            
                            original = result[0]
                            new_code = self.agent.apply_suggestion(original, sugg)
                            
                            self.state.conn.execute(
                                "UPDATE tabs SET content = ?, is_dirty = 1 WHERE path = ?",
                                (new_code, active_file)
                            )
                            self.state.conn.commit()
                            
                            suggestions.remove(sugg)
                            st.session_state.code_agent_suggestions = suggestions
                            st.rerun()
                        except ValueError as e:
                            st.sidebar.error(f"Cannot apply suggestion: {e}")
                            
                with col2:
                    if st.button("❌ Dismiss", key=f"dismiss_{sugg['id']}"):
                        suggestions.remove(sugg)
                        st.session_state.code_agent_suggestions = suggestions
                        st.rerun()

class GitPanel:
    """Enhanced Git panel"""
    
    def __init__(self, state: CodeLabStateV2):
        self.state = state
        
    def render(self):
        """Render Git panel"""
        st.sidebar.header("🌿 Git Operations")
        
        # Check if git tool is available
        if not self._is_git_available():
            st.sidebar.warning("Git operations not available in this session")
            return
            
        repo_path = st.sidebar.text_input("Repository path", value=".")
        
        if not self.state._validate_path(pathlib.Path(repo_path)):
            st.sidebar.error("Repository path outside sandbox!")
            return
            
        if st.sidebar.button("Refresh Status", key="git_status_btn"):
            status = _safe_tool_call("git_ops", operation="status", repo_path=repo_path)
            if status:
                st.sidebar.code(status)
            else:
                st.sidebar.error("Failed to get status")
                
        with st.sidebar.expander("Branch Management"):
            new_branch = st.sidebar.text_input("New branch name", key="new_branch_input")
            if st.sidebar.button("Create Branch", key="create_branch_btn") and new_branch:
                result = _safe_tool_call(
                    "git_ops", 
                    operation="create_branch", 
                    repo_path=repo_path, 
                    branch_name=new_branch
                )
                if result:
                    st.sidebar.success("✅ Branch created")
                else:
                    st.sidebar.error("Branch creation failed")
                    
        with st.sidebar.expander("Commit"):
            col1, col2 = st.sidebar.columns([2, 1])
            with col1:
                commit_msg = st.sidebar.text_input(
                    "Commit message", 
                    placeholder="Describe changes",
                    key="commit_msg_input"
                )
            with col2:
                if st.sidebar.button("Commit", key="commit_btn") and commit_msg:
                    result = _safe_tool_call(
                        "git_ops", 
                        operation="commit", 
                        repo_path=repo_path, 
                        message=commit_msg
                    )
                    if result:
                        st.sidebar.success("✅ Committed")
                    else:
                        st.sidebar.error("Commit failed")
                        
        if st.sidebar.button("Stage All", key="stage_all_btn"):
            result = _safe_tool_call("git_ops", operation="stage_all", repo_path=repo_path)
            if result:
                st.sidebar.code(result)
            else:
                st.sidebar.error("Staging failed")
    
    def _is_git_available(self) -> bool:
        """Check if git tool is available"""
        dispatcher = _get_state("TOOL_DISPATCHER", {})
        if "git_ops" not in dispatcher:
            logger.warning("Git operations tool not available in dispatcher")
            return False
        return True

class StaticAnalysisPanel:
    """Static analysis panel"""
    
    def __init__(self, state: CodeLabStateV2):
        self.state = state
        
    def render(self):
        """Render analysis panel"""
        st.sidebar.header("🔍 Static Analysis")
        
        active_file = st.session_state.code_active_file
        if not active_file:
            st.sidebar.info("Open a file to analyze")
            return
            
        # SAFE fetch with null check
        result = self.state.conn.execute(
            "SELECT content FROM tabs WHERE path = ?", (active_file,)
        ).fetchone()
        if result is None:
            st.sidebar.error("File content not available")
            return
            
        content = result[0]
        
        if st.sidebar.button("Analyze Now", key="analyze_btn"):
            analysis = self._analyze_code(content)
            self._render_analysis_results(analysis)
            
    def _analyze_code(self, code: str) -> Dict:
        """Analyze code"""
        try:
            tree = ast.parse(code)
            analyzer = CodeAnalyzer()
            analyzer.visit(tree)
            return analyzer.metrics
        except SyntaxError as e:
            return {"error": str(e)}
            
    def _render_analysis_results(self, analysis: Dict):
        """Render analysis results"""
        if "error" in analysis:
            st.sidebar.error(f"Syntax Error: {analysis['error']}")
            return
            
        st.sidebar.metric("Functions", analysis["functions"])
        st.sidebar.metric("Classes", analysis["classes"])
        st.sidebar.metric("Complexity", analysis["complexity"])
        
        if analysis["imports"]:
            st.sidebar.markdown("**Imports:**")
            for imp in analysis["imports"][:5]:
                st.sidebar.text(f"• {imp}")
        
        if analysis["comments"] > 0:
            st.sidebar.metric("Comments", analysis["comments"])

# ============================================================================
# MAIN PAGE
# ============================================================================

def main():
    st.set_page_config(
        page_title="Code-Lab-Pro v2",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state defaults
    _init_session_defaults()
    
    # Initialize UI components
    try:
        state = CodeLabStateV2()
    except Exception as e:
        logger.error(f"Failed to initialize state: {e}")
        st.error("Failed to initialize CodeLab. Check logs.")
        return
        
    file_browser = FileBrowser(state)
    editor = Editor(state)
    debugger = DebuggerPanel()
    agent_panel = AgentPanel(state)
    git_panel = GitPanel(state)
    analysis_panel = StaticAnalysisPanel(state)
    
    # Layout
    col_sidebar, col_main = st.columns([1, 4])
    
    with col_sidebar:
        file_browser.render()
        git_panel.render()
        analysis_panel.render()
        
        # Footer metrics
        st.sidebar.markdown("---")
        try:
            tab_count = state.conn.execute("SELECT COUNT(*) FROM tabs").fetchone()[0]
            st.sidebar.metric("Open Tabs", tab_count)
            st.sidebar.metric("Debug Session", "Active" if st.session_state.code_debug_session else "Inactive")
        except:
            st.sidebar.text("Metrics unavailable")
        
    with col_main:
        editor.render()
        debugger.render()
        
    # Floating panels
    agent_panel.render()

if __name__ == "__main__":
    main()
