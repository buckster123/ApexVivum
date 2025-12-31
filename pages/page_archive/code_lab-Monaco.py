# pages/code_lab.py
"""
Apex Aurum Code Lab - Fixed Monaco Integration
"""

import sys
import os
import json
import pathlib
import importlib.util
from typing import Dict, List, Optional, Any, Callable
import streamlit as st
from streamlit_monaco import st_monaco
import uuid
import time
import traceback
from datetime import datetime
import subprocess
import threading
import re
from collections import OrderedDict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# MONACO IMPORT HANDLER (FIX)
# =============================================================================

def get_monaco_editor():
    """
    Detects and returns the correct Monaco editor function
    Handles version differences and import quirks
    """
    try:
        import streamlit_monaco as monaco_module
        
        # List all available attributes
        attrs = dir(monaco_module)
        logger.info(f"streamlit_monaco attributes: {attrs}")
        
        # Try common API names
        if "st_monaco_editor" in attrs:
            return monaco_module.st_monaco_editor
        elif "monaco_editor" in attrs:
            return monaco_module.monaco_editor
        elif "editor" in attrs:
            return monaco_module.editor
        else:
            # Last resort: try to find any function with 'editor' in name
            editor_funcs = [a for a in attrs if "editor" in a.lower() and callable(getattr(monaco_module, a))]
            if editor_funcs:
                logger.warning(f"Using fallback: {editor_funcs[0]}")
                return getattr(monaco_module, editor_funcs[0])
            
    except ImportError as e:
        logger.error(f"Failed to import streamlit_monaco: {e}")
    
    # If all fails, return None and we'll use a fallback
    return None

# Store the detected function globally
monaco_editor_func = get_monaco_editor()

# =============================================================================
# DEPENDENCY MANAGEMENT
# =============================================================================

def ensure_dependencies():
    """Auto-install missing dependencies"""
    required = {
        "streamlit_monaco": "streamlit-monaco>=0.1.3",
        "RestrictedPython": "RestrictedPython>=6.0",
        "pygit2": "pygit2>=1.10.0"
    }
    
    missing = []
    for module_name, pip_spec in required.items():
        try:
            __import__(module_name)
        except ImportError:
            missing.append(pip_spec)
    
    if missing:
        st.warning(f"Installing: {', '.join(missing)}")
        for spec in missing:
            subprocess.check_call([sys.executable, "-m", "pip", "install", spec])
        st.success("Dependencies installed! Please refresh the page.")
        st.stop()

ensure_dependencies()

# =============================================================================
# RUNTIME ENVIRONMENT DETECTION
# =============================================================================

class RuntimeEnvironment:
    """Detects and bridges to main.py without requiring imports"""
    
    def __init__(self):
        self.is_standalone: bool = False
        self.main_py_available: bool = False
        self.sandbox_dir: pathlib.Path = pathlib.Path("./sandbox")
        self.tools: Dict[str, Callable] = {}
        self.session_state = st.session_state
        self.app_state = None
        self.chroma_client = None
        self.counter_lock = threading.Lock()
        
        # Check if monaco is available
        self.monaco_available = monaco_editor_func is not None
        
        self._detect()
    
    def _detect(self):
        # Check if running inside main.py's process
        if "main" in sys.modules:
            self._load_from_main_module()
        else:
            # Search for main.py file
            self._search_and_load_main()
        
        if not self.main_py_available:
            self.is_standalone = True
            self._create_standalone_tools()
    
    def _load_from_main_module(self):
        """Load from sys.modules['main']"""
        try:
            main = sys.modules["main"]
            self.app_state = main.AppState.get()
            self.sandbox_dir = pathlib.Path(main.state.sandbox_dir)
            self.tools = main.TOOL_DISPATCHER.copy()
            self.chroma_client = getattr(main.state, 'chroma_client', None)
            self.counter_lock = getattr(main.state, 'counter_lock', threading.Lock())
            self.main_py_available = True
            logger.info("✅ Loaded from imported main module")
        except Exception as e:
            logger.error(f"Failed to load from main: {e}")
    
    def _search_and_load_main(self):
        """Search filesystem for main.py and load it safely"""
        candidates = [
            pathlib.Path(__file__).parent.parent / "main.py",
            pathlib.Path.cwd() / "main.py",
            pathlib.Path(__file__).parent / "main.py"
        ]
        
        for path in candidates:
            if path.exists():
                try:
                    spec = importlib.util.spec_from_file_location("main", path)
                    if spec and spec.loader:
                        main_mod = importlib.util.module_from_spec(spec)
                        
                        # Prevent Streamlit execution during import
                        original_streamlit = sys.modules.get("streamlit")
                        mock_st = type(sys)("streamlit")
                        mock_st.session_state = st.session_state
                        mock_st.cache_data = lambda **kw: (lambda f: f)
                        mock_st.cache_resource = lambda **kw: (lambda f: f)
                        sys.modules["streamlit"] = mock_st
                        
                        try:
                            spec.loader.exec_module(main_mod)
                            if hasattr(main_mod, 'AppState'):
                                self.app_state = main_mod.AppState.get()
                                self.sandbox_dir = pathlib.Path(main_mod.state.sandbox_dir)
                                self.tools = main_mod.TOOL_DISPATCHER.copy()
                                self.chroma_client = getattr(main_mod.state, 'chroma_client', None)
                                self.counter_lock = getattr(main_mod.state, 'counter_lock', threading.Lock())
                                self.main_py_available = True
                                logger.info(f"✅ Loaded main.py from {path}")
                                return
                        finally:
                            if original_streamlit:
                                sys.modules["streamlit"] = original_streamlit
                except Exception as e:
                    logger.warning(f"Failed to load {path}: {e}")
    
    def _create_standalone_tools(self):
        """Create minimal but functional standalone tools"""
        logger.info("Creating standalone tools")
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)
        
        self.tools = {
            "fs_read_file": self._fs_read_file,
            "fs_write_file": self._fs_write_file,
            "fs_list_files": self._fs_list_files,
            "fs_mkdir": self._fs_mkdir,
            "code_execution": self._code_execution,
            "git_ops": self._git_ops,
            "shell_exec": self._shell_exec,
            "memory_insert": self._memory_stub,
            "memory_query": self._memory_stub,
        }
    
    def _sanitize_path(self, path_str: str) -> pathlib.Path:
        """Strict path sanitization for sandbox security"""
        if ".." in path_str or path_str.startswith("/"):
            raise ValueError("Path traversal detected")
        
        path = (self.sandbox_dir / path_str).resolve()
        if not path.is_relative_to(self.sandbox_dir.resolve()):
            raise ValueError("Path outside sandbox")
        
        return path
    
    def _fs_read_file(self, file_path: str) -> str:
        try:
            safe_path = self._sanitize_path(file_path)
            if not safe_path.exists():
                return f"Error: File not found: {file_path}"
            return safe_path.read_text(encoding='utf-8')
        except Exception as e:
            return f"Error: {e}"
    
    def _fs_write_file(self, file_path: str, content: str) -> str:
        try:
            safe_path = self._sanitize_path(file_path)
            safe_path.parent.mkdir(parents=True, exist_ok=True)
            safe_path.write_text(content, encoding='utf-8')
            return f"✅ File '{file_path}' saved"
        except Exception as e:
            return f"Error: {e}"
    
    def _fs_list_files(self, dir_path: str = "") -> str:
        try:
            safe_path = self._sanitize_path(dir_path)
            files = [f.name + ("/" if f.is_dir() else "") for f in safe_path.iterdir()]
            return json.dumps(files)
        except Exception as e:
            return f"Error: {e}"
    
    def _fs_mkdir(self, dir_path: str) -> str:
        try:
            safe_path = self._sanitize_path(dir_path)
            safe_path.mkdir(parents=True, exist_ok=True)
            return f"✅ Directory '{dir_path}' created"
        except Exception as e:
            return f"Error: {e}"
    
    def _code_execution(self, code: str, venv_path: str = None) -> str:
        """Execute in RestrictedPython sandbox"""
        try:
            from RestrictedPython import compile_restricted_exec
            
            restricted_globals = {
                "__builtins__": {
                    "print": print, "len": len, "range": range, "str": str, "int": int,
                    "float": float, "list": list, "dict": dict, "set": set, "tuple": tuple,
                    "abs": abs, "round": round, "max": max, "min": min, "sum": sum,
                    "sorted": sorted, "enumerate": enumerate, "zip": zip, "map": map,
                    "filter": filter, "any": any, "all": all, "type": type, "isinstance": isinstance,
                },
                "_print_": lambda *args: st.code(" ".join(map(str, args)), language="text"),
                "_getattr_": getattr, "_getitem_": lambda obj, key: obj[key], "_write_": lambda x: x,
            }
            
            result = compile_restricted_exec(code)
            if result.errors:
                return f"Restricted compile error: {'; '.join(result.errors)}"
            
            import io
            from contextlib import redirect_stdout
            
            output = io.StringIO()
            with redirect_stdout(output):
                exec(result.code, restricted_globals, {})
            
            return output.getvalue() or "✅ Execution successful (no output)"
        except Exception as e:
            return f"Error: {traceback.format_exc()}"
    
    def _git_ops(self, operation: str, repo_path: str, message: str = None, name: str = None) -> str:
        try:
            import pygit2
            safe_repo = self._sanitize_path(repo_path)
            repo = pygit2.discover_repository(str(safe_repo)) or pygit2.init_repository(str(safe_repo))
            
            if operation == "init":
                return "✅ Repository initialized"
            elif operation == "commit":
                if not message:
                    return "Error: Commit message required"
                repo.index.add_all()
                repo.index.write()
                tree = repo.index.write_tree()
                author = pygit2.Signature("CodeLab", "codelab@apex")
                repo.create_commit("HEAD", author, author, message, tree, [repo.head.target] if repo.head.is_branch else [])
                return "✅ Changes committed"
            elif operation == "branch":
                if not name:
                    return "Error: Branch name required"
                repo.create_branch(name, repo.head.peel())
                return f"✅ Branch '{name}' created"
            elif operation == "diff":
                return repo.diff().patch or "No changes"
            else:
                return f"Unknown operation: {operation}"
        except Exception as e:
            return f"Git error: {e}"
    
    def _shell_exec(self, command: str) -> str:
        """Whitelisted shell execution"""
        try:
            cmd_parts = command.split()
            if not cmd_parts:
                return "Error: Empty command"
            
            whitelist = {"ls", "grep", "sed", "awk", "cat", "echo", "wc", "tail", "head", "cp", "mv", "rm", "mkdir", "rmdir", "touch", "python", "pip", "git"}
            if cmd_parts[0] not in whitelist:
                return f"Error: Command '{cmd_parts[0]}' not whitelisted"
            
            if any(".." in part for part in cmd_parts):
                return "Error: Path traversal detected"
            
            result = subprocess.run(
                cmd_parts,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=self.sandbox_dir
            )
            return result.stdout if result.returncode == 0 else f"Error: {result.stderr}"
        except subprocess.TimeoutExpired:
            return "Error: Command timed out"
        except Exception as e:
            return f"Error: {e}"
    
    def _memory_stub(self, **kwargs) -> str:
        """Minimal memory stub for standalone mode"""
        return "Memory: Standalone mode (no persistence)"

# =============================================================================
# CODE LAB STATE MANAGEMENT
# =============================================================================

class CodeLabState:
    """Manages editor tabs and UI state"""
    
    def __init__(self, env: RuntimeEnvironment):
        self.env = env
        self.tabs: OrderedDict = OrderedDict()
        self.active_tab: Optional[str] = None
        self.terminal_history: List[Dict] = []
        self._load_persisted()
    
    def _load_persisted(self):
        """Load saved state from session"""
        self.tabs = OrderedDict(st.session_state.get("code_lab_tabs", {}))
        self.active_tab = st.session_state.get("code_lab_active_tab")
        self.terminal_history = st.session_state.get("code_lab_terminal_history", [])
    
    def save(self):
        """Persist state"""
        st.session_state["code_lab_tabs"] = self.tabs
        st.session_state["code_lab_active_tab"] = self.active_tab
        st.session_state["code_lab_terminal_history"] = self.terminal_history
    
    def open_file(self, file_path: str):
        """Open file in new tab"""
        if file_path not in self.tabs:
            content = self.env.tools["fs_read_file"](file_path)
            if content.startswith("Error:"):
                st.error(content)
                return
            
            self.tabs[file_path] = {
                "content": content,
                "original": content,
                "modified": False,
                "language": self._detect_language(file_path),
            }
        
        self.active_tab = file_path
        self.save()
    
    def close_tab(self, file_path: str):
        """Close tab and save state"""
        if file_path in self.tabs:
            del self.tabs[file_path]
            if self.active_tab == file_path:
                self.active_tab = next(iter(self.tabs.keys())) if self.tabs else None
            self.save()
    
    def update_content(self, file_path: str, content: str):
        """Update tab content and track modifications"""
        if file_path in self.tabs:
            self.tabs[file_path]["content"] = content
            self.tabs[file_path]["modified"] = content != self.tabs[file_path]["original"]
            self.save()
    
    @staticmethod
    def _detect_language(file_path: str) -> str:
        ext = pathlib.Path(file_path).suffix.lower()
        return {
            ".py": "python", ".js": "javascript", ".ts": "typescript",
            ".html": "html", ".css": "css", ".json": "json", ".md": "markdown",
            ".yaml": "yaml", ".yml": "yaml", ".xml": "xml", ".sh": "shell",
            ".sql": "sql", "": "text"
        }.get(ext, "text")

# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_file_tree(state: CodeLabState, path: pathlib.Path, level: int = 0):
    """Render interactive file tree"""
    try:
        items = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    except Exception as e:
        st.error(f"Cannot read {path}: {e}")
        return
    
    for item in items:
        rel_path = item.relative_to(state.env.sandbox_dir).as_posix()
        
        if item.is_dir():
            with st.expander(f"📁 {item.name}/", expanded=level < 2):
                render_file_tree(state, item, level + 1)
        else:
            col1, col2 = st.columns([4, 1])
            with col1:
                if st.button(f"📄 {item.name}", key=f"file_{rel_path}"):
                    state.open_file(rel_path)
            with col2:
                if st.button("🗑️", key=f"del_{rel_path}"):
                    if st.checkbox(f"Confirm?", key=f"conf_{rel_path}"):
                        try:
                            item.unlink()
                            state.close_tab(rel_path)
                            st.success("Deleted")
                            time.sleep(0.5)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Delete failed: {e}")

def render_editor(state: CodeLabState):
    """Main editor with Monaco"""
    if not state.tabs:
        st.info("📁 Select a file from the tree to start editing")
        return
    
    # Tab bar
    tab_cols = st.columns(len(state.tabs))
    for idx, (file_path, tab_data) in enumerate(state.tabs.items()):
        with tab_cols[idx]:
            label = pathlib.Path(file_path).name
            if tab_data["modified"]:
                label += " ●"
            st.button(label, key=f"tab_{file_path}", type="primary" if file_path == state.active_tab else "secondary")
    
    # Editor
    if state.active_tab in state.tabs:
        tab_data = state.tabs[state.active_tab]
        
        # Use detected Monaco function or fallback to Ace
        if state.env.monaco_available:
            try:
                editor_kwargs = {
                    "value": tab_data["content"],
                    "height": 600,
                    "language": tab_data["language"],
                    "theme": "vs-dark",
                    "key": f"monaco_{state.active_tab}",
                    "minimap": True,
                    "line_numbers": True,
                    "word_wrap": "on",
                    "font_size": 14,
                    "tab_size": 4
                }
                
                # Call the detected function with proper arguments
                if monaco_editor_func.__name__ == "st_monaco_editor":
                    edited = monaco_editor_func(**editor_kwargs)
                else:
                    # Some versions might not have the st_ prefix
                    edited = monaco_editor_func(**editor_kwargs)
                
            except Exception as e:
                st.error(f"Monaco error: {e}")
                st.info("Falling back to Ace editor...")
                # Fallback to Ace
                from streamlit_ace import st_ace
                edited = st_ace(
                    value=tab_data["content"],
                    language=tab_data["language"],
                    theme="monokai",
                    height=600,
                    key=f"ace_{state.active_tab}"
                )
        else:
            # Monaco not available, use Ace
            from streamlit_ace import st_ace
            edited = st_ace(
                value=tab_data["content"],
                language=tab_data["language"],
                theme="monokai",
                height=600,
                key=f"ace_{state.active_tab}"
            )
        
        state.update_content(state.active_tab, edited)
        
        # Action bar
        st.markdown("---")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            if st.button("💾 Save", key="save_btn"):
                result = state.env.tools["fs_write_file"](state.active_tab, edited)
                if "successfully" in result or "✅" in result:
                    st.success("✅ Saved!")
                    tab_data["original"] = edited
                    tab_data["modified"] = False
                    state.save()
                else:
                    st.error(result)
        
        with col2:
            if st.button("▶️ Run", key="run_btn"):
                if tab_data["language"] == "python":
                    output = state.env.tools["code_execution"](edited)
                    st.code(output, language="text")
                else:
                    st.warning("Run only for Python")
        
        with col3:
            if st.button("🔍 Lint", key="lint_btn"):
                try:
                    from black import FileMode, format_str
                    formatted = format_str(edited, mode=FileMode())
                    state.update_content(state.active_tab, formatted)
                    st.rerun()
                except:
                    st.warning("Black not available")
        
        with col4:
            if st.button("📋 Copy Path", key="copy_btn"):
                st.code(state.active_tab)
        
        with col5:
            if st.button("❌ Close", key="close_btn"):
                state.close_tab(state.active_tab)
                st.rerun()

def render_git_panel(state: CodeLabState):
    """Git operations"""
    st.subheader("🌿 Git")
    repo_path = st.text_input("Repo path", value=".", key="git_path")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Init Repo"):
            st.info(state.env.tools["git_ops"]("init", repo_path))
        
        msg = st.text_input("Commit message")
        if st.button("Commit") and msg:
            st.info(state.env.tools["git_ops"]("commit", repo_path, message=msg))
    
    with col2:
        branch = st.text_input("Branch name")
        if st.button("Create Branch") and branch:
            st.info(state.env.tools["git_ops"]("branch", repo_path, name=branch))
        
        if st.button("Show Diff"):
            st.code(state.env.tools["git_ops"]("diff", repo_path), language="diff")

def render_terminal(state: CodeLabState):
    """Interactive terminal"""
    st.subheader("🖥️ Terminal")
    
    cmd = st.text_input("Command", placeholder="ls -la", key="term_input")
    if cmd:
        state.terminal_history.append({"cmd": cmd, "time": datetime.now().strftime("%H:%M:%S")})
        state.save()
        
        output = state.env.tools["shell_exec"](cmd)
        if output.startswith("Error:"):
            st.error(output)
        else:
            st.code(output, language="text")
    
    with st.expander("History"):
        for h in reversed(state.terminal_history[-10:]):
            st.text(f"{h['time']} {h['cmd']}")

def render_sync_panel(state: CodeLabState):
    """Sync with chat session"""
    if state.env.is_standalone:
        st.info("Standalone mode - no session sync")
        return
    
    st.subheader("🔗 Session Sync")
    uuid = state.env.session_state.get("current_convo_uuid")
    if uuid:
        st.success(f"Connected: `{uuid[:8]}...`")
        
        msgs = state.env.session_state.get("messages", [])
        if msgs and st.button("Save Context"):
            content = "# Chat Context\n\n" + "\n\n".join([f"**{m['role']}**: {m['content']}" for m in msgs[-10:]])
            path = f"chat_context_{datetime.now():%Y%m%d_%H%M%S}.md"
            state.env.tools["fs_write_file"](path, content)
            st.success(f"Saved to {path}")
            state.open_file(path)
    else:
        st.warning("No active chat session")

def render_stats(state: CodeLabState):
    """Statistics"""
    st.subheader("📊 Stats")
    
    files = sum(1 for _ in state.env.sandbox_dir.rglob("*") if _.is_file())
    dirs = sum(1 for _ in state.env.sandbox_dir.rglob("*") if _.is_dir())
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Files", files)
    col2.metric("Dirs", dirs)
    col3.metric("Tabs", len(state.tabs))

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    st.set_page_config(page_title="Apex Aurum Code Lab", layout="wide")
    
    env = RuntimeEnvironment()
    state = CodeLabState(env)
    
    # Show Monaco status
    if not env.monaco_available:
        st.warning("⚠️ Monaco Editor not available - using Ace fallback")
    
    # Top bar
    st.title("🧪 Apex Aurum Code Lab")
    col1, col2, col3 = st.columns(3)
    col1.metric("Mode", "Standalone" if env.is_standalone else "Integrated")
    col2.metric("Sandbox", str(env.sandbox_dir))
    col3.metric("Tabs", len(state.tabs))
    
    # Layout
    col_tree, col_main = st.columns([1, 3])
    
    with col_tree:
        st.subheader("📁 Files")
        if st.button("➕ New File"):
            name = st.text_input("Filename", placeholder="script.py", key="new_file")
            if name:
                env.tools["fs_write_file"](name, "# New file\n")
                st.rerun()
        
        if env.sandbox_dir.exists():
            render_file_tree(state, env.sandbox_dir)
        else:
            st.warning("Sandbox not found")
            if st.button("Create Sandbox"):
                env.sandbox_dir.mkdir(parents=True, exist_ok=True)
                st.rerun()
    
    with col_main:
        tabs = st.tabs(["📝 Editor", "🌿 Git", "🖥️ Terminal", "🔗 Sync", "📊 Stats"])
        
        with tabs[0]:
            render_editor(state)
        with tabs[1]:
            render_git_panel(state)
        with tabs[2]:
            render_terminal(state)
        with tabs[3]:
            render_sync_panel(state)
        with tabs[4]:
            render_stats(state)

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
