# pages/02_Code_Lab_Pro.py
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
from typing import Optional, Dict, List
import sqlite3
import asyncio
import numpy
import builtins

# ============================================================================
# STANDALONE MODE FALLBACKS (if main.py not loaded)
# ============================================================================

def _get_sandbox_dir():
    """Get sandbox dir from session or default"""
    if "app_state" in st.session_state:
        return st.session_state.app_state.sandbox_dir
    return "./sandbox"

def _get_state(key, default=None):
    """Safe session state getter"""
    if key in st.session_state:
        return st.session_state[key]
    return default

def _safe_tool_call(func_name: str, **kwargs):
    """Safely call a tool if dispatcher exists, else return None"""
    if "TOOL_DISPATCHER" not in st.session_state:
        return None
    dispatcher = st.session_state.TOOL_DISPATCHER
    if func_name not in dispatcher:
        return None
    
    try:
        # Run in background thread to avoid blocking
        result = [None]
        def _call():
            result[0] = dispatcher[func_name](**kwargs)
        thread = threading.Thread(target=_call)
        thread.start()
        thread.join(timeout=30)  # 30s timeout
        return result[0]
    except Exception as e:
        st.error(f"Tool error: {e}")
        return None

# ============================================================================
# CODE LAB STATE MANAGEMENT
# ============================================================================

class CodeLabState:
    """
    Manages Code-Lab state with strict path validation and isolated module loading.
    Mirrors main.py's security model but bootstraps modules for page execution.
    """
    
    def __init__(self):
        self.sandbox_dir = _get_sandbox_dir()
        self._abs_sandbox = pathlib.Path(self.sandbox_dir).resolve()
        self._init_default_state()
        
    def _init_default_state(self):
        """Initialize session state defaults for Code-Lab"""
        if "code_editor_tabs" not in st.session_state:
            st.session_state.code_editor_tabs = {}
        if "code_active_file" not in st.session_state:
            st.session_state.code_active_file = None
        if "code_debug_session" not in st.session_state:
            st.session_state.code_debug_session = None
        if "code_agent_suggestions" not in st.session_state:
            st.session_state.code_agent_suggestions = []
            
    def _validate_path(self, path: pathlib.Path) -> bool:
        """
        Replicate main.py's strict path validation exactly.
        Uses string comparison after resolve() to prevent sandbox escape.
        """
        try:
            # Combine with sandbox and resolve (like main.py does)
            full_path = (self._abs_sandbox / path).resolve()
            # String prefix check - same as main.py's logic
            return str(full_path).startswith(str(self._abs_sandbox))
        except Exception:
            return False
            
    def list_files(self, path: str = "") -> List[str]:
        """List files with proper validation, matching main.py safety"""
        try:
            target = pathlib.Path(self.sandbox_dir) / path
            if not target.exists():
                return []
            
            files = []
            for item in target.iterdir():
                try:
                    # Get relative path first
                    rel = item.relative_to(self.sandbox_dir)
                    # Validate before adding
                    if self._validate_path(rel):
                        files.append(str(rel) + ("/" if item.is_dir() else ""))
                except ValueError:
                    continue  # Skip items outside sandbox
                    
            return sorted(files)
        except Exception as e:
            st.error(f"File list error: {e}")
            return []
    
    def read_file(self, file_path: str) -> Optional[str]:
        """Read file with validation, preferring tool system but with fallback"""
        # Validate BEFORE any operation
        if not self._validate_path(pathlib.Path(file_path)):
            return None
            
        # Prefer tool system for consistency
        if result := _safe_tool_call("fs_read_file", file_path=file_path):
            return result if "Error" not in result else None
        
        # Direct fallback (should also validate)
        try:
            safe_path = self._abs_sandbox / file_path
            if not self._validate_path(file_path):
                return None
            with open(safe_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception:
            return None
    
    def write_file(self, file_path: str, content: str) -> bool:
        """Write file with validation, preferring tool system"""
        # Validate first
        if not self._validate_path(pathlib.Path(file_path)):
            return False
            
        # Prefer tool system
        if result := _safe_tool_call("fs_write_file", file_path=file_path, content=content):
            return "successfully" in result
        
        # Direct fallback
        try:
            safe_path = self._abs_sandbox / file_path
            if not self._validate_path(file_path):
                return False
            safe_path.parent.mkdir(parents=True, exist_ok=True)
            with open(safe_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception as e:
            st.error(f"Write error: {e}")
            return False
    
    def get_repl_namespace(self) -> dict:
        """
        Get REPL namespace, bootstrapping modules from main venv if not already loaded.
        This is the KEY METHOD that makes Code-Lab self-sufficient.
        """
        
        # If main.py already set this up, use it directly
        if "repl_namespace" in st.session_state:
            return st.session_state.repl_namespace
        
        # === SAFE BUILTINS - Match main.py exactly ===
        safe_builtins = {
            b: getattr(builtins, b)
            for b in [
                "print", "len", "range", "str", "int", "float", "list", "dict",
                "set", "tuple", "abs", "round", "max", "min", "sum", "sorted",
                "enumerate", "zip", "map", "filter", "any", "all", "bool",
                "type", "isinstance", "hasattr", "getattr", "pow"
            ]
        }
        
        # Start with minimal safe namespace
        namespace = {
            "__builtins__": safe_builtins,
        }
        
        # === DYNAMIC MODULE LOADING - Replicate main.py's ADDITIONAL_LIBS ===
        
        # Math & Science Libraries (safe, lightweight)
        try:
            import numpy as np
            namespace["numpy"] = np
            namespace["np"] = np  # Add common alias
        except ImportError:
            pass
        
        try:
            import sympy
            namespace["sympy"] = sympy
        except ImportError:
            pass
        
        try:
            import mpmath
            namespace["mpmath"] = mpmath
        except ImportError:
            pass
        
        # Graph & Network Libraries
        try:
            import networkx as nx
            namespace["networkx"] = nx
            namespace["nx"] = nx
        except ImportError:
            pass
        
        # Game & Media Libraries (safe but optional)
        try:
            import chess
            namespace["chess"] = chess
        except ImportError:
            pass
        
        try:
            import pygame
            namespace["pygame"] = pygame
        except ImportError:
            pass
        
        # Quantum & ML Libraries (heavy but safe)
        try:
            import qutip
            namespace["qutip"] = qutip
        except ImportError:
            pass
        
        try:
            import qiskit
            namespace["qiskit"] = qiskit
        except ImportError:
            pass
        
        try:
            import torch
            namespace["torch"] = torch
        except ImportError:
            pass
        
        # SciPy ecosystem (conditionally available)
        try:
            import scipy
            namespace["scipy"] = scipy
        except ImportError:
            pass
        
        try:
            import matplotlib.pyplot as plt
            namespace["matplotlib"] = __import__('matplotlib')
            namespace["plt"] = plt
        except ImportError:
            pass
        
        try:
            import pandas as pd
            namespace["pandas"] = pd
            namespace["pd"] = pd
        except ImportError:
            pass
        
        try:
            import sklearn
            namespace["sklearn"] = sklearn
        except ImportError:
            pass
        
        # Biology & Chemistry Libraries
        try:
            import Bio
            namespace["biopython"] = Bio
        except ImportError:
            pass
        
        # Cache the namespace so we only build it once
        st.session_state.repl_namespace = namespace
        return namespace
    
    def get_venvs(self) -> List[str]:
        """List available venvs in sandbox"""
        venvs = []
        try:
            for item in pathlib.Path(self.sandbox_dir).iterdir():
                if item.is_dir() and (item / "bin" / "python").exists():
                    venvs.append(item.name)
        except Exception:
            pass
        return sorted(venvs)

# ============================================================================
# DEBUGGER COMPONENT
# ============================================================================

class SimpleDebugger:
    """Lightweight debugger that doesn't require pdb hooks"""
    
    def __init__(self, code: str, namespace: dict):
        self.code = code
        self.namespace = namespace.copy()
        self.lines = code.split('\n')
        self.breakpoints = set()
        self.current_line = 0
        self.paused = False
        
    def toggle_breakpoint(self, line: int):
        """Toggle breakpoint on line (0-indexed)"""
        if line in self.breakpoints:
            self.breakpoints.remove(line)
        else:
            self.breakpoints.add(line)
    
    def get_locals_snapshot(self) -> dict:
        """Get current local variables snapshot"""
        # Filter out private and large objects
        return {
            k: v for k, v in self.namespace.items() 
            if not k.startswith('_') and len(str(v)) < 1000
        }
    
    def step(self) -> tuple[bool, Optional[dict]]:
        """Execute one line, return (should_pause, locals_snapshot)"""
        if self.current_line >= len(self.lines):
            return False, None
            
        line = self.lines[self.current_line].strip()
        
        # Skip empty lines and comments
        if not line or line.startswith('#'):
            self.current_line += 1
            return self.current_line in self.breakpoints, self.get_locals_snapshot()
        
        # Check breakpoint
        if self.current_line in self.breakpoints:
            self.paused = True
            return True, self.get_locals_snapshot()
        
        # Execute line
        try:
            exec(line, self.namespace)
            self.current_line += 1
            return self.current_line in self.breakpoints, self.get_locals_snapshot()
        except Exception as e:
            self.paused = True
            return True, {"__error__": str(e)}
    
    def run_to_completion(self) -> tuple[str, dict]:
        """Run all code and return (output, final_locals)"""
        old_stdout = sys.stdout
        captured = io.StringIO()
        sys.stdout = captured
        
        try:
            for i, line in enumerate(self.lines):
                self.current_line = i
                if line.strip() and not line.strip().startswith('#'):
                    exec(line, self.namespace)
        except Exception as e:
            sys.stdout = old_stdout
            return f"Error at line {i+1}: {e}", self.get_locals_snapshot()
        
        sys.stdout = old_stdout
        return captured.getvalue(), self.get_locals_snapshot()

# ============================================================================
# AGENT PAIR-PROGRAMMING
# ============================================================================

def spawn_coder_agent(task: str, file_path: str, code_context: str) -> Optional[str]:
    """Spawn agent to suggest code improvements"""
    convo_uuid = _get_state("current_convo_uuid", str(uuid.uuid4()))
    
    # Use the main app's agent_spawn if available
    if "agent_spawn" in st.session_state.get("TOOL_DISPATCHER", {}):
        agent_task = f"""
        Analyze this code in {file_path}:
        
        ```python
        {code_context}
        ```
        
        Task: {task}
        Provide specific code suggestions with line numbers. Format:
        - [LINE X] Suggestion: ...
        - [LINE Y] Suggestion: ...
        """
        
        result = _safe_tool_call(
            "agent_spawn",
            sub_agent_type="coder",
            task=agent_task,
            convo_uuid=convo_uuid,
            model="kimi-k2-thinking"  # Pi-5 friendly: uses API, not local
        )
        return result
    
    return None

def parse_agent_suggestions(response: str) -> List[Dict]:
    """Parse agent suggestions into structured format"""
    suggestions = []
    for line in response.split('\n'):
        if "[LINE" in line and "]" in line:
            try:
                line_num_start = line.find("[LINE") + 5
                line_num_end = line.find("]", line_num_start)
                line_num = int(line[line_num_start:line_num_end].strip())
                suggestion = line.split("]")[-1].strip()
                if suggestion:
                    suggestions.append({
                        "line": line_num,
                        "text": suggestion,
                        "accepted": False,
                        "id": str(uuid.uuid4())[:8]
                    })
            except:
                continue
    return suggestions

# ============================================================================
# MAIN PAGE LAYOUT
# ============================================================================

def render_file_browser():
    """Left sidebar file tree with bulletproof path handling"""
    st.sidebar.header("📂 Sandbox Explorer")
    
    lab_state = CodeLabState()
    
    # Track expanded folders
    if "expanded_folders" not in st.session_state:
        st.session_state.expanded_folders = set()
    
    def _safe_render_directory(directory: pathlib.Path, level: int = 0):
        """Safely render directory contents"""
        indent = "    " * level
        
        try:
            # Get and sort items (folders first)
            items = []
            for item in directory.iterdir():
                # Validate each item's path
                try:
                    rel = item.relative_to(lab_state.sandbox_dir)
                    if lab_state._validate_path(rel):
                        items.append(item)
                except Exception:
                    continue
            
            items = sorted(items, key=lambda p: (not p.is_dir(), p.name.lower()))
            
            # Render each item
            for item in items:
                rel_path = item.relative_to(lab_state.sandbox_dir)
                
                if item.is_dir():
                    # For folders, create nested expander
                    folder_key = f"folder_{rel_path}_{level}"
                    with st.sidebar.expander(f"{indent}📁 {item.name}/", expanded=folder_key in st.session_state.expanded_folders):
                        if folder_key in st.session_state.expanded_folders:
                            _safe_render_directory(item, level + 1)
                else:
                    # For files, show button
                    file_key = f"file_{rel_path}_{level}"
                    if st.sidebar.button(f"{indent}📄 {item.name}", key=file_key):
                        st.session_state["workspace_selected_file"] = str(rel_path)
                        st.rerun()
                        
        except PermissionError:
            st.sidebar.error(f"{indent}Permission denied: {directory.name}")
        except Exception as e:
            st.sidebar.error(f"{indent}Error reading {directory.name}: {e}")
    
    # Start rendering from sandbox root
    try:
        root_path = pathlib.Path(lab_state.sandbox_dir)
        _safe_render_directory(root_path)
    except Exception as e:
        st.sidebar.error(f"Cannot access sandbox: {e}")
        return
    
    # New file controls
    st.sidebar.markdown("---")
    new_name = st.sidebar.text_input("New filename")
    
    col_create, col_refresh = st.sidebar.columns(2)
    
    with col_create:
        if st.sidebar.button("Create File") and new_name.strip():
            new_path = pathlib.Path(new_name.strip())
            if not lab_state._validate_path(new_path):
                st.sidebar.error("Path outside sandbox!")
            else:
                full_path = lab_state._abs_sandbox / new_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.touch()
                st.rerun()
    
    with col_refresh:
        if st.sidebar.button("Refresh"):
            st.rerun()

def _render_tree_recursive(path: pathlib.Path, indent: str = ""):
    """Internal recursive helper with validated paths"""
    lab_state = CodeLabState()
    abs_sandbox = pathlib.Path(lab_state.sandbox_dir).resolve()
    
    try:
        items = []
        for item in path.iterdir():
            abs_item = item.resolve()
            if str(abs_item).startswith(str(abs_sandbox)):
                items.append(item)
        
        items = sorted(items, key=lambda p: (not p.is_dir(), p.name.lower()))
    except PermissionError:
        st.sidebar.error(f"Permission denied: {path}")
        return
    
    for item in items:
        rel_path = str(item.relative_to(abs_sandbox))
        
        if item.is_dir():
            with st.sidebar.expander(f"{indent}📁 {item.name}/"):
                _render_tree_recursive(item, indent + "    ")
        else:
            if st.sidebar.button(f"{indent}📄 {item.name}", key=f"tree_select_{rel_path}"):
                st.session_state["workspace_selected_file"] = rel_path
                st.rerun()

def render_editor():
    """Main editor area with tabs - NOW WITH AUTO-LOADER"""
    
    # ========== AUTO-LOADER: Hydrate selected file into tabs ==========
    selected_file = st.session_state.get("workspace_selected_file")
    if selected_file and selected_file not in st.session_state.code_editor_tabs:
        lab_state = CodeLabState()
        content = lab_state.read_file(selected_file)
        if content is not None:
            st.session_state.code_editor_tabs[selected_file] = content
            st.session_state.code_active_file = selected_file
            st.rerun()  # Re-run to render the new tab
        else:
            st.error(f"Could not load file: {selected_file}")
            st.session_state.workspace_selected_file = None
    
    # If still no tabs, show the message
    if not st.session_state.code_editor_tabs:
        st.info("Select a file from the browser to start editing")
        return
    
    # ========== REST OF YOUR ORIGINAL CODE (unchanged) ==========
    st.header("🎨 Code Editor Pro")
    
    # Tabbed interface
    tab_names = list(st.session_state.code_editor_tabs.keys())
    tabs = st.tabs(tab_names)
    
    for i, (file_path, content) in enumerate(st.session_state.code_editor_tabs.items()):
        with tabs[i]:
            col_actions, col_debug = st.columns([3, 1])
            
            with col_actions:
                b1, b2, b3, b4 = st.columns(4)
                
                with b1:
                    if st.button("💾 Save", key=f"save_{file_path}"):
                        lab_state = CodeLabState()
                        edited_content = st.session_state.code_editor_tabs[file_path]
                        if lab_state.write_file(file_path, edited_content):
                            st.success("Saved!")
                
                with b2:
                    if file_path.endswith('.py'):
                        if st.button("▶️ Run", key=f"run_{file_path}"):
                            # Use current tab content, not saved file
                            current_code = st.session_state.code_editor_tabs[file_path]
                            debugger = SimpleDebugger(current_code, CodeLabState().get_repl_namespace())
                            output, final_locals = debugger.run_to_completion()
                            st.code(output, language="text")
                            st.session_state.code_debug_session = {
                                "file": file_path,
                                "output": output,
                                "locals": final_locals
                            }
                
                with b3:
                    if st.button("🐛 Debug", key=f"debug_{file_path}"):
                        current_code = st.session_state.code_editor_tabs[file_path]
                        st.session_state.code_debug_session = {
                            "file": file_path,
                            "debugger": SimpleDebugger(current_code, CodeLabState().get_repl_namespace())
                        }
                
                with b4:
                    if st.button("🤖 Agent Help", key=f"agent_{file_path}"):
                        st.session_state.show_agent_panel = True
            
            # Editor
            ext = pathlib.Path(file_path).suffix.lower()
            lang_map = {
                ".py": "python", ".js": "javascript", ".ts": "typescript",
                ".html": "html", ".css": "css", ".json": "json",
                ".md": "markdown", ".yaml": "yaml", ".yml": "yaml",
                ".txt": "text", ".sql": "sql", ".xml": "xml"
            }
            language = lang_map.get(ext, "text")
            
            # Use st_ace with current tab content
            edited = st_ace(
                value=content,
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
            
            # Update content immediately when edited
            st.session_state.code_editor_tabs[file_path] = edited
            st.session_state.code_active_file = file_path

def render_debugger():
    """Bottom debugger panel"""
    if not st.session_state.code_debug_session:
        return
    
    st.markdown("---")
    st.subheader("🐛 Debugger")
    
    session = st.session_state.code_debug_session
    debugger = session.get("debugger")
    
    if debugger:
        # Current line indicator
        st.info(f"Current line: {debugger.current_line + 1}")
        
        # Code with line numbers and breakpoint toggles
        for i, line in enumerate(debugger.lines):
            col1, col2 = st.columns([1, 20])
            with col1:
                is_bp = i in debugger.breakpoints
                if st.checkbox("⚫" if is_bp else "⚪", key=f"bp_{i}", value=is_bp):
                    debugger.toggle_breakpoint(i)
            with col2:
                if i == debugger.current_line:
                    st.markdown(f"**→ {i+1:3} | {line}**")
                else:
                    st.code(f"{i+1:3} | {line}", language=None)
        
        # Control buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Step"):
                paused, locals_snapshot = debugger.step()
                st.session_state.code_debug_session["locals"] = locals_snapshot
                if paused:
                    st.info(f"Paused at line {debugger.current_line + 1}")
                st.rerun()
        
        with col2:
            if st.button("Continue"):
                while True:
                    paused, locals_snapshot = debugger.step()
                    if not paused:
                        break
                st.success("Execution completed")
                st.session_state.code_debug_session["locals"] = locals_snapshot
                st.rerun()
        
        with col3:
            if st.button("Stop"):
                st.session_state.code_debug_session = None
                st.rerun()
        
        # Variables view
        if "locals" in st.session_state.code_debug_session:
            st.markdown("---")
            st.subheader("📊 Variables")
            locals_data = st.session_state.code_debug_session["locals"]
            if locals_data:
                for var, val in locals_data.items():
                    st.text(f"{var}: {val}")
            else:
                st.info("No local variables to display")
    else:
        # Simple output view if not debugger instance
        st.code(session.get("output", ""), language="text")

def render_agent_panel():
    """Right sidebar agent pair-programming"""
    if not st.session_state.get("show_agent_panel", False):
        return
    
    st.sidebar.markdown("---")
    st.sidebar.header("🤖 Agent Pair-Programmer")
    
    task = st.sidebar.text_area("What do you want the agent to help with?", 
                                "Review this code for bugs and suggest improvements")
    
    if st.sidebar.button("Spawn Coder Agent"):
        lab_state = CodeLabState()
        active_file = st.session_state.code_active_file
        if active_file:
            code = st.session_state.code_editor_tabs.get(active_file, "")
            with st.spinner("Agent analyzing..."):
                result = spawn_coder_agent(task, active_file, code)
            if result:
                suggestions = parse_agent_suggestions(result)
                st.session_state.code_agent_suggestions = suggestions
                st.sidebar.success("Agent suggestions ready!")
            else:
                st.sidebar.error("Agent spawn failed or not available")
    
    # Display suggestions
    if suggestions := st.session_state.code_agent_suggestions:
        st.sidebar.markdown("---")
        st.sidebar.subheader("💡 Suggestions")
        
        for sugg in suggestions:
            with st.sidebar.expander(f"Line {sugg['line']}: {sugg['text'][:50]}..."):
                st.text(sugg['text'])
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    if st.button("✅ Accept", key=f"accept_{sugg['id']}"):
                        # Apply suggestion (simple insert after line)
                        active_file = st.session_state.code_active_file
                        lines = st.session_state.code_editor_tabs[active_file].split('\n')
                        lines.insert(sugg['line'], f"# AGENT: {sugg['text']}")
                        st.session_state.code_editor_tabs[active_file] = '\n'.join(lines)
                        st.rerun()
                with col2:
                    if st.button("❌ Dismiss", key=f"dismiss_{sugg['id']}"):
                        st.session_state.code_agent_suggestions.remove(sugg)
                        st.rerun()

def render_env_manager():
    """Environment and package management"""
    st.sidebar.header("🐍 Environment Manager")
    
    lab_state = CodeLabState()
    venvs = lab_state.get_venvs()
    
    if venvs:
        selected_venv = st.sidebar.selectbox("Select Venv", ["default"] + venvs)
        if selected_venv != "default":
            venv_path = f"./{selected_venv}"
        else:
            venv_path = None
    else:
        venv_path = None
    
    # Package installation (uses your pip_install whitelist)
    packages = st.sidebar.text_input("Install packages (space-separated)")
    if st.sidebar.button("Install"):
        if packages:
            pkg_list = packages.split()
            result = _safe_tool_call("pip_install", venv_path=venv_path or "venv", packages=pkg_list)
            if result:
                st.sidebar.code(result)
    
    # Show installed packages
    if venv_path and st.sidebar.button("List Installed"):
        pip_path = pathlib.Path(lab_state.sandbox_dir) / venv_path / "bin" / "pip"
        try:
            result = subprocess.run([str(pip_path), "list"], capture_output=True, text=True, timeout=30)
            st.sidebar.code(result.stdout)
        except Exception as e:
            st.sidebar.error(f"Failed to list packages: {e}")

def render_git_panel():
    """Git operations panel with safe paths"""
    st.sidebar.header("🌿 Git")
    
    repo_path = st.sidebar.text_input("Repo path", value=".")
    
    # Validate repo path
    lab_state = CodeLabState()
    if not lab_state._validate_path(pathlib.Path(repo_path)):
        st.sidebar.error("Repository path outside sandbox!")
        return
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.sidebar.button("Status"):
            result = _safe_tool_call("git_ops", operation="status", repo_path=repo_path)
            if result:
                st.sidebar.code(result)
    
    with col2:
        commit_msg = st.sidebar.text_input("Commit message")
        if st.sidebar.button("Commit") and commit_msg:
            result = _safe_tool_call("git_ops", operation="commit", repo_path=repo_path, message=commit_msg)
            if result:
                st.sidebar.success(result)

# ============================================================================
# MAIN PAGE RENDER
# ============================================================================

def main():
    st.set_page_config(
        page_title="Code-Lab-Pro",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize state if not exists
    if "code_editor_tabs" not in st.session_state:
        st.session_state.code_editor_tabs = {}
    if "code_active_file" not in st.session_state:
        st.session_state.code_active_file = None
    if "show_agent_panel" not in st.session_state:
        st.session_state.show_agent_panel = False
    
    # Render main layout
    render_file_browser()
    render_env_manager()
    render_git_panel()
    
    # Main content area
    col_main, col_debug = st.columns([3, 2])
    
    with col_main:
        render_editor()
    
    with col_debug:
        render_debugger()
    
    # Floating agent panel
    render_agent_panel()
    
    # Footer metrics
    st.sidebar.markdown("---")
    st.sidebar.metric("Open Tabs", len(st.session_state.code_editor_tabs))
    st.sidebar.metric("Debug Session", "Active" if st.session_state.code_debug_session else "Inactive")

if __name__ == "__main__":
    main()
