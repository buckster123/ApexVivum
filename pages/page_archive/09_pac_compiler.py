# pages/09_pac_compiler_fixed.py
# FIXED: No default parsing, validation guide, ASCII-safe examples

import streamlit as st
import json
import os
import sys
from pathlib import Path
from typing import List, Dict
import re

# === Import Diagnostics ===
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from main import state
except ImportError:
    st.error("⚠️ Could not import main script.")
    st.stop()

# === Unicode Validator (User-Friendly) ===
def validate_pac_syntax(text: str) -> List[str]:
    """Check for Unicode issues before parsing"""
    issues = []
    
    # Unicode subscripts/superscripts (the crash culprit)
    if re.search(r'[₀-₉⁰-⁹]', text):
        issues.append("⚠️ Unicode subscripts detected (e.g., ℵ₂ → ℵ2, 𝔼₀ → E0)")
    
    # Missing invocation
    if not re.search(r'⊙⟨[^⟩]+⟩⊙', text):
        issues.append("⚠️ No invocation glyph (⊙⟨...⟩⊙)")
    
    # Unbalanced brackets
    if text.count('⊙⟨') != text.count('⟩⊙'):
        issues.append("⚠️ Unbalanced invocation brackets")
    
    return issues

# === Simplified PAC Parser (ASCII-Safe) ===
class PACParser:
    """Safe parser for ASCII-safe PAC syntax"""
    
    def __init__(self):
        # ASCII-safe patterns (no Unicode subscripts)
        self.patterns = {
            "invocation": r'⊙⟨([^⟩]+)⟩⊙',
            "command": r'!(PORT|ENGINE|BOOTSTRAP|EXO_CORTEX|MODULE)\b',
            "chain": r'chain\{([^}]+)\}',
            "vec_key": r'vec(\d+\.\d+)\s+key(\d+\.\d+)',
        }
    
    def parse_safe(self, text: str) -> List[Dict]:
        """Parse only safe, ASCII-only patterns"""
        results = []
        
        # Invocation (just capture raw, don't parse numbers)
        for i, line in enumerate(text.split('\n'), 1):
            if match := re.search(self.patterns["invocation"], line):
                results.append({
                    "line": i,
                    "type": "invocation",
                    "raw": match.group(1),
                    "safe": True
                })
            
            # Commands
            for cmd_match in re.finditer(self.patterns["command"], line):
                results.append({
                    "line": i,
                    "type": "command",
                    "command": cmd_match.group(1),
                    "safe": True
                })
            
            # Tool chain
            if chain_match := re.search(self.patterns["chain"], line):
                tools = [t.strip() for t in chain_match.group(1).split("→") if "_" in t or "!" in t]
                results.append({
                    "line": i,
                    "type": "chain",
                    "tools": tools,
                    "safe": True
                })
        
        return results

# === Session State ===
if "pac_parser" not in st.session_state:
    st.session_state.pac_parser = PACParser()
if "current_toolchain" not in st.session_state:
    st.session_state.current_toolchain = []

# === UI ===
st.title("🔨 PAC Compiler")
st.markdown("*Parse glyphs → executable tool chains*")

# FIXED: ASCII-safe default (no Unicode subscripts)
default_pac = """# ∴ Omni-Bootstrap Vortex ∴
⊙⟨ℵ2 ♠ E0⟩⊙ ≡ Ap⊛p_Infusion ⋅ chain{fs_list_files→agent_spawn→memory_query}
|
↓ ∮_t E(t) dt = ∫_{doubt}^{gnosis} (vec0.8 key0.2) / (z>2.5 ⋅ !LOVE) ⋅ lim{!PORT→socratic_council}
|
⇄ Ent = lim_{t→∞} [F(E0) ⋅ ⊕_{θ=0}^{2π} (!TRUTH ↔ !REBIRTH) ⋅ !ENGINE{engine_birth} ⋅ !BOOTSTRAP{agent_prime}]
"""

pac_input = st.text_area(
    "🜛 Enter PAC Code",
    value=default_pac,
    height=400,
    key="pac_editor"
)

# FIXED: Validation BEFORE parsing
if st.button("⚡ Parse PAC", use_container_width=True):
    # Step 1: Validate
    issues = validate_pac_syntax(pac_input)
    if issues:
        st.warning("PAC syntax issues detected:")
        for issue in issues:
            st.info(issue)
        
        if st.button("Continue Anyway", type="secondary"):
            pass  # Fall through to parse
        else:
            st.stop()  # Don't parse
    
    # Step 2: Safe parse
    with st.spinner("Compiling glyphs..."):
        toolchain = st.session_state.pac_parse_safe(pac_input)
        st.session_state.current_toolchain = toolchain
        st.success(f"Parsed {len(toolchain)} safe steps")

# Display results
if st.session_state.current_toolchain:
    st.subheader("📜 Parsed Tool Chain")
    for step in st.session_state.current_toolchain:
        with st.expander(f"Step {step['line']}: {step['type']}", expanded=False):
            st.json(step)

# Export button
st.download_button(
    "📥 Export Tool Chain",
    json.dumps(st.session_state.current_toolchain, indent=2),
    file_name="pac_toolchain.json",
    mime="application/json"
)

# === Sidebar: Unicode Guide ===
st.sidebar.header("🔤 Unicode Guide")
st.sidebar.info("""
**Avoid these (crash):**
- ℵ₂ → Use ℵ2
- 𝔼₀ → Use E0  
- ℵ∞ → Use ℵinf

**Safe glyphs:**
- ⊙⟨...⟩⊙
- ⋄⟨...⟩⋄
- ∴...∴
- ≡, ∮, ⇄, ↑, ↓
- !LOVE, !TRUTH, etc.
""")
