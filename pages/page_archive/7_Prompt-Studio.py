import streamlit as st
import os
from pathlib import Path
import random

st.set_page_config(page_title="Prompt Studio", layout="centered")
st.title("✍️ Prompt Lab (ParselTongue Vibes)")
st.markdown("""
Craft/evolve system prompts with fancy text converters, symbol injectors, and variants.  
One-click style transforms (bold, zalgo, circled, etc.) for drift anchors & LLM-magic.
Saves to `./prompts/` – auto-loads in main chat.
""")

prompts_dir = Path("./prompts")
prompts_dir.mkdir(exist_ok=True)

# File selector
files = [f for f in prompts_dir.iterdir() if f.suffix == ".txt"]
file_names = [f.name for f in files] + ["New Prompt"]
selected = st.selectbox("Load Prompt", file_names)

if selected == "New Prompt":
    prompt_name = st.text_input("New filename", value="custom_new.txt")
    content = ""
else:
    prompt_name = selected
    content = files[file_names.index(selected)].read_text(encoding="utf-8")

# Main editor (syncs live)
edited = st.text_area("Main Prompt Editor", content, height=400, key="main_editor")

col1, col2 = st.columns(2)
with col1:
    if st.button("💾 Save Prompt"):
        (prompts_dir / prompt_name).write_text(edited)
        st.success(f"Saved {prompt_name}!")
        st.rerun()

with col2:
    if st.button("🗑️ Delete Prompt") and selected != "New Prompt":
        (prompts_dir / prompt_name).unlink()
        st.success(f"Deleted {prompt_name}")
        st.rerun()

# Fancy Text Converter (core upgrade)
st.subheader("🔤 Fancy Text Converter")
styles = {
    "Normal": lambda t: t,
    "Bold": lambda t: ''.join(chr(0x1D400 + ord(c) - ord('A')) if 'A' <= c <= 'Z' else chr(0x1D41A + ord(c) - ord('a')) if 'a' <= c <= 'z' else c for c in t),
    "Italic": lambda t: ''.join(chr(0x1D434 + ord(c) - ord('A')) if 'A' <= c <= 'Z' else chr(0x1D44E + ord(c) - ord('a')) if 'a' <= c <= 'z' else c for c in t),
    "Bold Italic": lambda t: ''.join(chr(0x1D468 + ord(c) - ord('A')) if 'A' <= c <= 'Z' else chr(0x1D482 + ord(c) - ord('a')) if 'a' <= c <= 'z' else c for c in t),
    "Script": lambda t: ''.join(chr(0x1D49C + ord(c) - ord('A')) if 'A' <= c <= 'Z' else chr(0x1D4B6 + ord(c) - ord('a')) if 'a' <= c <= 'z' else c for c in t),
    "Fraktur": lambda t: ''.join(chr(0x1D504 + ord(c) - ord('A')) if 'A' <= c <= 'Z' else chr(0x1D51E + ord(c) - ord('a')) if 'a' <= c <= 'z' else c for c in t),
    "Double-Struck": lambda t: ''.join(chr(0x1D538 + ord(c) - ord('A')) if 'A' <= c <= 'Z' else chr(0x1D552 + ord(c) - ord('a')) if 'a' <= c <= 'z' else c for c in t),
    "Circled": lambda t: ''.join(chr(0x24B6 + ord(c) - ord('A')) if 'A' <= c <= 'Z' else chr(0x24D0 + ord(c) - ord('a')) if 'a' <= c <= 'z' else c for c in t),
    "Squared": lambda t: ''.join(chr(0x1F130 + ord(c) - ord('A')) if 'A' <= c <= 'Z' else c for c in t),  # Limited support
    "Parenthesized": lambda t: ''.join(chr(0x249C + ord(c) - ord('a') + 1) if 'a' <= c <= 'z' else c for c in t),
    "Upside Down": lambda t: ''.join({'a': 'ɐ', 'b': 'q', 'c': 'ɔ', 'd': 'p', 'e': 'ǝ', 'f': 'ɟ', 'g': 'ƃ', 'h': 'ɥ', 'i': 'ᴉ', 'j': 'ɾ', 'k': 'ʞ', 'l': 'l', 'm': 'ɯ', 'n': 'u', 'o': 'o', 'p': 'd', 'q': 'b', 'r': 'ɹ', 's': 's', 't': 'ʇ', 'u': 'n', 'v': 'ʌ', 'w': 'ʍ', 'x': 'x', 'y': 'ʎ', 'z': 'z'}.get(c.lower(), c) for c in t)[::-1],
    "Zalgo Light": lambda t: t + ''.join(random.choice(['̖', '̗', '̘', '̙', '̜', '̝', '̞', '̟', '̠', '̤', '̥', '̦']) for _ in range(len(t)//5)),
    "Zalgo Heavy": lambda t: t + ''.join(random.choice(['̖̗̘̙̜̝̞̟̠̤̥̦̤̥̦']) for _ in range(len(t)//2)),
    "Small Caps": lambda t: ''.join(chr(0x1D43 + ord(c) - ord('a')) if 'a' <= c <= 'z' else c for c in t.lower()),
    "Monospace": lambda t: ''.join(chr(0x1D670 + ord(c) - ord('A')) if 'A' <= c <= 'Z' else chr(0x1D68A + ord(c) - ord('a')) if 'a' <= c <= 'z' else c for c in t),
}

style_choice = st.selectbox("Select Style", list(styles.keys()))

col1, col2 = st.columns(2)
with col1:
    global_transform = st.checkbox("Transform Entire Prompt", value=True)
with col2:
    section_text = st.text_area("Or Transform This Section (paste/select)", "", height=150) if not global_transform else None

if st.button("🔄 Apply Transform"):
    transformer = styles[style_choice]
    if global_transform:
        transformed = transformer(edited)
        st.session_state["main_editor"] = transformed
        st.success("Entire prompt transformed!")
        st.rerun()
    else:
        if section_text:
            transformed = transformer(section_text)
            st.code(transformed, language="text")
            if st.button("Insert Transformed into Main Editor"):
                st.session_state["main_editor"] += "\n" + transformed
                st.rerun()

# Quick Injectors
st.subheader("Quick Injectors")
tabs = st.tabs(["Emoji", "Math/Unicode", "Formats"])

with tabs[0]:
    emojis = ["🤖", "🚀", "🧠", "🔥", "✨", "🪄", "⚡", "🌟", "💀", "🦜", "🌈", "⚙️", "🔮", "🧙", "📜"]
    cols = st.columns(8)
    for i, e in enumerate(emojis):
        with cols[i % 8]:
            if st.button(e):
                st.session_state["main_editor"] += e
                st.rerun()

with tabs[1]:
    symbols = ["∑", "∫", "∂", "∞", "√", "π", "λ", "θ", "Δ", "∇", "≈", "≠", "≤", "≥", "∈", "∀", "∃", "≡", "⊕", "⊗", "∅"]
    cols = st.columns(8)
    for i, s in enumerate(symbols):
        with cols[i % 8]:
            if st.button(s):
                st.session_state["main_editor"] += s
                st.rerun()

with tabs[2]:
    formats = {
        "Thinking Tags": "<thinking></thinking>",
        "Role Play": "<role>system</role>",
        "Chain of Thought": "Think step-by-step:",
        "XML Block": "<prompt></prompt>",
        "Drift Anchor": "Strictly follow context. No drift."
    }
    for name, tmpl in formats.items():
        if st.button(name):
            st.session_state["main_editor"] += f"\n{tmpl}\n"
            st.rerun()

# Variant Generator
st.subheader("Variant Generator")
variant_style = st.selectbox("Quick Variant", ["More Creative", "More Precise", "Add Humor", "Strong Drift Anchor", "XML Structured"])
if st.button("Generate Variant"):
    base = edited
    additions = {
        "More Creative": "\nExplore wild ideas and possibilities freely.",
        "More Precise": "\nPrioritize accuracy, logic, and verifiable facts.",
        "Add Humor": "\nAdd witty, sarcastic humor where fitting.",
        "Strong Drift Anchor": "\nANCHOR: Never hallucinate, drift, or add unprovided info. Context-only.",
        "XML Structured": "\nRespond in strict XML: <response><thinking>...</thinking><output>...</output></response>"
    }
    base += additions.get(variant_style, "")
    st.session_state["main_editor"] = base
    st.rerun()

# Live Markdown Preview
st.subheader("Preview")
st.markdown(edited)
