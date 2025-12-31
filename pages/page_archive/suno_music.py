# pages/suno_music.py
"""
Apex Aurum Suno Music Generator
Standalone page for AI-powered music creation using Suno API
"""

import os
import sys
import json
import time
import asyncio
import httpx
import base64
import streamlit as st
import openai
from pathlib import Path
from typing import Dict, Any, Optional
import re
from datetime import datetime
import uuid
import logging

# CRITICAL: Import OpenAI class (not just the module)
from openai import OpenAI

# Import from parent module (main.py) - adjust path if needed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import existing infrastructure from main.py
from main import (
    API_KEY,
    Models,
    container,
    tool_limiter_sync,
    safe_call,
    get_session_cache,
    memory_insert,
    inject_convo_uuid,
    TOOL_DISPATCHER
)

# Set up logger for this page
logger = logging.getLogger(__name__)

# ============================================================================
# SUNO API CLIENT
# ============================================================================

class SunoClient:
    """Handles Suno API communication"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.suno.ai/v1"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    async def generate_music(
        self,
        prompt: str,
        make_instrumental: bool = False,
        wait_for_completion: bool = True,
        poll_interval: int = 5
    ) -> Dict[str, Any]:
        """Generate music and optionally wait for completion"""
        
        tool_limiter_sync()
        
        # Parse structured prompt from markdown boxes
        parsed = self._parse_prompt_structure(prompt)
        
        payload = {
            "prompt": parsed.get("lyrics", prompt),
            "make_instrumental": make_instrumental,
            "tags": parsed.get("style", ""),
            "title": parsed.get("title", "Apex Aurum Creation")
        }
        
        # Add optional parameters if present
        if "duration" in parsed:
            payload["duration"] = parsed["duration"]
        
        # Start generation
        async with self.client as client:
            response = await client.post(
                f"{self.base_url}/generate",
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            task = response.json()
            
            if not wait_for_completion:
                return {
                    "status": "queued",
                    "task_id": task.get("id"),
                    "message": "Generation started. Poll for status."
                }
            
            # Poll for completion
            task_id = task.get("id")
            return await self._poll_until_ready(client, task_id, poll_interval)
    
    async def _poll_until_ready(
        self, 
        client: httpx.AsyncClient, 
        task_id: str, 
        interval: int
    ) -> Dict[str, Any]:
        """Poll task status until complete"""
        
        status_url = f"{self.base_url}/generate/{task_id}"
        
        while True:
            response = await client.get(status_url, headers=self.headers)
            response.raise_for_status()
            status_data = response.json()
            
            state = status_data.get("status")
            
            if state == "complete":
                return {
                    "status": "complete",
                    "task_id": task_id,
                    "audio_url": status_data.get("audio_url"),
                    "video_url": status_data.get("video_url"),
                    "duration": status_data.get("duration"),
                    "metadata": status_data.get("metadata", {})
                }
            elif state in ["error", "failed"]:
                return {
                    "status": "failed",
                    "task_id": task_id,
                    "error": status_data.get("error", "Unknown error")
                }
            
            # Still processing
            await asyncio.sleep(interval)
    
    def _parse_prompt_structure(self, prompt: str) -> Dict[str, str]:
        """Extract structured data from markdown boxes"""
        
        result = {}
        
        # Extract title from first line if it's a header
        lines = prompt.split('\n')
        if lines and lines[0].startswith('# '):
            result["title"] = lines[0][2:].strip()
            lines = lines[1:]
            prompt = '\n'.join(lines)
        
        # Find markdown code blocks for specific sections
        patterns = {
            "lyrics": r'```(?:lyrics|text)?\s*\n(.*?)```',
            "style": r'```(?:style|tags)?\s*\n(.*?)```',
            "genre": r'```(?:genre)?\s*\n(.*?)```',
            "duration": r'```(?:duration)?\s*\n(.*?)```'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, prompt, re.DOTALL)
            if match:
                result[key] = match.group(1).strip()
        
        # If no lyrics block found, use entire prompt as lyrics
        if "lyrics" not in result:
            result["lyrics"] = prompt
        
        return result
    
    async def download_audio(self, audio_url: str) -> bytes:
        """Download audio file from URL"""
        
        async with self.client as client:
            response = await client.get(audio_url, headers=self.headers)
            response.raise_for_status()
            return response.content

# ============================================================================
# SUNO TOOL REGISTRATION
# ============================================================================

@inject_convo_uuid
async def suno_generate_music(
    prompt: str,
    make_instrumental: bool = False,
    wait_for_completion: bool = True,
    convo_uuid: str = None,
    user: str = "shared"
) -> str:
    """
    Generate music using Suno API. Provide structured prompt with markdown boxes:
    
    # Song Title
    
    ```lyrics
    Your lyrics here
    ```
    
    ```style
    genre, mood, instruments
    ```
    
    ```duration
    120
    ```
    """
    
    tool_limiter_sync()
    
    # Check for API key
    suno_key = os.getenv("SUNO_API_KEY")
    if not suno_key:
        return "Error: SUNO_API_KEY not set in .env file"
    
    try:
        async with SunoClient(suno_key) as client:
            result = await client.generate_music(
                prompt=prompt,
                make_instrumental=make_instrumental,
                wait_for_completion=wait_for_completion
            )
            
            # Store generation metadata in memory
            memory_insert(
                "suno_last_generation",
                {
                    "summary": f"Generated track: {result.get('status')}",
                    "details": result,
                    "prompt": prompt,
                    "timestamp": datetime.now().isoformat(),
                    "salience": 0.8
                },
                convo_uuid=convo_uuid
            )
            
            return json.dumps(result, indent=2)
            
    except Exception as e:
        logger.error(f"Suno generation error: {e}")
        return f"Error generating music: {str(e)}"

# Register the tool
container.register_tool(suno_generate_music)

# Add to dispatcher with async wrapper
def suno_generate_music_sync(**kwargs):
    """Synchronous wrapper for the async Suno tool"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(suno_generate_music(**kwargs))
        return result
    finally:
        loop.close()

TOOL_DISPATCHER["suno_generate_music"] = suno_generate_music_sync

# ============================================================================
# SUNO PROMPT ENGINEERING SYSTEM
# ============================================================================

SUNO_SYSTEM_PROMPT = """You are a Suno music generation assistant. Your ONLY task is to create structured prompts for Suno API.

CRITICAL OUTPUT FORMAT - Use EXACT markdown structure:

# [Song Title]

```lyrics
[Write complete song lyrics here]
Verse 1:
Your lyrics...

Chorus:
Your chorus...
```

```style
genre: [e.g., electronic pop, acoustic indie]
mood: [e.g., uplifting, melancholic, energetic]
instruments: [e.g., guitar, synth, piano]
vocal_style: [e.g., female vocal, male rap]
```

```duration
[Length in seconds, e.g., 120]
```

RULES:
1. ALWAYS include all three code blocks (lyrics, style, duration)
2. Lyrics should be complete and formatted (Verse, Chorus, etc.)
3. Style should be specific and detailed
4. Duration between 60-300 seconds
5. NO extra text outside the format
6. NO explanations or apologies
7. NO tool calls - just output the prompt

When user asks for music, create the prompt. When they provide song ideas, convert them to this format. If they ask to modify, regenerate the entire prompt."""

# ============================================================================
# STREAMLIT UI
# ============================================================================

def render_suno_chat():
    """Main chat interface for Suno music generation"""
    
    st.title("🎵 Apex Aurum Suno Studio")
    st.markdown("Generate AI music with structured prompts and Suno API")
    
    # Initialize session state
    if "suno_messages" not in st.session_state:
        st.session_state.suno_messages = []
    if "suno_convo_uuid" not in st.session_state:
        st.session_state.suno_convo_uuid = f"suno_{str(uuid.uuid4())}"
    if "audio_cache" not in st.session_state:
        st.session_state.audio_cache = {}
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Suno Settings")
        st.text_input(
            "Suno API Key",
            value=os.getenv("SUNO_API_KEY", ""),
            type="password",
            key="suno_api_key_input",
            help="Set SUNO_API_KEY in .env for persistence"
        )
        
        st.selectbox(
            "Generation Mode",
            ["Lyrics + Instrumental", "Instrumental Only"],
            key="instrumental_mode"
        )
        
        st.checkbox("Wait for completion", value=True, key="wait_for_completion")
        st.slider("Poll interval (seconds)", 3, 30, 5, key="poll_interval")
        
        if st.button("Clear History"):
            st.session_state.suno_messages = []
            st.session_state.audio_cache = {}
            st.rerun()
    
    # Chat history
    for msg in st.session_state.suno_messages:
        with st.chat_message(msg["role"]):
            if "content" in msg:
                st.markdown(msg["content"])
            
            # Display audio if available
            if msg["role"] == "assistant" and "audio_data" in msg:
                st.audio(msg["audio_data"], format="audio/mp3")
                st.download_button(
                    "Download MP3",
                    msg["audio_data"],
                    file_name=msg.get("audio_filename", "suno_track.mp3"),
                    mime="audio/mp3",
                    key=f"dl_{msg.get('id', '')}"
                )
            
            # Show generation metadata
            if "metadata" in msg:
                with st.expander("Generation Details"):
                    st.json(msg["metadata"])
    
    # Chat input
    if prompt := st.chat_input("Describe your song idea or paste a prompt..."):
        # Store user message
        user_msg = {"role": "user", "content": prompt}
        st.session_state.suno_messages.append(user_msg)
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate structured prompt with Kimi
        with st.chat_message("assistant"):
            with st.status("🎵 Composing prompt...", expanded=False) as status:
                try:
                    client = OpenAI(api_key=API_KEY, base_url="https://api.moonshot.ai/v1 ")
                    
                    # Generate structured prompt
                    response = client.chat.completions.create(
                        model=Models.KIMI_K2_THINKING.value,
                        messages=[
                            {"role": "system", "content": SUNO_SYSTEM_PROMPT},
                            {"role": "user", "content": f"Create a Suno prompt for: {prompt}"}
                        ],
                        stream=False,
                        max_tokens=2000
                    )
                    
                    structured_prompt = response.choices[0].message.content
                    st.code(structured_prompt, language="markdown")
                    
                    status.update(label="🚀 Generating music...", state="running")
                    
                    # Call Suno tool
                    make_instrumental = st.session_state.get("instrumental_mode") == "Instrumental Only"
                    
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    try:
                        suno_result = loop.run_until_complete(
                            suno_generate_music(
                                prompt=structured_prompt,
                                make_instrumental=make_instrumental,
                                wait_for_completion=True,  # Force wait for demo
                                convo_uuid=st.session_state.suno_convo_uuid,
                                user=st.session_state.get("user", "shared")
                            )
                        )
                        
                        result_data = json.loads(suno_result)
                        
                        if result_data.get("status") == "complete":
                            status.update(label="✅ Complete! Playing...", state="complete")
                            
                            # Download audio
                            audio_data = loop.run_until_complete(
                                SunoClient(os.getenv("SUNO_API_KEY")).download_audio(
                                    result_data["audio_url"]
                                )
                            )
                            
                            # Display audio
                            st.audio(audio_data, format="audio/mp3")
                            
                            # Store in session
                            msg_id = str(uuid.uuid4())
                            st.session_state.audio_cache[msg_id] = audio_data
                            
                            # Add to message history
                            assistant_msg = {
                                "role": "assistant",
                                "content": "🎵 Music generated successfully!",
                                "audio_data": audio_data,
                                "audio_filename": f"suno_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3",
                                "metadata": result_data,
                                "id": msg_id
                            }
                            st.session_state.suno_messages.append(assistant_msg)
                            
                            # Download button
                            st.download_button(
                                "Download MP3",
                                audio_data,
                                file_name=assistant_msg["audio_filename"],
                                mime="audio/mp3",
                                key=f"dl_main_{msg_id}"
                            )
                            
                            # Show metadata
                            with st.expander("Generation Details"):
                                st.json(result_data)
                        
                        elif result_data.get("status") == "queued":
                            status.update(label="⏳ Queued", state="complete")
                            st.info(f"Task queued: {result_data.get('task_id')}")
                            
                        else:
                            status.update(label="❌ Failed", state="error")
                            st.error(f"Generation failed: {result_data.get('error')}")
                    
                    except json.JSONDecodeError as e:
                        status.update(label="❌ Invalid response format", state="error")
                        st.error(f"Failed to parse Suno response: {e}")
                        st.code(suno_result)
                    
                    finally:
                        loop.close()
                        
                except Exception as e:
                    status.update(label="❌ Error", state="error")
                    st.error(f"Error: {str(e)}")
                    logger.error(f"Suno chat error: {e}", exc_info=True)

def main():
    """Entry point for Streamlit page"""
    st.set_page_config(
        page_title="Apex Aurum Suno Studio",
        page_icon="🎵",
        layout="wide"
    )
    
    # Check authentication (reuse main app logic)
    if not st.session_state.get("logged_in", False):
        st.error("Please login from the main app first.")
        if st.button("Go to Login"):
            st.switch_page("main.py")
        return
    
    render_suno_chat()

if __name__ == "__main__":
    main()
