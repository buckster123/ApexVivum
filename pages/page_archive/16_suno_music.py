import streamlit as st
import requests
import time
import os
import logging
from pathlib import Path
import base64
import json
from datetime import datetime

# ==================== CONFIGURATION ====================

# Create logs directory
log_folder = "logs"
Path(log_folder).mkdir(exist_ok=True)
log_file = Path(log_folder) / f"suno_client_{datetime.now().strftime('%Y%m%d')}.log"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Suno Music Generator",
    page_icon="🎵",
    layout="centered"
)

# Constants
API_PROVIDER_URLS = {
    "sunoapi.org": "https://api.sunoapi.org/api/v1",
    "kie.ai": "https://api.kie.ai/api/v1",
}
AUDIO_FOLDER = "generated_music"
Path(AUDIO_FOLDER).mkdir(exist_ok=True)

# ==================== SESSION STATE ====================

def init_session_state():
    defaults = {
        'api_key': "",
        'api_provider': "sunoapi.org",
        'task_id': None,
        'generation_complete': False,
        'audio_data': None,
        'debug_mode': False,
        'connection_tested': False,
        'credits': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ==================== API FUNCTIONS ====================

def detect_provider(api_key):
    """Detect API provider"""
    if api_key.startswith('sk-') and len(api_key) < 50:
        return "kie.ai"
    return "sunoapi.org"

def validate_key(api_key):
    """Validate API key format"""
    if not api_key or len(api_key) < 20:
        return False, "Invalid key format"
    if ' ' in api_key:
        return False, "Key contains spaces"
    return True, "Valid format"

def test_connection(api_key, provider):
    """Test API connection and get credits"""
    base_url = API_PROVIDER_URLS[provider]
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Test credits endpoint
    try:
        endpoint = f"{base_url}/generate/credit" if provider == "sunoapi.org" else f"{base_url}/credit"
        response = requests.get(endpoint, headers=headers, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            credits = result.get('data', 0)
            return True, f"Connected! Credits: {credits}", credits
        elif response.status_code == 401:
            return False, "Invalid API key", 0
        else:
            return False, f"Error {response.status_code}", 0
    except Exception as e:
        return False, f"Connection failed: {str(e)}", 0

def generate_music(api_key, prompt, model, is_instrumental, title=None, style=None):
    """Generate music"""
    provider = st.session_state.api_provider
    base_url = API_PROVIDER_URLS[provider]
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Build payload
    payload = {
        "model": model,
        "instrumental": is_instrumental,
    }
    
    if provider == "sunoapi.org":
        payload["customMode"] = not is_instrumental
        payload["callBackUrl"] = "https://example.com/callback"
        payload["prompt"] = prompt
        if title:
            payload["title"] = title
        if style:
            payload["style"] = style
    else:
        payload["prompt"] = prompt
        if title:
            payload["title"] = title
        if style:
            payload["style"] = style
    
    logger.info(f"Generation payload: {json.dumps(payload, indent=2)}")
    
    response = requests.post(
        f"{base_url}/generate",
        headers=headers,
        json=payload,
        timeout=30
    )
    
    if response.status_code != 200:
        raise Exception(f"API Error {response.status_code}: {response.text}")
    
    result = response.json()
    if result.get("code") != 200:
        raise Exception(f"Generation failed: {result.get('msg')}")
    
    return result["data"]["taskId"]

def check_status(api_key, task_id):
    """Check generation status"""
    provider = st.session_state.api_provider
    base_url = API_PROVIDER_URLS[provider]
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    response = requests.get(
        f"{base_url}/generate/record-info",
        headers=headers,
        params={"taskId": task_id},
        timeout=10
    )
    
    if response.status_code != 200:
        return {"status": "ERROR", "error": f"Status check failed: {response.status_code}"}
    
    result = response.json()
    if result.get("code") != 200:
        return {"status": "ERROR", "error": result.get("msg")}
    
    return result.get("data", {})

def download_audio(url, filename):
    """Download audio file"""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        filepath = Path(AUDIO_FOLDER) / filename
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        logger.info(f"Downloaded: {filepath} ({Path(filepath).stat().st_size} bytes)")
        return str(filepath)
    except Exception as e:
        logger.error(f"Download failed: {str(e)}")
        raise

def play_audio(filepath):
    """Play audio in browser"""
    with open(filepath, 'rb') as f:
        audio_bytes = f.read()
        b64 = base64.b64encode(audio_bytes).decode()
        st.markdown(
            f'<audio controls style="width:100%;"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>',
            unsafe_allow_html=True
        )

def get_safe_filename(title):
    """Create safe filename"""
    safe = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
    return f"{safe}_{int(time.time())}.mp3"

# ==================== MAIN APP ====================

def main():
    init_session_state()
    
    st.title("🎵 Suno Music Generator")
    st.markdown("Generate AI music with Suno API. Log file: `{}`".format(log_file))
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🔑 Setup")
        
        api_key = st.text_input(
            "API Key",
            type="password",
            value=st.session_state.api_key,
            help="From sunoapi.org/api-key"
        )
        
        if api_key != st.session_state.api_key:
            st.session_state.api_key = api_key
            st.session_state.connection_tested = False
        
        if api_key:
            is_valid, msg = validate_key(api_key)
            if not is_valid:
                st.error(f"❌ Format: {msg}")
                return
            
            if st.button("🔍 Test Connection", type="secondary"):
                with st.spinner("Testing..."):
                    provider = detect_provider(api_key)
                    st.session_state.api_provider = provider
                    
                    success, msg, credits = test_connection(api_key, provider)
                    if success:
                        st.session_state.connection_tested = True
                        st.session_state.credits = credits
                        st.success(f"✅ {msg}")
                        logger.info(f"API connected. Credits: {credits}")
                    else:
                        st.error(f"❌ {msg}")
                        logger.error(f"Connection failed: {msg}")
                        return
        
        if st.session_state.credits is not None:
            st.info(f"💰 Credits: {st.session_state.credits}")
        
        st.markdown("---")
        st.markdown("### 📖 Help")
        st.markdown("""
        - Prompt: Describe your music
        - Model: V3_5 is stable
        - Instrumental: No vocals
        - Audio saved to `generated_music/`
        """)
        
        if st.button("🗑️ Clear Key"):
            st.session_state.api_key = ""
            st.session_state.connection_tested = False
            st.session_state.credits = None
            st.rerun()
    
    # Main interface
    if not (st.session_state.api_key and st.session_state.connection_tested):
        st.info("🔑 Enter your API key and test connection to start")
        return
    
    # Generation form
    with st.form("music_form"):
        prompt = st.text_area(
            "Music Prompt",
            placeholder="A calm piano track for meditation...",
            height=100
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            model = st.selectbox("Model", ["V3_5", "V4", "V4_5", "V4_5PLUS", "V4_5ALL", "V5"])
            is_instrumental = st.checkbox("Instrumental Only", True)
        
        with col2:
            title = st.text_input("Title (opt)")
            style = st.text_input("Style Tags (opt)")
        
        submitted = st.form_submit_button("🎵 Generate", type="primary")
    
    if submitted:
        if not prompt.strip():
            st.error("❌ Prompt is required")
            return
        
        st.session_state.generation_complete = False
        st.session_state.audio_data = None
        
        # Submit generation
        try:
            with st.spinner("🚀 Submitting..."):
                task_id = generate_music(
                    st.session_state.api_key,
                    prompt,
                    model,
                    is_instrumental,
                    title or None,
                    style or None
                )
                st.session_state.task_id = task_id
                st.success(f"✅ Task started: `{task_id[:12]}...`")
                logger.info(f"Task submitted: {task_id}")
        except Exception as e:
            logger.error(f"Submission error: {str(e)}")
            st.error(f"❌ Failed: {str(e)}")
            return
        
        # Poll status
        status_container = st.empty()
        progress = st.progress(0)
        
        with st.spinner("⏳ Generating (2-4 minutes)..."):
            start = time.time()
            while time.time() - start < 300:  # 5 min timeout
                status = check_status(st.session_state.api_key, st.session_state.task_id)
                status_text = status.get("status", "UNKNOWN")
                
                with status_container:
                    if status_text == "PENDING":
                        st.info("⏳ In queue...")
                        progress.progress(10)
                    elif status_text == "GENERATING":
                        st.info("🎼 Generating audio...")
                        progress.progress(50)
                    elif status_text == "SUCCESS":
                        st.success("✅ Complete!")
                        progress.progress(100)
                        st.session_state.generation_complete = True
                        st.session_state.audio_data = status
                        break
                    elif status_text == "ERROR":
                        st.error(f"❌ Failed: {status.get('error')}")
                        logger.error(f"Task failed: {status.get('error')}")
                        return
                    else:
                        st.warning(f"🤔 Status: {status_text}")
                        logger.debug(f"Unknown status: {status_text}")
                
                time.sleep(5)
            
            else:
                logger.error("Generation timeout")
                st.error("⏱️ Timeout (5 minutes)")
                return
        
        # Display results
        if st.session_state.generation_complete:
            data = st.session_state.audio_data
            suno_data = data.get("response", {}).get("sunoData", [])
            
            if suno_data:
                st.markdown("---")
                st.subheader("🎧 Your Music")
                
                for i, track in enumerate(suno_data):
                    track_title = track.get("title", f"Track {i+1}")
                    audio_url = track.get("audioUrl")
                    duration = track.get("duration", 0)
                    
                    with st.container():
                        st.markdown(f"**{track_title}**")
                        st.caption(f"Duration: {duration:.1f}s")
                        
                        try:
                            filename = get_safe_filename(track_title)
                            with st.spinner("📥 Downloading..."):
                                filepath = download_audio(audio_url, filename)
                            
                            col1, col2 = st.columns([4, 1])
                            
                            with col1:
                                play_audio(filepath)
                            
                            with col2:
                                with open(filepath, 'rb') as f:
                                    st.download_button(
                                        "💾 Save",
                                        f,
                                        Path(filepath).name,
                                        "audio/mpeg",
                                        key=f"dl_{i}"
                                    )
                            
                            st.caption(f"Saved: `{filepath}`")
                            logger.info(f"Track ready: {track_title}")
                                
                        except Exception as e:
                            logger.error(f"Track error: {str(e)}")
                            st.error(f"Failed: {str(e)}")
            else:
                logger.warning("No tracks in response")
                st.warning("No audio data received")

if __name__ == "__main__":
    main()
