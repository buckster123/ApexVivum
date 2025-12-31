import streamlit as st
import requests
import time
import os
import logging
import sqlite3
import json
import base64
from pathlib import Path
from datetime import datetime

# ==================== CONFIGURATION & LOGGING ====================

# Paths
AUDIO_FOLDER = "generated_music"
DB_FILE = "suno_history.db"
LOG_FOLDER = "logs"
Path(AUDIO_FOLDER).mkdir(exist_ok=True)
Path(LOG_FOLDER).mkdir(exist_ok=True)

# Logging
log_file = Path(LOG_FOLDER) / f"suno_radio_{datetime.now().strftime('%Y%m%d')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# V5 Limits
V5_LIMITS = {
    "prompt": 5000,
    "style": 1000,
    "title": 100,
    "negative_tags": 500
}

# Page Config
st.set_page_config(
    page_title="🎵 Suno AI Radio",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== SESSION STATE ====================

def init_session_state():
    defaults = {
        'api_key': "",
        'api_provider': "sunoapi.org",
        'connection_tested': False,
        'credits': 0,
        'current_task': None,
        'queue': [],
        'auto_play': False,
        'stats': {'total_gens': 0, 'credits_used': 0, 'avg_time': 0},
        'history_loaded': False,
        'debug_mode': False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ==================== DATABASE FUNCTIONS ====================

def get_db():
    """Initialize SQLite DB"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS generations (
            id TEXT PRIMARY KEY,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            prompt TEXT,
            title TEXT,
            model TEXT,
            is_instrumental BOOLEAN,
            duration REAL,
            audio_url TEXT,
            audio_file TEXT,
            cover_images TEXT,
            status TEXT
        )
    """)
    conn.commit()
    return conn

def save_generation(data):
    """Save generation to database"""
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO generations 
            (id, prompt, title, model, is_instrumental, duration, audio_url, audio_file, cover_images, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data['id'],
            data['prompt'],
            data['title'],
            data['model'],
            data['is_instrumental'],
            data['duration'],
            data['audio_url'],
            data['audio_file'],
            json.dumps(data.get('cover_images', [])),
            data['status']
        ))
        conn.commit()
        conn.close()
        logger.info(f"Saved generation {data['id']} to database")
    except Exception as e:
        logger.error(f"Failed to save generation: {str(e)}")

def load_history(limit=100, offset=0, search=None):
    """Load generation history"""
    try:
        conn = get_db()
        cursor = conn.cursor()
        
        query = "SELECT * FROM generations WHERE 1=1"
        params = []
        
        if search:
            query += " AND (prompt LIKE ? OR title LIKE ?)"
            params.extend([f"%{search}%", f"%{search}%"])
        
        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(zip(['id', 'timestamp', 'prompt', 'title', 'model', 'is_instrumental', 
                          'duration', 'audio_url', 'audio_file', 'cover_images', 'status'], row)) 
                for row in rows]
    except Exception as e:
        logger.error(f"Failed to load history: {str(e)}")
        return []

def get_stats():
    """Get generation statistics"""
    try:
        conn = get_db()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*), SUM(duration), AVG(duration) FROM generations WHERE status = 'SUCCESS'")
        total_gens, total_duration, avg_duration = cursor.fetchone()
        
        cursor.execute("SELECT model, COUNT(*) FROM generations GROUP BY model")
        model_stats = dict(cursor.fetchall())
        
        cursor.execute("""
            SELECT DATE(timestamp), COUNT(*) 
            FROM generations 
            WHERE timestamp > datetime('now', '-30 days')
            GROUP BY DATE(timestamp)
        """)
        daily_stats = dict(cursor.fetchall())
        
        conn.close()
        
        return {
            'total_gens': total_gens or 0,
            'total_duration': total_duration or 0,
            'avg_duration': avg_duration or 0,
            'model_stats': model_stats,
            'daily_stats': daily_stats
        }
    except Exception as e:
        logger.error(f"Failed to get stats: {str(e)}")
        return {'total_gens': 0, 'total_duration': 0, 'avg_duration': 0, 
                'model_stats': {}, 'daily_stats': {}}

# ==================== API CLIENT ====================

API_PROVIDER_URLS = {
    "sunoapi.org": "https://api.sunoapi.org/api/v1",
    "kie.ai": "https://api.kie.ai/api/v1",
}

def detect_provider(api_key):
    return "kie.ai" if api_key.startswith('sk-') and len(api_key) < 50 else "sunoapi.org"

def test_connection(api_key, provider):
    base_url = API_PROVIDER_URLS[provider]
    headers = {"Authorization": f"Bearer {api_key}"}
    
    endpoint = f"{base_url}/{'generate/credit' if provider == 'sunoapi.org' else 'credit'}"
    response = requests.get(endpoint, headers=headers, timeout=10)
    
    if response.status_code == 200:
        result = response.json()
        return True, result.get('data', 0)
    return False, 0

def generate_music(api_key, prompt, model, is_instrumental, **kwargs):
    """Generate music with detailed error logging"""
    provider = st.session_state.api_provider
    base_url = API_PROVIDER_URLS[provider]
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # FIX: Force customMode=True to bypass 500 char limit
    payload = {
        "model": model,
        "instrumental": is_instrumental,
        "customMode": True,  # Always use custom mode for full control
        "callBackUrl": "https://example.com/callback",
        "prompt": prompt
    }
    
    # Add optional fields
    if kwargs.get('title'):
        payload['title'] = kwargs['title']
    if kwargs.get('style'):
        payload['style'] = kwargs['style']
    if kwargs.get('negativeTags'):
        payload['negativeTags'] = kwargs['negativeTags']
    if kwargs.get('styleWeight'):
        payload['styleWeight'] = kwargs['styleWeight']
    if kwargs.get('weirdnessConstraint'):
        payload['weirdnessConstraint'] = kwargs['weirdnessConstraint']
    if kwargs.get('vocalGender'):
        payload['vocalGender'] = kwargs['vocalGender']
    
    # DEBUG LOGGING
    logger.info("="*60)
    logger.info("GENERATION REQUEST")
    logger.info(f"Provider: {provider}")
    logger.info(f"URL: {base_url}/generate")
    logger.info(f"Payload: {json.dumps(payload, indent=2)}")
    logger.info("="*60)
    
    try:
        response = requests.post(
            f"{base_url}/generate",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        logger.info(f"Response Status Code: {response.status_code}")
        logger.info(f"Response Text: {response.text[:500]}...")
        
        if response.status_code != 200:
            raise Exception(f"HTTP Error {response.status_code}: {response.text[:300]}")
        
        result = response.json()
        logger.info(f"Parsed JSON: {json.dumps(result, indent=2)}")
        
        # Check API response code
        if result.get("code") != 200:
            error_msg = result.get('msg', 'No error message provided')
            error_data = result.get('data', {})
            raise Exception(f"API Error Code {result.get('code')}: {error_msg} | Data: {error_data}")
        
        # Get task ID
        task_id = result.get("data", {}).get("taskId")
        if not task_id:
            raise Exception(f"No taskId in response. Full response: {json.dumps(result)}")
        
        logger.info(f"Task ID received successfully: {task_id}")
        return task_id
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON response from API: {str(e)}")
        raise Exception(f"API returned invalid JSON: {response.text[:200]}")
    except requests.exceptions.Timeout:
        logger.error("Request timed out after 30 seconds")
        raise Exception("API request timed out - server took too long to respond")
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error: {str(e)}")
        raise Exception(f"Failed to connect to API: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in generate_music: {str(e)}")
        raise

def check_status(api_key, task_id):
    """Check generation status with detailed logging"""
    provider = st.session_state.api_provider
    base_url = API_PROVIDER_URLS[provider]
    
    headers = {"Authorization": f"Bearer {api_key}"}
    
    logger.info("="*60)
    logger.info(f"STATUS CHECK for task: {task_id}")
    logger.info(f"URL: {base_url}/generate/record-info")
    logger.info("="*60)
    
    try:
        response = requests.get(
            f"{base_url}/generate/record-info",
            headers=headers,
            params={"taskId": task_id},
            timeout=10
        )
        
        logger.info(f"Status Response Code: {response.status_code}")
        logger.info(f"Status Response Text: {response.text[:500]}...")
        
        if response.status_code != 200:
            logger.warning(f"Status check returned HTTP {response.status_code}")
            return {"status": "ERROR", "error": f"HTTP {response.status_code}: {response.text[:200]}"}
        
        result = response.json()
        logger.info(f"Status JSON: {json.dumps(result, indent=2)}")
        
        if result.get("code") != 200:
            error_msg = result.get('msg', 'Unknown error in status check')
            logger.warning(f"Status API error: {error_msg}")
            return {"status": "ERROR", "error": error_msg}
        
        data = result.get("data", {})
        logger.info(f"Task status: {data.get('status')} | Full data: {json.dumps(data)[:300]}...")
        
        return data
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in status response: {str(e)}")
        return {"status": "ERROR", "error": f"Invalid JSON: {response.text[:100]}"}
    except Exception as e:
        logger.error(f"Exception in check_status: {str(e)}")
        return {"status": "ERROR", "error": str(e)}

def generate_cover(api_key, task_id):
    """Generate cover art"""
    provider = st.session_state.api_provider
    base_url = API_PROVIDER_URLS[provider]
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "taskId": task_id,
        "callBackUrl": "https://example.com/callback"
    }
    
    logger.info(f"Generating cover for task {task_id}")
    
    try:
        response = requests.post(
            f"{base_url}/suno/cover/generate",
            headers=headers,
            json=payload,
            timeout=20
        )
        
        if response.status_code != 200:
            logger.warning(f"Cover generation HTTP {response.status_code}: {response.text[:200]}")
            return None
        
        result = response.json()
        logger.info(f"Cover generation response: {json.dumps(result)}")
        
        if result.get("code") == 409:  # Already exists
            return result["data"]["taskId"]
        elif result.get("code") == 200:
            return result["data"]["taskId"]
        
        return None
    except Exception as e:
        logger.error(f"Cover generation error: {str(e)}")
        return None

def check_cover_status(api_key, cover_task_id):
    """Check cover generation status"""
    provider = st.session_state.api_provider
    base_url = API_PROVIDER_URLS[provider]
    
    headers = {"Authorization": f"Bearer {api_key}"}
    
    try:
        response = requests.get(
            f"{base_url}/suno/cover/details",
            headers=headers,
            params={"taskId": cover_task_id},
            timeout=10
        )
        
        if response.status_code != 200:
            return {"status": "ERROR", "error": f"HTTP {response.status_code}"}
        
        result = response.json()
        return result.get("data", {})
    except Exception as e:
        logger.error(f"Cover status check error: {str(e)}")
        return {"status": "ERROR", "error": str(e)}

def download_file(url, filepath):
    """Download file with error handling"""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        file_size = Path(filepath).stat().st_size
        logger.info(f"Downloaded: {filepath} ({file_size} bytes)")
        return str(filepath)
    except Exception as e:
        logger.error(f"Download failed: {str(e)}")
        raise

# ==================== UI COMPONENTS ====================

def render_audio_player(filepath, title=""):
    """Render audio player"""
    try:
        if not Path(filepath).exists():
            st.error(f"Audio file not found: {filepath}")
            return
        
        with open(filepath, 'rb') as f:
            audio_bytes = f.read()
            b64 = base64.b64encode(audio_bytes).decode()
        
        audio_html = f"""
            <div style="margin: 10px 0;">
                <div style="font-weight: bold; margin-bottom: 5px;">{title}</div>
                <audio controls style="width:100%;">
                    <source src="data:audio/mp3;base64,{b64}" type="audio/mpeg">
                </audio>
            </div>
        """
        st.markdown(audio_html, unsafe_allow_html=True)
    except Exception as e:
        logger.error(f"Failed to render audio player: {str(e)}")
        st.error(f"Could not load audio: {str(e)}")

def render_image_grid(image_paths):
    """Render image grid"""
    try:
        if len(image_paths) == 1:
            st.image(image_paths[0], use_container_width=True)
        elif len(image_paths) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                st.image(image_paths[0], use_container_width=True)
            with col2:
                st.image(image_paths[1], use_container_width=True)
    except Exception as e:
        logger.error(f"Failed to render images: {str(e)}")

def show_character_counter(text, max_chars, label):
    """Show character counter"""
    count = len(text) if text else 0
    color = "red" if count > max_chars else "orange" if count > max_chars * 0.9 else "green"
    st.caption(f"{label}: {count}/{max_chars}")

# ==================== MAIN APP ====================

def main():
    init_session_state()
    
    st.title("🎵 Suno AI Radio Station")
    st.markdown("*Your infinite AI music generator*")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🔑 API Setup")
        
        api_key = st.text_input("API Key", type="password", value=st.session_state.api_key)
        if api_key != st.session_state.api_key:
            st.session_state.api_key = api_key
            st.session_state.connection_tested = False
        
        if api_key:
            if st.button("🔍 Test Connection", type="secondary"):
                with st.spinner("Testing..."):
                    provider = detect_provider(api_key)
                    st.session_state.api_provider = provider
                    success, credits = test_connection(api_key, provider)
                    if success:
                        st.session_state.connection_tested = True
                        st.session_state.credits = credits
                        st.success(f"✅ Connected! Credits: {credits}")
                    else:
                        st.error("❌ Connection failed")
        
        if st.session_state.connection_tested:
            st.info(f"💰 Credits: {st.session_state.credits}")
        
        st.markdown("### ⚙️ Options")
        st.session_state.auto_play = st.checkbox("🔄 Auto-Play Mode")
        st.session_state.debug_mode = st.checkbox("🔧 Debug Mode")
        
        if st.button("🗑️ Clear API Key"):
            st.session_state.api_key = ""
            st.session_state.connection_tested = False
            st.rerun()
        
        if st.session_state.debug_mode:
            st.markdown("### 🐛 Debug Info")
            st.text(f"Provider: {st.session_state.api_provider}")
            st.text(f"Connection: {st.session_state.connection_tested}")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["🎼 Generate", "📜 History", "📊 Stats"])
    
    # ==================== GENERATE TAB ====================
    with tab1:
        if not (st.session_state.api_key and st.session_state.connection_tested):
            st.info("🔑 Connect your API key in the sidebar to start")
            return
        
        # Model selection
        model = st.selectbox("Model", ["V3_5", "V4", "V4_5", "V4_5PLUS", "V4_5ALL", "V5"])
        is_v5 = model == "V5"
        
        # Inputs
        col1, col2 = st.columns([2, 1])
        
        with col1:
            prompt = st.text_area(
                "📝 Music Prompt",
                placeholder="Describe your song...",
                height=150,
                max_chars=V5_LIMITS["prompt"] if is_v5 else 3000
            )
            show_character_counter(prompt, V5_LIMITS["prompt"] if is_v5 else 3000, "Prompt")
        
        with col2:
            is_instrumental = st.checkbox("🎵 Instrumental", value=True)
            title = st.text_input("🎵 Title (opt)", max_chars=V5_LIMITS["title"] if is_v5 else 80)
            show_character_counter(title, V5_LIMITS["title"] if is_v5 else 80, "Title")
        
        # Advanced options
        with st.expander("🎚️ Advanced Options"):
            col3, col4 = st.columns(2)
            
            with col3:
                style = st.text_input("Style Tags")
                show_character_counter(style, V5_LIMITS["style"] if is_v5 else 200, "Style")
                negative_tags = st.text_input("Exclude Tags")
            
            with col4:
                style_weight = st.slider("Style Weight", 0.0, 1.0, 0.65, 0.05)
                weirdness = st.slider("Weirdness", 0.0, 1.0, 0.5, 0.05)
                vocal_gender = st.selectbox("Vocal Gender", ["auto", "m", "f"]) if not is_instrumental else "auto"
        
        # Generate button
        if st.button("🚀 Generate Music", type="primary", use_container_width=True):
            if not prompt.strip():
                st.error("❌ Prompt is required")
                return
            
            try:
                # Submit generation
                with st.spinner("🚀 Submitting request..."):
                    logger.info("User initiated generation")
                    task_id = generate_music(
                        st.session_state.api_key,
                        prompt,
                        model,
                        is_instrumental,
                        title=title or None,
                        style=style or None,
                        negativeTags=negative_tags or None,
                        styleWeight=style_weight,
                        weirdnessConstraint=weirdness,
                        vocalGender=None if vocal_gender == "auto" else vocal_gender
                    )
                    st.success(f"✅ Task submitted: `{task_id[:12]}...`")
                
                # Poll for completion
                progress = st.progress(0)
                status_container = st.empty()
                
                with st.spinner("⏳ Generating (2-4 minutes)..."):
                    while True:
                        status = check_status(st.session_state.api_key, task_id)
                        status_val = status.get("status", "UNKNOWN")
                        
                        with status_container:
                            if status_val == "PENDING":
                                progress.progress(20)
                                st.info("⏳ In queue...")
                            elif status_val == "GENERATING":
                                progress.progress(60)
                                st.info("🎼 Generating audio...")
                            elif status_val == "SUCCESS":
                                progress.progress(100)
                                st.success("✅ Generation complete!")
                                break
                            elif status_val == "ERROR":
                                error_msg = status.get('error', 'Unknown error during generation')
                                logger.error(f"Task failed: {error_msg}")
                                st.error(f"❌ Generation failed: {error_msg}")
                                return
                            else:
                                st.warning(f"🤔 Status: {status_val}")
                                logger.warning(f"Unknown status: {status_val}")
                        
                        time.sleep(5)
                
                # Process results
                logger.info("Processing successful generation results")
                suno_data = status.get("response", {}).get("sunoData", [])
                
                if not suno_data:
                    logger.error("No sunoData in response")
                    st.error("❌ No audio data received from API")
                    logger.debug(f"Full status response: {json.dumps(status)}")
                    return
                
                st.markdown("### 🎧 Generated Tracks")
                
                for i, track in enumerate(suno_data):
                    track_id = track.get('id', f"track_{i}")
                    track_title = track.get('title', f"Track {i+1}")
                    audio_url = track.get('audioUrl')
                    duration = track.get('duration', 0)
                    
                    if not audio_url:
                        logger.warning(f"No audioUrl for track {track_id}")
                        continue
                    
                    st.markdown(f"**{track_title}**")
                    st.caption(f"Duration: {duration:.1f}s | Model: {model}")
                    
                    # Download audio
                    safe_title = "".join(c for c in track_title if c.isalnum() or c in (' ', '-', '_')).rstrip()
                    filename = f"{safe_title}_{track_id[-8:]}.mp3"
                    filepath = download_file(audio_url, Path(AUDIO_FOLDER) / filename)
                    
                    # Save to database
                    save_generation({
                        'id': track_id,
                        'prompt': prompt,
                        'title': track_title,
                        'model': model,
                        'is_instrumental': is_instrumental,
                        'duration': duration,
                        'audio_url': audio_url,
                        'audio_file': filepath,
                        'cover_images': [],
                        'status': 'SUCCESS'
                    })
                    
                    # Display
                    render_audio_player(filepath, track_title)
                    
                    col1, col2 = st.columns([4, 1])
                    with col2:
                        with open(filepath, 'rb') as f:
                            st.download_button(
                                "💾 Save MP3",
                                f,
                                filename,
                                "audio/mpeg",
                                key=f"dl_{task_id}_{i}"
                            )
                    
                    # Cover generation button
                    if st.button(f"🎨 Generate Cover", key=f"cover_{track_id}"):
                        with st.spinner("Creating cover art..."):
                            cover_task_id = generate_cover(st.session_state.api_key, task_id)
                            if cover_task_id:
                                # Poll cover status
                                while True:
                                    cover_status = check_cover_status(st.session_state.api_key, cover_task_id)
                                    if cover_status.get('status') == 'SUCCESS':
                                        images = cover_status.get('images', [])
                                        if images:
                                            cover_paths = []
                                            for j, img_url in enumerate(images):
                                                ext = img_url.split('.')[-1].split('?')[0]
                                                cover_filename = f"{safe_title}_cover_{j+1}.{ext}"
                                                cover_path = download_file(img_url, Path(AUDIO_FOLDER) / cover_filename)
                                                cover_paths.append(cover_path)
                                            
                                            # Update DB
                                            conn = get_db()
                                            cursor = conn.cursor()
                                            cursor.execute(
                                                "UPDATE generations SET cover_images = ? WHERE id = ?",
                                                (json.dumps(cover_paths), track_id)
                                            )
                                            conn.commit()
                                            conn.close()
                                            
                                            render_image_grid(cover_paths)
                                        break
                                    time.sleep(3)
                            else:
                                st.warning("Cover generation failed or already exists")
                
            except Exception as e:
                logger.error(f"Generation error: {str(e)}", exc_info=True)
                st.error(f"❌ Error: {str(e)}")
                if st.session_state.debug_mode:
                    with st.expander("🔍 Debug Details"):
                        st.code(str(e))
                        st.text(f"Log file: {log_file}")
    
    # ==================== HISTORY TAB ====================
    with tab2:
        st.subheader("📜 Generation History")
        
        search = st.text_input("🔍 Search", placeholder="Search prompts or titles...")
        
        history = load_history(search=search)
        
        if not history:
            st.info("No history yet. Generate some music!")
        else:
            for entry in history:
                with st.expander(f"🎵 {entry['title']} ({entry['timestamp'][:16]})"):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**Prompt:** {entry['prompt'][:150]}...")
                        st.caption(f"Model: {entry['model']} | Duration: {entry['duration']:.1f}s")
                    
                    with col2:
                        if entry['audio_file'] and Path(entry['audio_file']).exists():
                            render_audio_player(entry['audio_file'], "")
                            
                            with open(entry['audio_file'], 'rb') as f:
                                st.download_button(
                                    "Save",
                                    f,
                                    Path(entry['audio_file']).name,
                                    "audio/mpeg",
                                    key=f"hist_dl_{entry['id']}"
                                )
                    
                    if entry['cover_images']:
                        covers = json.loads(entry['cover_images'])
                        if covers:
                            st.markdown("**Covers:**")
                            render_image_grid(covers)
    
    # ==================== STATS TAB ====================
    with tab3:
        st.subheader("📊 Usage Statistics")
        
        stats = get_stats()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Generations", stats['total_gens'])
        with col2:
            hours = stats['total_duration'] / 3600
            st.metric("Total Hours", f"{hours:.1f}h")
        with col3:
            st.metric("Avg Duration", f"{stats['avg_duration'] or 0:.1f}s")
        
        st.markdown("### Model Usage")
        if stats['model_stats']:
            models = list(stats['model_stats'].keys())
            counts = list(stats['model_stats'].values())
            st.bar_chart(dict(zip(models, counts)))
        else:
            st.info("No data yet")

if __name__ == "__main__":
    main()
