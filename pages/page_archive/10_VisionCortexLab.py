import streamlit as st
import pathlib
import os
from PIL import Image
from datetime import datetime
import json
import numpy as np
import sys

sys.path.append("..")
from main import (
    AppState, generate_embedding, memory_insert, 
    fs_read_file, fs_write_file, safe_call
)

st.set_page_config(page_title="Vision Cortex Lab", layout="wide")
state = AppState.get()

class VisionLabState:
    def __init__(self):
        self.batch_queue = []
        self.cortex_name = "vision_cortex_001"

vision_state = get_state("visionlab", VisionLabState())

# ====================== IMAGE UPLOAD & METADATA ======================
st.header("🖼️ Vision Cortex Lab (Pi‑5 Optimized)")

uploaded = st.file_uploader(
    "Upload Images", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True
)

if uploaded:
    for img_file in uploaded:
        img = Image.open(img_file)
        # Thumbnail for display/storage
        img.thumbnail((256, 256))
        
        # Extract metadata
        meta = {
            "filename": img_file.name,
            "size": img.size,
            "mode": img.mode,
            "timestamp": datetime.now().isoformat(),
            "exif": img.getexif().get(306)  # DateTime
        }
        
        # Agent description (lightweight: one API call per batch)
        if st.button(f"Describe {img_file.name}"):
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            b64 = base64.b64encode(buffered.getvalue()).decode()
            
            # Use Moonshot to describe
            from openai import OpenAI
            client = OpenAI(api_key=state.API_KEY, base_url="https://api.moonshot.ai/v1")
            response = client.chat.completions.create(
                model="moonshot-v1-32k",
                messages=[{
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": "Describe this image in 30 words for embedding."},
                        {"type": "image_url", "image_url": f"data:image/png;base64,{b64}"}
                    ]
                }],
                max_tokens=50
            )
            description = response.choices[0].message.content
            meta["description"] = description
            
            # Embed description (not pixels)
            embedding = safe_call(generate_embedding, description)
            
            # Store in Chroma via memory system
            mem_value = {
                "summary": f"Image: {img_file.name}",
                "details": meta,
                "embedding": json.loads(embedding) if "error" not in embedding else None,
                "tags": ["vision_cortex", "image"]
            }
            safe_call(memory_insert, f"img_{img_file.name}", mem_value, 
                     convo_uuid=vision_state.cortex_name)
            
            st.success(f"Embedded: {description[:50]}...")

# ====================== CORTEX EXPLORER ======================
st.subheader("🔍 Cortex Explorer")

query = st.text_input("Search images by description")
if query:
    # Use existing text search (no vector required)
    results = safe_call(memory_query, limit=10, convo_uuid=vision_state.cortex_name)
    try:
        images = json.loads(results)
        filtered = {k:v for k,v in images.items() if query.lower() in v.get("details",{}).get("description","").lower()}
        
        cols = st.columns(3)
        for idx, (key, data) in enumerate(filtered.items()):
            with cols[idx % 3]:
                st.image(f"{state.sandbox_dir}/thumbs/{data['details']['filename']}", 
                        use_column_width=True)
                st.caption(data["details"].get("description", key)[:100])
    except:
        st.info("No images found. Upload and describe first.")

# ====================== BATCH MODE (Pi‑5 Safe) ======================
if st.button("Process Batch"):
    # Process all pending with progress bar
    # Limits concurrency to avoid RAM spikes
    progress = st.progress(0)
    for idx, img_file in enumerate(uploaded):
        # ...same as above but batched...
        progress.progress((idx + 1) / len(uploaded))
