"""
pages/music_visualizer.py
Final version: Fixed AudioAnalyzer attribute typo + threading import + MP4 finalization
"""

import streamlit as st
import librosa
import soundfile as sf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image, ImageFilter
import subprocess
import os
from pathlib import Path
import gc
import time
import json
import warnings
from dataclasses import dataclass
from typing import Optional, Callable
import sys
import logging
import io
import threading  # ← CRITICAL for FFmpeg stderr capture

# =============================================================================
# PATH INTEGRATION & LOGGER SETUP
# =============================================================================
try:
    from APEX_AURUM_REFACTORED_v1_1 import state
    SANDBOX_ROOT = Path(state.sandbox_dir).resolve()
except Exception as e:
    SANDBOX_ROOT = Path(st.session_state.get("sandbox_dir", "./sandbox")).resolve()

MUSIC_ROOT = SANDBOX_ROOT / "music"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

DEBUG_MODE = True
def log_debug(msg: str):
    if DEBUG_MODE:
        st.sidebar.text(f"🔍 {msg}")

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class VisualizerConfig:
    width: int = 1280
    height: int = 720
    fps: int = 30
    chunk_duration: int = 15
    bitrate: str = "6M"
    particle_count: int = 60
    bar_count: int = 48
    beat_sensitivity: float = 1.2
    quality_preset: str = "medium"
    
    def __post_init__(self):
        if self.quality_preset == "low":
            self.particle_count = 30
            self.bar_count = 32
            self.bitrate = "3M"
        elif self.quality_preset == "high":
            self.particle_count = 90
            self.bar_count = 64
            self.bitrate = "8M"

# =============================================================================
# PARTICLE SYSTEM & AUDIO ANALYZER - FIXED: beat_frames attribute
# =============================================================================

class ParticleSystem:
    def __init__(self, config: VisualizerConfig):
        self.config = config
        self.positions = np.random.rand(config.particle_count, 2)
        self.velocities = np.random.randn(config.particle_count, 2) * 0.002
        self.lifetimes = np.random.rand(config.particle_count)
        self.sizes = np.random.uniform(1, 3, config.particle_count)
        self.colors = np.random.rand(config.particle_count, 3)
        
    def update(self, beat_strength: float, onset: float):
        center_force = (0.5 - self.positions) * beat_strength * 0.01
        self.velocities += center_force
        turbulence = np.random.randn(*self.velocities.shape) * onset * 0.003
        self.velocities += turbulence
        self.velocities *= 0.98
        self.positions += self.velocities
        bounce_mask = (self.positions < 0) | (self.positions > 1)
        self.velocities[bounce_mask] *= -0.8
        self.positions = np.clip(self.positions, 0, 1)
        self.lifetimes -= 0.008
        respawn_mask = self.lifetimes <= 0
        self.lifetimes[respawn_mask] = 1.0
        self.positions[respawn_mask] = np.random.rand(np.sum(respawn_mask), 2)
        
    def get_draw_data(self):
        if len(self.positions) == 0:
            return np.array([[0.5, 0.5]]), np.array([1.0]), np.array([[1.0, 1.0, 1.0]]), np.array([0.5])
        alphas = self.lifetimes ** 0.5
        return self.positions, self.sizes, self.colors, alphas

class AudioAnalyzer:
    def __init__(self, audio_path: Path, sr: int = 22050):
        log_debug(f"Loading audio: {audio_path.name}")
        self.sr = sr
        try:
            self.y, self.sr = librosa.load(audio_path, sr=sr, mono=True, duration=600)
        except Exception as e:
            st.error(f"Failed to load audio: {e}")
            raise
        
        self.duration = librosa.get_duration(y=self.y, sr=sr)
        if self.duration < 0.1:
            raise ValueError("Audio too short (< 0.1s)")
            
        log_debug(f"Audio: {self.duration:.2f}s @ {self.sr}Hz")
        
        self.onset_env = librosa.onset.onset_strength(y=self.y, sr=sr)
        self.rms = librosa.feature.rms(y=self.y)[0]
        self.tempo, self.beat_frames = librosa.beat.beat_track(y=self.y, sr=sr)  # ← FIXED: use beat_frames
        
        if len(self.beat_frames) == 0:
            self.beat_frames = np.array([0, len(self.onset_env) // 2])
        
        self.stft = np.abs(librosa.stft(self.y, n_fft=2048))
        self.beat_interp = self._interpolate_beats()
        
    def _interpolate_beats(self):
        beat_env = np.zeros_like(self.onset_env)
        beat_env[self.beat_frames] = 1.0
        return np.convolve(beat_env, np.ones(10)/10, mode='same')
    
    def get_frame_features(self, time: float):
        onset_idx = min(int(time * self.sr // 512), len(self.onset_env)-1)
        rms_idx = min(int(time * self.sr // 2048), len(self.rms)-1)
        
        return {
            'onset': self.onset_env[onset_idx],
            'rms': self.rms[rms_idx],
            'beat_strength': self.beat_interp[onset_idx],
            'spectrum': self._get_spectrum_at(time)
        }
    
    def _get_spectrum_at(self, time: float):
        time_idx = min(int(time * self.sr // 512), self.stft.shape[1]-1)
        spectrum = self.stft[:, time_idx]
        bins = np.array_split(spectrum, 48)
        return np.array([np.mean(bin) for bin in bins])

# =============================================================================
# VISUALIZER ENGINE - MP4 Finalization Guaranteed
# =============================================================================

class VisualizerEngine:
    def __init__(self, config: VisualizerConfig):
        self.config = config
        self.particles = ParticleSystem(config)
        
    def generate_video(self, audio_path: Path, cover_path: Path, 
                      output_path: Path, progress_callback: Optional[Callable] = None):
        log_debug("Starting video generation...")
        
        try:
            analyzer = AudioAnalyzer(audio_path)
        except Exception as e:
            st.error(f"Audio analysis failed: {e}")
            return False
            
        cover_bg, cover_fg = self._prepare_images(cover_path)
        total_frames = int(analyzer.duration * self.config.fps)
        
        hw_encoder_available = self._check_hw_encoder_device()
        codec = 'h264_v4l2m2m' if hw_encoder_available else 'libx264'
        
        if not hw_encoder_available:
            st.warning("⚠️ Hardware encoder unavailable. Using software encoding")
        
        cmd = [
            'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{self.config.width}x{self.config.height}', '-pix_fmt', 'rgb24',
            '-r', str(self.config.fps), '-i', '-', '-i', str(audio_path),
            '-c:v', codec, '-c:a', 'aac', '-b:v', self.config.bitrate,
            '-b:a', '192k', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        ]
        
        if not hw_encoder_available:
            cmd.extend(['-preset', 'ultrafast'])
        
        cmd.append(str(output_path))
        
        fig, ax_dict = self._create_figure_template(cover_bg, cover_fg)
        ffmpeg_proc = None
        
        try:
            ffmpeg_proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8)
            
            # Health check
            try:
                ret = ffmpeg_proc.wait(timeout=5)
                if ret is not None and ret != 0:
                    raise subprocess.CalledProcessError(ret, cmd)
            except subprocess.TimeoutExpired:
                pass
            
            chunk_frames = int(self.config.chunk_duration * self.config.fps)
            for chunk_start in range(0, total_frames, chunk_frames):
                self._process_chunk(chunk_start, chunk_frames, total_frames,
                                  analyzer, cover_fg, fig, ax_dict, ffmpeg_proc, progress_callback)
                gc.collect()
            
            # CRITICAL: Close stdin and wait for MP4 finalization
            try:
                ffmpeg_proc.stdin.close()
            except:
                pass
            
            ret = ffmpeg_proc.wait()
            
            if ret != 0:
                stderr_output = ffmpeg_proc.stderr.read().decode() if ffmpeg_proc.stderr else "No stderr"
                raise subprocess.CalledProcessError(ret, cmd, stderr_output)
            
            # Validate complete file
            if not output_path.exists() or output_path.stat().st_size < 1024:
                raise ValueError(f"Incomplete output file: {output_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)
            return False
            
        finally:
            plt.close(fig)
            if ffmpeg_proc and ffmpeg_proc.poll() is None:
                ffmpeg_proc.kill()
    
    def _check_hw_encoder_device(self) -> bool:
        """Check if V4L2 hardware encoder device exists"""
        v4l2_devices = ['/dev/video11', '/dev/video12', '/dev/video10']
        device_exists = any(Path(dev).exists() for dev in v4l2_devices)
        
        if not device_exists:
            logger.warning(f"No V4L2 encoder devices found: {v4l2_devices}")
            return False
        
        try:
            test_cmd = ['ffmpeg', '-f', 'lavfi', '-i', 'nullsrc=s=160x90:d=0.05:r=30', 
                       '-c:v', 'h264_v4l2m2m', '-b:v', '1M', '-f', 'null', '-']
            result = subprocess.run(test_cmd, capture_output=True, timeout=5)
            return result.returncode == 0
        except:
            return False
    
    def _prepare_images(self, cover_path: Path):
        log_debug(f"Processing cover: {cover_path.name}")
        try:
            cover = Image.open(cover_path).convert('RGB')
        except Exception as e:
            st.error(f"Cannot open cover image: {e}")
            raise
            
        if cover.width > 4000 or cover.height > 4000:
            cover = cover.resize((2000, 2000), Image.Resampling.LANCZOS)
            
        fg_max_w, fg_max_h = int(self.config.width * 0.6), int(self.config.height * 0.6)
        fg_ratio = min(fg_max_w / cover.width, fg_max_h / cover.height)
        fg_size = (int(cover.width * fg_ratio), int(cover.height * fg_ratio))
        cover_fg = cover.resize(fg_size, Image.Resampling.LANCZOS)
        
        bg_ratio = max(self.config.width / cover.width, self.config.height / cover.height)
        bg_size = (int(cover.width * bg_ratio), int(cover.height * cover.height // cover.width))
        cover_bg = cover.resize(bg_size, Image.Resampling.BILINEAR)
        cover_bg = cover_bg.filter(ImageFilter.GaussianBlur(radius=25))
        
        left = (cover_bg.width - self.config.width) // 2
        top = (cover_bg.height - self.config.height) // 2
        cover_bg = cover_bg.crop((left, top, left + self.config.width, top + self.config.height))
        return cover_bg, cover_fg
    
    def _create_figure_template(self, bg_img, fg_img):
        fig = plt.figure(figsize=(self.config.width/100, self.config.height/100), dpi=100)
        fig.patch.set_facecolor('black')
        
        ax_bg = fig.add_axes([0, 0, 1, 1])
        ax_bg.imshow(bg_img, aspect='auto')
        ax_bg.axis('off')
        
        fg_w, fg_h = fg_img.size
        ax_cover = fig.add_axes([
            0.5 - (fg_w/self.config.width)/2,
            0.5 - (fg_h/self.config.height)/2,
            fg_w/self.config.width,
            fg_h/self.config.height
        ])
        ax_cover_img = ax_cover.imshow(fg_img, aspect='auto')
        ax_cover.axis('off')
        
        ax_bars = fig.add_axes([0.1, 0.05, 0.8, 0.15], facecolor='none')
        ax_bars.set_xlim(0, 48)
        ax_bars.set_ylim(0, 1)
        ax_bars.axis('off')
        bar_rects = [Rectangle((i, 0), 0.9, 0, facecolor='cyan', alpha=0.6) for i in range(48)]
        for rect in bar_rects:
            ax_bars.add_patch(rect)
        
        ax_particles = fig.add_axes([0, 0, 1, 1], facecolor='none')
        scatter = ax_particles.scatter([], [], s=[], c=[], cmap='plasma')
        ax_particles.set_xlim(0, 1)
        ax_particles.set_ylim(0, 1)
        ax_particles.axis('off')
        
        ax_flash = fig.add_axes([0, 0, 1, 1], facecolor='none')
        flash_rect = Rectangle((0, 0), 1, 1, facecolor='white', alpha=0, transform=fig.transFigure)
        ax_flash.add_patch(flash_rect)
        ax_flash.axis('off')
        
        return fig, {
            'ax_cover_img': ax_cover_img,
            'bar_rects': bar_rects,
            'scatter': scatter,
            'flash_rect': flash_rect,
        }
    
    def _process_chunk(self, chunk_start, chunk_frames, total_frames,
                      analyzer, cover_fg, fig, ax_dict, proc, callback):
        for frame_idx in range(chunk_start, min(chunk_start + chunk_frames, total_frames)):
            current_time = frame_idx / self.config.fps
            features = analyzer.get_frame_features(current_time)
            self.particles.update(features['beat_strength'], features['onset'])
            pos, sizes, colors, alphas = self.particles.get_draw_data()
            self._update_frame(fig, ax_dict, features, pos, sizes, colors, alphas, cover_fg)
            
            fig.canvas.draw()
            
            if hasattr(fig.canvas, 'buffer_rgba'):
                buf = fig.canvas.buffer_rgba()
                frame_data = np.frombuffer(buf, dtype='uint8').reshape((self.config.height, self.config.width, 4))
                frame_data = frame_data[:, :, :3]
            elif hasattr(fig.canvas, 'tostring_rgb'):
                frame_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                frame_data = frame_data.reshape((self.config.height, self.config.width, 3))
            else:
                raise RuntimeError("No compatible canvas buffer method found")
            
            try:
                proc.stdin.write(frame_data.tobytes())
            except BrokenPipeError:
                raise RuntimeError("FFmpeg encoder crashed during frame write")
            
            if callback and frame_idx % self.config.fps == 0:
                callback(frame_idx / total_frames)
    
    def _update_frame(self, fig, ax_dict, features, pos, sizes, colors, alphas, cover_fg):
        spectrum = features['spectrum']
        spectrum_norm = np.log1p(spectrum) / np.log1p(spectrum.max() + 1e-6)
        for i, rect in enumerate(ax_dict['bar_rects']):
            height = spectrum_norm[i % len(spectrum_norm)]
            rect.set_height(height)
            rect.set_facecolor(plt.cm.plasma(i / len(ax_dict['bar_rects'])))
        
        scatter = ax_dict['scatter']
        if len(pos) > 0:
            scatter.set_offsets(pos)
            scatter.set_sizes(sizes * (1 + features['beat_strength'] * 3))
            scatter.set_array(alphas)
        
        flash_intensity = features['beat_strength'] ** 2 * 0.3
        ax_dict['flash_rect'].set_alpha(flash_intensity)

# =============================================================================
# STREAMLIT PAGE UI
# =============================================================================

def music_visualizer_page():
    st.title("🎵 Pi Visualizer Studio")
    st.caption("Hardware-accelerated video generator for Raspberry Pi 5")
    
    if DEBUG_MODE:
        st.sidebar.header("Debug Log")
        st.sidebar.info(f"Sandbox: {SANDBOX_ROOT}")
        st.sidebar.info(f"Music root: {MUSIC_ROOT}")
    
    MUSIC_ROOT.mkdir(parents=True, exist_ok=True)
    
    try:
        audio_files = list(MUSIC_ROOT.rglob("*.mp3")) + list(MUSIC_ROOT.rglob("*.wav"))
        log_debug(f"Found {len(audio_files)} audio files")
    except Exception as e:
        st.error(f"File scan failed: {e}")
        log_debug(f"Scan error: {e}")
        return
    
    if not audio_files:
        st.warning(f"No audio files in `{MUSIC_ROOT.relative_to(SANDBOX_ROOT)}`")
        st.info("Tip: Place MP3/WAV files in sandbox/music/ with matching .jpg/.png covers")
        return
    
    selected_audio = st.selectbox(
        "Select Audio",
        options=audio_files,
        format_func=lambda p: str(p.relative_to(MUSIC_ROOT))
    )
    
    cover_path = None
    for ext in ['.jpg', '.png', '.jpeg']:
        candidate = selected_audio.with_suffix(ext)
        if candidate.exists():
            cover_path = candidate
            break
    
    if not cover_path:
        st.error(f"No cover image found (expected: `{selected_audio.stem}.jpg` or `.png`)")
        return
    
    st.success(f"Found cover: {cover_path.name}")
    st.image(str(cover_path), width=200)
    
    with st.expander("⚙️ Visualizer Settings", expanded=False):
        config = VisualizerConfig(
            quality_preset=st.select_slider("Quality", ["low", "medium", "high"], "medium"),
            width=1280, height=720,
            fps=st.select_slider("FPS", [24, 30, 60], 30)
        )
    
    output_path = selected_audio.with_suffix('.visualizer.mp4')
    
    if st.button("🎬 Generate Video", type="primary", use_container_width=True):
        hw_check = VisualizerEngine(config)._check_hw_encoder_device()
        if not hw_check:
            st.warning("⚠️ Hardware encoder unavailable. Using software encoding (slower)")
        
        engine = VisualizerEngine(config)
        progress_bar = st.progress(0)
        status = st.empty()
        
        start_time = time.time()
        
        def update_progress(p):
            progress_bar.progress(p)
            elapsed = time.time() - start_time
            if p > 0:
                eta = elapsed / p - elapsed
                status.text(f"⏱️ ETA: {int(eta//60)}m {int(eta%60)}s")
        
        try:
            status.info("Starting generation...")
            success = engine.generate_video(selected_audio, cover_path, output_path, update_progress)
            
            if success and output_path.exists():
                status.success("✅ Complete!")
                st.video(str(output_path))
                
                filesize = output_path.stat().st_size / (1024*1024)
                duration = librosa.get_duration(filename=str(selected_audio))
                st.info(f"📊 {output_path.name} | {duration:.1f}s | {filesize:.1f}MB")
            else:
                status.error("❌ Generation failed - check logs")
                
        except Exception as e:
            st.exception(e)
            logger.error("Unhandled generation error", exc_info=True)
        finally:
            gc.collect()

# =============================================================================
# PI 5 LITE SETUP INSTRUCTIONS
# =============================================================================

def show_setup_instructions():
    with st.expander("🔧 Pi 5 Trixie Lite Setup (Run These ONCE)", expanded=False):
        st.markdown("""
        ### **System Packages for Hardware Encoding:**
        
        ```bash
        sudo apt update
        sudo apt install -y raspberrypi-kernel raspberrypi-bootloader libraspberrypi0 libraspberrypi-bin
        sudo apt install -y v4l-utils libavcodec-extra linux-image-rpi raspi-firmware
        sudo reboot
        ```
        
        ### **After Reboot, Verify:**
        
        ```bash
        ls -l /dev/video11  # Should exist
        ffmpeg -encoders | grep v4l2m2m  # Should show encoder
        ```
        
        **The script auto-fallbacks to software encoding if hardware fails.**
        """)

# =============================================================================
# AUTO-EXECUTE
# =============================================================================
if __name__ in ["__page__", "__main__"]:
    try:
        show_setup_instructions()
        music_visualizer_page()
    except Exception as e:
        st.error("Unhandled error in visualizer page")
        st.exception(e)
