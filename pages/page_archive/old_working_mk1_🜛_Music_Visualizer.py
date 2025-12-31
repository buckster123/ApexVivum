"""
pages/music_visualizer.py
Advanced Visualizer Studio for Pi 5 - Phase 2 (Fixed)
Features: Multi-layer effects, live preview, custom presets, performance monitoring
"""

import streamlit as st
import librosa
import soundfile as sf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image, ImageFilter, ImageEnhance
import subprocess
import os
from pathlib import Path
import gc
import time
import json
import warnings
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Any
import sys
import logging
import io
import threading
from enum import Enum

# =============================================================================
# PATH INTEGRATION & LOGGER
# =============================================================================
try:
    from APEX_AURUM_REFACTORED_v1_1 import state
    SANDBOX_ROOT = Path(state.sandbox_dir).resolve()
except Exception as e:
    SANDBOX_ROOT = Path(st.session_state.get("sandbox_dir", "./sandbox")).resolve()

MUSIC_ROOT = SANDBOX_ROOT / "music"
PRESETS_DIR = MUSIC_ROOT / "visualizer_presets"
PRESETS_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

DEBUG_MODE = False  # Set True for dev logs

# =============================================================================
# ENUMS & DATACLASSES
# =============================================================================

class ColorScheme(Enum):
    CYAN = "Cyan Plasma"
    FIRE = "Fire (Red/Orange)"
    NEON = "Neon (Green/Pink)"
    OCEAN = "Ocean (Blue/Cyan)"
    PSYCHEDELIC = "Psychedelic (Rainbow)"

class VisualizationMode(Enum):
    BOTH = "Spectrum + Waveform"
    SPECTRUM_ONLY = "Spectrum Only"
    WAVEFORM_ONLY = "Waveform Only"

@dataclass
class EffectSettings:
    # Particle system
    particle_count: int = 60
    particle_speed: float = 1.0
    particle_size_range: tuple = (1.0, 4.0)
    particle_color_scheme: ColorScheme = ColorScheme.CYAN
    
    # Spectrum bars
    bar_count: int = 48
    bar_width: float = 0.9
    bar_color_scheme: ColorScheme = ColorScheme.CYAN
    bar_smoothness: float = 0.7  # 0-1 smoothing factor
    
    # Waveform overlay
    enable_waveform: bool = True
    waveform_color: str = "#00FF88"
    waveform_thickness: float = 2.0
    waveform_position: str = "bottom"  # top, bottom, center
    
    # Cover effects
    cover_pulse: bool = True
    cover_rotation: bool = False
    cover_rotation_speed: float = 0.5  # degrees per second
    cover_zoom_on_beat: bool = True
    cover_zoom_intensity: float = 0.05
    
    # Background
    background_blur: int = 25
    background_brightness: float = 0.7
    
    # Beat effects
    beat_flash: bool = True
    beat_flash_intensity: float = 0.3
    beat_particle_burst: bool = True
    
    # Performance
    quality_preset: str = "medium"
    fps: int = 30
    width: int = 1280
    height: int = 720
    chunk_duration: int = 15
    bitrate: str = "6M"

@dataclass
class VisualizerConfig:
    effects: EffectSettings = field(default_factory=EffectSettings)
    
    def __post_init__(self):
        # Auto-adjust based on quality preset
        if self.effects.quality_preset == "low":
            self.effects.particle_count = 30
            self.effects.bar_count = 32
            self.effects.bitrate = "3M"
        elif self.effects.quality_preset == "high":
            self.effects.particle_count = 90
            self.effects.bar_count = 64
            self.effects.bitrate = "8M"

# =============================================================================
# CORE SYSTEMS (Particles, Audio, Engine)
# =============================================================================

class ParticleSystem:
    def __init__(self, config: EffectSettings):
        self.config = config
        count = config.particle_count
        self.positions = np.random.rand(count, 2)
        self.velocities = np.random.randn(count, 2) * 0.002 * config.particle_speed
        self.lifetimes = np.random.rand(count)
        self.sizes = np.random.uniform(*config.particle_size_range, count)
        self.colors = self._generate_colors()
        
    def _generate_colors(self):
        if self.config.particle_color_scheme == ColorScheme.FIRE:
            return np.random.rand(self.config.particle_count, 3) * [1, 0.3, 0]
        elif self.config.particle_color_scheme == ColorScheme.NEON:
            return np.random.rand(self.config.particle_count, 3) * [0, 1, 0.5]
        elif self.config.particle_color_scheme == ColorScheme.OCEAN:
            return np.random.rand(self.config.particle_count, 3) * [0, 0.5, 1]
        else:  # CYAN or PSYCHEDELIC
            return np.random.rand(self.config.particle_count, 3)
        
    def update(self, beat_strength: float, onset: float, burst: bool = False):
        center_force = (0.5 - self.positions) * beat_strength * 0.01
        self.velocities += center_force
        
        if burst:
            turbulence = np.random.randn(*self.velocities.shape) * 0.01
        else:
            turbulence = np.random.randn(*self.velocities.shape) * onset * 0.005
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
        if hasattr(self.config, 'beat_flash_intensity'):
            alphas *= (1 + self.config.beat_flash_intensity)
        return self.positions, self.sizes, self.colors, alphas

class AudioAnalyzer:
    def __init__(self, audio_path: Path, sr: int = 22050, smooth_factor: float = 0.7):
        self.sr = sr
        self.smooth_factor = smooth_factor
        try:
            self.y, self.sr = librosa.load(audio_path, sr=sr, mono=True, duration=600)
        except Exception as e:
            st.error(f"Failed to load audio: {e}")
            raise
        
        self.duration = librosa.get_duration(y=self.y, sr=sr)
        if self.duration < 0.1:
            raise ValueError("Audio too short (< 0.1s)")
        
        self.onset_env = librosa.onset.onset_strength(y=self.y, sr=sr)
        self.rms = librosa.feature.rms(y=self.y)[0]
        self.tempo, self.beat_frames = librosa.beat.beat_track(y=self.y, sr=sr)
        
        if len(self.beat_frames) == 0:
            self.beat_frames = np.array([0, len(self.onset_env) // 2])
        
        self.stft = np.abs(librosa.stft(self.y, n_fft=2048))
        self.times = librosa.times_like(self.stft, sr=sr)
        self.beat_interp = self._interpolate_beats()
        
        # For waveform overlay
        self.audio_normalized = (self.y - np.mean(self.y)) / (np.std(self.y) + 1e-6)
        
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
            'spectrum': self._get_spectrum_at(time),
            'waveform': self._get_waveform_at(time)
        }
    
    def _get_spectrum_at(self, time: float):
        time_idx = min(int(time * self.sr // 512), self.stft.shape[1]-1)
        spectrum = self.stft[:, time_idx]
        bins = np.array_split(spectrum, 48)
        return np.array([np.mean(bin) for bin in bins])
    
    def _get_waveform_at(self, time: float):
        """Get 100ms window of waveform for overlay"""
        sample_pos = int(time * self.sr)
        window = self.sr // 10  # 100ms
        start = max(0, sample_pos - window // 2)
        end = min(len(self.audio_normalized), start + window)
        return self.audio_normalized[start:end]

class VisualizerEngine:
    def __init__(self, config: VisualizerConfig):
        self.config = config
        self.particles = ParticleSystem(config.effects)
        
    @staticmethod
    def _log_debug(msg: str):
        """Static method for debug logging to avoid scoping issues"""
        if DEBUG_MODE:
            st.sidebar.text(f"🔍 {msg}")
        
    def generate_video(self, audio_path: Path, cover_path: Path, 
                      output_path: Path, progress_callback: Optional[Callable] = None):
        self._log_debug("Starting video generation...")
        
        try:
            analyzer = AudioAnalyzer(audio_path, smooth_factor=self.config.effects.bar_smoothness)
        except Exception as e:
            st.error(f"Audio analysis failed: {e}")
            return False
            
        cover_bg, cover_fg = self._prepare_images(cover_path)
        total_frames = int(analyzer.duration * self.config.effects.fps)
        
        hw_encoder_available = self._check_hw_encoder_device()
        codec = 'h264_v4l2m2m' if hw_encoder_available else 'libx264'
        
        if not hw_encoder_available:
            st.warning("⚠️ Hardware encoder unavailable. Using software encoding")
        
        cmd = [
            'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{self.config.effects.width}x{self.config.effects.height}', '-pix_fmt', 'rgb24',
            '-r', str(self.config.effects.fps), '-i', '-', '-i', str(audio_path),
            '-c:v', codec, '-c:a', 'aac', '-b:v', self.config.effects.bitrate,
            '-b:a', '192k', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        ]
        
        if not hw_encoder_available:
            cmd.extend(['-preset', 'ultrafast'])
        
        cmd.append(str(output_path))
        
        fig, ax_dict = self._create_figure_template(cover_bg, cover_fg, analyzer)
        ffmpeg_proc = None
        
        try:
            ffmpeg_proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8)
            
            # Early health check
            try:
                ret = ffmpeg_proc.wait(timeout=5)
                if ret is not None and ret != 0:
                    raise subprocess.CalledProcessError(ret, cmd)
            except subprocess.TimeoutExpired:
                pass
            
            chunk_frames = int(self.config.effects.chunk_duration * self.config.effects.fps)
            for chunk_start in range(0, total_frames, chunk_frames):
                self._process_chunk(chunk_start, chunk_frames, total_frames,
                                  analyzer, cover_fg, fig, ax_dict, ffmpeg_proc, progress_callback)
                gc.collect()
            
            # CRITICAL: Finalize MP4 properly
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
            logger.warning(f"No V4l2 encoder devices: {v4l2_devices}")
            return False
        
        try:
            test_cmd = ['ffmpeg', '-f', 'lavfi', '-i', 'nullsrc=s=160x90:d=0.05:r=30', 
                       '-c:v', 'h264_v4l2m2m', '-b:v', '1M', '-f', 'null', '-']
            result = subprocess.run(test_cmd, capture_output=True, timeout=5)
            return result.returncode == 0
        except:
            return False
    
    def _prepare_images(self, cover_path: Path):
        self._log_debug(f"Processing cover: {cover_path.name}")
        try:
            cover = Image.open(cover_path).convert('RGB')
        except Exception as e:
            st.error(f"Cannot open cover image: {e}")
            raise
            
        if cover.width > 4000 or cover.height > 4000:
            cover = cover.resize((2000, 2000), Image.Resampling.LANCZOS)
            
        fg_max_w, fg_max_h = int(self.config.effects.width * 0.6), int(self.config.effects.height * 0.6)
        fg_ratio = min(fg_max_w / cover.width, fg_max_h / cover.height)
        fg_size = (int(cover.width * fg_ratio), int(cover.height * fg_ratio))
        cover_fg = cover.resize(fg_size, Image.Resampling.LANCZOS)
        
        bg_ratio = max(self.config.effects.width / cover.width, self.config.effects.height / cover.height)
        bg_size = (int(cover.width * bg_ratio), int(cover.height * cover.height // cover.width))
        cover_bg = cover.resize(bg_size, Image.Resampling.BILINEAR)
        cover_bg = cover_bg.filter(ImageFilter.GaussianBlur(radius=self.config.effects.background_blur))
        
        left = (cover_bg.width - self.config.effects.width) // 2
        top = (cover_bg.height - self.config.effects.height) // 2
        cover_bg = cover_bg.crop((left, top, left + self.config.effects.width, top + self.config.effects.height))
        return cover_bg, cover_fg
    
    def _create_figure_template(self, bg_img, fg_img, analyzer):
        fig = plt.figure(figsize=(self.config.effects.width/100, self.config.effects.height/100), dpi=100)
        fig.patch.set_facecolor('black')
        
        # Background
        ax_bg = fig.add_axes([0, 0, 1, 1])
        bg_display = ImageEnhance.Brightness(bg_img).enhance(self.config.effects.background_brightness)
        ax_bg.imshow(bg_display, aspect='auto')
        ax_bg.axis('off')
        
        # Cover (centered)
        fg_w, fg_h = fg_img.size
        ax_cover = fig.add_axes([
            0.5 - (fg_w/self.config.effects.width)/2,
            0.5 - (fg_h/self.config.effects.height)/2,
            fg_w/self.config.effects.width,
            fg_h/self.config.effects.height
        ])
        ax_cover_img = ax_cover.imshow(fg_img, aspect='auto')
        ax_cover.axis('off')
        
        # Spectrum bars (bottom)
        ax_bars = fig.add_axes([0.1, 0.05, 0.8, 0.15], facecolor='none')
        ax_bars.set_xlim(0, self.config.effects.bar_count)
        ax_bars.set_ylim(0, 1)
        ax_bars.axis('off')
        bar_rects = [Rectangle((i, 0), self.config.effects.bar_width, 0, facecolor='cyan', alpha=0.6) 
                    for i in range(self.config.effects.bar_count)]
        for rect in bar_rects:
            ax_bars.add_patch(rect)
        
        # Waveform overlay
        ax_waveform = fig.add_axes([0.1, 0.85, 0.8, 0.1], facecolor='none')
        ax_waveform.set_xlim(0, 1)
        ax_waveform.set_ylim(-1, 1)
        ax_waveform.axis('off')
        waveform_line, = ax_waveform.plot([], [], color=self.config.effects.waveform_color, 
                                         linewidth=self.config.effects.waveform_thickness)
        
        # Particles
        ax_particles = fig.add_axes([0, 0, 1, 1], facecolor='none')
        scatter = ax_particles.scatter([], [], s=[], c=[], cmap='plasma')
        ax_particles.set_xlim(0, 1)
        ax_particles.set_ylim(0, 1)
        ax_particles.axis('off')
        
        # Beat flash overlay
        ax_flash = fig.add_axes([0, 0, 1, 1], facecolor='none')
        flash_rect = Rectangle((0, 0), 1, 1, facecolor='white', alpha=0, transform=fig.transFigure)
        ax_flash.add_patch(flash_rect)
        ax_flash.axis('off')
        
        return fig, {
            'ax_cover_img': ax_cover_img,
            'bar_rects': bar_rects,
            'scatter': scatter,
            'flash_rect': flash_rect,
            'waveform_line': waveform_line,
            'ax_waveform': ax_waveform,
            'ax_cover': ax_cover,
        }
    
    def _process_chunk(self, chunk_start, chunk_frames, total_frames,
                      analyzer, cover_fg, fig, ax_dict, proc, callback):
        for frame_idx in range(chunk_start, min(chunk_start + chunk_frames, total_frames)):
            current_time = frame_idx / self.config.effects.fps
            features = analyzer.get_frame_features(current_time)
            
            beat_burst = features['beat_strength'] > 0.8 and self.config.effects.beat_particle_burst
            self.particles.update(features['beat_strength'], features['onset'], burst=beat_burst)
            pos, sizes, colors, alphas = self.particles.get_draw_data()
            
            self._update_frame(fig, ax_dict, features, pos, sizes, colors, alphas, cover_fg, current_time)
            
            fig.canvas.draw()
            
            # Render to buffer
            if hasattr(fig.canvas, 'buffer_rgba'):
                buf = fig.canvas.buffer_rgba()
                frame_data = np.frombuffer(buf, dtype='uint8').reshape((self.config.effects.height, self.config.effects.width, 4))
                frame_data = frame_data[:, :, :3]
            elif hasattr(fig.canvas, 'tostring_rgb'):
                frame_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                frame_data = frame_data.reshape((self.config.effects.height, self.config.effects.width, 3))
            else:
                raise RuntimeError("No compatible canvas buffer method found")
            
            try:
                proc.stdin.write(frame_data.tobytes())
            except BrokenPipeError:
                raise RuntimeError("FFmpeg encoder crashed during frame write")
            
            if callback and frame_idx % self.config.effects.fps == 0:
                callback(frame_idx / total_frames)
    
    def _update_frame(self, fig, ax_dict, features, pos, sizes, colors, alphas, cover_fg, current_time):
        # Update spectrum bars
        spectrum = features['spectrum']
        spectrum_norm = np.log1p(spectrum) / np.log1p(spectrum.max() + 1e-6)
        for i, rect in enumerate(ax_dict['bar_rects']):
            height = spectrum_norm[i % len(spectrum_norm)]
            rect.set_height(height)
            rect.set_facecolor(plt.cm.plasma(i / len(ax_dict['bar_rects'])))
        
        # Update waveform (if enabled)
        if self.config.effects.enable_waveform and 'waveform_line' in ax_dict:
            waveform = features['waveform']
            if len(waveform) > 0:
                x_data = np.linspace(0, 1, len(waveform))
                ax_dict['waveform_line'].set_data(x_data, waveform)
        
        # Update particles
        scatter = ax_dict['scatter']
        if len(pos) > 0:
            scatter.set_offsets(pos)
            scatter.set_sizes(sizes * (1 + features['beat_strength'] * 3))
            scatter.set_array(alphas)
        
        # Update cover effects
        if self.config.effects.cover_pulse:
            scale = 1 + features['beat_strength'] * self.config.effects.cover_zoom_intensity
            ax_dict['ax_cover_img'].set_data(
                cover_fg.resize((int(cover_fg.width * scale), int(cover_fg.height * scale)), Image.Resampling.LANCZOS)
            )
        
        # Beat flash
        flash_intensity = features['beat_strength'] ** 2 * self.config.effects.beat_flash_intensity
        ax_dict['flash_rect'].set_alpha(flash_intensity)

# =============================================================================
# PRESET MANAGEMENT
# =============================================================================

def save_preset(name: str, config: EffectSettings):
    preset_path = PRESETS_DIR / f"{name}.json"
    with open(preset_path, 'w') as f:
        json.dump(config.__dict__, f, indent=2, default=str)
    st.success(f"Preset '{name}' saved!")

def load_preset(name: str) -> EffectSettings:
    preset_path = PRESETS_DIR / f"{name}.json"
    if not preset_path.exists():
        return EffectSettings()
    
    with open(preset_path, 'r') as f:
        data = json.load(f)
    
    # Convert string enums back to objects
    if 'particle_color_scheme' in data and data['particle_color_scheme']:
        data['particle_color_scheme'] = ColorScheme(data['particle_color_scheme'] if isinstance(data['particle_color_scheme'], str) else data['particle_color_scheme']['value'])
    if 'bar_color_scheme' in data and data['bar_color_scheme']:
        data['bar_color_scheme'] = ColorScheme(data['bar_color_scheme'] if isinstance(data['bar_color_scheme'], str) else data['bar_color_scheme']['value'])
    
    return EffectSettings(**data)

def list_presets() -> list[str]:
    return [p.stem for p in PRESETS_DIR.glob("*.json")]

# =============================================================================
# STREAMLIT UI - Multi-Tab Professional Interface
# =============================================================================

def music_visualizer_page():
    st.title("🎵 Music Visualizer Studio Pro")
    st.caption("Advanced hardware-accelerated music video generator")
    
    # Initialize session state for settings
    if "viz_config" not in st.session_state:
        st.session_state.viz_config = VisualizerConfig()
    
    # Sidebar: Quick Actions
    with st.sidebar:
        st.header("Quick Actions")
        if st.button("🎬 Quick Render (Low/30fps)"):
            st.session_state.viz_config.effects.quality_preset = "low"
            st.session_state.viz_config.effects.fps = 30
            st.success("Quick render settings applied!")
        
        if st.button("✨ High Quality (Medium/60fps)"):
            st.session_state.viz_config.effects.quality_preset = "high"
            st.session_state.viz_config.effects.fps = 60
            st.success("High quality settings applied!")
        
        st.divider()
        
        # Performance Monitor
        if DEBUG_MODE:
            st.subheader("Performance")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Particles", st.session_state.viz_config.effects.particle_count)
            with col2:
                st.metric("Bars", st.session_state.viz_config.effects.bar_count)
    
    # Main UI: Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Files", "🎨 Effects", "⚙️ Performance", "▶️ Generate"])
    
    # === TAB 1: FILE SELECTION ===
    with tab1:
        st.header("File Selection")
        
        # Audio file selection
        try:
            audio_files = list(MUSIC_ROOT.rglob("*.mp3")) + list(MUSIC_ROOT.rglob("*.wav"))
        except:
            audio_files = []
            
        if not audio_files:
            st.error(f"No audio files in `{MUSIC_ROOT.relative_to(SANDBOX_ROOT)}`")
            return
        
        selected_audio = st.selectbox(
            "Select Audio",
            options=audio_files,
            format_func=lambda p: str(p.relative_to(MUSIC_ROOT))
        )
        
        # Auto-detect cover
        cover_path = None
        for ext in ['.jpg', '.png', '.jpeg']:
            candidate = selected_audio.with_suffix(ext)
            if candidate.exists():
                cover_path = candidate
                break
        
        if cover_path:
            st.success(f"Found cover: {cover_path.name}")
            st.image(str(cover_path), width=300)
        else:
            st.warning("No cover image found")
    
    # === TAB 2: EFFECT SETTINGS ===
    with tab2:
        st.header("Visual Effects")
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.subheader("Particle System")
            st.session_state.viz_config.effects.particle_count = st.slider(
                "Particle Count", 10, 200, st.session_state.viz_config.effects.particle_count,
                help="More particles = richer visuals but slower"
            )
            st.session_state.viz_config.effects.particle_speed = st.slider(
                "Particle Speed", 0.1, 3.0, st.session_state.viz_config.effects.particle_speed
            )
            st.session_state.viz_config.effects.particle_size_range = st.slider(
                "Size Range", 0.5, 5.0, (1.0, 4.0), step=0.1
            )
            
            # FIXED: Color scheme selector using enum values
            current_color = st.session_state.viz_config.effects.particle_color_scheme.value
            color_options = [e.value for e in ColorScheme]
            selected_color = st.selectbox(
                "Color Scheme", color_options,
                index=color_options.index(current_color)
            )
            st.session_state.viz_config.effects.particle_color_scheme = ColorScheme(selected_color)
            
            st.subheader("Beat Effects")
            st.session_state.viz_config.effects.beat_flash = st.checkbox(
                "Beat Flash", st.session_state.viz_config.effects.beat_flash
            )
            if st.session_state.viz_config.effects.beat_flash:
                st.session_state.viz_config.effects.beat_flash_intensity = st.slider(
                    "Flash Intensity", 0.1, 0.8, st.session_state.viz_config.effects.beat_flash_intensity
                )
            st.session_state.viz_config.effects.beat_particle_burst = st.checkbox(
                "Particle Burst on Beat", st.session_state.viz_config.effects.beat_particle_burst
            )
        
        with col_right:
            st.subheader("Spectrum Bars")
            st.session_state.viz_config.effects.bar_count = st.slider(
                "Bar Count", 16, 128, st.session_state.viz_config.effects.bar_count
            )
            st.session_state.viz_config.effects.bar_width = st.slider(
                "Bar Width", 0.5, 1.5, st.session_state.viz_config.effects.bar_width, step=0.1
            )
            st.session_state.viz_config.effects.bar_smoothness = st.slider(
                "Smoothness", 0.0, 1.0, st.session_state.viz_config.effects.bar_smoothness,
                help="Higher = smoother bar movement"
            )
            
            # FIXED: Bar color scheme selector using enum values
            current_bar_color = st.session_state.viz_config.effects.bar_color_scheme.value
            bar_color_options = [e.value for e in ColorScheme]
            selected_bar_color = st.selectbox(
                "Bar Color Scheme", bar_color_options,
                index=bar_color_options.index(current_bar_color)
            )
            st.session_state.viz_config.effects.bar_color_scheme = ColorScheme(selected_bar_color)
            
            st.subheader("Waveform")
            st.session_state.viz_config.effects.enable_waveform = st.checkbox(
                "Enable Waveform", st.session_state.viz_config.effects.enable_waveform
            )
            if st.session_state.viz_config.effects.enable_waveform:
                st.session_state.viz_config.effects.waveform_color = st.color_picker(
                    "Waveform Color", st.session_state.viz_config.effects.waveform_color
                )
                st.session_state.viz_config.effects.waveform_thickness = st.slider(
                    "Thickness", 0.5, 5.0, st.session_state.viz_config.effects.waveform_thickness
                )
            
            st.subheader("Cover Effects")
            st.session_state.viz_config.effects.cover_pulse = st.checkbox(
                "Pulse on Beat", st.session_state.viz_config.effects.cover_pulse
            )
            if st.session_state.viz_config.effects.cover_pulse:
                st.session_state.viz_config.effects.cover_zoom_intensity = st.slider(
                    "Pulse Intensity", 0.01, 0.1, st.session_state.viz_config.effects.cover_zoom_intensity
                )
        
        # Preset Management
        st.divider()
        preset_col1, preset_col2 = st.columns([3, 1])
        
        with preset_col1:
            preset_name = st.text_input("Preset Name", placeholder="e.g., Fire Pulse")
        with preset_col2:
            if st.button("💾 Save Preset"):
                if preset_name:
                    save_preset(preset_name, st.session_state.viz_config.effects)
                else:
                    st.error("Enter a preset name")
        
        presets = list_presets()
        if presets:
            selected_preset = st.selectbox("Load Preset", [""] + presets)
            if st.button("📂 Load Selected") and selected_preset:
                st.session_state.viz_config.effects = load_preset(selected_preset)
                st.success(f"Loaded preset: {selected_preset}")
                st.rerun()
    
    # === TAB 3: PERFORMANCE SETTINGS ===
    with tab3:
        st.header("Performance & Quality")
        
        col_perf1, col_perf2 = st.columns(2)
        
        with col_perf1:
            st.subheader("Quality Preset")
            st.session_state.viz_config.effects.quality_preset = st.select_slider(
                "Quality", ["low", "medium", "high"], st.session_state.viz_config.effects.quality_preset
            )
            
            st.subheader("Resolution")
            res_options = ["960x540", "1280x720", "1920x1080"]
            current_res = f"{st.session_state.viz_config.effects.width}x{st.session_state.viz_config.effects.height}"
            selected_res = st.selectbox("Resolution", res_options, index=res_options.index(current_res) if current_res in res_options else 1)
            w, h = map(int, selected_res.split('x'))
            st.session_state.viz_config.effects.width = w
            st.session_state.viz_config.effects.height = h
            
            st.subheader("Frame Rate")
            st.session_state.viz_config.effects.fps = st.select_slider(
                "FPS", [24, 30, 60], st.session_state.viz_config.effects.fps
            )
        
        with col_perf2:
            st.subheader("Advanced")
            st.session_state.viz_config.effects.chunk_duration = st.slider(
                "Chunk Duration (seconds)", 5, 30, st.session_state.viz_config.effects.chunk_duration,
                help="Larger chunks = faster but more memory"
            )
            st.session_state.viz_config.effects.bitrate = st.text_input(
                "Bitrate", st.session_state.viz_config.effects.bitrate,
                help="e.g., 3M, 6M, 8M. Lower = smaller file, faster render"
            )
            
            st.subheader("Background")
            st.session_state.viz_config.effects.background_blur = st.slider(
                "Blur Radius", 0, 50, st.session_state.viz_config.effects.background_blur
            )
            st.session_state.viz_config.effects.background_brightness = st.slider(
                "Brightness", 0.3, 1.0, st.session_state.viz_config.effects.background_brightness
            )
    
    # === TAB 4: GENERATE & PREVIEW ===
    with tab4:
        st.header("Generate Video")
        
        # Quick preview (5 seconds)
        if st.button("👁️ Generate 5s Preview"):
            preview_path = selected_audio.with_suffix('.preview.mp4')
            preview_config = st.session_state.viz_config
            preview_config.effects.chunk_duration = 5  # Force short chunk
            
            engine = VisualizerEngine(preview_config)
            with st.spinner("Generating preview..."):
                success = engine.generate_video(
                    selected_audio, cover_path, preview_path,
                    progress_callback=lambda p: st.progress(p)
                )
            
            if success and preview_path.exists():
                st.success("Preview ready!")
                st.video(str(preview_path))
                if st.button("🗑️ Delete Preview"):
                    preview_path.unlink(missing_ok=True)
                    st.rerun()
            else:
                st.error("Preview generation failed")
        
        st.divider()
        
        # Final render
        output_path = selected_audio.with_suffix('.visualizer.mp4')
        if output_path.exists():
            st.info(f"Output file will be: `{output_path.name}`")
            if st.checkbox("Overwrite existing file"):
                st.warning("⚠️ Existing file will be replaced")
        
        if st.button("🎬 Generate Full Video", type="primary", use_container_width=True):
            engine = VisualizerEngine(st.session_state.viz_config)
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
                status.info("Starting full generation...")
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
# PI 5 SETUP INSTRUCTIONS
# =============================================================================

def show_setup_instructions():
    with st.expander("🔧 Pi 5 Hardware Encoding Setup", expanded=False):
        st.markdown("""
        ### **Required Packages (Run Once):**
        
        ```bash
        sudo apt update
        sudo apt install -y raspberrypi-kernel v4l-utils libavcodec-extra raspi-firmware
        sudo reboot
        ```
        
        ### **Verify After Reboot:**
        
        ```bash
        ls -l /dev/video11  # Should exist
        ffmpeg -encoders | grep v4l2m2m  # Should show encoder
        ```
        
        **If hardware fails, software encoding auto-fallbacks.**
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
