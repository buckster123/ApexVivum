"""
pages/music_visualizer.py
Advanced Visualizer Studio for Pi 5 - Phase 4 (Orientation-Aware Rendering)
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
from dataclasses import dataclass, field, fields
from typing import Optional, Callable, Dict, Any, Tuple
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

DEBUG_MODE = False

# =============================================================================
# ENUMS & DATACLASSES
# =============================================================================

class ColorScheme(Enum):
    CYAN = "Cyan Plasma"
    FIRE = "Fire (Red/Orange)"
    NEON = "Neon (Green/Pink)"
    OCEAN = "Ocean (Blue/Cyan)"
    PSYCHEDELIC = "Psychedelic (Rainbow)"

class PositionOption(Enum):
    TOP = "Top"
    BOTTOM = "Bottom"
    LEFT = "Left"
    RIGHT = "Right"
    CENTER = "Center"

@dataclass
class EffectSettings:
    # VERSION CONTROL - increment when adding fields
    config_version: int = 4
    
    # Particle system
    particle_count: int = 60
    particle_speed: float = 1.0
    particle_size_range: tuple = (1.0, 4.0)
    particle_color_scheme: ColorScheme = ColorScheme.CYAN
    
    # Spectrum bars
    bar_count: int = 48
    bar_width: float = 0.9
    bar_color_scheme: ColorScheme = ColorScheme.CYAN
    bar_smoothness: float = 0.7
    
    # NEW: Bar enhancements
    bar_mirror: bool = False
    bar_gradient: bool = True
    bar_rounded: bool = False
    
    # Waveform overlay
    enable_waveform: bool = True
    waveform_color: str = "#00FF88"
    waveform_thickness: float = 2.0
    
    # NEW: Waveform enhancements
    waveform_fill: bool = False
    waveform_mirror: bool = False
    
    # Positioning and Sizing Controls
    bar_position: PositionOption = PositionOption.BOTTOM
    bar_width_pct: float = 80
    bar_height_pct: float = 15
    bar_offset_pct: float = 5
    
    waveform_position: PositionOption = PositionOption.TOP
    waveform_width_pct: float = 80
    waveform_height_pct: float = 10
    waveform_offset_pct: float = 5
    
    # Cover effects
    cover_pulse: bool = True
    cover_rotation: bool = False
    cover_rotation_speed: float = 0.5
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
        if self.effects.quality_preset == "low":
            self.effects.particle_count = 30
            self.effects.bar_count = 32
            self.effects.bitrate = "3M"
        elif self.effects.quality_preset == "high":
            self.effects.particle_count = 90
            self.effects.bar_count = 64
            self.effects.bitrate = "8M"

# =============================================================================
# CONFIG MIGRATION UTILITY
# =============================================================================
def migrate_config(old_config: Any) -> VisualizerConfig:
    """Migrate old configs to current schema"""
    new_config = VisualizerConfig()
    
    if isinstance(old_config, VisualizerConfig):
        # Copy over existing fields
        for field in fields(old_config.effects):
            if hasattr(old_config.effects, field.name):
                setattr(new_config.effects, field.name, getattr(old_config.effects, field.name))
        
        # Ensure new fields have defaults
        for field in fields(new_config.effects):
            if not hasattr(new_config.effects, field.name):
                setattr(new_config.effects, field.name, field.default)
    
    return new_config

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
        else:
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
        sample_pos = int(time * self.sr)
        window = self.sr // 10
        start = max(0, sample_pos - window // 2)
        end = min(len(self.audio_normalized), start + window)
        return self.audio_normalized[start:end]

class VisualizerEngine:
    def __init__(self, config: VisualizerConfig):
        self.config = config
        self.particles = ParticleSystem(config.effects)
        
    @staticmethod
    def _log_debug(msg: str):
        if DEBUG_MODE:
            st.sidebar.text(f"🔍 {msg}")
    
    def _is_vertical_orientation(self, position: PositionOption) -> bool:
        """Check if position requires vertical rendering"""
        return position in [PositionOption.LEFT, PositionOption.RIGHT]
    
    def _get_bar_color(self, index: int, total: int):
        """Generate bar color based on selected scheme"""
        scheme = self.config.effects.bar_color_scheme
        
        if scheme == ColorScheme.FIRE:
            return plt.cm.Reds(index / total * 0.5 + 0.3)
        elif scheme == ColorScheme.NEON:
            # Alternate green/pink for neon effect
            return (0, 1, 0, 0.6) if index % 2 == 0 else (1, 0, 0.5, 0.6)
        elif scheme == ColorScheme.OCEAN:
            return plt.cm.Blues(index / total * 0.5 + 0.5)
        elif scheme == ColorScheme.CYAN:
            return (0, 1, 0.8, 0.6)  # Cyan with alpha
        else:  # PSYCHEDELIC
            return plt.cm.hsv(index / total)
        
    def _calculate_element_bounds(self, position: PositionOption, width_pct: float, 
                                height_pct: float, offset_pct: float) -> Tuple[float, float, float, float]:
        """Calculate matplotlib axes bounds [left, bottom, width, height] from position enum and percentages"""
        width = width_pct / 100.0
        height = height_pct / 100.0
        offset = offset_pct / 100.0
        
        if position == PositionOption.TOP:
            left = (1 - width) / 2
            bottom = 1 - height - offset
        elif position == PositionOption.BOTTOM:
            left = (1 - width) / 2
            bottom = offset
        elif position == PositionOption.LEFT:
            left = offset
            bottom = (1 - height) / 2
        elif position == PositionOption.RIGHT:
            left = 1 - width - offset
            bottom = (1 - height) / 2
        elif position == PositionOption.CENTER:
            left = (1 - width) / 2
            bottom = (1 - height) / 2
        else:
            left = (1 - width) / 2
            bottom = offset
        
        return [left, bottom, width, height]
        
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
            st.warning("⚠️ Hardware encoder unavailable. Using CPU-based software encoding")
        
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
            
            try:
                ffmpeg_proc.stdin.close()
            except:
                pass
            
            ret = ffmpeg_proc.wait()
            
            if ret != 0:
                stderr_output = ffmpeg_proc.stderr.read().decode() if ffmpeg_proc.stderr else "No stderr"
                raise subprocess.CalledProcessError(ret, cmd, stderr_output)
            
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
        # Pi 5 has different encoder paths than Pi 4
        # For now, return False to force CPU encoding which works reliably
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
        
        # Dynamic Spectrum Bars positioning
        bar_bounds = self._calculate_element_bounds(
            self.config.effects.bar_position,
            self.config.effects.bar_width_pct,
            self.config.effects.bar_height_pct,
            self.config.effects.bar_offset_pct
        )
        ax_bars = fig.add_axes(bar_bounds, facecolor='none')
        
        # Set orientation based on position
        is_vertical = self._is_vertical_orientation(self.config.effects.bar_position)
        if is_vertical:
            ax_bars.set_xlim(0, 1)
            ax_bars.set_ylim(0, self.config.effects.bar_count)
        else:
            ax_bars.set_xlim(0, self.config.effects.bar_count)
            ax_bars.set_ylim(0, 1)
        ax_bars.axis('off')
        
        # Create bars with correct orientation
        bar_rects = []
        for i in range(self.config.effects.bar_count):
            if is_vertical:
                # Vertical bars (portrait mode) - grow horizontally
                rect = Rectangle((0, i), 0, self.config.effects.bar_width, 
                                facecolor='cyan', alpha=0.6)
            else:
                # Horizontal bars (landscape mode) - grow vertically
                rect = Rectangle((i, 0), self.config.effects.bar_width, 0, 
                                facecolor='cyan', alpha=0.6)
            bar_rects.append(rect)
            ax_bars.add_patch(rect)
        
        if self.config.effects.enable_waveform:
            waveform_bounds = self._calculate_element_bounds(
                self.config.effects.waveform_position,
                self.config.effects.waveform_width_pct,
                self.config.effects.waveform_height_pct,
                self.config.effects.waveform_offset_pct
            )
            ax_waveform = fig.add_axes(waveform_bounds, facecolor='none')
            
            # Set waveform orientation
            if self._is_vertical_orientation(self.config.effects.waveform_position):
                ax_waveform.set_xlim(-1, 1)
                ax_waveform.set_ylim(0, 1)
            else:
                ax_waveform.set_xlim(0, 1)
                ax_waveform.set_ylim(-1, 1)
                
            ax_waveform.axis('off')
            
            # Create line
            if self.config.effects.waveform_fill:
                waveform_line, = ax_waveform.fill([], [], color=self.config.effects.waveform_color, alpha=0.3)
            else:
                waveform_line, = ax_waveform.plot([], [], color=self.config.effects.waveform_color, 
                                                 linewidth=self.config.effects.waveform_thickness)
        else:
            ax_waveform = None
            waveform_line = None
        
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
            'ax_bars': ax_bars,
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
        spectrum = features['spectrum']
        spectrum_norm = np.log1p(spectrum) / np.log1p(spectrum.max() + 1e-6)
        
        # Apply mirroring if enabled
        if self.config.effects.bar_mirror and len(spectrum_norm) > 1:
            # Create symmetrical mirrored spectrum
            mirrored = np.concatenate([spectrum_norm[::-1], spectrum_norm])
            spectrum_norm = mirrored[:len(ax_dict['bar_rects'])]
        
        # Update bars with orientation awareness
        is_vertical = self._is_vertical_orientation(self.config.effects.bar_position)
        
        for i, rect in enumerate(ax_dict['bar_rects']):
            if i >= len(spectrum_norm):
                break
            height = spectrum_norm[i]
            
            if is_vertical:
                # Vertical bars (grow horizontally)
                rect.set_width(height)
                rect.set_height(self.config.effects.bar_width)
                rect.set_xy((0, i))
            else:
                # Horizontal bars (grow vertically)
                rect.set_height(height)
                rect.set_width(self.config.effects.bar_width)
                rect.set_xy((i, 0))
            
            # Set color based on scheme
            rect.set_facecolor(self._get_bar_color(i, min(len(spectrum_norm), len(ax_dict['bar_rects']))))
        
        if self.config.effects.enable_waveform and ax_dict.get('waveform_line'):
            waveform = features['waveform']
            if len(waveform) > 0:
                if self.config.effects.waveform_mirror:
                    # Mirror vertically for symmetrical effect
                    waveform = np.concatenate([waveform, -waveform[::-1]])
                
                if self._is_vertical_orientation(self.config.effects.waveform_position):
                    # Vertical waveform (x=amplitude, y=time)
                    y_data = np.linspace(0, 1, len(waveform))
                    if self.config.effects.waveform_fill:
                        x_fill = np.concatenate([waveform, waveform[::-1]])
                        y_fill = np.concatenate([y_data, y_data[::-1]])
                        ax_dict['waveform_line'].set_xy(np.column_stack([x_fill, y_fill]))
                    else:
                        ax_dict['waveform_line'].set_data(waveform, y_data)
                else:
                    # Horizontal waveform (x=time, y=amplitude)
                    x_data = np.linspace(0, 1, len(waveform))
                    if self.config.effects.waveform_fill:
                        ax_dict['waveform_line'].set_xy(np.column_stack([x_data, waveform]))
                    else:
                        ax_dict['waveform_line'].set_data(x_data, waveform)
        
        scatter = ax_dict['scatter']
        if len(pos) > 0:
            scatter.set_offsets(pos)
            scatter.set_sizes(sizes * (1 + features['beat_strength'] * 3))
            scatter.set_array(alphas)
        
        if self.config.effects.cover_pulse:
            scale = 1 + features['beat_strength'] * self.config.effects.cover_zoom_intensity
            ax_dict['ax_cover_img'].set_data(
                cover_fg.resize((int(cover_fg.width * scale), int(cover_fg.height * scale)), Image.Resampling.LANCZOS)
            )
        
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
    if 'bar_position' in data and data['bar_position']:
        data['bar_position'] = PositionOption(data['bar_position'] if isinstance(data['bar_position'], str) else data['bar_position']['value'])
    if 'waveform_position' in data and data['waveform_position']:
        data['waveform_position'] = PositionOption(data['waveform_position'] if isinstance(data['waveform_position'], str) else data['waveform_position']['value'])
    
    return EffectSettings(**data)

def list_presets() -> list[str]:
    return [p.stem for p in PRESETS_DIR.glob("*.json")]

# =============================================================================
# POSITION PREVIEW RENDERER
# =============================================================================

def render_position_preview(config: EffectSettings, preview_placeholder):
    """Renders a live preview of element positioning"""
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    fig.patch.set_facecolor('#1E1E1E')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_facecolor('#2D2D2D')
    
    # Draw screen border
    screen_rect = Rectangle((5, 5), 90, 90, linewidth=2, edgecolor='white', facecolor='none')
    ax.add_patch(screen_rect)
    
    # Draw bars with orientation preview
    bar_width = config.bar_width_pct
    bar_height = config.bar_height_pct
    bar_offset = config.bar_offset_pct
    
    if config.bar_position == PositionOption.TOP:
        bar_x = (100 - bar_width) / 2
        bar_y = 100 - bar_height - bar_offset
    elif config.bar_position == PositionOption.BOTTOM:
        bar_x = (100 - bar_width) / 2
        bar_y = bar_offset
    elif config.bar_position == PositionOption.LEFT:
        bar_x = bar_offset
        bar_y = (100 - bar_height) / 2
    elif config.bar_position == PositionOption.RIGHT:
        bar_x = 100 - bar_width - bar_offset
        bar_y = (100 - bar_height) / 2
    elif config.bar_position == PositionOption.CENTER:
        bar_x = (100 - bar_width) / 2
        bar_y = (100 - bar_height) / 2
    
    # Show orientation in preview
    if config.bar_position in [PositionOption.LEFT, PositionOption.RIGHT]:
        # Vertical bars
        for i in range(5):
            x = bar_x + (bar_width / 5) * i
            rect = Rectangle((x, bar_y + bar_height/4), bar_width/5, bar_height/2, 
                           facecolor='cyan', alpha=0.7, label='Spectrum Bars' if i == 0 else "")
            ax.add_patch(rect)
    else:
        # Horizontal bars
        for i in range(5):
            y = bar_y + (bar_height / 5) * i
            rect = Rectangle((bar_x + bar_width/4, y), bar_width/2, bar_height/5, 
                           facecolor='cyan', alpha=0.7, label='Spectrum Bars' if i == 0 else "")
            ax.add_patch(rect)
    
    # Draw waveform with orientation preview
    if config.enable_waveform:
        wf_width = config.waveform_width_pct
        wf_height = config.waveform_height_pct
        wf_offset = config.waveform_offset_pct
        
        if config.waveform_position == PositionOption.TOP:
            wf_x = (100 - wf_width) / 2
            wf_y = 100 - wf_height - wf_offset
        elif config.waveform_position == PositionOption.BOTTOM:
            wf_x = (100 - wf_width) / 2
            wf_y = wf_offset
        elif config.waveform_position == PositionOption.LEFT:
            wf_x = wf_offset
            wf_y = (100 - wf_height) / 2
        elif config.waveform_position == PositionOption.RIGHT:
            wf_x = 100 - wf_width - wf_offset
            wf_y = (100 - wf_height) / 2
        elif config.waveform_position == PositionOption.CENTER:
            wf_x = (100 - wf_width) / 2
            wf_y = (100 - wf_height) / 2
        
        # Show orientation
        if config.waveform_position in [PositionOption.LEFT, PositionOption.RIGHT]:
            # Vertical line for portrait mode
            ax.plot([wf_x + wf_width/2] * 2, [wf_y, wf_y + wf_height], 
                   color=config.waveform_color, linewidth=2, label='Waveform')
        else:
            # Horizontal line for landscape mode
            ax.plot([wf_x, wf_x + wf_width], [wf_y + wf_height/2] * 2, 
                   color=config.waveform_color, linewidth=2, label='Waveform')
    
    # Add labels and legend
    ax.text(50, 2, "Screen Preview", ha='center', va='bottom', color='white', fontsize=10)
    ax.legend(loc='upper right', fontsize=8, facecolor='#2D2D2D', edgecolor='white')
    ax.axis('off')
    
    # Render to placeholder
    preview_placeholder.pyplot(fig)
    plt.close(fig)

# =============================================================================
# STREAMLIT UI - Multi-Tab Professional Interface
# =============================================================================

def music_visualizer_page():
    st.title("🎵 Music Visualizer Studio Pro")
    st.caption("Advanced hardware-accelerated music video generator")
    
    # INITIALIZE OR MIGRATE CONFIG
    if "viz_config" not in st.session_state:
        st.session_state.viz_config = VisualizerConfig()
    else:
        # Check if config needs migration
        current_effects = st.session_state.viz_config.effects
        if not hasattr(current_effects, 'config_version') or current_effects.config_version < 4:
            st.session_state.viz_config = migrate_config(st.session_state.viz_config)
            st.info("✅ Configuration migrated to new version with orientation controls!")
    
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
        
        if DEBUG_MODE:
            st.subheader("Performance")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Particles", st.session_state.viz_config.effects.particle_count)
            with col2:
                st.metric("Bars", st.session_state.viz_config.effects.bar_count)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Files", "🎨 Effects", "⚙️ Performance", "▶️ Generate"])
    
    with tab1:
        st.header("File Selection")
        
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
    
    with tab2:
        st.header("Visual Effects")
        
        st.subheader("📐 Layout & Positioning")
        
        preview_placeholder = st.empty()
        
        col_layout1, col_layout2 = st.columns(2)
        
        with col_layout1:
            st.markdown("**Spectrum Bars**")
            st.session_state.viz_config.effects.bar_position = PositionOption(
                st.selectbox(
                    "Position", [p.value for p in PositionOption],
                    index=[p.value for p in PositionOption].index(st.session_state.viz_config.effects.bar_position.value),
                    key="bar_pos_select"
                )
            )
            st.session_state.viz_config.effects.bar_width_pct = st.slider(
                "Width (%)", 20, 100, st.session_state.viz_config.effects.bar_width_pct,
                help="Width as percentage of screen",
                key="bar_width_slider"
            )
            st.session_state.viz_config.effects.bar_height_pct = st.slider(
                "Height (%)", 5, 40, st.session_state.viz_config.effects.bar_height_pct,
                help="Height as percentage of screen",
                key="bar_height_slider"
            )
            st.session_state.viz_config.effects.bar_offset_pct = st.slider(
                "Edge Offset (%)", 0, 20, st.session_state.viz_config.effects.bar_offset_pct,
                help="Distance from screen edge",
                key="bar_offset_slider"
            )
            
            # NEW: Bar enhancement controls
            st.session_state.viz_config.effects.bar_mirror = st.checkbox(
                "Mirror Symmetry", st.session_state.viz_config.effects.bar_mirror,
                help="Creates mirrored effect (great for center position)"
            )
            st.session_state.viz_config.effects.bar_gradient = st.checkbox(
                "Gradient Fill", st.session_state.viz_config.effects.bar_gradient,
                help="Gradient top-to-bottom or left-to-right"
            )
        
        with col_layout2:
            st.markdown("**Waveform Overlay**")
            st.session_state.viz_config.effects.waveform_position = PositionOption(
                st.selectbox(
                    "Position", [p.value for p in PositionOption],
                    index=[p.value for p in PositionOption].index(st.session_state.viz_config.effects.waveform_position.value),
                    key="wave_pos_select"
                )
            )
            st.session_state.viz_config.effects.waveform_width_pct = st.slider(
                "Width (%)", 20, 100, st.session_state.viz_config.effects.waveform_width_pct,
                help="Width as percentage of screen",
                key="wave_width_slider"
            )
            st.session_state.viz_config.effects.waveform_height_pct = st.slider(
                "Height (%)", 5, 30, st.session_state.viz_config.effects.waveform_height_pct,
                help="Height as percentage of screen",
                key="wave_height_slider"
            )
            st.session_state.viz_config.effects.waveform_offset_pct = st.slider(
                "Edge Offset (%)", 0, 20, st.session_state.viz_config.effects.waveform_offset_pct,
                help="Distance from screen edge",
                key="wave_offset_slider"
            )
            
            # NEW: Waveform enhancement controls
            if st.session_state.viz_config.effects.enable_waveform:
                st.session_state.viz_config.effects.waveform_fill = st.checkbox(
                    "Fill Area", st.session_state.viz_config.effects.waveform_fill,
                    help="Fills area under waveform curve"
                )
                st.session_state.viz_config.effects.waveform_mirror = st.checkbox(
                    "Mirror Waveform", st.session_state.viz_config.effects.waveform_mirror,
                    help="Great for left/right positions"
                )
        
        render_position_preview(st.session_state.viz_config.effects, preview_placeholder)
        
        st.divider()
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.subheader("Particle System")
            st.session_state.viz_config.effects.particle_count = st.slider(
                "Particle Count", 10, 200, st.session_state.viz_config.effects.particle_count
            )
            st.session_state.viz_config.effects.particle_speed = st.slider(
                "Particle Speed", 0.1, 3.0, st.session_state.viz_config.effects.particle_speed
            )
            st.session_state.viz_config.effects.particle_size_range = st.slider(
                "Size Range", 0.5, 5.0, (1.0, 4.0), step=0.1
            )
            
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
                "Smoothness", 0.0, 1.0, st.session_state.viz_config.effects.bar_smoothness
            )
            
            current_bar_color = st.session_state.viz_config.effects.bar_color_scheme.value
            bar_color_options = [e.value for e in ColorScheme]
            selected_bar_color = st.selectbox(
                "Bar Color Scheme", bar_color_options,
                index=bar_color_options.index(current_bar_color)
            )
            st.session_state.viz_config.effects.bar_color_scheme = ColorScheme(selected_bar_color)
            
            st.subheader("Waveform Style")
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
                "Chunk Duration (seconds)", 5, 30, st.session_state.viz_config.effects.chunk_duration
            )
            st.session_state.viz_config.effects.bitrate = st.text_input(
                "Bitrate", st.session_state.viz_config.effects.bitrate
            )
            
            st.subheader("Background")
            st.session_state.viz_config.effects.background_blur = st.slider(
                "Blur Radius", 0, 50, st.session_state.viz_config.effects.background_blur
            )
            st.session_state.viz_config.effects.background_brightness = st.slider(
                "Brightness", 0.3, 1.0, st.session_state.viz_config.effects.background_brightness
            )
    
    with tab4:
        st.header("Generate Video")
        
        if st.button("👁️ Generate 5s Preview"):
            preview_path = selected_audio.with_suffix('.preview.mp4')
            preview_config = st.session_state.viz_config
            preview_config.effects.chunk_duration = 5
            
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
        ### **Pi 5 Software Encoding Mode Active**
        
        The Raspberry Pi 5 currently uses **CPU-based software encoding** (libx264) as the hardware codec 
        `h264_v4l2m2m` is not yet optimized for Pi 5 architecture. This implementation:
        
        - Automatically falls back to software encoding
        - Uses `ultrafast` preset for reasonable performance
        - Pi 5's CPU handles 720p/30fps efficiently
        - Software encoding provides better quality control
        
        ### **Performance Tips for Pi 5:**
        
        - Use **low** or **medium** quality preset
        - Keep resolution at **1280x720** for best performance
        - Limit particle count to under 100
        - Use chunk duration of **15 seconds** (default)
        
        ### **Monitor CPU Temperature:**
        
        ```bash
        watch -n 2 vcgencmd measure_temp
        ```
        
        If temp > 75°C, reduce quality settings or add active cooling.
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
