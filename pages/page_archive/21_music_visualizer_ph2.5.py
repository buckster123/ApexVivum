"""
pages/music_visualizer.py
Advanced Visualizer Studio for Pi 5 - Phase 2.5 (PRODUCTION READY)
Features: Particle physics, waveform patterns, bar geometry, audio channels
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
from typing import Optional, Callable, Any
from enum import Enum
import sys
import logging
import multiprocessing

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
# ENUMS & DATACLASSES - PHASE 2.5
# =============================================================================

class ColorScheme(Enum):
    CYAN = "Cyan Plasma"
    FIRE = "Fire (Red/Orange)"
    NEON = "Neon (Green/Pink)"
    OCEAN = "Ocean (Blue/Cyan)"
    PSYCHEDELIC = "Psychedelic (Rainbow)"
    MONOCHROME = "Monochrome"

class ParticlePhysics(Enum):
    BOUNCE = "Bounce (Box)"
    STREAM = "Stream (Wrap)"
    SPIRAL = "Spiral (Center)"
    GRAVITY = "Gravity (Fall)"
    VORTEX = "Vortex (Swirl)"

class WaveformPattern(Enum):
    LINE = "Line (Standard)"
    CIRCULAR = "Circular (Radial)"
    BARS = "Bars (Vertical)"

# FIXED: Removed ARC option causing ValueError
class BarGeometry(Enum):
    BOTTOM = "Bottom (Classic)"
    TOP = "Top"
    SIDES = "Sides (Left/Right)"
    CIRCLE = "Circle (Radial)"

class AudioMode(Enum):
    STEREO = "Stereo (Keep)"
    MONO = "Mono (Mixdown)"
    LEFT = "Left Channel Only"
    RIGHT = "Right Channel Only"

@dataclass
class EffectSettings:
    particle_count: int = 60
    particle_speed: float = 1.0
    particle_size_range: tuple = (1.0, 4.0)
    particle_color_scheme: ColorScheme = ColorScheme.CYAN
    particle_physics: ParticlePhysics = ParticlePhysics.BOUNCE
    particle_gravity: float = 0.001
    
    bar_count: int = 48
    bar_width: float = 0.9
    bar_color_scheme: ColorScheme = ColorScheme.CYAN
    bar_smoothness: float = 0.7
    bar_geometry: BarGeometry = BarGeometry.BOTTOM
    bar_height_scale: float = 0.2
    
    enable_waveform: bool = True
    waveform_color: str = "#00FF88"
    waveform_thickness: float = 2.0
    waveform_pattern: WaveformPattern = WaveformPattern.LINE
    
    cover_pulse: bool = True
    cover_rotation: bool = False
    cover_rotation_speed: float = 0.5
    cover_zoom_on_beat: bool = True
    cover_zoom_intensity: float = 0.05
    
    background_blur: int = 25
    background_brightness: float = 0.7
    
    beat_flash: bool = True
    beat_flash_intensity: float = 0.3
    beat_particle_burst: bool = True
    
    quality_preset: str = "medium"
    fps: int = 30
    width: int = 1280
    height: int = 720
    chunk_duration: int = 15
    bitrate: str = "6M"
    audio_mode: AudioMode = AudioMode.STEREO

    @classmethod
    def with_defaults(cls, **kwargs):
        """Safe constructor with enum validation"""
        defaults = cls()
        for f in fields(defaults):
            if f.name not in kwargs:
                kwargs[f.name] = getattr(defaults, f.name)
        
        enum_fields = {
            'particle_color_scheme': ColorScheme,
            'bar_color_scheme': ColorScheme,
            'particle_physics': ParticlePhysics,
            'waveform_pattern': WaveformPattern,
            'bar_geometry': BarGeometry,
            'audio_mode': AudioMode
        }
        
        for key, enum_class in enum_fields.items():
            if key in kwargs and kwargs[key]:
                value = kwargs[key]
                valid_values = [e.value for e in enum_class]
                
                if isinstance(value, str) and value in valid_values:
                    kwargs[key] = enum_class(value)
                elif isinstance(value, dict) and 'value' in value and value['value'] in valid_values:
                    kwargs[key] = enum_class(value['value'])
                else:
                    kwargs[key] = getattr(defaults, key)
        
        return cls(**kwargs)

@dataclass
class VisualizerConfig:
    effects: EffectSettings = field(default_factory=EffectSettings)
    
    def __post_init__(self):
        if self.effects.quality_preset == "low":
            self.effects.particle_count = max(30, self.effects.particle_count // 2)
            self.effects.bar_count = max(32, self.effects.bar_count // 2)
            self.effects.bitrate = "3M"
        elif self.effects.quality_preset == "high":
            self.effects.particle_count = min(90, int(self.effects.particle_count * 1.5))
            self.effects.bar_count = min(64, int(self.effects.bar_count * 1.5))
            self.effects.bitrate = "8M"

# =============================================================================
# CORE SYSTEMS
# =============================================================================

class ParticleSystem:
    def __init__(self, config: EffectSettings):
        self.config = config
        self.reset()
        
    def reset(self):
        """Re-initialize particles"""
        count = self.config.particle_count
        self.positions = np.random.rand(count, 2)
        self.velocities = np.random.randn(count, 2) * 0.002 * self.config.particle_speed
        self.lifetimes = np.random.rand(count)
        self.sizes = np.random.uniform(*self.config.particle_size_range, count)
        self.colors = self._generate_colors()
        
    def _generate_colors(self):
        scheme = self.config.particle_color_scheme
        if scheme == ColorScheme.FIRE:
            return np.random.rand(self.config.particle_count, 3) * [1, 0.3, 0]
        elif scheme == ColorScheme.NEON:
            return np.random.rand(self.config.particle_count, 3) * [0, 1, 0.5]
        elif scheme == ColorScheme.OCEAN:
            return np.random.rand(self.config.particle_count, 3) * [0, 0.5, 1]
        elif scheme == ColorScheme.MONOCHROME:
            gray = np.random.rand(self.config.particle_count)
            return np.column_stack([gray, gray, gray])
        else:
            return np.random.rand(self.config.particle_count, 3)
        
    def update(self, beat_strength: float, onset: float, burst: bool = False):
        physics = self.config.particle_physics
        
        if physics == ParticlePhysics.BOUNCE:
            self._physics_bounce(beat_strength, onset, burst)
        elif physics == ParticlePhysics.STREAM:
            self._physics_stream(beat_strength, onset, burst)
        elif physics == ParticlePhysics.SPIRAL:
            self._physics_spiral(beat_strength, onset, burst)
        elif physics == ParticlePhysics.GRAVITY:
            self._physics_gravity(beat_strength, onset, burst)
        elif physics == ParticlePhysics.VORTEX:
            self._physics_vortex(beat_strength, onset, burst)
        
        self.lifetimes -= 0.008
        respawn_mask = self.lifetimes <= 0
        self.lifetimes[respawn_mask] = 1.0
        self.positions[respawn_mask] = np.random.rand(np.sum(respawn_mask), 2)
        
        if physics == ParticlePhysics.SPIRAL:
            if not hasattr(self, 'angles'):
                self.angles = np.random.rand(len(self.positions)) * 2 * np.pi
            self.angles[respawn_mask] = np.random.rand(np.sum(respawn_mask)) * 2 * np.pi
        
    def _physics_bounce(self, beat_strength, onset, burst):
        center_force = (0.5 - self.positions) * beat_strength * 0.01
        self.velocities += center_force
        
        if burst:
            turbulence = np.random.randn(*self.velocities.shape) * 0.01 * self.config.particle_speed
        else:
            turbulence = np.random.randn(*self.velocities.shape) * onset * 0.005 * self.config.particle_speed
        
        self.velocities += turbulence
        self.velocities *= 0.98
        self.positions += self.velocities
        
        bounce_mask = (self.positions < 0) | (self.positions > 1)
        self.velocities[bounce_mask] *= -0.8
        self.positions = np.clip(self.positions, 0, 1)
    
    def _physics_stream(self, beat_strength, onset, burst):
        drift = np.array([0.02, 0]) * self.config.particle_speed
        self.velocities += drift
        
        if burst:
            turbulence = np.random.randn(*self.velocities.shape) * 0.01
        else:
            turbulence = np.random.randn(*self.velocities.shape) * onset * 0.005
        
        self.velocities += turbulence
        self.velocities *= 0.98
        self.positions += self.velocities
        self.positions %= 1.0
    
    def _physics_spiral(self, beat_strength, onset, burst):
        if not hasattr(self, 'angles'):
            self.angles = np.random.rand(len(self.positions)) * 2 * np.pi
            self.radii = np.random.rand(len(self.positions)) * 0.5
        
        self.angles += (0.05 + beat_strength * 0.1) * self.config.particle_speed
        self.radii += np.random.randn(len(self.radii)) * 0.001 - 0.0005
        
        self.radii = np.clip(self.radii, 0.1, 0.5)
        
        self.positions[:, 0] = 0.5 + self.radii * np.cos(self.angles)
        self.positions[:, 1] = 0.5 + self.radii * np.sin(self.angles)
        
        if burst:
            self.radii += np.random.randn(len(self.radii)) * 0.02
    
    def _physics_gravity(self, beat_strength, onset, burst):
        self.velocities[:, 1] += self.config.particle_gravity * self.config.particle_speed
        
        if burst:
            self.velocities[:, 1] -= beat_strength * 0.05
        
        self.velocities[:, 0] += np.random.randn(len(self.velocities)) * onset * 0.002
        self.velocities *= 0.98
        self.positions += self.velocities
        
        fall_mask = self.positions[:, 1] > 1
        self.positions[fall_mask, 1] = 0
        self.positions[fall_mask, 0] = np.random.rand(np.sum(fall_mask))
        self.velocities[fall_mask] = 0
    
    def _physics_vortex(self, beat_strength, onset, burst):
        dx = self.positions[:, 0] - 0.5
        dy = self.positions[:, 1] - 0.5
        angles = np.arctan2(dy, dx)
        radii = np.sqrt(dx**2 + dy**2)
        
        tangent_x = -np.sin(angles)
        tangent_y = np.cos(angles)
        
        swirl_strength = (0.01 + beat_strength * 0.02) * self.config.particle_speed
        self.velocities[:, 0] += tangent_x * swirl_strength
        self.velocities[:, 1] += tangent_y * swirl_strength
        
        if burst:
            radial_velocity = -radii * 0.05
            self.velocities[:, 0] += np.cos(angles) * radial_velocity
            self.velocities[:, 1] += np.sin(angles) * radial_velocity
        
        self.velocities *= 0.95
        self.positions += self.velocities
        self.positions = np.clip(self.positions, 0, 1)
        
    def get_draw_data(self):
        if len(self.positions) == 0:
            return np.array([[0.5, 0.5]]), np.array([1.0]), np.array([[1.0, 1.0, 1.0]]), np.array([0.5])
        
        alphas = self.lifetimes ** 0.5
        if hasattr(self.config, 'beat_flash_intensity'):
            alphas *= (1 + self.config.beat_flash_intensity * 0.5)
        
        return self.positions, self.sizes, self.colors, alphas

class AudioAnalyzer:
    def __init__(self, audio_path: Path, sr: int = 22050, smooth_factor: float = 0.7,
                 audio_mode: AudioMode = AudioMode.STEREO):
        self.sr = sr
        self.audio_mode = audio_mode
        self.smooth_factor = smooth_factor
        
        try:
            y, sr = librosa.load(audio_path, sr=sr, mono=False, duration=600)
            
            if self.audio_mode == AudioMode.STEREO and y.ndim > 1:
                self.y = y
            elif self.audio_mode == AudioMode.MONO:
                self.y = librosa.to_mono(y)
            elif self.audio_mode == AudioMode.LEFT and y.ndim > 1:
                self.y = y[0]
            elif self.audio_mode == AudioMode.RIGHT and y.ndim > 1:
                self.y = y[1]
            else:
                self.y = y
            
            y_mono = librosa.to_mono(y) if y.ndim > 1 else y
            
        except Exception as e:
            st.error(f"Failed to load audio: {e}")
            raise
        
        self.duration = librosa.get_duration(y=self.y, sr=sr)
        if self.duration < 0.1:
            raise ValueError("Audio too short (< 0.1s)")
        
        self.onset_env = librosa.onset.onset_strength(y=y_mono, sr=sr)
        self.rms = librosa.feature.rms(y=y_mono)[0]
        self.tempo, self.beat_frames = librosa.beat.beat_track(y=y_mono, sr=sr)
        
        if len(self.beat_frames) == 0:
            self.beat_frames = np.array([0, len(self.onset_env) // 2])
        
        self.stft = np.abs(librosa.stft(y_mono, n_fft=2048))
        self.times = librosa.times_like(self.stft, sr=sr)
        self.beat_interp = self._interpolate_beats()
        self.audio_normalized = (y_mono - np.mean(y_mono)) / (np.std(y_mono) + 1e-6)
        
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
        
        if len(self.audio_normalized) == 0:
            return np.array([])
        
        start = max(0, sample_pos - window // 2)
        end = min(len(self.audio_normalized), start + window)
        return self.audio_normalized[start:end]

class VisualizerEngine:
    def __init__(self, config: VisualizerConfig):
        self.config = config
        self.particles = ParticleSystem(config.effects)
        self._prev_spectrum = None
        
    @staticmethod
    def _log_debug(msg: str):
        if DEBUG_MODE:
            st.sidebar.text(f"🔍 {msg}")
        
    def generate_video(self, audio_path: Path, cover_path: Path, 
                      output_path: Path, progress_callback: Optional[Callable] = None):
        self._log_debug("Starting video generation...")
        
        try:
            analyzer = AudioAnalyzer(audio_path, smooth_factor=self.config.effects.bar_smoothness,
                                   audio_mode=self.config.effects.audio_mode)
        except Exception as e:
            st.error(f"Audio analysis failed: {e}")
            return False
            
        cover_bg, cover_fg = self._prepare_images(cover_path)
        total_frames = int(analyzer.duration * self.config.effects.fps)
        
        hw_encoder_available = self._check_hw_encoder_device() and not self._is_pi5()
        codec = 'h264_v4l2m2m' if hw_encoder_available else 'libx264'
        
        if not hw_encoder_available:
            st.warning("⚠️ Hardware encoder unavailable (Pi 5 or missing driver). Using software encoding")
        
        cmd = [
            'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{self.config.effects.width}x{self.config.effects.height}', '-pix_fmt', 'rgb24',
            '-r', str(self.config.effects.fps), '-i', '-', '-i', str(audio_path),
            '-c:v', codec, '-c:a', 'aac', '-b:v', self.config.effects.bitrate,
            '-b:a', self.config.effects.audio_bitrate, '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        ]
        
        if self.config.effects.audio_mode == AudioMode.MONO:
            cmd.extend(['-ac', '1'])
        
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
    
    def _is_pi5(self) -> bool:
        try:
            with open("/proc/cpuinfo", "r") as f:
                cpuinfo = f.read()
            return "BCM2712" in cpuinfo
        except:
            return False
    
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
                frame_data = np.frombuffer(buf, dtype='uint8').reshape(
                    (self.config.effects.height, self.config.effects.width, 4)
                )
                frame_data = frame_data[:, :, :3]
            else:
                frame_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                frame_data = frame_data.reshape(
                    (self.config.effects.height, self.config.effects.width, 3)
                )
            
            try:
                proc.stdin.write(frame_data.tobytes())
            except BrokenPipeError:
                raise RuntimeError("FFmpeg encoder crashed during frame write")
            
            if callback and frame_idx % self.config.effects.fps == 0:
                callback(frame_idx / total_frames)
    
    def _check_hw_encoder_device(self) -> bool:
        v4l2_devices = ['/dev/video11', '/dev/video12', '/dev/video10']
        device_exists = any(Path(dev).exists() for dev in v4l2_devices)
        
        if not device_exists:
            return False
        
        if self._is_pi5():
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
        
        # Spectrum bars
        bar_ax = self._get_bar_axes(fig, self.config.effects.bar_geometry)
        ax_bars = fig.add_axes(bar_ax, facecolor='none')
        ax_bars.set_xlim(0, self.config.effects.bar_count)
        ax_bars.set_ylim(0, 1)
        ax_bars.axis('off')
        bar_rects = [Rectangle((i, 0), self.config.effects.bar_width, 0, facecolor='cyan', alpha=0.6) 
                    for i in range(self.config.effects.bar_count)]
        for rect in bar_rects:
            ax_bars.add_patch(rect)
        
        # Waveform overlay
        waveform_ax = self._get_waveform_axes(fig, self.config.effects.waveform_pattern)
        ax_waveform = fig.add_axes(waveform_ax, facecolor='none')
        
        waveform_bars = None
        if self.config.effects.waveform_pattern == WaveformPattern.CIRCULAR:
            ax_waveform.set_xlim(-1, 1)
            ax_waveform.set_ylim(-1, 1)
            waveform_line, = ax_waveform.plot([], [], color=self.config.effects.waveform_color, 
                                             linewidth=self.config.effects.waveform_thickness)
            ax_waveform.set_aspect('equal')
        elif self.config.effects.waveform_pattern == WaveformPattern.BARS:
            ax_waveform.set_xlim(0, 1)
            ax_waveform.set_ylim(-1, 1)
            waveform_bars = [Rectangle((i, -1), 0.9, 2, facecolor=self.config.effects.waveform_color, alpha=0.6)
                           for i in np.linspace(0, 1, 50)]
            for bar in waveform_bars:
                ax_waveform.add_patch(bar)
            waveform_line = None
        else:
            ax_waveform.set_xlim(0, 1)
            ax_waveform.set_ylim(-1, 1)
            waveform_line, = ax_waveform.plot([], [], color=self.config.effects.waveform_color, 
                                             linewidth=self.config.effects.waveform_thickness)
        
        ax_waveform.axis('off')
        
        # Particles
        ax_particles = fig.add_axes([0, 0, 1, 1], facecolor='none')
        scatter = ax_particles.scatter([], [], s=[], c=[], cmap='plasma', alpha=0.7)
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
            'waveform_bars': waveform_bars,
        }
    
    def _get_bar_axes(self, fig, geometry):
        if geometry == BarGeometry.TOP:
            return [0.1, 0.85, 0.8, 0.1]
        elif geometry == BarGeometry.SIDES:
            return [0.05, 0.3, 0.1, 0.4]
        elif geometry == BarGeometry.CIRCLE:
            return [0.3, 0.3, 0.4, 0.4]
        else:  # BOTTOM
            return [0.1, 0.05, 0.8, 0.15]
    
    def _get_waveform_axes(self, fig, pattern):
        if pattern == WaveformPattern.CIRCULAR:
            return [0.35, 0.35, 0.3, 0.3]
        else:
            return [0.1, 0.85, 0.8, 0.1]
    
    def _update_frame(self, fig, ax_dict, features, pos, sizes, colors, alphas, cover_fg, current_time):
        spectrum = features['spectrum']
        spectrum_norm = np.log1p(spectrum) / np.log1p(spectrum.max() + 1e-6)
        
        if hasattr(self, '_prev_spectrum'):
            spectrum_norm = (spectrum_norm * (1 - self.config.effects.bar_smoothness) + 
                           self._prev_spectrum * self.config.effects.bar_smoothness)
        self._prev_spectrum = spectrum_norm
        
        for i, rect in enumerate(ax_dict['bar_rects']):
            height = spectrum_norm[i % len(spectrum_norm)] * self.config.effects.bar_height_scale
            rect.set_height(height)
            
            if self.config.effects.bar_color_scheme == ColorScheme.MONOCHROME:
                rect.set_facecolor((0.5, 0.5, 0.5))
            else:
                rect.set_facecolor(plt.cm.plasma(i / len(ax_dict['bar_rects'])))
        
        if self.config.effects.enable_waveform and 'waveform_line' in ax_dict:
            waveform = features['waveform']
            if len(waveform) > 0:
                if self.config.effects.waveform_pattern == WaveformPattern.CIRCULAR:
                    theta = np.linspace(0, 2*np.pi, len(waveform))
                    radius = 0.3 + waveform * 0.2
                    x_data = radius * np.cos(theta)
                    y_data = radius * np.sin(theta)
                    ax_dict['waveform_line'].set_data(x_data, y_data)
                elif self.config.effects.waveform_pattern == WaveformPattern.BARS and ax_dict['waveform_bars']:
                    bar_heights = (waveform + 1) / 2
                    for i, bar in enumerate(ax_dict['waveform_bars']):
                        if i < len(bar_heights):
                            bar.set_height(bar_heights[i] * 2)
                else:
                    x_data = np.linspace(0, 1, len(waveform))
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
    data = {}
    for f in fields(config):
        value = getattr(config, f.name)
        if isinstance(value, Enum):
            data[f.name] = value.value
        elif isinstance(value, tuple):
            data[f.name] = value
        else:
            data[f.name] = value
    
    with open(preset_path, 'w') as f:
        json.dump(data, f, indent=2)
    st.success(f"Preset '{name}' saved!")

def load_preset(name: str) -> EffectSettings:
    preset_path = PRESETS_DIR / f"{name}.json"
    if not preset_path.exists():
        return EffectSettings()
    
    with open(preset_path, 'r') as f:
        data = json.load(f)
    
    return EffectSettings.with_defaults(**data)

def list_presets() -> list[str]:
    return [p.stem for p in PRESETS_DIR.glob("*.json")]

def ensure_session_state():
    if "viz_config" not in st.session_state:
        st.session_state.viz_config = VisualizerConfig()
    
    effects = st.session_state.viz_config.effects
    defaults = EffectSettings()
    
    for f in fields(defaults):
        if not hasattr(effects, f.name):
            setattr(effects, f.name, getattr(defaults, f.name))
    
    # Validate enums
    enum_fields = {
        'particle_color_scheme': ColorScheme,
        'bar_color_scheme': ColorScheme,
        'particle_physics': ParticlePhysics,
        'waveform_pattern': WaveformPattern,
        'bar_geometry': BarGeometry,
        'audio_mode': AudioMode
    }
    
    for key, enum_class in enum_fields.items():
        current_value = getattr(effects, key)
        valid_values = [e.value for e in enum_class]
        if isinstance(current_value, str) and current_value not in valid_values:
            setattr(effects, key, getattr(defaults, key))
    
    cpu_cores = multiprocessing.cpu_count()
    if not hasattr(effects, 'core_count') or effects.core_count < 1:
        effects.core_count = max(1, cpu_cores // 2)
    
    effects.core_count = min(effects.core_count, cpu_cores)
    effects.core_count = max(1, effects.core_count)

# =============================================================================
# UI HELPER
# =============================================================================

def safe_enum_index(enum_class: Enum, current_value: Any, default_index: int = 0) -> int:
    """Safely get enum index with fallback"""
    valid_values = [e.value for e in enum_class]
    
    if isinstance(current_value, enum_class):
        current_value = current_value.value
    elif isinstance(current_value, dict) and 'value' in current_value:
        current_value = current_value['value']
    
    return valid_values.index(current_value) if current_value in valid_values else default_index

# =============================================================================
# STREAMLIT UI
# =============================================================================

def music_visualizer_page():
    st.title("🎵 Pi Visualizer Studio Pro v2.5")
    st.caption("Multi-core, customizable music video generator")
    
    ensure_session_state()
    
    # Sidebar: System Info
    with st.sidebar:
        st.header("System Info")
        cpu_count = multiprocessing.cpu_count()
        st.metric("CPU Cores", cpu_count)
        
        try:
            with open("/proc/cpuinfo", "r") as f:
                cpuinfo = f.read()
            is_pi5 = "BCM2712" in cpuinfo
        except:
            is_pi5 = False
        
        st.metric("Hardware", "Pi 5 (No HW Enc)" if is_pi5 else "Pi 4 (HW Enc Available)")
        
        if is_pi5:
            st.warning("Pi 5 detected: Software encoding only")
        
        st.divider()
        
        # Quick presets
        st.header("Quick Presets")
        if st.button("🎬 Fast Render"):
            st.session_state.viz_config.effects.quality_preset = "low"
            st.session_state.viz_config.effects.fps = 30
            st.session_state.viz_config.effects.core_count = max(1, cpu_count // 2)
            st.success("Fast render applied!")
            st.rerun()
        
        if st.button("✨ Cinema Quality"):
            st.session_state.viz_config.effects.quality_preset = "high"
            st.session_state.viz_config.effects.fps = 60
            st.session_state.viz_config.effects.width = 1920
            st.session_state.viz_config.effects.height = 1080
            st.success("Cinema quality applied!")
            st.rerun()
    
    # Main Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📁 Files", "🎨 Effects", "⚙️ Performance", "🔊 Audio", "▶️ Generate"])
    
    # === TAB 1: FILE SELECTION ===
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
    
    # === TAB 2: EFFECTS ===
    with tab2:
        st.header("Visual Effects")
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.subheader("Particle System")
            
            # Physics mode
            current_physics = st.session_state.viz_config.effects.particle_physics
            selected_physics = st.selectbox(
                "Physics Mode",
                options=[e.value for e in ParticlePhysics],
                index=safe_enum_index(ParticlePhysics, current_physics)
            )
            st.session_state.viz_config.effects.particle_physics = ParticlePhysics(selected_physics)
            
            # Particle count with reset
            new_count = st.slider("Particle Count", 10, 200, st.session_state.viz_config.effects.particle_count)
            if new_count != st.session_state.viz_config.effects.particle_count:
                st.session_state.viz_config.effects.particle_count = new_count
                st.session_state.viz_config.effects._needs_reset = True
            
            # Particle speed
            st.session_state.viz_config.effects.particle_speed = st.slider(
                "Particle Speed", 0.1, 5.0, st.session_state.viz_config.effects.particle_speed
            )
            
            # Gravity slider (conditional)
            if st.session_state.viz_config.effects.particle_physics == ParticlePhysics.GRAVITY:
                st.session_state.viz_config.effects.particle_gravity = st.slider(
                    "Gravity Strength", 0.0001, 0.02, st.session_state.viz_config.effects.particle_gravity
                )
            
            # Color scheme with reset
            current_color = st.session_state.viz_config.effects.particle_color_scheme
            selected_color = st.selectbox(
                "Color Scheme", [e.value for e in ColorScheme],
                index=safe_enum_index(ColorScheme, current_color)
            )
            if selected_color != current_color.value:
                st.session_state.viz_config.effects.particle_color_scheme = ColorScheme(selected_color)
                st.session_state.viz_config.effects._needs_reset = True
            
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
            
            # Bar count
            new_bar_count = st.slider("Bar Count", 16, 128, st.session_state.viz_config.effects.bar_count)
            if new_bar_count != st.session_state.viz_config.effects.bar_count:
                st.session_state.viz_config.effects.bar_count = new_bar_count
            
            # Bar geometry
            current_geom = st.session_state.viz_config.effects.bar_geometry
            selected_geom = st.selectbox(
                "Bar Position", [e.value for e in BarGeometry],
                index=safe_enum_index(BarGeometry, current_geom)
            )
            st.session_state.viz_config.effects.bar_geometry = BarGeometry(selected_geom)
            
            # Bar color
            current_bar_color = st.session_state.viz_config.effects.bar_color_scheme
            selected_bar_color = st.selectbox(
                "Bar Color Scheme", [e.value for e in ColorScheme],
                index=safe_enum_index(ColorScheme, current_bar_color)
            )
            st.session_state.viz_config.effects.bar_color_scheme = ColorScheme(selected_bar_color)
            
            # Bar height scale
            st.session_state.viz_config.effects.bar_height_scale = st.slider(
                "Bar Height Scale", 0.05, 0.5, st.session_state.viz_config.effects.bar_height_scale
            )
            
            # Smoothness
            st.session_state.viz_config.effects.bar_smoothness = st.slider(
                "Smoothness", 0.0, 1.0, st.session_state.viz_config.effects.bar_smoothness
            )
            
            st.subheader("Waveform")
            st.session_state.viz_config.effects.enable_waveform = st.checkbox(
                "Enable Waveform", st.session_state.viz_config.effects.enable_waveform
            )
            
            if st.session_state.viz_config.effects.enable_waveform:
                current_pattern = st.session_state.viz_config.effects.waveform_pattern
                selected_pattern = st.selectbox(
                    "Waveform Pattern", [e.value for e in WaveformPattern],
                    index=safe_enum_index(WaveformPattern, current_pattern)
                )
                st.session_state.viz_config.effects.waveform_pattern = WaveformPattern(selected_pattern)
                
                st.session_state.viz_config.effects.waveform_color = st.color_picker(
                    "Waveform Color", st.session_state.viz_config.effects.waveform_color
                )
                
                st.session_state.viz_config.effects.waveform_thickness = st.slider(
                    "Waveform Thickness", 0.5, 5.0, st.session_state.viz_config.effects.waveform_thickness
                )
            
            st.subheader("Background")
            st.session_state.viz_config.effects.background_blur = st.slider(
                "Blur Radius", 0, 50, st.session_state.viz_config.effects.background_blur
            )
            st.session_state.viz_config.effects.background_brightness = st.slider(
                "Brightness", 0.3, 1.0, st.session_state.viz_config.effects.background_brightness
            )
        
        # Cover effects
        st.divider()
        st.subheader("Cover Effects")
        st.session_state.viz_config.effects.cover_pulse = st.checkbox(
            "Pulse on Beat", st.session_state.viz_config.effects.cover_pulse
        )
        if st.session_state.viz_config.effects.cover_pulse:
            st.session_state.viz_config.effects.cover_zoom_intensity = st.slider(
                "Pulse Intensity", 0.01, 0.2, st.session_state.viz_config.effects.cover_zoom_intensity
            )
        
        # Preset Management
        st.divider()
        preset_col1, preset_col2 = st.columns([3, 1])
        
        with preset_col1:
            preset_name = st.text_input("Preset Name", placeholder="e.g., Fire Vortex")
        with preset_col2:
            if st.button("💾 Save Preset"):
                if preset_name:
                    save_preset(preset_name, st.session_state.viz_config.effects)
                    st.success(f"Preset '{preset_name}' saved!")
                    st.rerun()
                else:
                    st.error("Enter a preset name")
        
        presets = list_presets()
        if presets:
            selected_preset = st.selectbox("Load Preset", [""] + presets)
            if st.button("📂 Load Selected") and selected_preset:
                st.session_state.viz_config.effects = load_preset(selected_preset)
                st.success(f"Loaded: {selected_preset}")
                st.rerun()
    
    # === TAB 3: PERFORMANCE ===
    with tab3:
        st.header("Performance Settings")
        
        col_perf1, col_perf2 = st.columns(2)
        
        with col_perf1:
            st.subheader("Quality")
            st.session_state.viz_config.effects.quality_preset = st.select_slider(
                "Preset", ["low", "medium", "high"], st.session_state.viz_config.effects.quality_preset
            )
            
            cpu_cores = multiprocessing.cpu_count()
            st.session_state.viz_config.effects.core_count = st.slider(
                "CPU Cores", 1, cpu_cores, st.session_state.viz_config.effects.core_count,
                help=f"Use up to {cpu_cores} cores. More = faster but hotter"
            )
            
            res_options = ["960x540", "1280x720", "1920x1080"]
            current_res = f"{st.session_state.viz_config.effects.width}x{st.session_state.viz_config.effects.height}"
            selected_res = st.selectbox(
                "Resolution", res_options,
                index=res_options.index(current_res) if current_res in res_options else 1
            )
            w, h = map(int, selected_res.split('x'))
            st.session_state.viz_config.effects.width = w
            st.session_state.viz_config.effects.height = h
            
            st.session_state.viz_config.effects.fps = st.select_slider(
                "FPS", [24, 30, 60], st.session_state.viz_config.effects.fps
            )
        
        with col_perf2:
            st.subheader("Advanced")
            st.session_state.viz_config.effects.chunk_duration = st.slider(
                "Chunk Duration", 5, 30, st.session_state.viz_config.effects.chunk_duration
            )
            
            if st.session_state.viz_config.effects.width >= 1920:
                st.warning("⚠️ 1080p may produce unfinished files on Pi 5")
            
            st.session_state.viz_config.effects.bitrate = st.text_input(
                "Video Bitrate", st.session_state.viz_config.effects.bitrate
            )
    
    # === TAB 4: AUDIO ===
    with tab4:
        st.header("Audio Settings")
        
        current_audio = st.session_state.viz_config.effects.audio_mode
        selected_audio_mode = st.radio(
            "Audio Mode",
            options=[e.value for e in AudioMode],
            index=safe_enum_index(AudioMode, current_audio)
        )
        st.session_state.viz_config.effects.audio_mode = AudioMode(selected_audio_mode)
        
        st.subheader("Audio Encoding")
        st.session_state.viz_config.effects.audio_bitrate = st.selectbox(
            "Audio Bitrate", ["128k", "192k", "256k", "320k"],
            index=["128k", "192k", "256k", "320k"].index(st.session_state.viz_config.effects.audio_bitrate)
        )
        
        if st.session_state.viz_config.effects.audio_mode != AudioMode.STEREO:
            st.info(f"Audio will be converted to {st.session_state.viz_config.effects.audio_mode.value}")
    
    # === TAB 5: GENERATE ===
    with tab5:
        st.header("Generate Video")
        
        # CRITICAL SAFETY CHECK
        if not isinstance(selected_audio, Path) or not selected_audio.exists():
            st.error("⚠️ Audio file not selected or invalid. Please return to Files tab.")
            st.stop()
        
        # Quick preview
        if st.button("👁️ Generate 5s Preview"):
            preview_path = selected_audio.with_suffix('.preview.mp4')
            preview_config = st.session_state.viz_config
            
            engine = VisualizerEngine(preview_config)
            if hasattr(st.session_state.viz_config.effects, '_needs_reset'):
                engine.particles.reset()
                delattr(st.session_state.viz_config.effects, '_needs_reset')
            
            with st.spinner("Rendering preview..."):
                success = engine.generate_video(selected_audio, cover_path, preview_path,
                                              progress_callback=lambda p: st.progress(p))
            
            if success and preview_path.exists():
                st.success("✅ Preview ready!")
                st.video(str(preview_path))
                
                if st.button("🗑️ Delete Preview"):
                    preview_path.unlink(missing_ok=True)
                    st.rerun()
            else:
                st.error("❌ Preview failed")
        
        st.divider()
        
        # Full render
        output_path = selected_audio.with_suffix('.visualizer.mp4')
        
        if st.session_state.viz_config.effects.width >= 1920:
            st.warning("⚠️ 1080p may produce unfinished files on Pi 5 - test before long renders")
        
        if output_path.exists():
            st.info(f"Output: `{output_path.name}`")
            overwrite = st.checkbox("Overwrite existing file")
        else:
            overwrite = True
        
        render_btn = st.button("🎬 Generate Full Video", type="primary", use_container_width=True,
                             disabled=not overwrite and output_path.exists())
        
        if render_btn:
            # Final safety check
            if not isinstance(selected_audio, Path):
                selected_audio = Path(selected_audio)
                
            engine = VisualizerEngine(st.session_state.viz_config)
            if hasattr(st.session_state.viz_config.effects, '_needs_reset'):
                engine.particles.reset()
                delattr(st.session_state.viz_config.effects, '_needs_reset')
            
            progress_bar = st.progress(0)
            status = st.empty()
            eta_text = st.empty()
            
            start_time = time.time()
            
            def update_progress(p):
                progress_bar.progress(p)
                if p > 0:
                    elapsed = time.time() - start_time
                    eta = elapsed / p - elapsed
                    eta_text.text(f"⏱️ ETA: {int(eta//60)}m {int(eta%60)}s")
            
            try:
                status.info("Starting full generation...")
                success = engine.generate_video(selected_audio, cover_path, output_path, update_progress)
                
                if success and output_path.exists():
                    status.success("✅ Complete!")
                    st.video(str(output_path))
                    
                    filesize = output_path.stat().st_size / (1024*1024)
                    duration = librosa.get_duration(filename=str(selected_audio))
                    st.info(f"📊 {duration:.1f}s | {filesize:.1f}MB | "
                           f"{st.session_state.viz_config.effects.width}x{st.session_state.viz_config.effects.height} @ "
                           f"{st.session_state.viz_config.effects.fps}fps")
                else:
                    status.error("❌ Generation failed")
                    
            except Exception as e:
                st.exception(e)
                logger.error("Unhandled generation error", exc_info=True)
            finally:
                gc.collect()

# =============================================================================
# PI SETUP & HARDWARE INFO
# =============================================================================

def show_setup_instructions():
    with st.expander("🔧 Pi Hardware & Setup Info", expanded=False):
        try:
            with open("/proc/cpuinfo", "r") as f:
                cpuinfo = f.read()
            if "BCM2712" in cpuinfo:
                pi_model = "Pi 5 (No Hardware Encoder)"
                hw_status = "❌ Not Available"
            elif "BCM2711" in cpuinfo:
                pi_model = "Pi 4 (Hardware Encoder Available)"
                hw_status = "✅ Available if drivers loaded"
            else:
                pi_model = "Unknown Pi Model"
                hw_status = "⚠️ Unknown"
        except:
            pi_model = "Not a Pi"
            hw_status = "⚠️ Non-Pi system"
        
        st.markdown(f"""
        ### **Detected Hardware:**
        - **Model**: {pi_model}
        - **HW Encoder**: {hw_status}
        
        ### **Pi 4 Setup (for hardware encoding):**
        ```bash
        sudo apt update
        sudo apt install -y raspberrypi-kernel v4l-utils libavcodec-extra
        sudo modprobe bcm2835-codec
        ls -l /dev/video11  # Should exist
        ```
        
        ### **Pi 5 Note:**
        Pi 5's VideoCore VII has a broken V4L2 encoder in current kernels. 
        Software encoding (`libx264`) auto-fallbacks. Still fast on 4-8 cores!
        
        ### **Performance Tips:**
        - **720p @ 30fps**: ~0.5x real-time on Pi 5
        - **1080p @ 60fps**: May produce unfinished files - test first
        - **Use 2-4 cores**: Sweet spot for speed vs. heat
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
        logger.error("Fatal error in main", exc_info=True)
