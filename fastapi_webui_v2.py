#!/usr/bin/env python3
"""
FastAPI Web Interface for IndexTTS vLLM v2
A single-file FastAPI application that combines webui_with_presets.py functionality
with the API structure from deploy_vllm_indextts.py, using IndexTTS vLLM v2 as backend.

Features:
- IndexTTS vLLM v2 backend for ultra-fast inference
- Speaker preset management with persistent storage
- API compatibility for external integrations
- Modern web interface with Chinese support
- Parallel chunk processing for long texts
- MP3 output support
- Advanced translate/edit mode with segment editing and selective regeneration
- Gemini model selection (Flash vs Pro) with optional API key override
- Translation/transcription toggle with custom prompt support
- Per-segment generation control for efficient audio processing
"""

import os
import sys
import asyncio
import time
import tempfile
import traceback
import uuid
from pathlib import Path
from typing import Any, List, Dict, Optional, Literal, Tuple, Set
from contextlib import asynccontextmanager
from io import BytesIO
import base64
import functools
import json
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
import copy
from dataclasses import dataclass, field


# Audio processing
import numpy as np
import soundfile as sf
from pydub import AudioSegment

try:
    from clearvoice import ClearVoice  # type: ignore[import]
except ImportError:
    ClearVoice = None

try:
    from google import genai  # type: ignore[import]
    from google.genai import types  # type: ignore[import]
except ImportError:
    genai = None
    types = None


# FastAPI and web interface
from fastapi import FastAPI, File, UploadFile, Form, Request, BackgroundTasks, HTTPException, Depends
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field

# IndexTTS v2 and speaker management
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, "indextts"))

from indextts.infer_vllm_v2 import IndexTTS2
from speaker_preset_manager import SpeakerPresetManager, initialize_preset_manager

# Configuration
import argparse

# Global thread executor for blocking operations
executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="fastapi_async")

# Global ClearVoice models (initialized lazily and reused)
_enhancement_model: Optional[Any] = None
_super_res_model: Optional[Any] = None

# Gemini configuration
GEMINI_API_KEY_ENV_VAR = "GEMINI_API_KEY"
GOOGLE_API_KEY_ENV_VAR = "GOOGLE_API_KEY"
GEMINI_MODEL_ENV_VAR = "GEMINI_MODEL_NAME"
DEFAULT_GEMINI_MODEL_NAME = "gemini-2.5-pro"
JSON_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
TRANSLATION_PROMPT_TEMPLATE = (
    "You are a professional interpreter. "
    "Transcribe the speech from the provided audio and translate it into {dest_language} using natural, conversational wording. "
    "Return a JSON array where each item contains exactly the keys: "
    "\"start\" (timestamp in mm:ss or mm:ss.xxx), "
    "\"end\" (same format), "
    "\"source_text\" (original-language transcript), "
    "\"translated_text\" (translation in {dest_language}). "
    "Ensure the timestamps align closely with the audio and split into coherent segments. "
    "Respond with JSON only—no explanations, markdown, or additional text."
)
TRANSCRIPTION_PROMPT_TEMPLATE = (
    "You are a meticulous transcription expert. "
    "Transcribe the provided speech audio in its original language without translating it. "
    "Return a JSON array where each item contains the keys: "
    "\"start\" (timestamp in mm:ss or mm:ss.xxx), "
    "\"end\" (same format), "
    "\"source_text\" (the transcript in the original language), "
    "\"translated_text\" (leave this empty string to indicate no translation). "
    "Ensure timestamps align with the audio and segments remain coherent. "
    "Respond with JSON only—no markdown or commentary."
)
DEFAULT_GEMINI_TEMPERATURE = 0.2
DEFAULT_GEMINI_TOP_P = 0.9
TRANSLATE_DEFAULT_OUTPUT_FORMAT = "mp3"
TRANSLATE_DEFAULT_BITRATE = "192k"
AUDIO_GENERATION_MARGIN_MS = 20
TRANSLATION_TTS_CONCURRENCY = 20
MIN_SPEECH_DURATION_MS = 3000
MAX_MERGE_INTERVAL_MS = 300


@dataclass
class TranslateSessionData:
    session_id: str
    original_audio: AudioSegment
    dest_language: str
    prompt: str
    translate_enabled: bool
    response_format: str
    bitrate: str
    input_mime_type: Optional[str]
    clearvoice_settings: Dict[str, bool]
    base_segments: List[Dict[str, Any]]
    gemini_chunks: List[Dict[str, Any]]
    gemini_model: str
    gemini_api_key: Optional[str]
    backing_track_audio: Optional[AudioSegment] = None
    merge_with_backing: bool = False
    created_at: float = field(default_factory=lambda: time.time())


ADVANCED_TRANSLATE_SESSIONS: Dict[str, TranslateSessionData] = {}
ADVANCED_TRANSLATE_SESSION_LOCK = asyncio.Lock()
ADVANCED_TRANSLATE_SESSION_TTL_SECONDS = 60 * 60  # 1 hour


@functools.lru_cache(maxsize=4)
def _create_gemini_client(api_key: str):
    if genai is None:
        raise RuntimeError(
            "The google-genai package is required for translation. Install it with `pip install google-genai`."
        )
    return genai.Client(api_key=api_key)


def _get_gemini_client(api_key: str):
    return _create_gemini_client(api_key)


def estimate_speech_duration(text: str, language: str = "auto") -> int:
    """
    Estimate speech duration in milliseconds based on text length.
    
    Args:
        text: Input text
        language: Language hint ("zh", "en", or "auto")
    
    Returns:
        Estimated duration in milliseconds
    """
    if not text or not text.strip():
        return 0
    
    # Clean text
    text = text.strip()
    
    # Detect language if auto
    if language == "auto":
        # Simple heuristic: if more than 30% Chinese characters, treat as Chinese
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        if chinese_chars / max(len(text), 1) > 0.3:
            language = "zh"
        else:
            language = "en"
    
    # Speech rate estimates (characters per second)
    # These are conservative estimates based on typical TTS output
    if language == "zh":
        # Chinese: ~4-5 characters per second
        chars_per_second = 4.5
        char_count = len([c for c in text if '\u4e00' <= c <= '\u9fff' or c.isalnum()])
    else:
        # English: ~12-15 characters per second (including spaces)
        # Or ~150-180 words per minute = 2.5-3 words per second
        chars_per_second = 13.0
        char_count = len(text)
    
    # Calculate base duration
    duration_seconds = char_count / chars_per_second
    
    # Add padding for punctuation pauses (10% extra time)
    punctuation_count = sum(1 for c in text if c in ',.!?;:。，！？；：')
    pause_time = punctuation_count * 0.3  # 300ms per punctuation
    
    total_duration_seconds = duration_seconds + pause_time
    
    # Add 10% buffer for natural speech variations
    total_duration_seconds *= 1.1
    
    # Convert to milliseconds and round up to nearest 100ms
    duration_ms = int(total_duration_seconds * 1000)
    duration_ms = ((duration_ms + 99) // 100) * 100  # Round up to nearest 100ms
    
    return duration_ms

parser = argparse.ArgumentParser(description="IndexTTS vLLM v2 FastAPI WebUI")
parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose mode")
parser.add_argument("--port", type=int, default=8000, help="Port to run the web API on")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the web API on")
parser.add_argument("--model_dir", type=str, default="checkpoints", help="Model checkpoints directory")
parser.add_argument("--is_fp16", action="store_true", default=False, help="Fp16 infer")
parser.add_argument("--use_torch_compile", action="store_true", default=False, help="use torch compile")
parser.add_argument("--gpu_memory_utilization", type=float, default=0.25, help="GPU memory utilization")

# Parse args if run as script, otherwise use defaults
try:
    cmd_args = parser.parse_args()
except SystemExit:
    # If running in interactive mode, use defaults
    cmd_args = argparse.Namespace(
        verbose=False,
        port=8000,
        host="0.0.0.0",
        model_dir="checkpoints",
        is_fp16=False,
        use_torch_compile=False,
        gpu_memory_utilization=0.25
    )

# Create directories
os.makedirs("outputs/tasks", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("prompts", exist_ok=True)
os.makedirs("speaker_presets", exist_ok=True)

# Async wrapper functions for blocking operations
async def async_write_file(file_path: str, data: bytes) -> None:
    """Async wrapper for writing file data"""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(executor, _write_file_sync, file_path, data)

def _write_file_sync(file_path: str, data: bytes) -> None:
    """Synchronous file write operation"""
    with open(file_path, 'wb') as f:
        f.write(data)

async def async_read_file(file_path: str) -> bytes:
    """Async wrapper for reading file data"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, _read_file_sync, file_path)

def _read_file_sync(file_path: str) -> bytes:
    """Synchronous file read operation"""
    with open(file_path, 'rb') as f:
        return f.read()

async def async_remove_file(file_path: str) -> None:
    """Async wrapper for removing files"""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(executor, _remove_file_sync, file_path)

def _remove_file_sync(file_path: str) -> None:
    """Synchronous file removal operation"""
    try:
        os.unlink(file_path)
    except Exception:
        pass  # Ignore errors when removing temporary files

async def async_audio_read(file_path: str):
    """Async wrapper for soundfile.read()"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, sf.read, file_path)

async def async_cut_audio_to_duration(input_path: str, max_duration: float = 10.0):
    """Async wrapper for smart audio cutting at silence intervals"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, _smart_cut_audio_at_silence, input_path, max_duration)

def _detect_silence_intervals(audio_data, sample_rate, min_silence_duration=0.3, silence_threshold=-40):
    """
    Detect silence intervals in audio data.
    
    Args:
        audio_data: Audio samples (mono or stereo)
        sample_rate: Sample rate in Hz
        min_silence_duration: Minimum duration of silence in seconds (default: 0.3s)
        silence_threshold: Silence threshold in dB (default: -40dB)
    
    Returns:
        List of tuples (start_sample, end_sample) for each silence interval
    """
    import numpy as np
    
    # Convert to mono if stereo
    if len(audio_data.shape) > 1:
        audio_mono = np.mean(audio_data, axis=1)
    else:
        audio_mono = audio_data
    
    # Calculate RMS energy in dB
    frame_length = int(0.02 * sample_rate)  # 20ms frames
    hop_length = int(0.01 * sample_rate)    # 10ms hop
    
    # Calculate energy for each frame
    energy_db = []
    for i in range(0, len(audio_mono) - frame_length, hop_length):
        frame = audio_mono[i:i + frame_length]
        rms = np.sqrt(np.mean(frame ** 2))
        db = 20 * np.log10(rms + 1e-10)  # Add small value to avoid log(0)
        energy_db.append(db)
    
    energy_db = np.array(energy_db)
    
    # Find silence regions (below threshold)
    is_silence = energy_db < silence_threshold
    
    # Convert frame indices to sample indices
    min_silence_frames = int(min_silence_duration * sample_rate / hop_length)
    
    # Find continuous silence regions
    silence_intervals = []
    in_silence = False
    silence_start = 0
    
    for i, silent in enumerate(is_silence):
        if silent and not in_silence:
            # Start of silence
            in_silence = True
            silence_start = i
        elif not silent and in_silence:
            # End of silence
            silence_length = i - silence_start
            if silence_length >= min_silence_frames:
                # Convert frame indices to sample indices
                start_sample = silence_start * hop_length
                end_sample = i * hop_length
                silence_intervals.append((start_sample, end_sample))
            in_silence = False
    
    # Check last region
    if in_silence:
        silence_length = len(is_silence) - silence_start
        if silence_length >= min_silence_frames:
            start_sample = silence_start * hop_length
            end_sample = len(audio_mono)
            silence_intervals.append((start_sample, end_sample))
    
    return silence_intervals

def _smart_cut_audio_at_silence(input_path: str, max_duration: float = 10.0):
    """
    Smart audio cutting that uses silence intervals as natural cut points.
    Tries to find a segment between 3s and 15s that ends at a silence.
    
    Args:
        input_path: Path to input audio file
        max_duration: Maximum duration (unused, kept for compatibility)
    
    Returns:
        Path to cut audio file
    """
    try:
        # Load audio data
        audio_data, sample_rate = sf.read(input_path)
        
        total_duration = len(audio_data) / sample_rate
        
        # If audio is shorter than 3 seconds, return original
        if total_duration < 3.0:
            print(f"📏 Audio duration ({total_duration:.1f}s) is too short, keeping original")
            return input_path
        
        # If audio is between 3s and 15s, return original
        if total_duration <= 15.0:
            print(f"📏 Audio duration ({total_duration:.1f}s) is within ideal range (3-15s)")
            return input_path
        
        print(f"🔍 Analyzing audio ({total_duration:.1f}s) for silence intervals...")
        
        # Detect silence intervals
        silence_intervals = _detect_silence_intervals(audio_data, sample_rate)
        
        if not silence_intervals:
            print(f"⚠️ No silence intervals found, cutting at 10 seconds")
            cut_sample = int(10.0 * sample_rate)
            cut_audio = audio_data[:cut_sample]
        else:
            print(f"✓ Found {len(silence_intervals)} silence intervals")
            
            # Find the best cut point
            best_cut_sample = None
            
            for start_silence, end_silence in silence_intervals:
                # Use the middle of the silence interval as cut point
                cut_sample = (start_silence + end_silence) // 2
                cut_duration = cut_sample / sample_rate
                
                # Check if this cut point gives us a good duration (3s to 15s)
                if 3.0 <= cut_duration <= 15.0:
                    best_cut_sample = cut_sample
                    print(f"✓ Found ideal cut point at {cut_duration:.1f}s (at silence interval)")
                    break
            
            # If no ideal cut point found, try to get closest to target
            if best_cut_sample is None:
                # Find the silence interval closest to 10 seconds
                target_sample = int(10.0 * sample_rate)
                closest_silence = min(silence_intervals, 
                                    key=lambda x: abs((x[0] + x[1]) // 2 - target_sample))
                best_cut_sample = (closest_silence[0] + closest_silence[1]) // 2
                cut_duration = best_cut_sample / sample_rate
                print(f"✓ Using closest silence interval at {cut_duration:.1f}s")
            
            cut_audio = audio_data[:best_cut_sample]
        
        # Create output path with _cut suffix
        input_name = os.path.splitext(input_path)[0]
        output_path = f"{input_name}_cut.wav"
        
        # Save cut audio
        sf.write(output_path, cut_audio, sample_rate)
        
        cut_duration = len(cut_audio) / sample_rate
        
        print(f"✂️ Smart cut: {total_duration:.1f}s → {cut_duration:.1f}s (saved to {os.path.basename(output_path)})")
        
        # Remove original file and return cut file path
        try:
            os.remove(input_path)
        except Exception as cleanup_error:
            print(f"⚠️ Could not remove original audio file: {cleanup_error}")
        
        return output_path
        
    except Exception as e:
        print(f"❌ Error cutting audio: {e}")
        # Return original path if cutting fails
        return input_path


def _append_suffix_to_path(file_path: str, suffix: str) -> str:
    """Create a new path with the given suffix before the extension."""
    path_obj = Path(file_path)
    return str(path_obj.with_name(f"{path_obj.stem}{suffix}{path_obj.suffix}"))


def _extract_backing_track_from_vocals(
    original_mix: AudioSegment,
    enhanced_vocals: AudioSegment,
) -> Optional[AudioSegment]:
    """
    Approximate an instrumental backing track by subtracting the ClearVoice
    MossFormer2 vocals estimate from the original mixture.
    """
    if original_mix is None or enhanced_vocals is None:
        return None

    try:
        target_rate = int(original_mix.frame_rate or enhanced_vocals.frame_rate or 44100)
        target_channels = int(original_mix.channels or enhanced_vocals.channels or 1)
        target_sample_width = 2  # work in int16 space for predictable clipping

        mix_aligned = (
            original_mix.set_frame_rate(target_rate)
            .set_channels(target_channels)
            .set_sample_width(target_sample_width)
        )
        vocals_aligned = (
            enhanced_vocals.set_frame_rate(target_rate)
            .set_channels(target_channels)
            .set_sample_width(target_sample_width)
        )

        mix_samples = np.array(mix_aligned.get_array_of_samples(), dtype=np.float32)
        vocal_samples = np.array(vocals_aligned.get_array_of_samples(), dtype=np.float32)

        if target_channels > 1:
            mix_samples = mix_samples.reshape((-1, target_channels))
            vocal_samples = vocal_samples.reshape((-1, target_channels))

        min_len = min(len(mix_samples), len(vocal_samples))
        if min_len == 0:
            return None

        residual = np.copy(mix_samples)
        residual[:min_len] = mix_samples[:min_len] - vocal_samples[:min_len]
        if len(mix_samples) > min_len:
            residual[min_len:] = mix_samples[min_len:]

        clip_min = np.iinfo(np.int16).min
        clip_max = np.iinfo(np.int16).max
        residual = np.clip(residual, clip_min, clip_max).astype(np.int16)

        if target_channels > 1:
            residual = residual.reshape(-1)

        return AudioSegment(
            residual.tobytes(),
            frame_rate=target_rate,
            sample_width=target_sample_width,
            channels=target_channels,
        )
    except Exception as exc:
        print(f"⚠️ Failed to extract backing track from ClearVoice output: {exc}")
        return None


def _apply_clearvoice_processing_sync(
    input_path: str,
    apply_enhancement: bool,
    apply_super_resolution: bool,
) -> Tuple[str, List[str], Optional[str]]:
    """Run ClearVoice enhancement/super-resolution synchronously."""
    global _enhancement_model, _super_res_model
    
    if ClearVoice is None:
        raise RuntimeError("ClearVoice package is not available in the environment.")
    
    generated_paths: List[str] = []
    enhancement_output_path: Optional[str] = None
    current_input = input_path
    final_path = input_path
    
    try:
        if apply_enhancement:
            print("✨ ClearVoice: Applying MossFormer2_SE_48K enhancement...")
            # Initialize enhancement model if not already created
            if _enhancement_model is None:
                print("🔧 Initializing enhancement model (first use)...")
                _enhancement_model = ClearVoice(task="speech_enhancement", model_names=["MossFormer2_SE_48K"])
            enhancement_output = _enhancement_model(input_path=current_input, online_write=False)
            enhanced_path = _append_suffix_to_path(current_input, "_se")
            _enhancement_model.write(enhancement_output, output_path=enhanced_path)
            generated_paths.append(enhanced_path)
            final_path = enhanced_path
            current_input = enhanced_path
            enhancement_output_path = enhanced_path
            print(f"✅ ClearVoice: Enhancement saved to {os.path.basename(enhanced_path)}")
        
        if apply_super_resolution:
            print("🎛️ ClearVoice: Applying MossFormer2_SR_48K super-resolution...")
            # Initialize super-resolution model if not already created
            if _super_res_model is None:
                print("🔧 Initializing super-resolution model (first use)...")
                _super_res_model = ClearVoice(task="speech_super_resolution", model_names=["MossFormer2_SR_48K"])
            super_res_output = _super_res_model(input_path=current_input, online_write=False)
            super_res_path = _append_suffix_to_path(current_input, "_sr")
            _super_res_model.write(super_res_output, output_path=super_res_path)
            generated_paths.append(super_res_path)
            final_path = super_res_path
            print(f"✅ ClearVoice: Super-resolution saved to {os.path.basename(super_res_path)}")
        
        return final_path, generated_paths, enhancement_output_path
    except Exception:
        for created_path in generated_paths:
            try:
                os.remove(created_path)
            except Exception:
                pass
        raise


async def apply_clearvoice_processing(
    input_path: str,
    apply_enhancement: bool,
    apply_super_resolution: bool,
) -> Tuple[str, List[str], Optional[str]]:
    """Async wrapper for ClearVoice processing."""
    if not (apply_enhancement or apply_super_resolution):
        return input_path, []
    
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor,
        _apply_clearvoice_processing_sync,
        input_path,
        apply_enhancement,
        apply_super_resolution,
    )

async def convert_audio_to_format(wav_data, sample_rate, output_format="mp3", bitrate="128k"):
    """Convert audio data to specified format (MP3 or WAV)"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, _convert_audio_to_format_sync, wav_data, sample_rate, output_format, bitrate)

def _convert_audio_to_format_sync(wav_data, sample_rate, output_format="mp3", bitrate="128k"):
    """Synchronous audio format conversion"""
    try:
        # Convert numpy array to AudioSegment
        # Ensure wav_data is in the right format
        if wav_data.dtype != 'int16':
            # Convert float to int16
            wav_data = (wav_data * 32767).astype('int16')
        
        # Create AudioSegment from raw audio data
        audio_segment = AudioSegment(
            wav_data.tobytes(),
            frame_rate=sample_rate,
            sample_width=wav_data.dtype.itemsize,
            channels=1 if len(wav_data.shape) == 1 else wav_data.shape[1]
        )
        
        # Export to desired format
        with BytesIO() as output_buffer:
            if output_format.lower() == "mp3":
                audio_segment.export(output_buffer, format="mp3", bitrate=bitrate)
                media_type = "audio/mpeg"
                file_extension = "mp3"
            else:
                audio_segment.export(output_buffer, format="wav")
                media_type = "audio/wav" 
                file_extension = "wav"
            
            return output_buffer.getvalue(), media_type, file_extension
            
    except Exception as e:
        print(f"⚠️ Audio conversion failed, falling back to WAV: {e}")
        # Fallback to original soundfile method
        with BytesIO() as wav_buffer:
            sf.write(wav_buffer, wav_data, sample_rate, format='WAV')
            return wav_buffer.getvalue(), "audio/wav", "wav"


# Gemini translation helpers
def _get_gemini_model_name() -> str:
    return os.getenv(GEMINI_MODEL_ENV_VAR, DEFAULT_GEMINI_MODEL_NAME)


def _strip_code_fences(text: str) -> str:
    if not text:
        return ""
    match = JSON_FENCE_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        return text[3:-3].strip()
    # Fallback: extract JSON array if present
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1].strip()
    return text


def _parse_gemini_json(text: str) -> List[Dict[str, Any]]:
    cleaned = _strip_code_fences(text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Gemini returned invalid JSON: {exc}") from exc
    if not isinstance(data, list):
        raise ValueError("Gemini response must be a JSON array of segments.")
    return data


def _extract_text_from_gemini_response(response: Any) -> str:
    if response is None:
        return ""

    for attr in ("text", "output_text", "output_texts"):
        value = getattr(response, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list) and value:
            concatenated = "\n".join(str(item) for item in value if item)
            if concatenated.strip():
                return concatenated.strip()

    parts_text: List[str] = []

    candidates = getattr(response, "candidates", None)
    if candidates:
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            if content is not None:
                parts = getattr(content, "parts", None)
                if parts:
                    for part in parts:
                        part_text = getattr(part, "text", None)
                        if isinstance(part_text, str) and part_text.strip():
                            parts_text.append(part_text.strip())
            if isinstance(candidate, dict):
                for part in candidate.get("content", {}).get("parts", []):
                    part_text = part.get("text")
                    if isinstance(part_text, str) and part_text.strip():
                        parts_text.append(part_text.strip())

    if not parts_text:
        contents = getattr(response, "contents", None)
        if contents:
            for content in contents:
                parts = getattr(content, "parts", None)
                if parts:
                    for part in parts:
                        part_text = getattr(part, "text", None)
                        if isinstance(part_text, str) and part_text.strip():
                            parts_text.append(part_text.strip())
                elif isinstance(content, dict):
                    for part in content.get("parts", []):
                        part_text = part.get("text")
                        if isinstance(part_text, str) and part_text.strip():
                            parts_text.append(part_text.strip())

    return "\n".join(parts_text).strip()


def _parse_timestamp_to_ms(timestamp_value: Any) -> Optional[int]:
    if timestamp_value is None:
        return None
    if isinstance(timestamp_value, (int, float)):
        # Treat integer millisecond values separately
        if isinstance(timestamp_value, int) and timestamp_value >= 1000:
            return max(0, int(timestamp_value))
        seconds = float(timestamp_value)
        return max(0, int(round(seconds * 1000)))
    value = str(timestamp_value).strip()
    if not value:
        return None
    value = value.replace(",", ".")
    # Try simple numeric parse
    if re.fullmatch(r"\d+(?:\.\d+)?", value):
        seconds = float(value)
        return max(0, int(seconds * 1000))
    # Handle colon-delimited formats (e.g., hh:mm:ss.xxx or mm:ss)
    parts = value.split(":")
    try:
        parts = [p.strip() for p in parts if p.strip()]
        if not parts:
            return None
        total_seconds = 0.0
        multiplier = 1.0
        for part in reversed(parts):
            if "." in part:
                seconds = float(part)
            else:
                seconds = float(int(part))
            total_seconds += seconds * multiplier
            multiplier *= 60.0
        return max(0, int(round(total_seconds * 1000)))
    except ValueError:
        return None


def _format_ms_to_timestamp(milliseconds: int) -> str:
    ms = max(0, int(milliseconds))
    total_seconds = ms / 1000.0
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    if abs(seconds - round(seconds)) < 1e-3:
        seconds_str = f"{int(round(seconds)):02d}"
    else:
        seconds_str = f"{seconds:06.3f}".rstrip("0").rstrip(".")
        if seconds < 10 and not seconds_str.startswith("0"):
            seconds_str = "0" + seconds_str
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds_str}"
    return f"{minutes:02d}:{seconds_str}"


def _find_case_insensitive_key(data: Dict[str, Any], candidates: List[str], ignore: Set[str]) -> Optional[str]:
    lowered = {str(k).lower(): k for k in data.keys()}
    for candidate in candidates:
        candidate_lower = candidate.lower()
        if candidate_lower in lowered:
            if candidate_lower in ignore:
                continue
            return lowered[candidate_lower]
    return None


def _extract_text_fields(entry: Dict[str, Any], dest_language: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    ignore_keys = {
        "start",
        "end",
        "start_time",
        "end_time",
        "start_ms",
        "end_ms",
        "duration",
        "duration_ms",
        "timestamp",
        "time",
        "index",
        "type",
        "speaker",
        "language",
        "source_language",
        "target_language",
    }
    lower_dest = dest_language.lower() if dest_language else ""
    translation_candidates = [
        "translated_text",
        "translation",
        "translation_text",
        "target_text",
        "target",
        "output",
        "output_text",
        "translation_result",
        "english",
        "en",
    ]
    if lower_dest:
        translation_candidates.extend([
            lower_dest,
            lower_dest.replace("-", "_"),
            lower_dest.replace(" ", "_"),
        ])
    source_candidates = [
        "source_text",
        "source",
        "original_text",
        "original",
        "transcript",
        "transcription",
        "input_text",
        "utterance",
        "text",
        "chinese",
        "zh",
        "cn",
    ]
    translation_key = _find_case_insensitive_key(entry, translation_candidates, ignore_keys)
    source_key = _find_case_insensitive_key(entry, source_candidates, ignore_keys)

    def _fallback_key(preferred_key: Optional[str]) -> Optional[str]:
        if preferred_key is not None:
            return preferred_key
        for key, value in entry.items():
            key_lower = str(key).lower()
            if key_lower in ignore_keys:
                continue
            if isinstance(value, str) and value.strip():
                return key
        return None

    source_key = _fallback_key(source_key)
    translation_key = _fallback_key(translation_key)

    source_text = entry.get(source_key) if source_key else None
    translated_text = entry.get(translation_key) if translation_key else None
    return source_text, translated_text, source_key, translation_key


def _coerce_to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _coerce_positive_int(
    value: Any,
    default: int,
    *,
    min_value: int = 0,
    max_value: Optional[int] = None,
) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    if parsed < min_value:
        return min_value
    if max_value is not None and parsed > max_value:
        return max_value
    return parsed


def _parse_manual_segments_input(raw: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    """
    Parse user-supplied Gemini segment JSON.
    Accepts either a JSON array or an object with a top-level "segments" array.
    """
    if raw is None:
        return None
    raw_str = str(raw).strip()
    if not raw_str:
        return None
    try:
        parsed = json.loads(raw_str)
    except json.JSONDecodeError as exc:
        raise ValueError(f"segments_json is not valid JSON: {exc}") from exc

    if isinstance(parsed, dict) and "segments" in parsed:
        parsed = parsed["segments"]

    if not isinstance(parsed, list):
        raise ValueError("segments_json must be a JSON array of segment objects.")
    if not parsed:
        raise ValueError("segments_json array is empty.")

    cleaned: List[Dict[str, Any]] = []
    for idx, entry in enumerate(parsed):
        if not isinstance(entry, dict):
            raise ValueError(f"segments_json[{idx}] is not an object.")
        cleaned.append(entry)
    return cleaned


def _merge_short_speech_segments(
    segments: List[Dict[str, Any]],
    min_duration_ms: int,
    max_merge_interval_ms: int = MAX_MERGE_INTERVAL_MS,
) -> List[Dict[str, Any]]:
    merged_segments: List[Dict[str, Any]] = []
    i = 0
    total = len(segments)

    def _concat_text(a: Optional[str], b: Optional[str]) -> str:
        parts = [part.strip() for part in [a or "", b or ""] if part and part.strip()]
        return " ".join(parts)

    while i < total:
        segment = segments[i]
        if segment.get("type") != "speech":
            merged_segments.append(segment)
            i += 1
            continue

        start_ms = int(segment.get("start_ms", 0))
        end_ms = int(segment.get("end_ms", start_ms))
        source_text = segment.get("source_text", "")
        translated_text = segment.get("translated_text", "")
        raw_chunks: List[Any] = [segment.get("raw_chunk")]
        raw_indices: List[Any] = [segment.get("raw_chunk_index")]

        j = i + 1

        # Merge forward with following silence/speech until threshold reached
        # Only merge if silence intervals are short (<= max_merge_interval_ms)
        while end_ms - start_ms < min_duration_ms and j < total:
            next_segment = segments[j]
            seg_type = next_segment.get("type")
            if seg_type == "silence":
                silence_duration = int(next_segment.get("duration_ms", 0))
                # Only merge if silence is short enough
                if silence_duration <= max_merge_interval_ms:
                    end_ms = int(next_segment.get("end_ms", end_ms))
                    j += 1
                    continue
                else:
                    # Silence gap too long, stop merging forward
                    break
            if seg_type == "speech":
                end_ms = int(next_segment.get("end_ms", end_ms))
                source_text = _concat_text(source_text, next_segment.get("source_text"))
                translated_text = _concat_text(translated_text, next_segment.get("translated_text"))
                raw_chunks.append(next_segment.get("raw_chunk"))
                raw_indices.append(next_segment.get("raw_chunk_index"))
                j += 1
                continue
            break

        duration_ms = end_ms - start_ms

        if duration_ms < min_duration_ms and merged_segments:
            # Merge backward with previous speech if forward merge insufficient
            # Find the last speech segment in merged_segments
            prev_speech_idx = len(merged_segments) - 1
            while prev_speech_idx >= 0 and merged_segments[prev_speech_idx].get("type") != "speech":
                prev_speech_idx -= 1
            
            if prev_speech_idx >= 0:
                prev_segment = merged_segments[prev_speech_idx]
                prev_end_ms = int(prev_segment.get("end_ms", 0))
                gap_ms = start_ms - prev_end_ms
                # Only merge backward if gap is short enough
                if gap_ms <= max_merge_interval_ms:
                    prev_segment["end_ms"] = end_ms
                    prev_segment["duration_ms"] = end_ms - int(prev_segment.get("start_ms", start_ms))
                    prev_segment["end"] = _format_ms_to_timestamp(end_ms)
                    prev_segment["source_text"] = _concat_text(prev_segment.get("source_text"), source_text)
                    prev_segment["translated_text"] = _concat_text(prev_segment.get("translated_text"), translated_text)

                    prev_raw = prev_segment.get("raw_chunk")
                    if isinstance(prev_raw, list):
                        prev_raw.extend(raw_chunks)
                    elif prev_raw is None:
                        prev_segment["raw_chunk"] = raw_chunks
                    else:
                        prev_segment["raw_chunk"] = [prev_raw] + raw_chunks

                    prev_indices = prev_segment.get("raw_chunk_index")
                    raw_indices_clean = [idx for idx in raw_indices if idx is not None]
                    if raw_indices_clean:
                        if isinstance(prev_indices, list):
                            prev_indices.extend(raw_indices_clean)
                        elif prev_indices is None:
                            prev_segment["raw_chunk_index"] = raw_indices_clean
                        else:
                            prev_segment["raw_chunk_index"] = [prev_indices] + raw_indices_clean

                    i = j
                    continue

        new_segment = dict(segment)
        new_segment["start_ms"] = start_ms
        new_segment["end_ms"] = end_ms
        new_segment["duration_ms"] = duration_ms
        new_segment["start"] = _format_ms_to_timestamp(start_ms)
        new_segment["end"] = _format_ms_to_timestamp(end_ms)
        new_segment["source_text"] = source_text
        new_segment["translated_text"] = translated_text

        raw_chunks_clean = [chunk for chunk in raw_chunks if chunk is not None]
        if raw_chunks_clean:
            new_segment["raw_chunk"] = raw_chunks_clean if len(raw_chunks_clean) > 1 else raw_chunks_clean[0]
        raw_indices_clean = [idx for idx in raw_indices if idx is not None]
        if raw_indices_clean:
            new_segment["raw_chunk_index"] = raw_indices_clean if len(raw_indices_clean) > 1 else raw_indices_clean[0]

        merged_segments.append(new_segment)
        i = j

    # Reassign indices
    for idx, seg in enumerate(merged_segments):
        seg["index"] = idx

    return merged_segments


def _segment_audio_data_uri(
    audio: AudioSegment,
    start_ms: int,
    end_ms: int,
    fmt: str = "wav",
) -> Optional[str]:
    start = max(0, int(start_ms))
    end = max(start, int(end_ms))
    audio_len = len(audio)
    if start >= audio_len or start == end:
        return None
    if end > audio_len:
        end = audio_len
    snippet = audio[start:end]
    if len(snippet) == 0:
        return None
    buffer = BytesIO()
    snippet.export(buffer, format=fmt)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:audio/{fmt};base64,{encoded}"


def _serialize_segments_for_ui(
    segments: List[Dict[str, Any]],
    audio: AudioSegment,
) -> List[Dict[str, Any]]:
    ui_segments: List[Dict[str, Any]] = []
    for segment in segments:
        seg_type = segment.get("type", "speech")
        start_ms = int(segment.get("start_ms", 0))
        end_ms = int(segment.get("end_ms", start_ms))
        duration_ms = int(segment.get("duration_ms", max(0, end_ms - start_ms)))
        base_payload = {
            "index": segment.get("index"),
            "type": seg_type,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "duration_ms": duration_ms,
            "start": segment.get("start"),
            "end": segment.get("end"),
            "source_text": segment.get("source_text", ""),
            "translated_text": segment.get("translated_text", ""),
            "text_keys": segment.get("text_keys", {}),
        }
        if seg_type == "speech":
            base_payload["generate"] = True
            preview = _segment_audio_data_uri(audio, start_ms, end_ms)
            if preview:
                base_payload["audio_preview"] = preview
        else:
            base_payload["generate"] = False
        ui_segments.append(base_payload)
    return ui_segments


def _cleanup_expired_translate_sessions_locked(now: Optional[float] = None) -> None:
    if now is None:
        now = time.time()
    expired: List[str] = [
        session_id
        for session_id, session in ADVANCED_TRANSLATE_SESSIONS.items()
        if now - session.created_at > ADVANCED_TRANSLATE_SESSION_TTL_SECONDS
    ]
    for session_id in expired:
        ADVANCED_TRANSLATE_SESSIONS.pop(session_id, None)


async def _create_translate_session(
    original_audio: AudioSegment,
    dest_language: str,
    prompt: str,
    translate_enabled: bool,
    response_format: str,
    bitrate: str,
    input_mime_type: Optional[str],
    clearvoice_settings: Dict[str, bool],
    base_segments: List[Dict[str, Any]],
    gemini_chunks: List[Dict[str, Any]],
    gemini_model: str,
    gemini_api_key: Optional[str],
    backing_track_audio: Optional[AudioSegment] = None,
    merge_with_backing: bool = False,
) -> TranslateSessionData:
    session_id = uuid.uuid4().hex
    session = TranslateSessionData(
        session_id=session_id,
        original_audio=original_audio,
        dest_language=dest_language,
        prompt=prompt,
        translate_enabled=translate_enabled,
        response_format=response_format,
        bitrate=bitrate,
        input_mime_type=input_mime_type,
        clearvoice_settings=dict(clearvoice_settings or {}),
        base_segments=copy.deepcopy(base_segments),
        gemini_chunks=copy.deepcopy(gemini_chunks),
        gemini_model=gemini_model,
        gemini_api_key=gemini_api_key,
        backing_track_audio=backing_track_audio,
        merge_with_backing=merge_with_backing,
    )
    async with ADVANCED_TRANSLATE_SESSION_LOCK:
        _cleanup_expired_translate_sessions_locked()
        ADVANCED_TRANSLATE_SESSIONS[session_id] = session
    return session


async def _get_translate_session(session_id: str) -> Optional[TranslateSessionData]:
    async with ADVANCED_TRANSLATE_SESSION_LOCK:
        _cleanup_expired_translate_sessions_locked()
        session = ADVANCED_TRANSLATE_SESSIONS.get(session_id)
        if session:
            session.created_at = time.time()
        return session


async def _update_translate_session_segments(
    session_id: str, segments: List[Dict[str, Any]]
) -> None:
    async with ADVANCED_TRANSLATE_SESSION_LOCK:
        session = ADVANCED_TRANSLATE_SESSIONS.get(session_id)
        if session:
            session.base_segments = copy.deepcopy(segments)
            session.created_at = time.time()


async def _update_translate_session_metadata(
    session_id: str,
    *,
    response_format: Optional[str] = None,
    bitrate: Optional[str] = None,
    gemini_model: Optional[str] = None,
) -> None:
    async with ADVANCED_TRANSLATE_SESSION_LOCK:
        session = ADVANCED_TRANSLATE_SESSIONS.get(session_id)
        if session:
            if response_format:
                session.response_format = response_format
            if bitrate:
                session.bitrate = bitrate
            if gemini_model:
                session.gemini_model = gemini_model
            session.created_at = time.time()


def _guess_audio_format_from_mime(mime_type: Optional[str]) -> Optional[str]:
    if not mime_type:
        return None
    mime = mime_type.lower()
    mapping = {
        "audio/wav": "wav",
        "audio/x-wav": "wav",
        "audio/wave": "wav",
        "audio/mpeg": "mp3",
        "audio/mp3": "mp3",
        "audio/ogg": "ogg",
        "audio/ogg; codecs=opus": "ogg",
        "audio/webm": "webm",
        "audio/opus": "opus",
        "audio/flac": "flac",
        "audio/aac": "aac",
        "audio/mp4": "mp4",
        "audio/m4a": "mp4",
        "audio/x-m4a": "mp4",
        "audio/vnd.wave": "wav",
    }
    return mapping.get(mime)


def _prepare_translation_segments(
    original_audio: AudioSegment,
    chunk_data: List[Dict[str, Any]],
    dest_language: str,
    *,
    min_speech_duration_ms: int = MIN_SPEECH_DURATION_MS,
    max_merge_interval_ms: int = MAX_MERGE_INTERVAL_MS,
) -> List[Dict[str, Any]]:
    total_duration_ms = len(original_audio)
    segments: List[Dict[str, Any]] = []
    current_ms = 0

    def _get_timestamp(entry: Dict[str, Any], keys: List[str]) -> Optional[int]:
        for key in keys:
            if key in entry:
                parsed = _parse_timestamp_to_ms(entry[key])
                if parsed is not None:
                    return parsed
        return None

    def _ensure_silence(duration_ms: int, position: str) -> Optional[Dict[str, Any]]:
        if duration_ms <= 0:
            return None
        start_ms = current_ms
        end_ms = current_ms + duration_ms
        segment_payload = {
            "type": "silence",
            "start_ms": start_ms,
            "end_ms": end_ms,
            "duration_ms": duration_ms,
            "start": _format_ms_to_timestamp(start_ms),
            "end": _format_ms_to_timestamp(end_ms),
            "position": position,
        }
        return segment_payload

    if not chunk_data:
        raise ValueError("Gemini returned no transcription segments.")

    for idx, entry in enumerate(chunk_data):
        if not isinstance(entry, dict):
            continue

        start_ms = _get_timestamp(
            entry,
            [
                "start_ms",
                "startMilliseconds",
                "start_milliseconds",
                "start",
                "start_time",
                "begin",
                "from",
            ],
        )
        end_ms = _get_timestamp(
            entry,
            [
                "end_ms",
                "endMilliseconds",
                "end_milliseconds",
                "end",
                "end_time",
                "stop",
                "to",
            ],
        )

        if start_ms is None:
            start_ms = current_ms
        if end_ms is None:
            duration_ms = _get_timestamp(
                entry,
                ["duration_ms", "durationMilliseconds", "duration", "length", "segment_duration"],
            )
            if duration_ms is not None:
                end_ms = start_ms + duration_ms

        if end_ms is None:
            # Fallback: assume contiguous audio up to detected order
            # Use remainder of audio divided equally over remaining segments
            remaining_segments = max(len(chunk_data) - idx, 1)
            remaining_ms = max(total_duration_ms - start_ms, 0)
            avg_duration = remaining_ms // remaining_segments if remaining_segments else remaining_ms
            end_ms = start_ms + avg_duration

        start_ms = max(0, min(start_ms, total_duration_ms))
        end_ms = max(0, min(end_ms, total_duration_ms))
        if start_ms < current_ms:
            start_ms = current_ms
        if end_ms <= start_ms:
            continue

        # Add silence if there is a gap before this segment
        if start_ms > current_ms:
            silence_payload = _ensure_silence(start_ms - current_ms, "leading" if current_ms == 0 else "between")
            if silence_payload:
                silence_payload["index"] = len(segments)
                segments.append(silence_payload)
            current_ms = start_ms

        source_text, translated_text, source_key, translation_key = _extract_text_fields(entry, dest_language)
        if isinstance(source_text, str):
            source_text_value = source_text.strip()
        elif source_text is None:
            source_text_value = ""
        else:
            source_text_value = str(source_text)

        if isinstance(translated_text, str):
            translated_text_value = translated_text.strip()
        elif translated_text is None:
            translated_text_value = ""
        else:
            translated_text_value = str(translated_text)

        segment_payload: Dict[str, Any] = {
            "type": "speech",
            "start_ms": start_ms,
            "end_ms": end_ms,
            "duration_ms": end_ms - start_ms,
            "start": _format_ms_to_timestamp(start_ms),
            "end": _format_ms_to_timestamp(end_ms),
            "source_text": source_text_value,
            "translated_text": translated_text_value,
            "text_keys": {
                "source": source_key,
                "translated": translation_key,
            },
            "raw_chunk_index": idx,
            "raw_chunk": entry,
        }
        segment_payload["index"] = len(segments)
        segments.append(segment_payload)
        current_ms = end_ms

    # Add trailing silence if needed
    if current_ms < total_duration_ms:
        remaining = total_duration_ms - current_ms
        silence_payload = _ensure_silence(remaining, "trailing")
        if silence_payload:
            silence_payload["index"] = len(segments)
            segments.append(silence_payload)

    segments = _merge_short_speech_segments(
        segments,
        max(0, int(min_speech_duration_ms)),
        max_merge_interval_ms=max(0, int(max_merge_interval_ms)),
    )

    return segments


def _create_silence_segment(duration_ms: int, frame_rate: int, sample_width: int, channels: int) -> AudioSegment:
    duration_ms = max(0, int(duration_ms))
    silence = AudioSegment.silent(duration=duration_ms, frame_rate=frame_rate)
    silence = silence.set_sample_width(sample_width)
    silence = silence.set_channels(channels)
    return silence


def _match_segment_duration(
    segment: AudioSegment,
    target_ms: int,
    frame_rate: int,
    sample_width: int,
    channels: int,
) -> AudioSegment:
    target_ms = max(0, int(target_ms))
    segment = segment.set_frame_rate(frame_rate)
    segment = segment.set_sample_width(sample_width)
    segment = segment.set_channels(channels)

    if target_ms == 0:
        return _create_silence_segment(0, frame_rate, sample_width, channels)

    current_ms = len(segment)
    tolerance_ms = 100
    diff = target_ms - current_ms
    if diff > 0:
        if diff <= tolerance_ms:
            return segment + _create_silence_segment(diff, frame_rate, sample_width, channels)
        return segment + _create_silence_segment(diff, frame_rate, sample_width, channels)
    # Never trim segments; return original even if longer than target.
    return segment


async def _synthesize_translated_audio(
    original_audio: AudioSegment,
    segments: List[Dict[str, Any]],
    dest_language: str,
    response_format: str = TRANSLATE_DEFAULT_OUTPUT_FORMAT,
    bitrate: str = TRANSLATE_DEFAULT_BITRATE,
    input_mime_type: Optional[str] = None,
    clearvoice_settings: Optional[Dict[str, bool]] = None,
    backing_track_audio: Optional[AudioSegment] = None,
    merge_with_backing: bool = False,
) -> Tuple[bytes, str, Dict[str, Any]]:
    tts = tts_manager.get_tts()
    frame_rate = int(original_audio.frame_rate or 22050)
    sample_width = int(original_audio.sample_width or 2)
    channels = int(original_audio.channels or 1)

    combined_audio = _create_silence_segment(0, frame_rate, sample_width, channels)
    generation_log: List[Dict[str, Any]] = []
    semaphore = asyncio.Semaphore(TRANSLATION_TTS_CONCURRENCY)

    async def process_segment(index: int, segment: Dict[str, Any]):
        seg_type = segment.get("type")
        start_ms = int(segment.get("start_ms", 0))
        end_ms = int(segment.get("end_ms", start_ms))
        duration_ms = max(0, int(segment.get("duration_ms", max(0, end_ms - start_ms))))

        if seg_type == "silence":
            audio_seg = _create_silence_segment(duration_ms, frame_rate, sample_width, channels)
            log_entry = {
                "index": index,
                "type": "silence",
                "duration_ms": duration_ms,
                "start_ms": start_ms,
                "end_ms": end_ms,
            }
            return index, audio_seg, log_entry

        translated_text = (segment.get("translated_text") or "").strip()
        source_text = (segment.get("source_text") or "").strip()
        generate_segment = segment.get("generate", True)
        keep_original = segment.get("keep_original", False) or (not generate_segment)

        chunk_audio = original_audio[start_ms:end_ms]

        if keep_original:
            audio_seg = _match_segment_duration(chunk_audio, duration_ms, frame_rate, sample_width, channels)
            log_entry = {
                "index": index,
                "type": "speech",
                "status": "preserved",
                "duration_ms": duration_ms,
                "source_text": source_text,
                "translated_text": translated_text,
            }
            return index, audio_seg, log_entry

        if not translated_text:
            audio_seg = _create_silence_segment(duration_ms, frame_rate, sample_width, channels)
            log_entry = {
                "index": index,
                "type": "speech",
                "status": "skipped",
                "reason": "empty_translation",
                "source_text": source_text,
                "duration_ms": duration_ms,
            }
            return index, audio_seg, log_entry

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_chunk:
            chunk_audio.export(tmp_chunk.name, format="wav")
            chunk_path = tmp_chunk.name

        generation_target_ms = max(0, duration_ms - AUDIO_GENERATION_MARGIN_MS)
        generated_path = os.path.join("outputs", f"translate_{uuid.uuid4().hex}.wav")

        status = "success"
        error_message = None
        generated_audio: Optional[AudioSegment] = None

        try:
            async with semaphore:
                inference_path = await tts.infer(
                    spk_audio_prompt=chunk_path,
                    text=translated_text,
                    output_path=generated_path,
                    interval_silence=0,
                    speech_length=generation_target_ms,
                    diffusion_steps=10,
                    verbose=cmd_args.verbose,
                )
            generated_audio = AudioSegment.from_file(inference_path)
            generated_audio = _match_segment_duration(generated_audio, duration_ms, frame_rate, sample_width, channels)
        except Exception as exc:
            status = "error"
            error_message = str(exc)
            print(f"⚠️ Translation synthesis failed for segment {index}: {error_message}")
            generated_audio = _create_silence_segment(duration_ms, frame_rate, sample_width, channels)
        finally:
            if os.path.exists(chunk_path):
                try:
                    os.remove(chunk_path)
                except Exception:
                    pass
            if os.path.exists(generated_path):
                try:
                    await async_remove_file(generated_path)
                except Exception:
                    pass

        log_entry: Dict[str, Any] = {
            "index": index,
            "type": "speech",
            "status": status,
            "duration_ms": duration_ms,
            "source_text": source_text,
            "translated_text": translated_text,
        }
        if status == "error" and error_message:
            log_entry["error"] = error_message

        return index, generated_audio, log_entry

    segment_tasks = [process_segment(idx, segment) for idx, segment in enumerate(segments)]
    results = await asyncio.gather(*segment_tasks)
    results.sort(key=lambda item: item[0])

    for _, audio_segment, log_entry in results:
        combined_audio += audio_segment
        generation_log.append(log_entry)

    original_duration_ms = len(original_audio)
    final_duration_ms = len(combined_audio)
    if final_duration_ms < original_duration_ms:
        combined_audio += _create_silence_segment(original_duration_ms - final_duration_ms, frame_rate, sample_width, channels)

    backing_applied = False
    if merge_with_backing and backing_track_audio is not None:
        try:
            prepared_backing = (
                backing_track_audio.set_frame_rate(frame_rate)
                .set_sample_width(sample_width)
                .set_channels(channels)
            )
            prepared_backing = _match_segment_duration(
                prepared_backing,
                len(combined_audio),
                frame_rate,
                sample_width,
                channels,
            )
            combined_audio = prepared_backing.overlay(combined_audio)
            backing_applied = True
        except Exception as merge_error:
            print(f"⚠️ Failed to merge translated audio with backing track: {merge_error}")

    buffer = BytesIO()
    export_kwargs: Dict[str, Any] = {}
    audio_format = (response_format or TRANSLATE_DEFAULT_OUTPUT_FORMAT).lower()
    if audio_format == "mp3" and bitrate:
        export_kwargs["bitrate"] = bitrate
    combined_audio.export(buffer, format=audio_format, **export_kwargs)
    audio_bytes = buffer.getvalue()

    media_type_map = {
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "aac": "audio/aac",
        "flac": "audio/flac",
        "opus": "audio/opus",
        "ogg": "audio/ogg",
        "webm": "audio/webm",
    }
    media_type = media_type_map.get(audio_format, f"audio/{audio_format}")

    metadata = {
        "dest_language": dest_language,
        "segment_count": len(segments),
        "speech_segment_count": sum(1 for s in segments if s["type"] == "speech"),
        "silence_segment_count": sum(1 for s in segments if s["type"] == "silence"),
        "original_duration_ms": original_duration_ms,
        "generated_duration_ms": len(combined_audio),
        "generation_log": generation_log,
    }
    if input_mime_type:
        metadata["input_mime_type"] = input_mime_type
    if clearvoice_settings:
        metadata["clearvoice"] = clearvoice_settings
    metadata["backing_track"] = {
        "available": backing_track_audio is not None,
        "merged": backing_applied,
    }

    return audio_bytes, media_type, metadata


async def _gemini_transcribe_translate(
    audio_bytes: bytes,
    mime_type: str,
    dest_language: str,
    prompt_text: Optional[str] = None,
    *,
    model_name: Optional[str] = None,
    api_key_override: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if genai is None:
        raise RuntimeError(
            "The google-genai package is required for translation. Install it with `pip install google-genai`."
        )

    api_key = (api_key_override or "").strip() or os.getenv(GEMINI_API_KEY_ENV_VAR) or os.getenv(GOOGLE_API_KEY_ENV_VAR)
    if not api_key:
        raise RuntimeError(
            f"Neither {GEMINI_API_KEY_ENV_VAR} nor {GOOGLE_API_KEY_ENV_VAR} environment variables are set."
        )

    prompt = (prompt_text or "").strip() or TRANSLATION_PROMPT_TEMPLATE.format(dest_language=dest_language)
    model_name = (model_name or "").strip() or _get_gemini_model_name()

    if types is None:
        raise RuntimeError(
            "google-genai types module is unavailable. Ensure the `google-genai` package is installed and up to date."
        )

    client = _get_gemini_client(api_key)

    user_content = types.Content(
        role="user",
        parts=[
            types.Part.from_text(text=prompt),
            types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
        ],
    )

    def _call_gemini() -> str:
        response = client.models.generate_content(
            model=model_name,
            contents=[user_content],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=DEFAULT_GEMINI_TEMPERATURE,
                top_p=DEFAULT_GEMINI_TOP_P,
            ),
        )

        prompt_feedback = getattr(response, "prompt_feedback", None)
        if prompt_feedback is not None:
            block_reason = getattr(prompt_feedback, "block_reason", None)
            if block_reason:
                raise RuntimeError(f"Gemini blocked the prompt: {block_reason}")

        text = _extract_text_from_gemini_response(response)
        if not text:
            raise RuntimeError("Gemini returned an empty response.")
        return text

    loop = asyncio.get_event_loop()
    raw_text = await loop.run_in_executor(executor, _call_gemini)
    return _parse_gemini_json(raw_text)

# Global TTS manager
class TTSManager:
    _instance = None
    _initialized = False
    
    def __init__(self):
        self.tts = None
        self.speaker_manager = None
        
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    async def initialize(self):
        if not self._initialized:
            try:
                print("🚀 Initializing IndexTTS2 vLLM v2...")
                
                # Check model directory
                if not os.path.exists(cmd_args.model_dir):
                    raise FileNotFoundError(f"Model directory {cmd_args.model_dir} does not exist")
                
                # Check required files
                required_files = [
                    "bpe.model",
                    "gpt.pth", 
                    "config.yaml",
                    "s2mel.pth",
                    "wav2vec2bert_stats.pt"
                ]
                
                for file in required_files:
                    file_path = os.path.join(cmd_args.model_dir, file)
                    if not os.path.exists(file_path):
                        raise FileNotFoundError(f"Required file {file_path} does not exist")
                
                # Initialize IndexTTS2
                self.tts = IndexTTS2(
                    model_dir=cmd_args.model_dir,
                    is_fp16=cmd_args.is_fp16,
                    use_torch_compile=cmd_args.use_torch_compile,
                    gpu_memory_utilization=cmd_args.gpu_memory_utilization
                )
                
                # Initialize speaker preset manager
                self.speaker_manager = initialize_preset_manager(self.tts)
                
                # Initialize speaker API wrapper
                global speaker_api
                speaker_api = SpeakerAPIWrapper(self.speaker_manager)
                
                self._initialized = True
                print("✅ IndexTTS2 vLLM v2 initialized successfully!")
                print(f"🎭 Speaker preset manager initialized with {len(self.speaker_manager.list_presets())} existing presets")
                
                return True
                
            except Exception as e:
                print(f"❌ Failed to initialize IndexTTS2: {e}")
                traceback.print_exc()
                self._initialized = False
                return False
        return True
    
    def get_tts(self):
        if not self._initialized or self.tts is None:
            raise Exception("IndexTTS2 not initialized")
        return self.tts
    
    def is_ready(self):
        return self._initialized and self.tts is not None

# Create global TTS manager
tts_manager = TTSManager.get_instance()

# Speaker Management Functions using existing SpeakerPresetManager
class SpeakerAPIWrapper:
    """Wrapper to adapt SpeakerPresetManager for FastAPI endpoints"""
    
    def __init__(self, preset_manager: SpeakerPresetManager):
        self.preset_manager = preset_manager
    
    async def add_speaker(
        self,
        speaker_name: str,
        audio_files: List[bytes],
        filenames: List[str],
        apply_enhancement: bool = False,
        apply_super_resolution: bool = False,
    ) -> Dict[str, str]:
        """Add a new speaker with audio files and optional ClearVoice processing."""
        try:
            # Check if speaker already exists - run in thread pool
            loop = asyncio.get_event_loop()
            clearvoice_requested = apply_enhancement or apply_super_resolution

            if clearvoice_requested and ClearVoice is None:
                return {
                    "status": "error",
                    "message": "ClearVoice package is required for enhancement or super-resolution. Install the `clearvoice` package to enable these options."
                }

            existing_presets = await loop.run_in_executor(executor, self.preset_manager.list_presets)
            
            if speaker_name in existing_presets:
                preset_info = existing_presets[speaker_name]
                return {
                    "status": "success", 
                    "message": f"Speaker '{speaker_name}' already exists",
                    "info": "already_exists",
                    "audio_count": 1,  # SpeakerPresetManager uses single audio file
                    "description": preset_info.get("description", ""),
                    "created_at": preset_info.get("created_at", 0)
                }
            
            # Save the first audio file (SpeakerPresetManager expects single file)
            if not audio_files:
                return {"status": "error", "message": "No audio files provided"}
            
            # Create temporary file for the first audio
            audio_data = audio_files[0]
            filename = filenames[0] if filenames else "audio.wav"
            
            # Save to temporary location
            temp_dir = Path("speaker_presets") / "temp"
            # Create directory in thread pool to avoid blocking
            await loop.run_in_executor(executor, lambda: temp_dir.mkdir(parents=True, exist_ok=True))
            
            temp_path = temp_dir / f"{speaker_name}_{filename}"
            await async_write_file(str(temp_path), audio_data)
            
            processed_paths: Set[str] = set()
            processed_paths.add(str(temp_path))
            final_audio_path = str(temp_path)

            try:
                # Cut audio to 10 seconds if it exceeds the limit
                cut_temp_path = await async_cut_audio_to_duration(str(temp_path), max_duration=10.0)
                processed_paths.add(cut_temp_path)
                final_audio_path = cut_temp_path
                
                if clearvoice_requested:
                    try:
                        final_audio_path, clearvoice_paths, _ = await apply_clearvoice_processing(
                            final_audio_path,
                            apply_enhancement,
                            apply_super_resolution,
                        )
                        processed_paths.update(clearvoice_paths)
                    except Exception as cv_error:
                        return {"status": "error", "message": f"ClearVoice processing failed: {str(cv_error)}"}
                
                processed_paths.add(final_audio_path)

                description_parts = [
                    f"Added via API with {len(audio_files)} audio files (auto-cut to 10s)"
                ]
                if clearvoice_requested:
                    cv_features = []
                    if apply_enhancement:
                        cv_features.append("MossFormer2_SE_48K enhancement")
                    if apply_super_resolution:
                        cv_features.append("MossFormer2_SR_48K super-resolution")
                    description_parts.append("ClearVoice: " + " + ".join(cv_features))
                description = " | ".join(description_parts)
                
                # Use SpeakerPresetManager to add the speaker - run in thread pool to avoid blocking
                # This operation processes audio through TTS pipeline and can take several seconds
                success = await loop.run_in_executor(
                    executor,
                    self.preset_manager.add_speaker_preset,
                    speaker_name,
                    final_audio_path,
                    description
                )
                
                if success:
                    response: Dict[str, Any] = {
                        "status": "success", 
                        "message": f"Speaker '{speaker_name}' added successfully",
                        "info": "newly_added",
                        "audio_count": len(audio_files)
                    }
                    if clearvoice_requested:
                        response["clearvoice"] = {
                            "enhancement": apply_enhancement,
                            "super_resolution": apply_super_resolution
                        }
                    return response
                else:
                    return {"status": "error", "message": f"Failed to add speaker '{speaker_name}'"}
            
            finally:
                # Clean up temporary files asynchronously
                try:
                    for cleanup_path in processed_paths:
                        await async_remove_file(cleanup_path)
                except:
                    pass
            
        except Exception as e:
            return {"status": "error", "message": f"Failed to add speaker: {str(e)}"}
    
    async def delete_speaker(self, speaker_name: str) -> Dict[str, str]:
        """Delete a speaker"""
        try:
            # Run deletion in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(executor, self.preset_manager.delete_preset, speaker_name)
            
            if success:
                return {
                    "status": "success",
                    "message": f"Speaker '{speaker_name}' deleted successfully"
                }
            else:
                return {"status": "error", "message": f"Speaker '{speaker_name}' not found"}
            
        except Exception as e:
            return {"status": "error", "message": f"Failed to delete speaker: {str(e)}"}
    
    async def list_speakers(self) -> Dict[str, Any]:
        """List all speakers with metadata"""
        try:
            # Run the synchronous operation in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            presets = await loop.run_in_executor(executor, self.preset_manager.list_presets)
            
            speaker_info = {}
            for speaker_name, preset_data in presets.items():
                # Calculate cache file size in thread pool to avoid blocking
                cache_file = preset_data.get('cache_file', '')
                total_size = 0
                if cache_file:
                    def get_file_size(filepath):
                        if os.path.exists(filepath):
                            return os.path.getsize(filepath)
                        return 0
                    total_size = await loop.run_in_executor(executor, get_file_size, cache_file)
                
                speaker_info[speaker_name] = {
                    "audio_count": 1,  # SpeakerPresetManager uses single audio file
                    "audio_files": [os.path.basename(preset_data.get('audio_path', ''))],
                    "total_size_mb": total_size / (1024 * 1024),
                    "description": preset_data.get('description', ''),
                    "created_at": preset_data.get('created_at', 0),
                    "last_used": preset_data.get('last_used', 0)
                }
            
            return {
                "status": "success",
                "speakers": speaker_info,
                "total_speakers": len(speaker_info)
            }
            
        except Exception as e:
            return {"status": "error", "message": f"Failed to list speakers: {str(e)}"}
    
    async def get_speaker_audio_paths(self, speaker_name: str) -> Optional[List[str]]:
        """Get audio paths for a specific speaker"""
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            presets = await loop.run_in_executor(executor, self.preset_manager.list_presets)
            
            if speaker_name in presets:
                audio_path = presets[speaker_name].get('audio_path', '')
                if audio_path:
                    # Check file existence in thread pool
                    exists = await loop.run_in_executor(executor, os.path.exists, audio_path)
                    if exists:
                        return [audio_path]
            return None
        except Exception as e:
            print(f"Error getting speaker audio paths: {e}")
            return None
    
    def speaker_exists(self, speaker_name: str) -> bool:
        """Check if a speaker exists - simple check, no async needed"""
        try:
            presets = self.preset_manager.list_presets()
            return speaker_name in presets
        except Exception as e:
            print(f"Error checking speaker existence: {e}")
            return False

# Global speaker API wrapper (will be initialized after TTS)
speaker_api = None

# API Models
class TranslateRequest(BaseModel):
    audio: Optional[str] = Field(default=None, description="Base64-encoded audio or download URL.")
    dest_language: str = Field(..., description="Target language for translation, e.g., 'English'.")
    audio_mime_type: Optional[str] = Field(default=None, description="MIME type of the audio, e.g., 'audio/wav'.")
    prompt: Optional[str] = Field(default=None, description="Optional custom prompt for Gemini.")
    response_format: Optional[Literal["mp3", "wav", "flac", "aac", "opus", "ogg", "webm"]] = Field(
        default=None, description="Desired audio format for the translated output."
    )
    bitrate: Optional[str] = Field(
        default=None,
        description="Optional bitrate (e.g., '192k') when using lossy codecs such as MP3.",
    )
    enhance_voice: Optional[bool] = Field(
        default=False,
        description="Apply ClearVoice MossFormer2_SE_48K enhancement before translation.",
    )
    super_resolution_voice: Optional[bool] = Field(
        default=False,
        description="Apply ClearVoice MossFormer2_SR_48K super-resolution before translation.",
    )
    merge_backing_track: Optional[bool] = Field(
        default=False,
        description="When enabled, mix the regenerated speech back onto an instrumental backing track extracted during ClearVoice enhancement.",
    )
    segments_json: Optional[str] = Field(
        default=None,
        description="Optional JSON array of pre-generated Gemini-like segments to skip inference.",
    )
    min_speech_ms: Optional[int] = Field(
        default=None,
        description="Override the minimum speech segment duration (ms) when merging short segments.",
    )
    max_merge_ms: Optional[int] = Field(
        default=None,
        description="Override the maximum silence gap (ms) allowed when merging neighboring segments.",
    )
    gemini_model: Optional[str] = Field(
        default=None,
        description="Override the Gemini model used for transcription/translation.",
    )
    gemini_api_key: Optional[str] = Field(
        default=None,
        description="Provide a Gemini API key for this request if environment key is not set.",
    )


class TranslateSegmentInput(BaseModel):
    index: int
    type: Literal["speech", "silence"] = "speech"
    start_ms: int
    end_ms: int
    translated_text: Optional[str] = ""
    source_text: Optional[str] = ""
    generate: Optional[bool] = True


class TranslateGenerateRequest(BaseModel):
    session_id: str = Field(..., description="Session identifier returned from /api/translate_segments.")
    segments: List[TranslateSegmentInput] = Field(..., description="Segments to render, including edits and selection.")
    response_format: Optional[Literal["mp3", "wav", "flac", "aac", "opus", "ogg", "webm"]] = Field(
        default=None, description="Desired audio format for output. Defaults to the original request value."
    )
    bitrate: Optional[str] = Field(default=None, description="Optional bitrate for lossy codecs.")
    merge_backing_track: Optional[bool] = Field(
        default=None,
        description="Override whether to mix generated speech with the stored backing track (requires ClearVoice enhancement).",
    )


class CloneRequest(BaseModel):
    text: str = Field(..., description="The text to generate audio for.")
    reference_audio: Optional[str] = Field(default=None, description="Reference audio URL or base64")
    reference_text: Optional[str] = Field(default=None, description="Optional transcript")
    pitch: Optional[Literal["very_low", "low", "moderate", "high", "very_high"]] = Field(default=None)
    speed: Optional[Literal["very_low", "low", "moderate", "high", "very_high"]] = Field(default=None)
    temperature: float = Field(default=0.9)
    top_k: int = Field(default=50)
    top_p: float = Field(default=0.95)
    repetition_penalty: float = Field(default=1.0)
    max_tokens: int = Field(default=4096)
    length_threshold: int = Field(default=50)
    window_size: int = Field(default=50)
    stream: bool = Field(default=False)
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field(default="mp3")
    emotion_text: Optional[str] = Field(default="", description="Emotion description text for emotion control")
    emotion_weight: float = Field(default=0.6, description="Emotion control weight (0.0 to 1.0)")
    speech_length: int = Field(default=0, description="Target audio duration in milliseconds. If 0, uses default duration calculation.")
    diffusion_steps: int = Field(default=10, description="Number of diffusion steps for mel-spectrogram generation (1-50). Higher values improve quality but increase latency.")
    max_text_tokens_per_sentence: int = Field(default=120, ge=80, le=200, description="Maximum tokens per sentence for text splitting (80-200). Higher values = longer sentences but may impact quality.")

class SpeakRequest(BaseModel):
    text: str = Field(..., description="The text to generate audio for.")
    name: Optional[str] = Field(default=None, description="The name of the voice character")
    pitch: Optional[Literal["very_low", "low", "moderate", "high", "very_high"]] = Field(default=None)
    speed: Optional[Literal["very_low", "low", "moderate", "high", "very_high"]] = Field(default=None)
    temperature: float = Field(default=0.9)
    top_k: int = Field(default=50)
    top_p: float = Field(default=0.95)
    repetition_penalty: float = Field(default=1.0)
    max_tokens: int = Field(default=4096)
    length_threshold: int = Field(default=50)
    window_size: int = Field(default=50)
    stream: bool = Field(default=False)
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field(default="mp3")
    emotion_text: Optional[str] = Field(default="", description="Emotion description text for emotion control")
    emotion_weight: float = Field(default=0.6, description="Emotion control weight (0.0 to 1.0)")
    speech_length: int = Field(default=0, description="Target audio duration in milliseconds. If 0, uses default duration calculation.")
    diffusion_steps: int = Field(default=10, description="Number of diffusion steps for mel-spectrogram generation (1-50). Higher values improve quality but increase latency.")
    max_text_tokens_per_sentence: int = Field(default=120, ge=80, le=200, description="Maximum tokens per sentence for text splitting (80-200). Higher values = longer sentences but may impact quality.")

async def warmup_model():
    """Run warmup inferences to fully preload the model"""
    try:
        print("🔥 Running model warmup (2 inferences for full load)...")
        tts = tts_manager.get_tts()
        
        # First warmup inference
        warmup_audio_1 = os.path.join(current_dir, "examples", "voice_01.wav")
        warmup_text_1 = "你好！欢迎使用IndexTTS中文语音合成系统。这是一个功能强大的AI语音生成工具，能够准确处理中文语音合成任务。床前明月光，疑是地上霜。举头望明月，低头思故乡。这首《静夜思》是李白的名作，表达了诗人对故乡的深深思念之情。系统支持多种语音风格，让您的文本转换为自然流畅的语音。今天是2025年1月11日，时间是下午3点30分。这款产品的价格是12,999元，性价比很高。我的电话号码是138-8888-8888，欢迎联系。我正在使用IndexTTS和vLLM技术进行AI语音合成。This system supports both Chinese and English perfectly. 这个系统的RTF约为0.1，比原版快3倍！GPU memory utilization设置为85%。"
        
        # Check if first warmup audio exists
        if not os.path.exists(warmup_audio_1):
            print(f"⚠️ Warmup audio file not found: {warmup_audio_1}")
            return
        
        # Create temporary output file for first warmup
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            warmup_output_1 = tmp_file.name
        
        try:
            # Run first warmup inference
            print("🔥 Warmup 1/2: Modern text with voice_01.wav...")
            await tts.infer(
                spk_audio_prompt=warmup_audio_1,
                text=warmup_text_1,
                output_path=warmup_output_1,
                emo_audio_prompt=None,
                emo_alpha=0.6,
                emo_vector=None,
                use_emo_text=True,
                emo_text="兴奋",
                use_random=False,
                interval_silence=200,
                verbose=False,
                max_text_tokens_per_sentence=120,
                speaker_preset=None,
                speech_length=0,
                diffusion_steps=10
            )
            print("✅ Warmup 1/2 completed!")
        finally:
            # Clean up first temporary warmup file
            if os.path.exists(warmup_output_1):
                os.remove(warmup_output_1)
        
        # Second warmup inference
        warmup_audio_2 = os.path.join(current_dir, "examples", "voice_02.wav")
        warmup_text_2 = "人工智能是百年来最宏大的科技建设项目。它究竟是什么样子的？美国经济已经一分为二。一边是热火朝天的 AI 经济，另一边则是萎靡不振的消费经济。你可以在经济统计数据中看到这一点。上个季度，人工智能领域的支出增长超过了消费者支出的增长。如果没有 AI，美国的经济增长将会微不足道。你可以在股市中看到这一点。在过去两年里，股市增长的约 60% 来自与 AI 相关的公司，如微软、英伟达和 Meta。如果没有 AI 热潮，股市的回报率将惨不忍睹。"
        
        # Check if second warmup audio exists
        if not os.path.exists(warmup_audio_2):
            print(f"⚠️ Warmup audio file not found: {warmup_audio_2}")
            print("✅ Model warmup completed with 1/2 inferences")
            return
        
        # Create temporary output file for second warmup
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            warmup_output_2 = tmp_file.name
        
        try:
            # Run second warmup inference
            print("🔥 Warmup 2/2: Ancient poetry with voice_02.wav...")
            await tts.infer(
                spk_audio_prompt=warmup_audio_2,
                text=warmup_text_2,
                output_path=warmup_output_2,
                emo_audio_prompt=None,
                emo_alpha=0.6,
                emo_vector=None,
                use_emo_text=True,
                emo_text="无聊",
                use_random=False,
                interval_silence=200,
                verbose=False,
                max_text_tokens_per_sentence=120,
                speaker_preset=None,
                speech_length=0,
                diffusion_steps=10
            )
            print("✅ Warmup 2/2 completed!")
        finally:
            # Clean up second temporary warmup file
            if os.path.exists(warmup_output_2):
                os.remove(warmup_output_2)
        
        print("✅ Model warmup fully completed (2/2 inferences)!")
                
    except Exception as e:
        print(f"⚠️ Warmup failed (non-critical): {e}")
        traceback.print_exc()

# FastAPI lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    # Startup
    print("🚀 Starting IndexTTS vLLM v2 FastAPI WebUI...")
    await tts_manager.initialize()
    
    # Run warmup inference
    await warmup_model()
    
    yield
    # Shutdown (if needed)
    print("🔄 Shutting down IndexTTS vLLM v2...")
    # Shutdown the thread executor
    executor.shutdown(wait=True)

# Create FastAPI app
app = FastAPI(
    title="IndexTTS vLLM v2 FastAPI WebUI",
    description="Ultra-fast TTS with vLLM backend, speaker presets, and advanced translate/edit mode with Gemini integration",
    lifespan=lifespan
)

# Web Interface
@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🚀 IndexTTS vLLM v2 - FastAPI WebUI</title>
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                overflow: hidden;
            }
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 40px;
                text-align: center;
            }
            .header h1 { font-size: 3em; margin-bottom: 10px; }
            .subtitle { font-size: 1.3em; opacity: 0.9; margin-bottom: 15px; }
            .performance-badge {
                background: rgba(255,255,255,0.2);
                padding: 10px 20px;
                border-radius: 25px;
                font-size: 0.95em;
                display: inline-block;
                margin: 5px;
            }
            .content { padding: 40px; }
            .tabs {
                display: flex;
                border-bottom: 2px solid #f0f0f0;
                margin-bottom: 30px;
            }
            .tab {
                padding: 15px 25px;
                cursor: pointer;
                border-bottom: 3px solid transparent;
                transition: all 0.3s;
            }
            .tab.active {
                border-bottom-color: #667eea;
                color: #667eea;
                font-weight: 600;
            }
            .tab-content { display: none; }
            .tab-content.active { display: block; }
            .form-section {
                background: #fff;
                padding: 30px;
                border-radius: 15px;
                border: 2px solid #f0f0f0;
                margin-bottom: 25px;
            }
            .form-group { margin-bottom: 25px; }
            label {
                display: block;
                margin-bottom: 8px;
                font-weight: 600;
                color: #333;
            }
            textarea, input[type="file"], input[type="text"], select {
                width: 100%;
                padding: 15px;
                border: 2px solid #e1e5e9;
                border-radius: 10px;
                font-size: 16px;
                transition: border-color 0.3s;
            }
            textarea:focus, input:focus, select:focus {
                outline: none;
                border-color: #667eea;
            }
            textarea { resize: vertical; min-height: 120px; }
            .btn {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                padding: 15px 30px;
                font-size: 16px;
                border-radius: 10px;
                cursor: pointer;
                transition: transform 0.2s;
                margin: 5px;
            }
            .btn:hover { transform: translateY(-2px); }
            .btn:disabled {
                opacity: 0.6;
                cursor: not-allowed;
                transform: none;
            }
            .btn-danger {
                background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
            }
            .status {
                margin-top: 20px;
                padding: 15px;
                border-radius: 10px;
                display: none;
            }
            .status.success {
                background: #d4edda;
                border: 1px solid #c3e6cb;
                color: #155724;
            }
            .status.error {
                background: #f8d7da;
                border: 1px solid #f5c6cb;
                color: #721c24;
            }
            .segment-panel {
                margin-top: 25px;
                border: 2px dashed #d7dcff;
                padding: 20px;
                border-radius: 15px;
                background: #f7f8ff;
            }
            .segment-controls {
                display: flex;
                flex-wrap: wrap;
                gap: 12px;
                align-items: center;
                justify-content: space-between;
                margin-bottom: 15px;
            }
            .segment-list {
                display: flex;
                flex-direction: column;
                gap: 16px;
                max-height: 520px;
                overflow-y: auto;
                padding-right: 8px;
            }
            .segment-card {
                border-radius: 12px;
                border: 1px solid #dce1fa;
                padding: 16px;
                background: white;
                box-shadow: 0 4px 16px rgba(102,126,234,0.08);
            }
            .segment-card.speech { border-left: 4px solid #667eea; }
            .segment-card.silence { border-left: 4px solid #9aa0b6; }
            .segment-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 12px;
            }
            .segment-meta {
                font-size: 0.9em;
                color: #666;
            }
            .segment-body {
                margin-top: 12px;
                display: grid;
                gap: 14px;
            }
            .segment-body textarea {
                min-height: 80px;
                font-size: 0.95em;
            }
            .segment-timing {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
                gap: 12px;
                align-items: end;
            }
            .segment-timing label {
                font-weight: 500;
                color: #444;
            }
            .segment-timing input {
                width: 100%;
            }
            .segment-duration-label {
                font-size: 0.85em;
                color: #888;
            }
            .segment-checkbox {
                display: flex;
                align-items: center;
                gap: 8px;
                font-weight: 600;
                color: #333;
            }
            .segment-audio {
                width: 100%;
                margin-top: 8px;
            }
            .segment-empty {
                padding: 20px;
                text-align: center;
                background: rgba(102, 126, 234, 0.08);
                border-radius: 12px;
                color: #445;
                font-weight: 500;
            }
            .speaker-list {
                background: #f8f9fa;
                padding: 20px;
                border-radius: 10px;
                margin-top: 20px;
            }
            .speaker-item {
                background: white;
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 10px;
                border-left: 4px solid #667eea;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .speaker-info h4 { margin: 0; color: #333; }
            .speaker-info small { color: #666; }
            .loading {
                display: inline-block;
                width: 20px;
                height: 20px;
                border: 3px solid #f3f3f3;
                border-top: 3px solid #667eea;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin-right: 10px;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 IndexTTS vLLM v2</h1>
                <p class="subtitle">Ultra-Fast TTS with vLLM Backend / 超快速中英文语音合成</p>
                <div>
                    <span class="performance-badge">⚡ vLLM v2 Backend</span>
                    <span class="performance-badge">🇨🇳 Chinese Support</span>
                    <span class="performance-badge">🎭 Speaker Presets</span>
                    <span class="performance-badge">🎵 MP3 Output</span>
                    <span class="performance-badge">🔌 API Integration</span>
                    <span class="performance-badge">😊 Emotion Text Control</span>
                    <span class="performance-badge">🌊 Streaming Mode</span>
                    <span class="performance-badge">🌐 Translate/Edit Mode</span>
                    <span class="performance-badge">✂️ Segment Editing</span>
                </div>
            </div>
            <div class="content">
                <div class="tabs">
                    <div class="tab active" onclick="switchTab('synthesis')">🎵 Speech Synthesis</div>
                    <div class="tab" onclick="switchTab('translate')">🌐 Speech Translate/Edit</div>
                    <div class="tab" onclick="switchTab('speakers')">🎭 Speaker Management</div>
                    <div class="tab" onclick="switchTab('api')">📚 API Documentation</div>
                </div>

                <!-- Speech Synthesis Tab -->
                <div id="synthesis" class="tab-content active">
                    <div class="form-section">
                        <h3>🎵 Generate Speech</h3>
                        
                        <!-- Chinese Demo Section -->
                        <div style="background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%); padding: 20px; border-radius: 15px; margin-bottom: 20px;">
                            <h4 style="color: #333; margin-bottom: 15px;">🇨🇳 中文语音合成演示</h4>
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                                <button class="btn" onclick="setChineseDemo('现代文本')" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                                    现代文本演示
                                </button>
                                <button class="btn" onclick="setChineseDemo('古诗词')" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                                    古诗词演示
                                </button>
                                <button class="btn" onclick="setChineseDemo('数字日期')" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                                    数字日期处理
                                </button>
                                <button class="btn" onclick="setChineseDemo('中英混合')" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                                    中英混合文本
                                </button>
                                <button class="btn" onclick="setEmotionDemo('开心')" style="background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);">
                                    😊 开心情感
                                </button>
                                <button class="btn" onclick="setEmotionDemo('悲伤')" style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);">
                                    😢 悲伤情感
                                </button>
                                <button class="btn" onclick="setEmotionDemo('愤怒')" style="background: linear-gradient(135deg, #ff6b6b 0%, #ffa500 100%);">
                                    😠 愤怒情感
                                </button>
                                <button class="btn" onclick="setEmotionDemo('平静')" style="background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);">
                                    😌 平静情感
                                </button>
                            </div>
                            <p style="margin-top: 10px; font-size: 0.9em; color: #666;">
                                ✨ IndexTTS内置强大的中文文本规范化，支持数字转换、标点处理、拼音声调等
                            </p>
                        </div>
                        
                        <form id="ttsForm" enctype="multipart/form-data">
                            <div class="form-group">
                                <label for="text">Text to Synthesize / 输入要合成的文本:</label>
                                <textarea id="text" name="text" placeholder="Enter the text you want to convert to speech...&#10;输入您想要转换为语音的文本...&#10;&#10;中文示例：&#10;你好世界！今天是2025年1月11日，天气很好。&#10;这个AI语音合成系统支持中英混合文本。" required></textarea>
                            </div>
                            
                            <div class="form-group">
                                <label for="voice_files">Upload Voice Files (Optional):</label>
                                <input type="file" id="voice_files" name="voice_files" accept=".wav,.mp3,.m4a,.flac" multiple>
                            </div>
                            
                            <div class="form-group">
                                <label for="speaker">Use Speaker Preset:</label>
                                <select id="speaker" name="speaker">
                                    <option value="">Select a speaker...</option>
                                </select>
                            </div>
                            
                            <div class="form-group">
                                <label style="display: flex; align-items: center; cursor: pointer;">
                                    <input type="checkbox" id="streamingMode" name="streamingMode" style="width: auto; margin-right: 10px;">
                                    <span>⚡ Enable Streaming Mode (Play audio as it's generated)</span>
                                </label>
                                <small style="color: #666; margin-top: 5px; display: block;">
                                    Streaming mode starts playback immediately when the first chunk is ready
                                </small>
                            </div>
                            
                            <div class="form-group" id="streamingSettings" style="display: none; background: #f8f9fa; padding: 15px; border-radius: 10px; margin-top: 10px;">
                                <label for="firstChunkSize">⚡ First Chunk Size: <span id="firstChunkSizeValue">40</span> tokens</label>
                                <input type="range" id="firstChunkSize" name="firstChunkSize" 
                                       min="20" max="80" step="10" value="40"
                                       style="width: 100%; margin: 10px 0;"
                                       oninput="document.getElementById('firstChunkSizeValue').textContent = this.value">
                                <div style="display: flex; justify-content: space-between; font-size: 0.85em; color: #666;">
                                    <span>⚡ Faster (20)</span>
                                    <span>Balanced (40)</span>
                                    <span>Quality (80)</span>
                                </div>
                                <small style="color: #666; margin-top: 10px; display: block;">
                                    💡 Smaller = faster first response but more chunks. Recommended: 30-50 tokens.
                                </small>
                            </div>
                            
                            <!-- Emotion Control Section -->
                            <div style="background: linear-gradient(135deg, #ffd89b 0%, #19547b 100%); padding: 20px; border-radius: 15px; margin: 20px 0;">
                                <h4 style="color: white; margin-bottom: 15px;">😊 Emotion Text Control / 情感文本控制</h4>
                                <div class="form-group">
                                    <label for="emotionText" style="color: white;">Emotion Description / 情感描述:</label>
                                    <input type="text" id="emotionText" name="emotionText" 
                                           placeholder="e.g., happy and excited, sad and melancholic, angry and frustrated... 例如：开心兴奋，悲伤忧郁，愤怒沮丧..." 
                                           style="margin-bottom: 15px;">
                                </div>
                                <div class="form-group">
                                    <label for="emotionWeight" style="color: white;">Emotion Strength / 情感强度: <span id="emotionWeightValue">0.6</span></label>
                                    <input type="range" id="emotionWeight" name="emotionWeight" 
                                           min="0.0" max="1.0" step="0.1" value="0.6"
                                           style="width: 100%; margin-bottom: 10px;"
                                           oninput="document.getElementById('emotionWeightValue').textContent = this.value">
                                </div>
                                <p style="color: #fff; font-size: 0.9em; margin: 0;">
                                    💡 输入情感描述文本可以让AI更精准地控制语音的情感表达。留空则使用默认情感。
                                </p>
                            </div>
                            
                            <!-- Duration Control Section -->
                            <div style="background: linear-gradient(135deg, #36d1dc 0%, #5b86e5 100%); padding: 20px; border-radius: 15px; margin: 20px 0;">
                                <h4 style="color: white; margin-bottom: 15px;">⏱️ Duration Control / 时长控制</h4>
                                <div class="form-group">
                                    <label for="speechLength" style="color: white;">Target Duration / 目标时长 (milliseconds):</label>
                                    <input type="number" id="speechLength" name="speechLength" 
                                           value="0" min="0" max="6000000" step="100"
                                           placeholder="0 = auto duration"
                                           style="margin-bottom: 15px;">
                                    <button type="button" class="btn" onclick="estimateDuration()" style="background: rgba(255,255,255,0.3); margin-top: 5px;">
                                        📊 Estimate Duration from Text
                                    </button>
                                </div>
                                <div id="durationEstimate" style="color: white; font-weight: bold; margin-top: 10px; padding: 10px; background: rgba(255,255,255,0.2); border-radius: 8px; display: none;"></div>
                                <p style="color: #fff; font-size: 0.9em; margin: 10px 0 0 0;">
                                    💡 设置为 0 表示自动时长。指定毫秒数可用于视频配音/时间控制。Set to 0 for auto duration. Specify milliseconds for video dubbing/timing control.
                                </p>
                            </div>
                            
                            <!-- Diffusion Steps Control Section -->
                            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 15px; margin: 20px 0;">
                                <h4 style="color: white; margin-bottom: 15px;">🎨 Quality Control / 质量控制</h4>
                                <div class="form-group">
                                    <label for="diffusionSteps" style="color: white;">Diffusion Steps / 扩散步数: <span id="diffusionStepsValue">10</span></label>
                                    <input type="range" id="diffusionSteps" name="diffusionSteps" 
                                           min="1" max="50" step="1" value="10"
                                           style="width: 100%; margin-bottom: 10px;"
                                           oninput="document.getElementById('diffusionStepsValue').textContent = this.value">
                                </div>
                                <p style="color: #fff; font-size: 0.9em; margin: 0;">
                                    💡 更高的步数可以提高音质但会增加延迟。建议值: 快速=5, 默认=10, 高质量=20-30。Higher steps improve quality but increase latency. Recommended: Fast=5, Default=10, High-quality=20-30.
                                </p>
                            </div>
                            
                            <!-- Text Tokens Per Sentence Control Section -->
                            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 20px; border-radius: 15px; margin: 20px 0;">
                                <h4 style="color: white; margin-bottom: 15px;">✂️ Text Splitting / 文本分句</h4>
                                <div class="form-group">
                                    <label for="maxTextTokens" style="color: white;">Max Tokens Per Sentence / 每句最大Token数: <span id="maxTextTokensValue">120</span></label>
                                    <input type="range" id="maxTextTokens" name="maxTextTokens" 
                                           min="80" max="200" step="10" value="120"
                                           style="width: 100%; margin-bottom: 10px;"
                                           oninput="document.getElementById('maxTextTokensValue').textContent = this.value">
                                </div>
                                <div style="display: flex; justify-content: space-between; font-size: 0.85em; color: #fff;">
                                    <span>Short (80)</span>
                                    <span>Balanced (120)</span>
                                    <span>Long (200)</span>
                                </div>
                                <p style="color: #fff; font-size: 0.9em; margin: 10px 0 0 0;">
                                    💡 控制每个句子的最大长度。较短=更多句子但处理更快，较长=更少句子但减少断句。Controls max sentence length. Shorter = more sentences but faster processing, Longer = fewer sentences but fewer breaks.
                                </p>
                            </div>
                            
                            <button type="submit" class="btn" id="generateBtn">
                                🎵 Generate Speech
                            </button>
                            <button type="button" class="btn btn-danger" onclick="clearOutputs()">
                                🗑️ Clear All Outputs
                            </button>
                        </form>
                        
                        <div id="status" class="status"></div>
                        <div id="audioResult"></div>
                    </div>
                </div>

                <!-- Speech Translation Tab -->
                <div id="translate" class="tab-content">
                    <div class="form-section">
                        <h3>🌐 Speech Translate/Edit</h3>
                        <p style="color: #666; margin-bottom: 20px;">
                            Upload source speech audio, pick a destination language, and optionally enter advanced mode to audition Gemini segments, tweak timings/text, and regenerate only the pieces you need.
                        </p>
                        <form id="translateForm" enctype="multipart/form-data">
                            <div class="form-group">
                                <label for="translateAudioFile">Source Audio:</label>
                                <input type="file" id="translateAudioFile" name="audio_file" accept=".wav,.mp3,.m4a,.flac,.aac,.ogg,.opus" required>
                                <small style="color: #666; display: block; margin-top: 6px;">
                                    Supported formats: WAV, MP3, M4A, FLAC, AAC, OGG, OPUS. Audio is processed locally then sent to Gemini for transcription.
                                </small>
                            </div>
                            <div class="form-group">
                                <label for="translateDestLanguage">Destination Language:</label>
                                <select id="translateDestLanguage" name="dest_language" required>
                                    <option value="">Select a language...</option>
                                    <option value="English">English</option>
                                    <option value="Chinese">Chinese</option>
                                </select>
                            </div>
                            <div class="form-group">
                                <label for="translateGeminiModel">Gemini Model:</label>
                                <select id="translateGeminiModel" name="gemini_model">
                                    <option value="gemini-2.5-pro" selected>Gemini 2.5 Pro (highest accuracy)</option>
                                    <option value="gemini-flash-latest">Gemini Flash Latest (fast)</option>
                                </select>
                                <small style="color: #666; display: block; margin-top: 6px;">
                                    Choose the Gemini model used for transcription/translation. Flash is faster; Pro is more accurate.
                                </small>
                            </div>
                            <div class="form-group">
                                <label for="translateGeminiApiKey">Gemini API Key (optional):</label>
                                <input type="password" id="translateGeminiApiKey" name="gemini_api_key" placeholder="Use this key instead of the system default..." autocomplete="off">
                                <small style="color: #666; display: block; margin-top: 6px;">
                                    Provide a key if the server environment does not have one configured or you need to override it for this request.
                                </small>
                            </div>
                            <div class="form-group">
                                <label for="translateOutputFormat">Output Format:</label>
                                <select id="translateOutputFormat" name="response_format">
                                    <option value="mp3" selected>MP3 (default)</option>
                                    <option value="wav">WAV</option>
                                    <option value="flac">FLAC</option>
                                    <option value="aac">AAC</option>
                                    <option value="opus">OPUS</option>
                                </select>
                            </div>
                            <div class="form-group">
                                <label>ClearVoice Preprocessing (Optional):</label>
                                <div style="display: flex; gap: 16px; flex-wrap: wrap;">
                                    <label style="display: flex; align-items: center; gap: 8px;">
                                        <input type="checkbox" id="translateEnhancement">
                                        <span>Enhance with MossFormer2_SE_48K</span>
                                    </label>
                                    <label style="display: flex; align-items: center; gap: 8px;">
                                        <input type="checkbox" id="translateSuperResolution">
                                        <span>Super Resolution with MossFormer2_SR_48K</span>
                                    </label>
                                    <label style="display: flex; align-items: center; gap: 8px;">
                                        <input type="checkbox" id="translateMergeBack" disabled>
                                        <span>Mix translated speech back into instrumental (requires enhancement)</span>
                                    </label>
                                </div>
                                <small style="color: #666; margin-top: 5px; display: block;">
                                    Defaults to off. Requires the ClearVoice package to be installed.
                                </small>
                            </div>
                            <div class="form-group">
                                <label>Segment Merge Settings (Optional):</label>
                                <div style="display: flex; gap: 16px; flex-wrap: wrap;">
                                    <div style="flex: 1 1 220px;">
                                        <label for="translateMinSpeech" style="font-weight: 500;">Minimum speech duration (ms):</label>
                                        <input type="number" id="translateMinSpeech" min="500" step="100" placeholder="Default 3000">
                                    </div>
                                    <div style="flex: 1 1 220px;">
                                        <label for="translateMaxMerge" style="font-weight: 500;">Max merge silence gap (ms):</label>
                                        <input type="number" id="translateMaxMerge" min="50" step="50" placeholder="Default 300">
                                    </div>
                                </div>
                                <small style="color: #666; margin-top: 5px; display: block;">
                                    These values control Gemini segment stitching. Lower min duration keeps shorter phrases; higher max merge gap allows merging across longer silences.
                                </small>
                            </div>
                            <div class="form-group">
                                <label style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
                                    <input type="checkbox" id="translateManualSegmentsToggle">
                                    <span>Provide manual Gemini segments JSON (skip Gemini inference)</span>
                                </label>
                                <div id="translateManualSegmentsPanel" style="display: none;">
                                    <textarea id="translateManualSegments" rows="6" placeholder='Paste the raw JSON array returned from Gemini (or another AI). Each entry should include "start", "end", "source_text", and "translated_text".'></textarea>
                                    <small style="color: #666; display: block; margin-top: 6px;">
                                        Use the prompt templates below with any LLM to generate the JSON response, then paste it here to bypass the Gemini API step.
                                    </small>
                                </div>
                            </div>
                            <div class="form-group" id="translatePromptTemplates" style="display: none; background: rgba(102,126,234,0.08); padding: 16px; border-radius: 12px;">
                                <label>Gemini Prompt Templates</label>
                                <div style="display: flex; flex-wrap: wrap; gap: 16px;">
                                    <div style="flex: 1 1 250px;">
                                        <label style="font-weight: 500;">Translation mode prompt:</label>
                                        <textarea id="translatePromptTranslation" readonly style="min-height: 140px;"></textarea>
                                    </div>
                                    <div style="flex: 1 1 250px;">
                                        <label style="font-weight: 500;">Transcription-only prompt:</label>
                                        <textarea id="translatePromptTranscription" readonly style="min-height: 140px;"></textarea>
                                    </div>
                                </div>
                                <small style="color: #666; display: block; margin-top: 6px;">
                                    Copy these prompts when generating manual segments with your preferred AI model. Replace <code>{'{dest_language}'}</code> as needed.
                                </small>
                            </div>
                            <div class="form-group" style="margin-top: 20px;">
                                <label style="display: flex; align-items: center; gap: 10px;">
                                    <input type="checkbox" id="translateAdvancedMode">
                                    <span>Enable advanced translate/edit workflow</span>
                                </label>
                                <small style="color: #666; margin-top: 6px; display: block;">
                                    When enabled we will analyze segments first so you can listen, edit, and choose which parts to regenerate before final synthesis.
                                </small>
                            </div>
                            <div id="translateAdvancedSettings" style="display: none; background: rgba(102,126,234,0.08); padding: 16px; border-radius: 12px; margin-bottom: 20px;">
                                <div class="form-group" style="margin-bottom: 16px;">
                                    <label style="display: flex; align-items: center; gap: 10px; margin-bottom: 6px;">
                                        <input type="checkbox" id="translateDebugTranslate" checked>
                                        <span>Ask Gemini to translate while transcribing</span>
                                    </label>
                                    <small style="color: #666; display: block;">
                                        Uncheck to only transcribe with timestamps; you can enter translation text manually per segment.
                                    </small>
                                </div>
                                <div class="form-group" style="margin-bottom: 0;">
                                    <label for="translateCustomPrompt">Custom Gemini Prompt (optional):</label>
                                    <textarea id="translateCustomPrompt" rows="3" placeholder="Override the default Gemini prompt for segment analysis..."></textarea>
                                    <small style="color: #666; margin-top: 6px; display: block;">
                                        Leave blank to use the optimized defaults for translate/transcribe modes.
                                    </small>
                                </div>
                            </div>
                            <button type="submit" class="btn" id="translateBtn">🌐 Translate Speech</button>
                        </form>
                        <div id="translateStatus" class="status"></div>
                        <div id="translateAdvancedPanel" class="segment-panel" style="display: none;">
                            <div id="translateBackingPreview" style="display: none; margin-bottom: 16px;"></div>
                            <div class="segment-controls">
                                <label class="segment-checkbox">
                                    <input type="checkbox" id="translateSegmentsSelectAll" checked>
                                    <span>Select all segments for generation</span>
                                </label>
                                <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                                    <button type="button" class="btn" id="translateGenerateBtn">🎧 Generate Selected Segments</button>
                                </div>
                            </div>
                            <div id="translateSegmentsStatus" class="status"></div>
                            <div id="translateSegmentsList" class="segment-list"></div>
                        </div>
                        <div id="translateResult"></div>
                    </div>
                </div>

                <!-- Speaker Management Tab -->
                <div id="speakers" class="tab-content">
                    <div class="form-section">
                        <h3>➕ Add New Speaker</h3>
                        <form id="addSpeakerForm" enctype="multipart/form-data">
                            <div class="form-group">
                                <label for="speakerName">Speaker Name:</label>
                                <input type="text" id="speakerName" name="speakerName" placeholder="Enter speaker name..." required>
                            </div>
                            
                            <div class="form-group">
                                <label for="speakerAudioFiles">Audio Files:</label>
                                <input type="file" id="speakerAudioFiles" name="speakerAudioFiles" accept=".wav,.mp3,.m4a,.flac" multiple required>
                                <small style="color: #666; margin-top: 5px; display: block;">
                                    Upload multiple audio files for better voice quality<br>
                                    ✂️ Audio will be smartly cut at silence intervals (3-15s) for optimal performance
                                </small>
                            </div>
                            
                            <div class="form-group">
                                <label>Pure Voice Extraction (ClearVoice, optional):</label>
                                <div style="display: flex; gap: 16px; flex-wrap: wrap;">
                                    <label style="display: flex; align-items: center; gap: 8px;">
                                        <input type="checkbox" id="applyEnhancement">
                                        <span>Enhance with MossFormer2_SE_48K</span>
                                    </label>
                                    <label style="display: flex; align-items: center; gap: 8px;">
                                        <input type="checkbox" id="applySuperResolution">
                                        <span>Super Resolution with MossFormer2_SR_48K</span>
                                    </label>
                                </div>
                                <small style="color: #666; margin-top: 5px; display: block;">
                                    Enable ClearVoice MossFormer2 models to clean and upscale the reference audio. When both are selected, enhancement runs before super-resolution.
                                </small>
                            </div>
                            
                            <button type="submit" class="btn">➕ Add Speaker</button>
                        </form>
                        
                        <div id="speakerStatus" class="status"></div>
                    </div>

                    <div class="form-section">
                        <h3>🎭 Manage Speakers</h3>
                        <button class="btn" onclick="loadSpeakerList()">🔄 Refresh Speaker List</button>
                        <div id="speakerList" class="speaker-list"></div>
                    </div>
                </div>

                <!-- API Documentation Tab -->
                <div id="api" class="tab-content">
                    <div class="form-section">
                        <h3>📚 API Endpoints</h3>
                        
                        <h4 style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 10px; border-radius: 8px;">🔷 API Endpoints (Recommended for External Use)</h4>
                        
                        <h5>🔍 Server Information</h5>
                        <ul style="margin-left: 20px; line-height: 1.6;">
                            <li><strong>GET /server_info</strong> - Get server information, model details, and available speakers
                                <ul style="margin-left: 20px; margin-top: 5px; color: #666;">
                                    <li>Returns: Server version, model name, speaker list, capabilities</li>
                                </ul>
                            </li>
                        </ul>
                        
                        <h5>👥 Speaker Management</h5>
                        <ul style="margin-left: 20px; line-height: 1.6;">
                            <li><strong>GET /audio_roles</strong> - List all available speaker presets
                                <ul style="margin-left: 20px; margin-top: 5px; color: #666;">
                                    <li>Returns: <code>{"success": true, "roles": ["speaker1", "speaker2", ...]}</code></li>
                                </ul>
                            </li>
                            <li><strong>POST /add_speaker</strong> - Register a new speaker with reference audio
                                <ul style="margin-left: 20px; margin-top: 5px; color: #666;">
                                    <li>Form data: <code>name</code> (string), <code>audio_file</code> (file upload)</li>
                                    <li>Optional form data: <code>enhance_voice</code> (bool), <code>super_resolution_voice</code> (bool) — toggles ClearVoice MossFormer2_SE_48K and MossFormer2_SR_48K (both default to <code>false</code>); <code>merge_backing_track</code> (bool) mixes regenerated speech onto the extracted instrumental (requires enhancement); <code>min_speech_ms</code>/<code>max_merge_ms</code> override the segment-merging heuristics; <code>segments_json</code> lets you supply Gemini-style JSON to skip inference.</li>
                                    <li>Audio will be automatically trimmed to 3-15 seconds at silence points; when both toggles are enabled, enhancement runs before super-resolution</li>
                                </ul>
                            </li>
                            <li><strong>POST /delete_speaker</strong> - Remove an existing speaker preset
                                <ul style="margin-left: 20px; margin-top: 5px; color: #666;">
                                    <li>Form data: <code>name</code> (string)</li>
                                </ul>
                            </li>
                        </ul>

                        <h5>🌐 Speech Translation</h5>
                        <ul style="margin-left: 20px; line-height: 1.6;">
                            <li><strong>POST /api/translate_audio</strong> - Translate speech audio and regenerate voice in the target language while preserving timing.
                                <ul style="margin-left: 20px; margin-top: 5px; color: #666;">
                                    <li>Multipart form fields: <code>audio_file</code> (file), <code>dest_language</code> (string); optional <code>response_format</code> (mp3/wav/flac/aac/opus/ogg/webm), <code>prompt</code> (custom Gemini instructions), <code>enhance_voice</code> (bool), <code>super_resolution_voice</code> (bool) to run ClearVoice preprocessing, <code>merge_backing_track</code> (bool) to blend the result with the extracted instrumental backing, plus <code>min_speech_ms</code>/<code>max_merge_ms</code> to override segment-merging heuristics and <code>segments_json</code> to supply pre-generated Gemini-like segments.</li>
                                    <li>JSON alternative: <code>{"audio": "&lt;base64&gt;", "dest_language": "English", "audio_mime_type": "audio/wav", "response_format": "mp3", "enhance_voice": true, "super_resolution_voice": false, "merge_backing_track": true, "segments_json": "[...]","min_speech_ms": 3000, "max_merge_ms": 300}</code></li>
                                    <li>Response: Audio stream. Inspect headers like <code>X-Translation-Model</code> and <code>X-Translation-Segments</code> for run metadata.</li>
                                </ul>
                            </li>
                        </ul>
                        
                        <h5>🎙️ Speech Generation - Non-Streaming (Standard Mode)</h5>
                        <ul style="margin-left: 20px; line-height: 1.6;">
                            <li><strong>POST /speak</strong> - Generate speech using a registered speaker preset
                                <ul style="margin-left: 20px; margin-top: 5px; color: #666;">
                                    <li>Required: <code>text</code> (string), <code>name</code> (speaker name)</li>
                                    <li>Optional: <code>response_format</code> (mp3/opus/aac/flac/wav/pcm, default: mp3)</li>
                                    <li>Optional: <code>emotion_text</code> (emotion description), <code>emotion_weight</code> (0.0-1.0)</li>
                                    <li>Optional: <code>diffusion_steps</code> (int, default: 10)</li>
                                    <li>Optional: <code>max_text_tokens_per_sentence</code> (int, 80-200, default: 120) - Controls text splitting</li>
                                    <li>Returns: Audio file in specified format</li>
                                </ul>
                            </li>
                            <li><strong>POST /clone_voice</strong> - Clone voice using uploaded reference audio (zero-shot)
                                <ul style="margin-left: 20px; margin-top: 5px; color: #666;">
                                    <li>Required: <code>text</code> (string), <code>reference_audio_file</code> (file upload)</li>
                                    <li>Optional: Same as /speak (response_format, emotion_text, emotion_weight, diffusion_steps, max_text_tokens_per_sentence)</li>
                                    <li>Returns: Audio file cloned from reference voice</li>
                                </ul>
                            </li>
                        </ul>
                        
                        <h5>⚡ Speech Generation - Streaming (Low Latency Mode)</h5>
                        <ul style="margin-left: 20px; line-height: 1.6;">
                            <li><strong>POST /speak_stream</strong> - Generate speech with streaming chunks
                                <ul style="margin-left: 20px; margin-top: 5px; color: #666;">
                                    <li>Same parameters as <code>/speak</code></li>
                                    <li>Streams audio chunks as they're generated</li>
                                    <li>Response format: <code>CHUNK:{idx}:{size}:{status}\n{audio_bytes}</code></li>
                                    <li>Status: CONTINUE (more chunks coming) or LAST (final chunk)</li>
                                </ul>
                            </li>
                            <li><strong>POST /clone_voice_stream</strong> - Clone voice with streaming
                                <ul style="margin-left: 20px; margin-top: 5px; color: #666;">
                                    <li>Same parameters as <code>/clone_voice</code></li>
                                    <li>Same streaming format as /speak_stream</li>
                                </ul>
                            </li>
                        </ul>
                        
                        <hr style="margin: 20px 0; border: none; border-top: 2px solid #f0f0f0;">
                        
                        <h4 style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%); color: white; padding: 10px; border-radius: 8px;">🔧 Utility API (WebUI Internal)</h4>
                        
                        <h5>🛠️ Helper Endpoints</h5>
                        <ul style="margin-left: 20px; line-height: 1.6;">
                            <li><strong>POST /api/estimate_duration</strong> - Estimate speech duration from text
                                <ul style="margin-left: 20px; margin-top: 5px; color: #666;">
                                    <li>JSON body: <code>{"text": "...", "language": "auto"}</code></li>
                                    <li>Returns: Estimated duration in seconds and milliseconds</li>
                                </ul>
                            </li>
                            <li><strong>POST /api/clear_outputs</strong> - Clear all generated output files
                                <ul style="margin-left: 20px; margin-top: 5px; color: #666;">
                                    <li>No parameters required</li>
                                    <li>Returns: Number of files deleted and disk space freed</li>
                                </ul>
                            </li>
                        </ul>
                        
                        <hr style="margin: 20px 0; border: none; border-top: 2px solid #f0f0f0;">
                        
                        <h4>🆕 Emotion Text Control Feature</h4>
                        <ul style="margin-left: 20px; line-height: 1.6;">
                            <li><strong>emotion_text</strong> (optional): Natural language emotion description
                                <ul style="margin-left: 20px; margin-top: 5px; color: #666;">
                                    <li>Examples: "happy and excited", "sad and melancholic", "calm and peaceful", "angry and frustrated"</li>
                                </ul>
                            </li>
                            <li><strong>emotion_weight</strong> (optional): Control emotion intensity (0.0 = no emotion, 1.0 = maximum)
                                <ul style="margin-left: 20px; margin-top: 5px; color: #666;">
                                    <li>Default: 0.6 (moderate intensity)</li>
                                    <li>Recommended range: 0.3-0.9</li>
                                </ul>
                            </li>
                            <li><strong>Example usage:</strong>
                                <pre style="background: #f5f5f5; padding: 10px; border-radius: 5px; margin-top: 10px; overflow-x: auto;"><code>{
  "text": "Hello, how are you today?",
  "name": "speaker1",
  "emotion_text": "cheerful and friendly",
  "emotion_weight": 0.7,
  "response_format": "mp3"
}</code></pre>
                            </li>
                        </ul>
                        
                        <hr style="margin: 20px 0; border: none; border-top: 2px solid #f0f0f0;">
                        
                        <h4>✂️ Text Splitting Control</h4>
                        <ul style="margin-left: 20px; line-height: 1.6;">
                            <li><strong>max_text_tokens_per_sentence</strong> (optional): Maximum tokens per sentence for text splitting (80-200)
                                <ul style="margin-left: 20px; margin-top: 5px; color: #666;">
                                    <li>Default: 120 tokens</li>
                                    <li>Range: 80-200 tokens</li>
                                    <li>Lower values (80-100): More sentences, faster processing, may sound choppy</li>
                                    <li>Balanced (110-130): Good balance of quality and processing speed</li>
                                    <li>Higher values (140-200): Fewer sentences, slower processing, may impact quality for very long sentences</li>
                                </ul>
                            </li>
                            <li><strong>Example usage:</strong>
                                <pre style="background: #f5f5f5; padding: 10px; border-radius: 5px; margin-top: 10px; overflow-x: auto;"><code>{
  "text": "This is a long text that will be split into manageable chunks for processing.",
  "name": "speaker1",
  "max_text_tokens_per_sentence": 120,
  "response_format": "mp3"
}</code></pre>
                            </li>
                        </ul>
                        
                        <hr style="margin: 20px 0; border: none; border-top: 2px solid #f0f0f0;">
                        
                        <h4>📊 Complete Endpoint Summary</h4>
                        <table style="width: 100%; border-collapse: collapse; margin-top: 15px;">
                            <thead>
                                <tr style="background: #f5f5f5;">
                                    <th style="padding: 10px; text-align: left; border: 1px solid #ddd;">Method</th>
                                    <th style="padding: 10px; text-align: left; border: 1px solid #ddd;">Endpoint</th>
                                    <th style="padding: 10px; text-align: left; border: 1px solid #ddd;">Purpose</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td style="padding: 8px; border: 1px solid #ddd;">GET</td>
                                    <td style="padding: 8px; border: 1px solid #ddd;"><code>/</code></td>
                                    <td style="padding: 8px; border: 1px solid #ddd;">This web interface</td>
                                </tr>
                                <tr>
                                    <td style="padding: 8px; border: 1px solid #ddd;">GET</td>
                                    <td style="padding: 8px; border: 1px solid #ddd;"><code>/server_info</code></td>
                                    <td style="padding: 8px; border: 1px solid #ddd;">Server & model information</td>
                                </tr>
                                <tr>
                                    <td style="padding: 8px; border: 1px solid #ddd;">GET</td>
                                    <td style="padding: 8px; border: 1px solid #ddd;"><code>/audio_roles</code></td>
                                    <td style="padding: 8px; border: 1px solid #ddd;">List speakers</td>
                                </tr>
                                <tr>
                                    <td style="padding: 8px; border: 1px solid #ddd;">POST</td>
                                    <td style="padding: 8px; border: 1px solid #ddd;"><code>/add_speaker</code></td>
                                    <td style="padding: 8px; border: 1px solid #ddd;">Add speaker preset</td>
                                </tr>
                                <tr>
                                    <td style="padding: 8px; border: 1px solid #ddd;">POST</td>
                                    <td style="padding: 8px; border: 1px solid #ddd;"><code>/delete_speaker</code></td>
                                    <td style="padding: 8px; border: 1px solid #ddd;">Remove speaker preset</td>
                                </tr>
                                <tr style="background: #f9f9ff;">
                                    <td style="padding: 8px; border: 1px solid #ddd;">POST</td>
                                    <td style="padding: 8px; border: 1px solid #ddd;"><code>/speak</code></td>
                                    <td style="padding: 8px; border: 1px solid #ddd;"><strong>Generate speech (standard)</strong></td>
                                </tr>
                                <tr style="background: #f9f9ff;">
                                    <td style="padding: 8px; border: 1px solid #ddd;">POST</td>
                                    <td style="padding: 8px; border: 1px solid #ddd;"><code>/clone_voice</code></td>
                                    <td style="padding: 8px; border: 1px solid #ddd;"><strong>Clone voice (standard)</strong></td>
                                </tr>
                                <tr style="background: #fff9f0;">
                                    <td style="padding: 8px; border: 1px solid #ddd;">POST</td>
                                    <td style="padding: 8px; border: 1px solid #ddd;"><code>/speak_stream</code></td>
                                    <td style="padding: 8px; border: 1px solid #ddd;"><strong>⚡ Generate speech (streaming)</strong></td>
                                </tr>
                                <tr style="background: #fff9f0;">
                                    <td style="padding: 8px; border: 1px solid #ddd;">POST</td>
                                    <td style="padding: 8px; border: 1px solid #ddd;"><code>/clone_voice_stream</code></td>
                                    <td style="padding: 8px; border: 1px solid #ddd;"><strong>⚡ Clone voice (streaming)</strong></td>
                                </tr>
                                <tr style="background: #fff0f0;">
                                    <td style="padding: 8px; border: 1px solid #ddd;">POST</td>
                                    <td style="padding: 8px; border: 1px solid #ddd;"><code>/api/estimate_duration</code></td>
                                    <td style="padding: 8px; border: 1px solid #ddd;">Estimate speech length</td>
                                </tr>
                                <tr style="background: #fff0f0;">
                                    <td style="padding: 8px; border: 1px solid #ddd;">POST</td>
                                    <td style="padding: 8px; border: 1px solid #ddd;"><code>/api/clear_outputs</code></td>
                                    <td style="padding: 8px; border: 1px solid #ddd;">Clean output files</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>

        <script>
            function switchTab(tabName) {
                // Hide all tab contents
                document.querySelectorAll('.tab-content').forEach(content => {
                    content.classList.remove('active');
                });
                
                // Remove active class from all tabs
                document.querySelectorAll('.tab').forEach(tab => {
                    tab.classList.remove('active');
                });
                
                // Show selected tab content
                document.getElementById(tabName).classList.add('active');
                
                // Add active class to clicked tab
                event.target.classList.add('active');
                
                // Load speakers if switching to speakers tab
                if (tabName === 'speakers') {
                    loadSpeakerList();
                }
            }

            function setChineseDemo(type) {
                const textArea = document.getElementById('text');
                const demos = {
                    '现代文本': '你好！欢迎使用IndexTTS中文语音合成系统。这是一个功能强大的AI语音生成工具，能够准确处理中文语音合成任务。系统支持多种语音风格，让您的文本转换为自然流畅的语音。',
                    '古诗词': '床前明月光，疑是地上霜。举头望明月，低头思故乡。这首《静夜思》是李白的名作，表达了诗人对故乡的深深思念之情。',
                    '数字日期': '今天是2025年1月11日，时间是下午3点30分。这款产品的价格是12,999元，性价比很高。我的电话号码是138-8888-8888，欢迎联系。',
                    '中英混合': '我正在使用IndexTTS和vLLM技术进行AI语音合成。This system supports both Chinese and English perfectly. 这个系统的RTF约为0.1，比原版快3倍！GPU memory utilization设置为85%。'
                };
                
                textArea.value = demos[type];
                textArea.focus();
                
                // Show a brief tooltip
                showStatus(`已设置${type}演示文本`, 'success');
                setTimeout(() => hideStatus(), 2000);
            }

            function setEmotionDemo(emotionType) {
                const textArea = document.getElementById('text');
                const emotionText = document.getElementById('emotionText');
                const emotionWeight = document.getElementById('emotionWeight');
                
                const emotionDemos = {
                    '开心': {
                        text: '今天真是太开心了！我收到了好消息，心情特别愉快。阳光明媚，鸟儿在歌唱，一切都是那么美好！',
                        emotion: 'happy and joyful',
                        weight: 0.8
                    },
                    '悲伤': {
                        text: '雨滴轻敲着窗台，就像我内心的忧伤。离别的时刻总是让人难过，回忆如潮水般涌来。',
                        emotion: 'sad and melancholic',
                        weight: 0.7
                    },
                    '愤怒': {
                        text: '这实在太过分了！我再也无法忍受这种不公正的待遇。愤怒在我心中燃烧，必须要说出来！',
                        emotion: 'angry and frustrated',
                        weight: 0.6
                    },
                    '平静': {
                        text: '静坐在湖边，微风轻拂过脸颊。内心如湖水般平静，思绪缓缓流淌，享受这宁静的时光。',
                        emotion: 'calm and peaceful',
                        weight: 0.3
                    }
                };
                
                const demo = emotionDemos[emotionType];
                if (demo) {
                    textArea.value = demo.text;
                    emotionText.value = demo.emotion;
                    emotionWeight.value = demo.weight;
                    document.getElementById('emotionWeightValue').textContent = demo.weight;
                    textArea.focus();
                    
                    // Show a brief tooltip
                    showStatus(`已设置${emotionType}情感演示 (${demo.emotion})`, 'success');
                    setTimeout(() => hideStatus(), 3000);
                }
            }

            function showStatus(message, type, elementId = 'status') {
                const status = document.getElementById(elementId);
                status.textContent = message;
                status.className = `status ${type}`;
                status.style.display = 'block';
            }

            function hideStatus(elementId = 'status') {
                document.getElementById(elementId).style.display = 'none';
            }

            async function loadSpeakers() {
                try {
                    const response = await fetch('/audio_roles');
                    const data = await response.json();
                    const select = document.getElementById('speaker');
                    
                    // Clear existing options except first
                    select.innerHTML = '<option value="">Select a speaker...</option>';
                    
                    if (data.success && data.roles) {
                        data.roles.forEach(speaker => {
                            const option = document.createElement('option');
                            option.value = speaker;
                            option.textContent = speaker;
                            select.appendChild(option);
                        });
                    }
                } catch (error) {
                    console.error('Failed to load speakers:', error);
                }
            }

            async function loadSpeakerList() {
                try {
                    const response = await fetch('/audio_roles');
                    const data = await response.json();
                    const listDiv = document.getElementById('speakerList');
                    
                    if (data.success && data.roles) {
                        const speakers = data.roles;
                        let html = `<h4>📊 ${speakers.length} Speakers Available</h4>`;
                        
                        for (const name of speakers) {
                            html += `
                                <div class="speaker-item">
                                    <div class="speaker-info">
                                        <h4>🎭 ${name}</h4>
                                        <small>Speaker preset</small>
                                    </div>
                                    <button class="btn btn-danger" onclick="deleteSpeaker('${name}')">🗑️ Delete</button>
                                </div>
                            `;
                        }
                        
                        listDiv.innerHTML = html;
                    } else {
                        listDiv.innerHTML = '<p>No speakers found.</p>';
                    }
                } catch (error) {
                    console.error('Failed to load speaker list:', error);
                    document.getElementById('speakerList').innerHTML = '<p>Error loading speakers.</p>';
                }
            }

            async function deleteSpeaker(speakerName) {
                if (!confirm(`Are you sure you want to delete speaker "${speakerName}"?`)) {
                    return;
                }
                
                try {
                    const formData = new FormData();
                    formData.append('name', speakerName);
                    
                    const response = await fetch('/delete_speaker', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    showStatus(result.success ? 'Speaker deleted successfully' : result.error, result.success ? 'success' : 'error', 'speakerStatus');
                    
                    if (result.success) {
                        loadSpeakerList();
                        loadSpeakers(); // Refresh dropdown
                    }
                } catch (error) {
                    showStatus(`Error deleting speaker: ${error.message}`, 'error', 'speakerStatus');
                }
            }

            async function estimateDuration() {
                const text = document.getElementById('text').value;
                if (!text.trim()) {
                    showStatus('Please enter text first', 'error');
                    return;
                }
                
                try {
                    showStatus('Estimating duration...', 'success');
                    
                    const response = await fetch('/api/estimate_duration', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({text: text, language: 'auto'})
                    });
                    
                    const result = await response.json();
                    if (result.status === 'success') {
                        const estimateDiv = document.getElementById('durationEstimate');
                        estimateDiv.innerHTML = `📊 Estimated: <strong>${result.duration_s}s</strong> (${result.duration_ms}ms)<br>🌐 Language: ${result.detected_language} | 📝 Characters: ${result.char_count}`;
                        estimateDiv.style.display = 'block';
                        document.getElementById('speechLength').value = result.duration_ms;
                        showStatus(`Duration estimated: ${result.duration_s}s`, 'success');
                    } else {
                        showStatus(`Error: ${result.message}`, 'error');
                    }
                } catch (error) {
                    showStatus(`Error estimating duration: ${error.message}`, 'error');
                }
            }

            async function clearOutputs() {
                if (!confirm('Are you sure you want to clear all generated output files? This action cannot be undone.')) {
                    return;
                }
                
                try {
                    showStatus('Clearing outputs...', 'success');
                    
                    const response = await fetch('/api/clear_outputs', {
                        method: 'POST'
                    });
                    
                    const result = await response.json();
                    
                    if (result.status === 'success') {
                        const message = `✅ ${result.message}\n📁 Files deleted: ${result.files_deleted}\n💾 Space freed: ${result.space_freed_mb} MB`;
                        showStatus(message, 'success');
                        
                        // Clear the audio result display
                        document.getElementById('audioResult').innerHTML = '';
                    } else {
                        showStatus(`Error: ${result.message}`, 'error');
                    }
                } catch (error) {
                    showStatus(`Error clearing outputs: ${error.message}`, 'error');
                }
            }

            // Add Speaker Form
            document.getElementById('addSpeakerForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const speakerName = document.getElementById('speakerName').value;
                const audioFiles = document.getElementById('speakerAudioFiles').files;
                
                if (!audioFiles || audioFiles.length === 0) {
                    showStatus('Please select at least one audio file', 'error', 'speakerStatus');
                    return;
                }
                
                try {
                    showStatus('Adding speaker...', 'success', 'speakerStatus');
                    
                    const formData = new FormData();
                    formData.append('name', speakerName);
                    formData.append('audio_file', audioFiles[0]); // /add_speaker uses single file
                    formData.append('enhance_voice', document.getElementById('applyEnhancement').checked ? 'true' : 'false');
                    formData.append('super_resolution_voice', document.getElementById('applySuperResolution').checked ? 'true' : 'false');
                    
                    const response = await fetch('/add_speaker', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (result.success) {
                        showStatus(`Speaker "${speakerName}" added successfully!`, 'success', 'speakerStatus');
                        this.reset();
                        loadSpeakerList();
                        loadSpeakers(); // Refresh dropdown
                    } else {
                        showStatus(`Error: ${result.error}`, 'error', 'speakerStatus');
                    }
                } catch (error) {
                    showStatus(`Error adding speaker: ${error.message}`, 'error', 'speakerStatus');
                }
            });

            // TTS Form
            document.getElementById('ttsForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const formData = new FormData(this);
                const text = formData.get('text');
                const speaker = formData.get('speaker');
                const emotionText = document.getElementById('emotionText').value;
                const emotionWeight = parseFloat(document.getElementById('emotionWeight').value);
                const diffusionSteps = parseInt(document.getElementById('diffusionSteps').value);
                const maxTextTokens = parseInt(document.getElementById('maxTextTokens').value);
                const streamingMode = document.getElementById('streamingMode').checked;
                
                if (!text.trim()) {
                    showStatus('Please enter some text to synthesize.', 'error');
                    return;
                }
                
                try {
                    const startTime = performance.now();
                    
                    if (streamingMode) {
                        // Streaming mode
                        await handleStreamingRequest(text, speaker, emotionText, emotionWeight, diffusionSteps, maxTextTokens, formData, startTime);
                    } else {
                        // Regular mode
                        await handleRegularRequest(text, speaker, emotionText, emotionWeight, diffusionSteps, maxTextTokens, formData, startTime);
                    }
                } catch (error) {
                    showStatus(`Network error: ${error.message}`, 'error');
                }
            });

            const translateBtn = document.getElementById('translateBtn');
            const translateAdvancedToggle = document.getElementById('translateAdvancedMode');
            const translateAdvancedSettings = document.getElementById('translateAdvancedSettings');
            const translateAdvancedPanel = document.getElementById('translateAdvancedPanel');
            const translateSegmentsList = document.getElementById('translateSegmentsList');
            const translateSegmentsStatus = document.getElementById('translateSegmentsStatus');
            const translateSegmentsSelectAll = document.getElementById('translateSegmentsSelectAll');
            const translateGenerateBtn = document.getElementById('translateGenerateBtn');
            const translateDebugTranslate = document.getElementById('translateDebugTranslate');
            const translateCustomPrompt = document.getElementById('translateCustomPrompt');
            const translateGeminiModel = document.getElementById('translateGeminiModel');
            const translateGeminiApiKey = document.getElementById('translateGeminiApiKey');
            const translateDestLanguageSelect = document.getElementById('translateDestLanguage');
            const translateEnhanceEl = document.getElementById('translateEnhancement');
            const translateSuperEl = document.getElementById('translateSuperResolution');
            const translateMergeBackEl = document.getElementById('translateMergeBack');
            const translateBackingPreview = document.getElementById('translateBackingPreview');
            const translateMinSpeechInput = document.getElementById('translateMinSpeech');
            const translateMaxMergeInput = document.getElementById('translateMaxMerge');
            const translateManualSegmentsToggle = document.getElementById('translateManualSegmentsToggle');
            const translateManualSegmentsPanel = document.getElementById('translateManualSegmentsPanel');
            const translateManualSegmentsInput = document.getElementById('translateManualSegments');
            const translatePromptTranslation = document.getElementById('translatePromptTranslation');
            const translatePromptTranscription = document.getElementById('translatePromptTranscription');
            const translatePromptTemplates = document.getElementById('translatePromptTemplates');
            let currentTranslateSessionId = null;
            let currentTranslateSegments = [];
            let promptTemplates = {
                translation: '',
                transcription: '',
            };

            function formatTimestamp(ms) {
                const totalMs = Math.max(0, Math.round(ms || 0));
                const minutes = Math.floor(totalMs / 60000);
                const seconds = (totalMs % 60000) / 1000;
                const secondsStr =
                    seconds < 10
                        ? `0${seconds.toFixed(3)}`.replace(/([.][0-9]*?[1-9])0+$/,'$1').replace(/[.]0+$/,'')
                        : `${seconds.toFixed(3)}`.replace(/([.][0-9]*?[1-9])0+$/,'$1').replace(/[.]0+$/,'');
                return `${String(minutes).padStart(2, '0')}:${secondsStr}`;
            }

            function setTranslateButtonLabel() {
                if (!translateBtn) return;
                if (translateAdvancedToggle && translateAdvancedToggle.checked) {
                    translateBtn.textContent = '🧠 Analyze Segments';
                } else {
                    translateBtn.textContent = '🌐 Translate Speech';
                }
            }

            function syncTranslateMergeBackState() {
                if (!translateMergeBackEl) {
                    return;
                }
                const enhancementEnabled = translateEnhanceEl && translateEnhanceEl.checked;
                translateMergeBackEl.disabled = !enhancementEnabled;
                if (!enhancementEnabled) {
                    translateMergeBackEl.checked = false;
                }
            }

            if (translateEnhanceEl) {
                translateEnhanceEl.addEventListener('change', syncTranslateMergeBackState);
                syncTranslateMergeBackState();
            }
            if (translateDestLanguageSelect) {
                translateDestLanguageSelect.addEventListener('change', refreshPromptTemplates);
            }

            function appendSegmentParameters(formData) {
                if (!formData) {
                    return;
                }
                if (translateMinSpeechInput) {
                    const minValue = (translateMinSpeechInput.value || '').trim();
                    if (minValue) {
                        formData.append('min_speech_ms', minValue);
                    }
                }
                if (translateMaxMergeInput) {
                    const maxValue = (translateMaxMergeInput.value || '').trim();
                    if (maxValue) {
                        formData.append('max_merge_ms', maxValue);
                    }
                }
            }

            function appendManualSegments(formData) {
                if (
                    !formData ||
                    !translateManualSegmentsToggle ||
                    !translateManualSegmentsInput ||
                    !translateManualSegmentsToggle.checked
                ) {
                    return;
                }
                const manualText = translateManualSegmentsInput.value.trim();
                if (manualText) {
                    formData.append('segments_json', manualText);
                }
            }

            function refreshPromptTemplates() {
                const destLang = translateDestLanguageSelect ? (translateDestLanguageSelect.value || '').trim() : '';
                const replacement = destLang || '{dest_language}';
                if (translatePromptTranslation) {
                    translatePromptTranslation.value = promptTemplates.translation
                        ? promptTemplates.translation.split('{dest_language}').join(replacement)
                        : '';
                }
                if (translatePromptTranscription) {
                    translatePromptTranscription.value = promptTemplates.transcription
                        ? promptTemplates.transcription.split('{dest_language}').join(replacement)
                        : '';
                }
            }
            if (translateManualSegmentsToggle && translateManualSegmentsPanel) {
                const updateManualSegmentsVisibility = () => {
                    const enabled = translateManualSegmentsToggle.checked;
                    translateManualSegmentsPanel.style.display = enabled ? 'block' : 'none';
                    if (translatePromptTemplates) {
                        translatePromptTemplates.style.display = enabled ? 'block' : 'none';
                    }
                };
                translateManualSegmentsToggle.addEventListener('change', updateManualSegmentsVisibility);
                updateManualSegmentsVisibility();
            }

            async function loadPromptTemplates() {
                if (!translatePromptTranslation && !translatePromptTranscription) {
                    return;
                }
                try {
                    const response = await fetch('/api/prompt_templates');
                    if (!response.ok) {
                        return;
                    }
                    const data = await response.json();
                    if (typeof data.translation === 'string') {
                        promptTemplates.translation = data.translation;
                    }
                    if (typeof data.transcription === 'string') {
                        promptTemplates.transcription = data.transcription;
                    }
                    refreshPromptTemplates();
                } catch (error) {
                    console.warn('Failed to load prompt templates', error);
                }
            }

            loadPromptTemplates();

            function resetAdvancedPanel(clearSession = true) {
                if (clearSession) {
                    currentTranslateSessionId = null;
                }
                currentTranslateSegments = [];
                if (translateAdvancedPanel) {
                    translateAdvancedPanel.style.display = 'none';
                }
                if (translateSegmentsList) {
                    translateSegmentsList.innerHTML = '';
                }
                if (translateBackingPreview) {
                    translateBackingPreview.style.display = 'none';
                    translateBackingPreview.innerHTML = '';
                }
                if (translateSegmentsStatus) {
                    hideStatus('translateSegmentsStatus');
                }
                if (translateSegmentsSelectAll) {
                    translateSegmentsSelectAll.checked = true;
                }
            }

            function updateTranslateSegmentsSummary() {
                if (!translateSegmentsStatus) {
                    return;
                }
                if (!translateSegmentsList) {
                    hideStatus('translateSegmentsStatus');
                    return;
                }
                const speechCards = translateSegmentsList.querySelectorAll('.segment-card.speech');
                if (!speechCards.length) {
                    hideStatus('translateSegmentsStatus');
                    return;
                }
                let selected = 0;
                speechCards.forEach(card => {
                    const checkbox = card.querySelector('input.segment-generate');
                    if (checkbox && checkbox.checked) {
                        selected += 1;
                    }
                });
                const preserved = speechCards.length - selected;
                showStatus(
                    `Selected ${selected}/${speechCards.length} speech segments • Preserving ${preserved}`,
                    'success',
                    'translateSegmentsStatus'
                );
                if (translateSegmentsSelectAll) {
                    translateSegmentsSelectAll.checked = selected === speechCards.length;
                }
            }

            function renderTranslateSegments(segments = []) {
                if (!translateSegmentsList) {
                    return;
                }
                translateSegmentsList.innerHTML = '';
                const hasSpeech = segments.some(seg => seg.type === 'speech');
                if (!segments.length) {
                    translateSegmentsList.innerHTML = '<div class="segment-empty">No segments returned from Gemini.</div>';
                    updateTranslateSegmentsSummary();
                    return;
                }
                if (translateSegmentsSelectAll) {
                    const allSelected = segments.filter(seg => seg.type === 'speech').every(seg => seg.generate !== false);
                    translateSegmentsSelectAll.checked = allSelected;
                }
                segments.forEach(segment => {
                    const startMsVal = Number.isFinite(segment.start_ms) ? segment.start_ms : 0;
                    const endMsVal = Number.isFinite(segment.end_ms) ? segment.end_ms : startMsVal;
                    const durationVal = Number.isFinite(segment.duration_ms)
                        ? segment.duration_ms
                        : Math.max(0, endMsVal - startMsVal);
                    const card = document.createElement('div');
                    card.className = `segment-card ${segment.type}`;
                    card.dataset.index = segment.index;
                    card.dataset.type = segment.type;

                    const header = document.createElement('div');
                    header.className = 'segment-header';

                    const title = document.createElement('div');
                    title.innerHTML = `<strong>#${segment.index}</strong> ${segment.type === 'speech' ? 'Speech Segment' : 'Silence Segment'}`;
                    header.appendChild(title);

                    if (segment.type === 'speech') {
                        const checkboxLabel = document.createElement('label');
                        checkboxLabel.className = 'segment-checkbox';
                        const checkbox = document.createElement('input');
                        checkbox.type = 'checkbox';
                        checkbox.className = 'segment-generate';
                        checkbox.checked = segment.generate !== false;
                        checkbox.addEventListener('change', updateTranslateSegmentsSummary);
                        checkboxLabel.appendChild(checkbox);
                        const span = document.createElement('span');
                        span.textContent = 'Generate';
                        checkboxLabel.appendChild(span);
                        header.appendChild(checkboxLabel);
                    } else {
                        const meta = document.createElement('span');
                        meta.className = 'segment-meta';
                        meta.textContent = 'Preserved silence';
                        header.appendChild(meta);
                    }

                    card.appendChild(header);

                    const metaInfo = document.createElement('div');
                    metaInfo.className = 'segment-meta';
                    metaInfo.textContent = `${segment.start || formatTimestamp(startMsVal)} → ${segment.end || formatTimestamp(endMsVal)} (${durationVal} ms)`;
                    card.appendChild(metaInfo);

                    const body = document.createElement('div');
                    body.className = 'segment-body';

                    const timing = document.createElement('div');
                    timing.className = 'segment-timing';
                    timing.innerHTML = `
                        <label>Start (ms)
                            <input type="number" class="segment-start" value="${startMsVal}" min="0">
                        </label>
                        <label>End (ms)
                            <input type="number" class="segment-end" value="${endMsVal}" min="0">
                        </label>
                        <div class="segment-duration-label">Duration: <span class="segment-duration">${durationVal}</span> ms</div>
                    `;
                    body.appendChild(timing);

                    const startInput = timing.querySelector('.segment-start');
                    const endInput = timing.querySelector('.segment-end');
                    const durationLabel = timing.querySelector('.segment-duration');
                    const updateDuration = () => {
                        const startVal = parseInt(startInput.value || '0', 10);
                        const endVal = parseInt(endInput.value || '0', 10);
                        const diff = Math.max(0, endVal - startVal);
                        durationLabel.textContent = diff;
                    };
                    startInput.addEventListener('input', updateDuration);
                    endInput.addEventListener('input', updateDuration);

                    if (segment.type === 'speech') {
                        const sourceGroup = document.createElement('div');
                        sourceGroup.innerHTML = `
                            <label>Source Text</label>
                            <textarea class="segment-source">${segment.source_text || ''}</textarea>
                        `;
                        body.appendChild(sourceGroup);

                        const translationGroup = document.createElement('div');
                        translationGroup.innerHTML = `
                            <label>Translation Text</label>
                            <textarea class="segment-translation">${segment.translated_text || ''}</textarea>
                        `;
                        body.appendChild(translationGroup);

                        if (segment.audio_preview) {
                            const audioEl = document.createElement('audio');
                            audioEl.className = 'segment-audio';
                            audioEl.controls = true;
                            audioEl.src = segment.audio_preview;
                            body.appendChild(audioEl);
                        }
                    }

                    card.appendChild(body);
                    translateSegmentsList.appendChild(card);
                });
                if (!hasSpeech) {
                    translateSegmentsList.insertAdjacentHTML('beforeend', '<div class="segment-empty">No speech segments detected.</div>');
                }
                updateTranslateSegmentsSummary();
            }

            function renderBackingPreview(sessionId, metadata) {
                if (!translateBackingPreview) {
                    return;
                }
                const backingMeta = metadata && metadata.backing_track;
                if (!backingMeta || !backingMeta.available) {
                    translateBackingPreview.style.display = 'none';
                    translateBackingPreview.innerHTML = '';
                    return;
                }
                const previewUrl = (backingMeta.preview_url && backingMeta.preview_url.trim())
                    ? backingMeta.preview_url
                    : `/api/translate_backing_track/${sessionId}`;
                const cacheKey = Date.now();
                translateBackingPreview.innerHTML = `
                    <div class="segment-card">
                        <div class="segment-header">🎼 Instrumental Backing Preview</div>
                        <audio controls style="width: 100%; margin-top: 6px;">
                            <source src="${previewUrl}?session=${sessionId}&t=${cacheKey}" type="audio/wav">
                        </audio>
                    </div>
                `;
                translateBackingPreview.style.display = 'block';
            }

            function syncSegmentRulesFromMetadata(rules) {
                if (!rules) {
                    return;
                }
                if (translateMinSpeechInput) {
                    if (rules.min_speech_ms !== undefined && rules.min_speech_ms !== null) {
                        translateMinSpeechInput.value = rules.min_speech_ms;
                    } else {
                        translateMinSpeechInput.value = '';
                    }
                }
                if (translateMaxMergeInput) {
                    if (rules.max_merge_ms !== undefined && rules.max_merge_ms !== null) {
                        translateMaxMergeInput.value = rules.max_merge_ms;
                    } else {
                        translateMaxMergeInput.value = '';
                    }
                }
            }

            if (translateAdvancedToggle) {
                translateAdvancedToggle.addEventListener('change', () => {
                    if (translateAdvancedSettings) {
                        translateAdvancedSettings.style.display = translateAdvancedToggle.checked ? 'block' : 'none';
                    }
                    if (!translateAdvancedToggle.checked) {
                        resetAdvancedPanel();
                    }
                    setTranslateButtonLabel();
                });
                setTranslateButtonLabel();
            }

            if (translateSegmentsSelectAll && translateSegmentsList) {
                translateSegmentsSelectAll.addEventListener('change', () => {
                    const speechCheckboxes = translateSegmentsList.querySelectorAll('.segment-card.speech input.segment-generate');
                    speechCheckboxes.forEach(cb => {
                        cb.checked = translateSegmentsSelectAll.checked;
                    });
                    updateTranslateSegmentsSummary();
                });
            } else {
                setTranslateButtonLabel();
            }

            const translateForm = document.getElementById('translateForm');
            if (translateForm) {
                translateForm.addEventListener('submit', async function(e) {
                    e.preventDefault();

                    const statusId = 'translateStatus';
                    const resultDiv = document.getElementById('translateResult');
                    const audioInput = document.getElementById('translateAudioFile');
                    const destInput = document.getElementById('translateDestLanguage');
                    const formatSelect = document.getElementById('translateOutputFormat');

                    hideStatus(statusId);
                    hideStatus('translateSegmentsStatus');
                    resultDiv.innerHTML = '';

                    if (!audioInput.files || audioInput.files.length === 0) {
                        showStatus('Please select a source audio file.', 'error', statusId);
                        return;
                    }

                    const destLanguage = destInput.value.trim();
                    if (!destLanguage) {
                        showStatus('Please select a destination language.', 'error', statusId);
                        return;
                    }

                    const selectedFormat = (formatSelect.value || 'mp3').toLowerCase();

                    const advancedEnabled = translateAdvancedToggle && translateAdvancedToggle.checked;

                    if (advancedEnabled) {
                        resetAdvancedPanel();
                        const formData = new FormData();
                        formData.append('audio_file', audioInput.files[0]);
                        formData.append('dest_language', destLanguage);
                        formData.append('response_format', selectedFormat);
                        formData.append('enhance_voice', translateEnhanceEl && translateEnhanceEl.checked ? 'true' : 'false');
                        formData.append('super_resolution_voice', translateSuperEl && translateSuperEl.checked ? 'true' : 'false');
                        formData.append('merge_backing_track', translateMergeBackEl && translateMergeBackEl.checked ? 'true' : 'false');
                        if (translateGeminiModel && translateGeminiModel.value) {
                            formData.append('gemini_model', translateGeminiModel.value);
                        }
                        if (translateGeminiApiKey && translateGeminiApiKey.value.trim()) {
                            formData.append('gemini_api_key', translateGeminiApiKey.value.trim());
                        }
                        if (translateDebugTranslate) {
                            formData.append('translate_text', translateDebugTranslate.checked ? 'true' : 'false');
                        }
                        const customPromptValue = translateCustomPrompt ? translateCustomPrompt.value.trim() : '';
                        if (customPromptValue) {
                            formData.append('prompt', customPromptValue);
                        }
                        appendSegmentParameters(formData);
                        appendManualSegments(formData);

                        try {
                            if (translateBtn) {
                                translateBtn.disabled = true;
                            }
                            showStatus('Analyzing audio and preparing editable segments... ⏳', 'success', statusId);

                            const response = await fetch('/api/translate_segments', {
                                method: 'POST',
                                body: formData
                            });

                            const contentType = response.headers.get('Content-Type') || '';

                            if (!response.ok) {
                                let errorMessage = `Segment preparation failed (${response.status})`;
                                if (contentType.includes('application/json')) {
                                    try {
                                        const errorData = await response.json();
                                        errorMessage = errorData.message || errorData.error || errorMessage;
                                    } catch (jsonError) {
                                        console.warn('Failed to parse error response:', jsonError);
                                    }
                                }
                                showStatus(errorMessage, 'error', statusId);
                                return;
                            }

                            if (!contentType.includes('application/json')) {
                                showStatus('Segment preparation failed: unexpected response format.', 'error', statusId);
                                return;
                            }

                            const data = await response.json();
                            if (data.status !== 'success' || !data.session_id) {
                                const message = data.message || data.error || 'Failed to prepare segments.';
                                showStatus(message, 'error', statusId);
                                return;
                            }

                            currentTranslateSessionId = data.session_id;
                            if (data.metadata && data.metadata.gemini_model && translateGeminiModel) {
                                translateGeminiModel.value = data.metadata.gemini_model;
                            }
                            renderBackingPreview(currentTranslateSessionId, data.metadata || null);
                            if (data.metadata && data.metadata.segment_rules) {
                                syncSegmentRulesFromMetadata(data.metadata.segment_rules);
                            }
                            const translateEnabledNow = translateDebugTranslate ? translateDebugTranslate.checked : true;
                            currentTranslateSegments = Array.isArray(data.segments)
                                ? data.segments.map(seg => ({
                                      ...seg,
                                      generate: translateEnabledNow ? seg.generate !== false : false,
                                  }))
                                : [];
                            renderTranslateSegments(currentTranslateSegments);
                            if (translateAdvancedPanel) {
                                translateAdvancedPanel.style.display = 'block';
                            }

                            const speechCount =
                                (data.metadata && data.metadata.speech_segment_count) ||
                                currentTranslateSegments.filter(seg => seg.type === 'speech').length;
                            const totalCount = currentTranslateSegments.length;
                            let statusMessage = `✅ Segments ready: ${totalCount}`;
                            if (typeof speechCount === 'number') {
                                statusMessage += ` total • ${speechCount} speech`;
                            }
                            showStatus(`${statusMessage}. Review below and choose segments to regenerate.`, 'success', statusId);
                            if (!currentTranslateSegments.length) {
                                showStatus('No segments detected. Try adjusting the audio or prompt.', 'error', 'translateSegmentsStatus');
                            }
                        } catch (error) {
                            console.error('Segment preparation error:', error);
                            showStatus(`Segment preparation error: ${error.message}`, 'error', statusId);
                        } finally {
                            if (translateBtn) {
                                translateBtn.disabled = false;
                            }
                        }
                        return;
                    } else {
                        resetAdvancedPanel();
                    }

                    const formData = new FormData();
                    formData.append('audio_file', audioInput.files[0]);
                    formData.append('dest_language', destLanguage);
                    formData.append('response_format', selectedFormat);
                    formData.append('enhance_voice', translateEnhanceEl && translateEnhanceEl.checked ? 'true' : 'false');
                    formData.append('super_resolution_voice', translateSuperEl && translateSuperEl.checked ? 'true' : 'false');
                    formData.append('merge_backing_track', translateMergeBackEl && translateMergeBackEl.checked ? 'true' : 'false');
                    if (translateGeminiModel && translateGeminiModel.value) {
                        formData.append('gemini_model', translateGeminiModel.value);
                    }
                    if (translateGeminiApiKey && translateGeminiApiKey.value.trim()) {
                        formData.append('gemini_api_key', translateGeminiApiKey.value.trim());
                    }
                    appendSegmentParameters(formData);
                    appendManualSegments(formData);

                    try {
                        if (translateBtn) {
                            translateBtn.disabled = true;
                        }
                        showStatus('Translating speech... this may take a moment ⏳', 'success', statusId);

                        const response = await fetch('/api/translate_audio', {
                            method: 'POST',
                            body: formData
                        });

                        const contentType = response.headers.get('Content-Type') || '';

                        if (!response.ok) {
                            let errorMessage = `Translation failed (${response.status})`;
                            if (contentType.includes('application/json')) {
                                try {
                                    const errorData = await response.json();
                                    errorMessage = errorData.message || errorData.error || errorMessage;
                                } catch (jsonError) {
                                    console.warn('Failed to parse error response:', jsonError);
                                }
                            }
                            showStatus(errorMessage, 'error', statusId);
                            return;
                        }

                        if (contentType.startsWith('audio/')) {
                            const blob = await response.blob();
                            const audioUrl = URL.createObjectURL(blob);
                            const downloadName = `translated_speech.${selectedFormat}`;

                            resultDiv.innerHTML = `
                                <audio controls src="${audioUrl}" style="width: 100%; margin-top: 20px;"></audio>
                                <div style="margin-top: 12px;">
                                    <a href="${audioUrl}" download="${downloadName}" class="btn">💾 Download</a>
                                </div>
                            `;

                            const segmentCount = response.headers.get('X-Translation-Segments');
                            const modelHeader = response.headers.get('X-Translation-Model');
                            let statusMessage = '✅ Translation complete!';
                            if (segmentCount) {
                                statusMessage += ` (${segmentCount} segments)`;
                            }
                            if (modelHeader) {
                                statusMessage += ` • Gemini model: ${modelHeader}`;
                            }
                            showStatus(statusMessage, 'success', statusId);
                        } else {
                            try {
                                const data = await response.json();
                                const message = data.message || data.error || 'Translation failed.';
                                showStatus(message, 'error', statusId);
                            } catch (parseError) {
                                showStatus('Translation failed: unexpected response format.', 'error', statusId);
                            }
                        }
                    } catch (error) {
                        console.error('Translation error:', error);
                        showStatus(`Translation error: ${error.message}`, 'error', statusId);
                    } finally {
                        if (translateBtn) {
                            translateBtn.disabled = false;
                        }
                    }
                });
            }

            if (translateGenerateBtn) {
                translateGenerateBtn.addEventListener('click', async () => {
                    const statusId = 'translateStatus';
                    const resultDiv = document.getElementById('translateResult');
                    hideStatus('translateSegmentsStatus');

                    if (!currentTranslateSessionId) {
                        showStatus('Analyze audio first to load segments.', 'error', 'translateSegmentsStatus');
                        return;
                    }
                    if (!translateSegmentsList) {
                        showStatus('Segment list unavailable.', 'error', 'translateSegmentsStatus');
                        return;
                    }
                    const segmentCards = translateSegmentsList.querySelectorAll('.segment-card');
                    if (!segmentCards.length) {
                        showStatus('No segments to generate.', 'error', 'translateSegmentsStatus');
                        return;
                    }

                    const segmentsPayload = [];
                    let hasError = false;

                    segmentCards.forEach(card => {
                        if (hasError) {
                            return;
                        }
                        const index = parseInt(card.dataset.index, 10);
                        const type = card.dataset.type || 'speech';
                        const startInput = card.querySelector('.segment-start');
                        const endInput = card.querySelector('.segment-end');
                        const startMs = parseInt(startInput ? startInput.value : '0', 10);
                        const endMs = parseInt(endInput ? endInput.value : '0', 10);
                        if (Number.isNaN(startMs) || Number.isNaN(endMs)) {
                            showStatus(`Segment #${index}: invalid timing.`, 'error', 'translateSegmentsStatus');
                            hasError = true;
                            return;
                        }
                        if (endMs <= startMs) {
                            showStatus(`Segment #${index}: end time must be greater than start time.`, 'error', 'translateSegmentsStatus');
                            hasError = true;
                            return;
                        }
                        let generate = false;
                        let sourceText = '';
                        let translatedText = '';
                        if (type === 'speech') {
                            const checkbox = card.querySelector('input.segment-generate');
                            generate = checkbox ? checkbox.checked : true;
                            const sourceInput = card.querySelector('.segment-source');
                            const translationInput = card.querySelector('.segment-translation');
                            sourceText = sourceInput ? sourceInput.value : '';
                            translatedText = translationInput ? translationInput.value : '';
                        }

                        segmentsPayload.push({
                            index,
                            type,
                            start_ms: startMs,
                            end_ms: endMs,
                            generate,
                            source_text: sourceText,
                            translated_text: translatedText,
                        });
                    });

                    if (hasError || !segmentsPayload.length) {
                        return;
                    }

                    const formatSelect = document.getElementById('translateOutputFormat');
                    const selectedFormat = (formatSelect && formatSelect.value ? formatSelect.value : 'mp3').toLowerCase();

                    const payload = {
                        session_id: currentTranslateSessionId,
                        segments: segmentsPayload,
                        response_format: selectedFormat,
                        merge_backing_track: translateMergeBackEl && translateMergeBackEl.checked ? true : false,
                    };

                    try {
                        translateGenerateBtn.disabled = true;
                        showStatus('Generating selected segments... 🎧', 'success', statusId);

                        const response = await fetch('/api/translate_generate_segments', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify(payload),
                        });

                        const contentType = response.headers.get('Content-Type') || '';

                        if (!response.ok) {
                            let errorMessage = `Generation failed (${response.status})`;
                            if (contentType.includes('application/json')) {
                                try {
                                    const errorData = await response.json();
                                    errorMessage = errorData.message || errorData.error || errorMessage;
                                } catch (jsonError) {
                                    console.warn('Failed to parse error response:', jsonError);
                                }
                            }
                            showStatus(errorMessage, 'error', 'translateSegmentsStatus');
                            showStatus(errorMessage, 'error', statusId);
                            return;
                        }

                        if (contentType.startsWith('audio/')) {
                            const blob = await response.blob();
                            const audioUrl = URL.createObjectURL(blob);
                            const downloadName = `translated_speech.${selectedFormat}`;

                            if (resultDiv) {
                                resultDiv.innerHTML = `
                                    <audio controls src="${audioUrl}" style="width: 100%; margin-top: 20px;"></audio>
                                    <div style="margin-top: 12px;">
                                        <a href="${audioUrl}" download="${downloadName}" class="btn">💾 Download</a>
                                    </div>
                                `;
                            }

                            const generatedHeader = parseInt(response.headers.get('X-Translation-Generated') || '0', 10);
                            const preservedHeader = parseInt(response.headers.get('X-Translation-Preserved') || '0', 10);
                            const segmentHeader = response.headers.get('X-Translation-Segments');
                            let statusMessage = '✅ Advanced translation complete!';
                            if (segmentHeader) {
                                statusMessage += ` (${segmentHeader} segments)`;
                            }
                            if (!Number.isNaN(generatedHeader) && !Number.isNaN(preservedHeader)) {
                                statusMessage += ` • Generated ${generatedHeader}, preserved ${preservedHeader}`;
                            }
                            showStatus(statusMessage, 'success', statusId);
                            showStatus(`Generated ${generatedHeader} segments • Preserved ${preservedHeader}`, 'success', 'translateSegmentsStatus');

                            const segmentMap = new Map(segmentsPayload.map(seg => [seg.index, seg]));
                            currentTranslateSegments = currentTranslateSegments.map(seg => {
                                const updated = segmentMap.get(seg.index);
                                if (!updated) {
                                    return seg;
                                }
                                const duration = Math.max(0, updated.end_ms - updated.start_ms);
                                return {
                                    ...seg,
                                    start_ms: updated.start_ms,
                                    end_ms: updated.end_ms,
                                    duration_ms: duration,
                                    start: formatTimestamp(updated.start_ms),
                                    end: formatTimestamp(updated.end_ms),
                                    source_text: updated.source_text || '',
                                    translated_text: updated.translated_text || '',
                                    generate: updated.generate,
                                };
                            });
                            renderTranslateSegments(currentTranslateSegments);
                        } else {
                            try {
                                const data = await response.json();
                                const message = data.message || data.error || 'Generation failed.';
                                showStatus(message, 'error', 'translateSegmentsStatus');
                                showStatus(message, 'error', statusId);
                            } catch (parseError) {
                                showStatus('Generation failed: unexpected response format.', 'error', 'translateSegmentsStatus');
                                showStatus('Generation failed: unexpected response format.', 'error', statusId);
                            }
                        }
                    } catch (error) {
                        console.error('Segment generation error:', error);
                        showStatus(`Segment generation error: ${error.message}`, 'error', 'translateSegmentsStatus');
                        showStatus(`Segment generation error: ${error.message}`, 'error', statusId);
                    } finally {
                        translateGenerateBtn.disabled = false;
                    }
                });
            }

            async function handleRegularRequest(text, speaker, emotionText, emotionWeight, diffusionSteps, maxTextTokens, formData, startTime) {
                    let response;
                    const voiceFiles = document.getElementById('voice_files').files;
                    
                    if (speaker) {
                        // Use /speak endpoint with speaker preset
                        const requestData = {
                            text: text, 
                            name: speaker,  // API uses 'name' not 'speaker'
                            emotion_text: emotionText || "",
                            emotion_weight: emotionWeight,
                            speech_length: parseInt(document.getElementById('speechLength').value) || 0,
                            diffusion_steps: diffusionSteps,
                            max_text_tokens_per_sentence: maxTextTokens,
                            response_format: "mp3"
                        };
                        
                        response = await fetch('/speak', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify(requestData)
                        });
                    } else if (voiceFiles && voiceFiles.length > 0) {
                        // Use /clone_voice endpoint with uploaded voice file
                        const cloneFormData = new FormData();
                        cloneFormData.append('text', text);
                        cloneFormData.append('reference_audio_file', voiceFiles[0]);
                        cloneFormData.append('emotion_text', emotionText || "");
                        cloneFormData.append('emotion_weight', emotionWeight.toString());
                        cloneFormData.append('speech_length', (parseInt(document.getElementById('speechLength').value) || 0).toString());
                        cloneFormData.append('diffusion_steps', diffusionSteps.toString());
                        cloneFormData.append('max_text_tokens_per_sentence', maxTextTokens.toString());
                        cloneFormData.append('response_format', 'mp3');
                        
                        response = await fetch('/clone_voice', {
                            method: 'POST',
                            body: cloneFormData
                        });
                    } else {
                        showStatus('Please select a speaker preset or upload a voice file', 'error');
                        return;
                    }
                    
                    if (response.ok) {
                        const endTime = performance.now();
                        const duration = ((endTime - startTime) / 1000).toFixed(2);
                        
                        const blob = await response.blob();
                        const audioUrl = URL.createObjectURL(blob);
                        
                        document.getElementById('audioResult').innerHTML = `
                            <h3>🎵 Generated Speech (${duration}s)</h3>
                        <audio controls autoplay style="width: 100%; margin: 10px 0;">
                                <source src="${audioUrl}" type="audio/mpeg">
                            </audio>
                            <br>
                            <a href="${audioUrl}" download="speech.mp3" class="btn">💾 Download</a>
                        `;
                        // Show enhanced status message with emotion info
                        let statusMessage = `Speech generated in ${duration}s! 🚀`;
                        if (emotionText && emotionText.trim()) {
                            statusMessage += ` 😊 Emotion: "${emotionText}" (${emotionWeight})`;
                        }
                        showStatus(statusMessage, 'success');
                    } else {
                        const error = await response.text();
                        showStatus(`Error: ${error}`, 'error');
                    }
            }

            async function handleStreamingRequest(text, speaker, emotionText, emotionWeight, diffusionSteps, maxTextTokens, formData, startTime) {
                showStatus('⚡ Streaming: Waiting for first chunk...', 'success');
                
                // Get first chunk size setting
                const firstChunkSize = parseInt(document.getElementById('firstChunkSize').value) || 40;
                
                let endpoint, requestOptions;
                
                const voiceFiles = document.getElementById('voice_files').files;
                
                if (speaker) {
                    // Use /speak_stream endpoint with speaker preset
                    endpoint = '/speak_stream';
                    requestOptions = {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            text: text,
                            name: speaker,  // API uses 'name' not 'speaker'
                            emotion_text: emotionText || "",
                            emotion_weight: emotionWeight,
                            speech_length: parseInt(document.getElementById('speechLength').value) || 0,
                            diffusion_steps: diffusionSteps,
                            max_text_tokens_per_sentence: maxTextTokens,
                            response_format: "mp3"
                        })
                    };
                } else if (voiceFiles && voiceFiles.length > 0) {
                    // Use /clone_voice_stream endpoint with uploaded voice file
                    endpoint = '/clone_voice_stream';
                    const cloneFormData = new FormData();
                    cloneFormData.append('text', text);
                    cloneFormData.append('reference_audio_file', voiceFiles[0]);
                    cloneFormData.append('emotion_text', emotionText || "");
                    cloneFormData.append('emotion_weight', emotionWeight.toString());
                    cloneFormData.append('speech_length', (parseInt(document.getElementById('speechLength').value) || 0).toString());
                    cloneFormData.append('diffusion_steps', diffusionSteps.toString());
                    cloneFormData.append('max_text_tokens_per_sentence', maxTextTokens.toString());
                    cloneFormData.append('response_format', 'mp3');
                    requestOptions = {
                        method: 'POST',
                        body: cloneFormData
                    };
                } else {
                    showStatus('Please select a speaker preset or upload a voice file for streaming', 'error');
                    return;
                }
                
                const response = await fetch(endpoint, requestOptions);
                
                if (!response.ok) {
                    const error = await response.text();
                    showStatus(`Error: ${error}`, 'error');
                    return;
                }
                
                const reader = response.body.getReader();
                const audioChunks = [];
                let buffer = new Uint8Array();
                let chunkCount = 0;
                let firstChunkTime = null;
                let audioContext = null;
                let audioSource = null;
                let nextStartTime = 0;
                
                // Create audio context for streaming playback
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
                
                try {
                    while (true) {
                        const {done, value} = await reader.read();
                        
                        if (done) {
                            break;
                        }
                        
                        // Append new data to buffer
                        const newBuffer = new Uint8Array(buffer.length + value.length);
                        newBuffer.set(buffer);
                        newBuffer.set(value, buffer.length);
                        buffer = newBuffer;
                        
                        // Try to parse chunks from buffer
                        while (true) {
                            // Look for header: CHUNK:idx:size:status\\n
                            // Find newline character (10 = '\\n' in ASCII)
                            let headerEnd = -1;
                            for (let i = 0; i < buffer.length; i++) {
                                if (buffer[i] === 10) {
                                    headerEnd = i;
                                    break;
                                }
                            }
                            
                            if (headerEnd === -1) break;
                            
                            const headerText = new TextDecoder().decode(buffer.slice(0, headerEnd));
                            
                            if (headerText.startsWith('ERROR:')) {
                                showStatus(`Streaming error: ${headerText.substring(6)}`, 'error');
                                return;
                            }
                            
                            if (!headerText.startsWith('CHUNK:')) break;
                            
                            const parts = headerText.split(':');
                            if (parts.length !== 4) break;
                            
                            const chunkIdx = parseInt(parts[1]);
                            const chunkSize = parseInt(parts[2]);
                            const isLast = parts[3] === 'LAST';
                            
                            // Check if we have the complete chunk
                            const chunkStart = headerEnd + 1;
                            const chunkEnd = chunkStart + chunkSize;
                            
                            if (buffer.length < chunkEnd) break;
                            
                            // Extract chunk data
                            const chunkData = buffer.slice(chunkStart, chunkEnd);
                            buffer = buffer.slice(chunkEnd);
                            
                            chunkCount++;
                            
                        if (firstChunkTime === null) {
                            firstChunkTime = performance.now();
                            const ttfb = ((firstChunkTime - startTime) / 1000).toFixed(2);
                            
                            // Show first chunk performance prominently
                            const firstChunkSize = document.getElementById('firstChunkSize').value;
                            showStatus(`⚡ First chunk ready in ${ttfb}s! (${firstChunkSize} tokens) Playing now...`, 'success');
                            
                            // Show real-time performance indicator
                            document.getElementById('audioResult').innerHTML = `
                                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 15px; color: white; margin: 10px 0;">
                                    <h3 style="margin: 0; display: flex; align-items: center; gap: 10px;">
                                        <span class="loading"></span>
                                        Streaming in progress...
                                    </h3>
                                    <div style="margin-top: 15px; background: rgba(255,255,255,0.2); padding: 15px; border-radius: 10px;">
                                        <div style="font-size: 1.2em; margin-bottom: 5px;">
                                            ⚡ First Chunk Generated
                                        </div>
                                        <div style="font-size: 2em; font-weight: bold;">
                                            ${ttfb}s
                                        </div>
                                        <div style="font-size: 0.9em; opacity: 0.9; margin-top: 5px;">
                                            🎵 Audio playing • Receiving chunk ${chunkCount}/${chunkCount}...
                                        </div>
                                    </div>
                                </div>
                            `;
                        } else {
                            // Update chunk counter during streaming
                            const currentDisplay = document.getElementById('audioResult').innerHTML;
                            if (currentDisplay.includes('Streaming in progress')) {
                                const ttfb = ((firstChunkTime - startTime) / 1000).toFixed(2);
                                document.getElementById('audioResult').innerHTML = `
                                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 15px; color: white; margin: 10px 0;">
                                        <h3 style="margin: 0; display: flex; align-items: center; gap: 10px;">
                                            <span class="loading"></span>
                                            Streaming in progress...
                                        </h3>
                                        <div style="margin-top: 15px; background: rgba(255,255,255,0.2); padding: 15px; border-radius: 10px;">
                                            <div style="font-size: 1.2em; margin-bottom: 5px;">
                                                ⚡ First Chunk Generated
                                            </div>
                                            <div style="font-size: 2em; font-weight: bold;">
                                                ${ttfb}s
                                            </div>
                                            <div style="font-size: 0.9em; opacity: 0.9; margin-top: 5px;">
                                                🎵 Audio playing • Received ${chunkCount} chunks...
                                            </div>
                                        </div>
                                    </div>
                                `;
                            }
                        }
                            
                            // Decode and play audio chunk
                            try {
                                const audioBlob = new Blob([chunkData], {type: 'audio/mpeg'});
                                const arrayBuffer = await audioBlob.arrayBuffer();
                                const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
                                
                                // Schedule playback
                                const source = audioContext.createBufferSource();
                                source.buffer = audioBuffer;
                                source.connect(audioContext.destination);
                                
                                const currentTime = audioContext.currentTime;
                                if (nextStartTime < currentTime) {
                                    nextStartTime = currentTime;
                                }
                                
                                source.start(nextStartTime);
                                nextStartTime += audioBuffer.duration;
                                
                                // Store for later download
                                audioChunks.push(chunkData);
                                
                                showStatus(`⚡ Streaming: Playing chunk ${chunkIdx + 1}...`, 'success');
                            } catch (decodeError) {
                                console.error('Error decoding audio chunk:', decodeError);
                                showStatus(`⚠️ Error decoding chunk ${chunkIdx}: ${decodeError.message}`, 'error');
                            }
                            
                        if (isLast) {
                            const endTime = performance.now();
                            const duration = ((endTime - startTime) / 1000).toFixed(2);
                            const firstChunkDuration = ((firstChunkTime - startTime) / 1000).toFixed(2);
                            
                            // Combine all chunks for download
                            const combinedBlob = new Blob(audioChunks, {type: 'audio/mpeg'});
                            const audioUrl = URL.createObjectURL(combinedBlob);
                            
                            // Calculate performance metrics
                            const totalGenTime = duration;
                            const firstChunkPercent = ((firstChunkDuration / totalGenTime) * 100).toFixed(0);
                            
                            document.getElementById('audioResult').innerHTML = `
                                <h3>🎵 Streamed Speech (${chunkCount} chunks)</h3>
                                <audio controls src="${audioUrl}" style="width: 100%; margin: 10px 0;"></audio>
                                <br>
                                <div style="background: #f8f9fa; padding: 15px; border-radius: 10px; margin: 10px 0;">
                                    <h4 style="margin-top: 0;">⚡ Performance Metrics</h4>
                                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 10px;">
                                        <div style="background: white; padding: 10px; border-radius: 5px; border-left: 4px solid #667eea;">
                                            <strong style="color: #667eea;">⏱️ First Chunk:</strong><br>
                                            <span style="font-size: 1.5em; font-weight: bold;">${firstChunkDuration}s</span>
                                        </div>
                                        <div style="background: white; padding: 10px; border-radius: 5px; border-left: 4px solid #764ba2;">
                                            <strong style="color: #764ba2;">🕐 Total Time:</strong><br>
                                            <span style="font-size: 1.5em; font-weight: bold;">${totalGenTime}s</span>
                                        </div>
                                    </div>
                                    <div style="background: white; padding: 10px; border-radius: 5px;">
                                        <strong>📊 First Chunk Speed:</strong> ${firstChunkPercent}% of total time<br>
                                        <div style="background: #e1e5e9; height: 10px; border-radius: 5px; margin-top: 5px; overflow: hidden;">
                                            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); height: 100%; width: ${firstChunkPercent}%;"></div>
                                        </div>
                                    </div>
                                </div>
                                <a href="${audioUrl}" download="speech.mp3" class="btn">💾 Download</a>
                            `;
                            
                            let statusMessage = `✅ Streaming complete! First chunk: ${firstChunkDuration}s, Total: ${totalGenTime}s (${chunkCount} chunks)`;
                            if (emotionText && emotionText.trim()) {
                                statusMessage += ` 😊 Emotion: "${emotionText}" (${emotionWeight})`;
                            }
                            showStatus(statusMessage, 'success');
                            return;
                        }
                        }
                    }
                } catch (streamError) {
                    console.error('Streaming error:', streamError);
                    showStatus(`Network error: ${streamError.message}`, 'error');
                }
            }

            // Toggle streaming settings visibility
            document.getElementById('streamingMode').addEventListener('change', function() {
                const streamingSettings = document.getElementById('streamingSettings');
                if (this.checked) {
                    streamingSettings.style.display = 'block';
                } else {
                    streamingSettings.style.display = 'none';
                }
            });

            // Load speakers on page load
            document.addEventListener('DOMContentLoaded', function() {
                loadSpeakers();
            });
        </script>
    </body>
    </html>
    """

# Health check endpoint
# Health check removed - use /server_info endpoint instead

# Utility endpoints
@app.post("/api/estimate_duration")
async def api_estimate_duration(request: Request):
    """API: Estimate speech duration from text"""
    try:
        data = await request.json()
        text = data.get("text", "")
        language = data.get("language", "auto")
        
        if not text or not text.strip():
            return JSONResponse(content={
                "status": "error",
                "message": "No text provided"
            })
        
        duration_ms = estimate_speech_duration(text, language)
        duration_s = duration_ms / 1000.0
        
        # Detect language for display
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        detected_lang = "Chinese" if chinese_chars / max(len(text), 1) > 0.3 else "English"
        
        return JSONResponse(content={
            "status": "success",
            "duration_ms": duration_ms,
            "duration_s": round(duration_s, 1),
            "detected_language": detected_lang,
            "char_count": len(text)
        })
        
    except Exception as e:
        print(f"❌ Error estimating duration: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.post("/api/clear_outputs")
async def api_clear_outputs():
    """API: Clear all generated output files"""
    try:
        outputs_dir = "outputs"
        
        if not os.path.exists(outputs_dir):
            return {
                "status": "success",
                "message": "Outputs directory does not exist",
                "files_deleted": 0,
                "space_freed_mb": 0
            }
        
        # Count files and size before deletion
        files_deleted = 0
        total_size = 0
        
        # Remove all files in outputs directory and subdirectories
        # Run file deletion in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        
        def delete_files_sync():
            deleted = 0
            size = 0
            for root, dirs, files in os.walk(outputs_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        file_size = os.path.getsize(file_path)
                        os.remove(file_path)
                        deleted += 1
                        size += file_size
                        print(f"🗑️ Deleted: {file_path}")
                    except Exception as e:
                        print(f"⚠️ Failed to delete {file_path}: {e}")
            return deleted, size
        
        files_deleted, total_size = await loop.run_in_executor(executor, delete_files_sync)
        
        space_freed_mb = total_size / (1024 * 1024)
        
        return {
            "status": "success",
            "message": f"Successfully cleared outputs directory",
            "files_deleted": files_deleted,
            "space_freed_mb": round(space_freed_mb, 2)
        }
        
    except Exception as e:
        print(f"❌ Error clearing outputs: {e}")
        traceback.print_exc()
        return {
            "status": "error",
            "message": f"Failed to clear outputs: {str(e)}"
        }


@app.get("/api/prompt_templates")
async def api_prompt_templates():
    """Return the default Gemini prompt templates so users can run them elsewhere."""
    return {
        "translation": TRANSLATION_PROMPT_TEMPLATE,
        "transcription": TRANSCRIPTION_PROMPT_TEMPLATE,
    }


@app.post("/api/translate_segments")
async def api_translate_segments(
    request: Request,
    dest_language: Optional[str] = Form(None),
    response_format: Optional[str] = Form(None),
    bitrate: Optional[str] = Form(None),
    audio: Optional[str] = Form(None),
    audio_mime_type: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    translate_text: Optional[bool] = Form(True),
    gemini_model: Optional[str] = Form(None),
    gemini_api_key: Optional[str] = Form(None),
    enhance_voice: Optional[bool] = Form(False),
    super_resolution_voice: Optional[bool] = Form(False),
    merge_backing_track: Optional[bool] = Form(False),
    min_speech_ms: Optional[int] = Form(None),
    max_merge_ms: Optional[int] = Form(None),
    segments_json: Optional[str] = Form(None),
    audio_file: Optional[UploadFile] = File(None),
):
    """API: Prepare translation segments for advanced translate/edit workflow."""
    try:
        payload: Optional[Dict[str, Any]] = None
        dest_language_value = dest_language
        response_format_value = response_format
        bitrate_value = bitrate
        audio_reference = audio
        audio_mime_type_value = audio_mime_type
        prompt_override = prompt
        translate_flag_value = translate_text
        gemini_model_value = gemini_model
        gemini_api_key_value = gemini_api_key
        enhance_voice_value = enhance_voice
        super_resolution_voice_value = super_resolution_voice
        merge_backing_track_value = merge_backing_track
        min_speech_duration_value = min_speech_ms
        max_merge_interval_value = max_merge_ms
        segments_override_value = segments_json
        min_speech_duration_value = min_speech_ms
        max_merge_interval_value = max_merge_ms
        segments_override_value = segments_json
        min_speech_duration_value = min_speech_ms
        max_merge_interval_value = max_merge_ms
        merge_backing_track_value = merge_backing_track
        min_speech_duration_value = min_speech_ms
        max_merge_interval_value = max_merge_ms
        merge_backing_track_value = merge_backing_track
        merge_backing_track_value = merge_backing_track
        merge_backing_track_value = merge_backing_track

        content_type = request.headers.get("content-type", "")
        if (
            dest_language is None
            and not audio_reference
            and audio_file is None
            and "application/json" in content_type.lower()
        ):
            try:
                payload = await request.json()
            except Exception as exc:
                return JSONResponse(
                    status_code=400,
                    content={"status": "error", "message": f"Invalid JSON payload: {str(exc)}"},
                )

        if payload is not None:
            dest_language_value = payload.get("dest_language", dest_language_value)
            response_format_value = payload.get("response_format", response_format_value)
            bitrate_value = payload.get("bitrate", bitrate_value)
            audio_reference = payload.get("audio", audio_reference)
            audio_mime_type_value = payload.get("audio_mime_type", audio_mime_type_value)
            prompt_override = payload.get("prompt", prompt_override)
            translate_flag_value = payload.get("translate", translate_flag_value)
            gemini_model_value = payload.get("gemini_model", gemini_model_value)
            gemini_api_key_value = payload.get("gemini_api_key", gemini_api_key_value)
            enhance_voice_value = payload.get("enhance_voice", enhance_voice_value)
            super_resolution_voice_value = payload.get("super_resolution_voice", super_resolution_voice_value)
            merge_backing_track_value = payload.get("merge_backing_track", merge_backing_track_value)
            min_speech_duration_value = payload.get("min_speech_ms", min_speech_duration_value)
            max_merge_interval_value = payload.get("max_merge_ms", max_merge_interval_value)
            segments_override_value = payload.get("segments_json", segments_override_value)

        dest_language_value = (dest_language_value or "").strip()
        if not dest_language_value:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Destination language (dest_language) is required."},
            )

        response_format_value = (response_format_value or TRANSLATE_DEFAULT_OUTPUT_FORMAT).lower()
        allowed_formats = {"mp3", "wav", "flac", "aac", "opus", "ogg", "webm"}
        if response_format_value not in allowed_formats:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": f"Unsupported response_format '{response_format_value}'. Allowed: {', '.join(sorted(allowed_formats))}",
                },
            )

        bitrate_value = bitrate_value or TRANSLATE_DEFAULT_BITRATE
        translate_enabled = _coerce_to_bool(translate_flag_value if translate_flag_value is not None else True)
        apply_enhancement = _coerce_to_bool(enhance_voice_value)
        apply_super_resolution = _coerce_to_bool(super_resolution_voice_value)
        requested_merge_backing = _coerce_to_bool(merge_backing_track_value)
        if requested_merge_backing and not apply_enhancement:
            print("⚠️ Merge-back requested without MossFormer2_SE_48K enhancement; ignoring request.")
            requested_merge_backing = False
        min_speech_duration = _coerce_positive_int(
            min_speech_duration_value,
            MIN_SPEECH_DURATION_MS,
            min_value=500,
        )
        max_merge_interval = _coerce_positive_int(
            max_merge_interval_value,
            MAX_MERGE_INTERVAL_MS,
            min_value=50,
        )
        allowed_gemini_models = {"gemini-flash-latest", "gemini-2.5-pro"}
        gemini_model_value = (gemini_model_value or "").strip()
        if gemini_model_value and gemini_model_value not in allowed_gemini_models:
            gemini_model_value = ""
        resolved_gemini_model = gemini_model_value or _get_gemini_model_name()
        gemini_api_key_value = (gemini_api_key_value or "").strip()

        audio_io = await load_audio_bytes_from_request(audio_file, audio_reference)
        if audio_io is None:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "No audio provided for translation."},
            )

        audio_bytes = audio_io.read()
        if not audio_bytes:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Provided audio data is empty."},
            )

        input_mime_type = audio_mime_type_value or (audio_file.content_type if audio_file else None) or "audio/wav"
        audio_format = _guess_audio_format_from_mime(input_mime_type)

        try:
            audio_buffer = BytesIO(audio_bytes)
            if audio_format:
                original_audio = AudioSegment.from_file(audio_buffer, format=audio_format)
            else:
                original_audio = AudioSegment.from_file(audio_buffer)
        except Exception as exc:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": f"Failed to decode audio: {str(exc)}"},
            )

        backing_track_audio: Optional[AudioSegment] = None
        pre_clearvoice_mix_audio: Optional[AudioSegment] = original_audio if apply_enhancement else None

        processed_paths: Set[str] = set()
        if apply_enhancement or apply_super_resolution:
            if ClearVoice is None:
                return JSONResponse(
                    status_code=500,
                    content={
                        "status": "error",
                        "message": "ClearVoice package is required for enhancement or super-resolution.",
                    },
                )

            temp_input_path = None
            final_processed_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_input:
                    original_audio.export(tmp_input.name, format="wav")
                    temp_input_path = tmp_input.name
                processed_paths.add(temp_input_path)

                final_processed_path, clearvoice_paths, enhancement_output_path = await apply_clearvoice_processing(
                    temp_input_path,
                    apply_enhancement,
                    apply_super_resolution,
                )
                processed_paths.update(clearvoice_paths)
                processed_paths.add(final_processed_path)

                enhancement_audio = None
                if apply_enhancement:
                    enhancement_source = enhancement_output_path or final_processed_path
                    try:
                        enhancement_audio = AudioSegment.from_file(enhancement_source, format="wav")
                    except Exception as enhancement_load_error:
                        print(f"⚠️ Failed to load ClearVoice enhancement output: {enhancement_load_error}")
                original_audio = AudioSegment.from_file(final_processed_path, format="wav")
                if apply_enhancement and pre_clearvoice_mix_audio is not None:
                    backing_track_audio = _extract_backing_track_from_vocals(
                        pre_clearvoice_mix_audio,
                        enhancement_audio or original_audio,
                    )
                    if backing_track_audio is not None:
                        print("🎶 Extracted instrumental backing track via MossFormer2_SE_48K.")
            except Exception as cv_error:
                for path in processed_paths:
                    try:
                        os.remove(path)
                    except Exception:
                        pass
                return JSONResponse(
                    status_code=500,
                    content={"status": "error", "message": f"ClearVoice processing failed: {str(cv_error)}"},
                )
            finally:
                for path in processed_paths:
                    try:
                        os.remove(path)
                    except Exception:
                        pass

        with BytesIO() as processed_buffer:
            original_audio.export(processed_buffer, format="wav")
            processed_audio_bytes = processed_buffer.getvalue()
        gemini_mime_type = "audio/wav"
        merge_with_backing = requested_merge_backing and backing_track_audio is not None
        if requested_merge_backing and backing_track_audio is None:
            print("⚠️ Unable to merge with backing track because no instrumental was derived.")

        final_prompt = (prompt_override or "").strip()
        if not final_prompt:
            if translate_enabled:
                final_prompt = TRANSLATION_PROMPT_TEMPLATE.format(dest_language=dest_language_value)
            else:
                final_prompt = TRANSCRIPTION_PROMPT_TEMPLATE

        manual_chunk_data = None
        manual_segments_used = False
        if segments_override_value:
            try:
                manual_chunk_data = _parse_manual_segments_input(segments_override_value)
                manual_segments_used = True
            except ValueError as exc:
                return JSONResponse(
                    status_code=400,
                    content={"status": "error", "message": str(exc)},
                )

        if manual_chunk_data is not None:
            gemini_chunks = manual_chunk_data
        else:
            gemini_chunks = await _gemini_transcribe_translate(
                processed_audio_bytes,
                gemini_mime_type,
                dest_language_value,
                final_prompt,
                model_name=resolved_gemini_model,
                api_key_override=gemini_api_key_value or None,
            )
        segments = _prepare_translation_segments(
            original_audio,
            gemini_chunks,
            dest_language_value,
            min_speech_duration_ms=min_speech_duration,
            max_merge_interval_ms=max_merge_interval,
        )

        if not translate_enabled:
            for segment in segments:
                if segment.get("type") != "speech":
                    continue
                translated_text = (segment.get("translated_text") or "").strip()
                source_text = (segment.get("source_text") or "").strip()
                if not translated_text and source_text:
                    segment["translated_text"] = source_text

        session = await _create_translate_session(
            original_audio,
            dest_language_value,
            final_prompt,
            translate_enabled,
            response_format_value,
            bitrate_value,
            input_mime_type,
            {
                "enhancement": apply_enhancement,
                "super_resolution": apply_super_resolution,
            },
            segments,
            gemini_chunks,
            resolved_gemini_model,
            gemini_api_key_value or None,
            backing_track_audio=backing_track_audio,
            merge_with_backing=merge_with_backing,
        )
        ui_segments = _serialize_segments_for_ui(segments, original_audio)

        metadata = {
            "dest_language": dest_language_value,
            "segment_count": len(segments),
            "speech_segment_count": sum(1 for seg in segments if seg.get("type") == "speech"),
            "silence_segment_count": sum(1 for seg in segments if seg.get("type") == "silence"),
            "audio_duration_ms": len(original_audio),
            "translate_enabled": translate_enabled,
            "response_format": response_format_value,
            "bitrate": bitrate_value,
            "prompt": final_prompt,
            "gemini_model": resolved_gemini_model,
            "clearvoice": {
                "enhancement": apply_enhancement,
                "super_resolution": apply_super_resolution,
            },
            "backing_track": {
                "available": backing_track_audio is not None,
                "requested": requested_merge_backing,
                "merge_with_backing": merge_with_backing,
                "preview_url": f"/api/translate_backing_track/{session.session_id}" if backing_track_audio is not None else None,
            },
            "segment_rules": {
                "min_speech_ms": min_speech_duration,
                "max_merge_ms": max_merge_interval,
            },
            "manual_segments": manual_segments_used,
        }

        return JSONResponse(
            content={
                "status": "success",
                "session_id": session.session_id,
                "segments": ui_segments,
                "metadata": metadata,
            }
        )
    except RuntimeError as runtime_error:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(runtime_error)},
        )
    except Exception as exc:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Failed to prepare translation segments: {str(exc)}"},
        )


@app.post("/api/translate_generate_segments")
async def api_translate_generate_segments(payload: TranslateGenerateRequest):
    """API: Generate translated audio from edited segments."""
    try:
        session = await _get_translate_session(payload.session_id)
        if session is None:
            return JSONResponse(
                status_code=404,
                content={"status": "error", "message": "Translate session not found or expired."},
            )

        merge_preference = session.merge_with_backing
        if payload.merge_backing_track is not None:
            merge_preference = _coerce_to_bool(payload.merge_backing_track)
            session.merge_with_backing = merge_preference
        merge_with_backing = merge_preference and (session.backing_track_audio is not None)
        if merge_preference and session.backing_track_audio is None:
            print("⚠️ Merge-back preference is enabled but no backing track is stored for this session.")

        allowed_formats = {"mp3", "wav", "flac", "aac", "opus", "ogg", "webm"}
        response_format_value = (
            payload.response_format or session.response_format or TRANSLATE_DEFAULT_OUTPUT_FORMAT
        ).lower()
        if response_format_value not in allowed_formats:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": f"Unsupported response_format '{response_format_value}'. Allowed: {', '.join(sorted(allowed_formats))}",
                },
            )

        bitrate_value = payload.bitrate or session.bitrate or TRANSLATE_DEFAULT_BITRATE
        max_duration = len(session.original_audio)
        base_segment_map = {seg.get("index"): seg for seg in session.base_segments}

        final_segments: List[Dict[str, Any]] = []
        sanitized_segments: List[Dict[str, Any]] = []

        for seg_input in payload.segments:
            start_ms = max(0, int(seg_input.start_ms))
            end_ms = max(0, int(seg_input.end_ms))
            if end_ms > max_duration:
                end_ms = max_duration
            if end_ms <= start_ms:
                return JSONResponse(
                    status_code=400,
                    content={
                        "status": "error",
                        "message": f"Segment index {seg_input.index} has invalid timing (end <= start).",
                    },
                )
            duration_ms = end_ms - start_ms
            start_label = _format_ms_to_timestamp(start_ms)
            end_label = _format_ms_to_timestamp(end_ms)
            translated_text = (seg_input.translated_text or "").strip()
            source_text = (seg_input.source_text or "").strip()

            is_speech = seg_input.type == "speech"
            generate_flag = bool(seg_input.generate) if is_speech else False
            keep_original = not generate_flag if is_speech else True

            segment_payload: Dict[str, Any] = {
                "index": seg_input.index,
                "type": seg_input.type,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "duration_ms": duration_ms,
                "start": start_label,
                "end": end_label,
                "source_text": source_text,
                "translated_text": translated_text,
                "generate": generate_flag if is_speech else False,
                "keep_original": keep_original,
            }

            base_info = base_segment_map.get(seg_input.index)
            if base_info and base_info.get("text_keys"):
                segment_payload["text_keys"] = base_info["text_keys"]

            final_segments.append(segment_payload)

            sanitized_segment: Dict[str, Any] = {
                "index": seg_input.index,
                "type": seg_input.type,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "duration_ms": duration_ms,
                "start": start_label,
                "end": end_label,
                "source_text": source_text,
                "translated_text": translated_text,
            }
            if base_info and base_info.get("text_keys"):
                sanitized_segment["text_keys"] = base_info["text_keys"]
            sanitized_segments.append(sanitized_segment)

        final_segments.sort(key=lambda seg: (int(seg.get("start_ms", 0)), int(seg.get("index", 0))))
        sanitized_segments.sort(key=lambda seg: (int(seg.get("start_ms", 0)), int(seg.get("index", 0))))

        audio_payload, media_type, metadata = await _synthesize_translated_audio(
            session.original_audio,
            final_segments,
            session.dest_language,
            response_format=response_format_value,
            bitrate=bitrate_value,
            input_mime_type=session.input_mime_type,
            clearvoice_settings=session.clearvoice_settings,
            backing_track_audio=session.backing_track_audio,
            merge_with_backing=merge_with_backing,
        )

        metadata.setdefault("backing_track", {})["requested"] = session.merge_with_backing
        generated_count = sum(
            1 for seg in final_segments if seg.get("type") == "speech" and not seg.get("keep_original", False)
        )
        preserved_count = sum(
            1 for seg in final_segments if seg.get("type") == "speech" and seg.get("keep_original", False)
        )

        metadata["selected_generated_count"] = generated_count
        metadata["selected_preserved_count"] = preserved_count
        metadata["session_id"] = session.session_id
        metadata["gemini_model"] = session.gemini_model or _get_gemini_model_name()

        await _update_translate_session_segments(session.session_id, sanitized_segments)
        await _update_translate_session_metadata(
            session.session_id,
            response_format=response_format_value,
            bitrate=bitrate_value,
            gemini_model=session.gemini_model,
        )

        headers = {
            "Content-Disposition": f"attachment; filename=translated_speech.{response_format_value}",
            "X-Translation-Model": session.gemini_model or _get_gemini_model_name(),
            "X-Translation-Segments": str(len(final_segments)),
            "X-Translation-Generated": str(generated_count),
            "X-Translation-Preserved": str(preserved_count),
            "X-Translation-Input-Mime": session.input_mime_type or "",
            "X-Translate-Session": session.session_id,
        }
        clearvoice_settings = session.clearvoice_settings or {}
        if clearvoice_settings:
            headers["X-Translation-ClearVoice"] = (
                f"enhancement={str(clearvoice_settings.get('enhancement', False)).lower()};"
                f"super_resolution={str(clearvoice_settings.get('super_resolution', False)).lower()}"
            )
        headers["X-Translation-Backing"] = (
            f"available={str(bool(session.backing_track_audio)).lower()};merged={str(merge_with_backing).lower()}"
        )

        return Response(content=audio_payload, media_type=media_type, headers=headers)
    except RuntimeError as runtime_error:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(runtime_error)},
        )
    except Exception as exc:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Failed to synthesize translation: {str(exc)}"},
        )


@app.get("/api/translate_backing_track/{session_id}")
async def api_translate_backing_track(session_id: str):
    """API: Stream the stored instrumental backing track for an advanced translate session."""
    session = await _get_translate_session(session_id)
    if session is None:
        return JSONResponse(
            status_code=404,
            content={"status": "error", "message": "Translate session not found or expired."},
        )
    if session.backing_track_audio is None:
        return JSONResponse(
            status_code=404,
            content={"status": "error", "message": "No backing track is available for this session."},
        )

    try:
        buffer = BytesIO()
        session.backing_track_audio.export(buffer, format="wav")
        audio_bytes = buffer.getvalue()
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Failed to export backing track: {str(exc)}"},
        )

    headers = {
        "Content-Disposition": f'inline; filename="translate_backing_{session_id}.wav"',
        "Cache-Control": "no-store, no-cache",
    }
    return Response(content=audio_bytes, media_type="audio/wav", headers=headers)


@app.post("/api/translate_audio")
async def api_translate_audio(
    request: Request,
    dest_language: Optional[str] = Form(None),
    response_format: Optional[str] = Form(None),
    bitrate: Optional[str] = Form(None),
    audio: Optional[str] = Form(None),
    audio_mime_type: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    gemini_model: Optional[str] = Form(None),
    gemini_api_key: Optional[str] = Form(None),
    enhance_voice: Optional[bool] = Form(False),
    super_resolution_voice: Optional[bool] = Form(False),
    merge_backing_track: Optional[bool] = Form(False),
    min_speech_ms: Optional[int] = Form(None),
    max_merge_ms: Optional[int] = Form(None),
    segments_json: Optional[str] = Form(None),
    audio_file: Optional[UploadFile] = File(None),
):
    """API: Translate speech audio to a target language and return synthesized audio."""
    try:
        payload: Optional[Dict[str, Any]] = None
        dest_language_value = dest_language
        audio_reference = audio
        audio_mime_type_value = audio_mime_type
        prompt_override = prompt
        response_format_value = response_format
        bitrate_value = bitrate
        gemini_model_value = gemini_model
        gemini_api_key_value = gemini_api_key
        enhance_voice_value = enhance_voice
        super_resolution_voice_value = super_resolution_voice
        merge_backing_track_value = merge_backing_track
        min_speech_duration_value = min_speech_ms
        max_merge_interval_value = max_merge_ms
        segments_override_value = segments_json

        content_type = request.headers.get("content-type", "")
        if (
            dest_language is None
            and not audio_reference
            and audio_file is None
            and "application/json" in content_type.lower()
        ):
            try:
                payload = await request.json()
            except Exception as exc:
                return JSONResponse(
                    status_code=400,
                    content={"status": "error", "message": f"Invalid JSON payload: {str(exc)}"},
                )

        if payload is not None:
            try:
                translate_req = TranslateRequest(**payload)
            except Exception as exc:
                return JSONResponse(
                    status_code=400,
                    content={"status": "error", "message": f"Invalid translate request: {str(exc)}"},
                )
            dest_language_value = translate_req.dest_language
            audio_reference = translate_req.audio or audio_reference
            audio_mime_type_value = translate_req.audio_mime_type or audio_mime_type_value
            prompt_override = translate_req.prompt or prompt_override
            response_format_value = translate_req.response_format or response_format_value
            bitrate_value = translate_req.bitrate or bitrate_value
            gemini_model_value = translate_req.gemini_model or gemini_model_value
            gemini_api_key_value = translate_req.gemini_api_key or gemini_api_key_value
            enhance_voice_value = translate_req.enhance_voice if translate_req.enhance_voice is not None else enhance_voice_value
            super_resolution_voice_value = (
                translate_req.super_resolution_voice if translate_req.super_resolution_voice is not None else super_resolution_voice_value
            )
            merge_backing_track_value = (
                translate_req.merge_backing_track if translate_req.merge_backing_track is not None else merge_backing_track_value
            )
            min_speech_duration_value = (
                translate_req.min_speech_ms if translate_req.min_speech_ms is not None else min_speech_duration_value
            )
            max_merge_interval_value = (
                translate_req.max_merge_ms if translate_req.max_merge_ms is not None else max_merge_interval_value
            )
            segments_override_value = (
                translate_req.segments_json if translate_req.segments_json is not None else segments_override_value
            )
            min_speech_duration_value = (
                translate_req.min_speech_ms if translate_req.min_speech_ms is not None else min_speech_duration_value
            )
            max_merge_interval_value = (
                translate_req.max_merge_ms if translate_req.max_merge_ms is not None else max_merge_interval_value
            )

        dest_language_value = (dest_language_value or "").strip()
        if not dest_language_value:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Destination language (dest_language) is required."},
            )

        response_format_value = (response_format_value or TRANSLATE_DEFAULT_OUTPUT_FORMAT).lower()
        allowed_formats = {"mp3", "wav", "flac", "aac", "opus", "ogg", "webm"}
        if response_format_value not in allowed_formats:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": f"Unsupported response_format '{response_format_value}'. Allowed: {', '.join(sorted(allowed_formats))}",
                },
            )

        bitrate_value = bitrate_value or TRANSLATE_DEFAULT_BITRATE
        apply_enhancement = _coerce_to_bool(enhance_voice_value)
        apply_super_resolution = _coerce_to_bool(super_resolution_voice_value)
        requested_merge_backing = _coerce_to_bool(merge_backing_track_value)
        if requested_merge_backing and not apply_enhancement:
            print("⚠️ Merge-back requested without MossFormer2_SE_48K enhancement; ignoring request.")
            requested_merge_backing = False
        min_speech_duration = _coerce_positive_int(
            min_speech_duration_value,
            MIN_SPEECH_DURATION_MS,
            min_value=500,
        )
        max_merge_interval = _coerce_positive_int(
            max_merge_interval_value,
            MAX_MERGE_INTERVAL_MS,
            min_value=50,
        )
        allowed_gemini_models = {"gemini-flash-latest", "gemini-2.5-pro"}
        gemini_model_value = (gemini_model_value or "").strip()
        if gemini_model_value and gemini_model_value not in allowed_gemini_models:
            gemini_model_value = ""
        resolved_gemini_model = gemini_model_value or _get_gemini_model_name()
        gemini_api_key_value = (gemini_api_key_value or "").strip()

        audio_io = await load_audio_bytes_from_request(audio_file, audio_reference)
        if audio_io is None:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "No audio provided for translation."},
            )

        audio_bytes = audio_io.read()
        if not audio_bytes:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Provided audio data is empty."},
            )

        input_mime_type = audio_mime_type_value or (audio_file.content_type if audio_file else None) or "audio/wav"
        audio_format = _guess_audio_format_from_mime(input_mime_type)

        try:
            audio_buffer = BytesIO(audio_bytes)
            if audio_format:
                original_audio = AudioSegment.from_file(audio_buffer, format=audio_format)
            else:
                original_audio = AudioSegment.from_file(audio_buffer)
        except Exception as exc:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": f"Failed to decode audio: {str(exc)}"},
            )

        backing_track_audio: Optional[AudioSegment] = None
        pre_clearvoice_mix_audio: Optional[AudioSegment] = original_audio if apply_enhancement else None

        processed_paths: Set[str] = set()
        if apply_enhancement or apply_super_resolution:
            if ClearVoice is None:
                return JSONResponse(
                    status_code=500,
                    content={
                        "status": "error",
                        "message": "ClearVoice package is required for enhancement or super-resolution.",
                    },
                )

            temp_input_path = None
            final_processed_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_input:
                    original_audio.export(tmp_input.name, format="wav")
                    temp_input_path = tmp_input.name
                processed_paths.add(temp_input_path)

                final_processed_path, clearvoice_paths, enhancement_output_path = await apply_clearvoice_processing(
                    temp_input_path,
                    apply_enhancement,
                    apply_super_resolution,
                )
                processed_paths.update(clearvoice_paths)
                processed_paths.add(final_processed_path)

                original_audio = AudioSegment.from_file(final_processed_path, format="wav")
                enhancement_audio = None
                if apply_enhancement:
                    enhancement_source = enhancement_output_path or final_processed_path
                    try:
                        enhancement_audio = AudioSegment.from_file(enhancement_source, format="wav")
                    except Exception as enhancement_load_error:
                        print(f"⚠️ Failed to load ClearVoice enhancement output: {enhancement_load_error}")
                if apply_enhancement and pre_clearvoice_mix_audio is not None:
                    backing_track_audio = _extract_backing_track_from_vocals(
                        pre_clearvoice_mix_audio,
                        enhancement_audio or original_audio,
                    )
                    if backing_track_audio is not None:
                        print("🎶 Extracted instrumental backing track via MossFormer2_SE_48K (direct translate).")
            except Exception as cv_error:
                for path in processed_paths:
                    try:
                        os.remove(path)
                    except Exception:
                        pass
                return JSONResponse(
                    status_code=500,
                    content={"status": "error", "message": f"ClearVoice processing failed: {str(cv_error)}"},
                )
            finally:
                for path in processed_paths:
                    try:
                        os.remove(path)
                    except Exception:
                        pass

        with BytesIO() as processed_buffer:
            original_audio.export(processed_buffer, format="wav")
            processed_audio_bytes = processed_buffer.getvalue()
        gemini_mime_type = "audio/wav"
        merge_with_backing = requested_merge_backing and backing_track_audio is not None
        if requested_merge_backing and backing_track_audio is None:
            print("⚠️ Unable to merge with backing track because no instrumental was derived.")

        final_prompt = (prompt_override or "").strip()
        if not final_prompt:
            final_prompt = TRANSLATION_PROMPT_TEMPLATE.format(dest_language=dest_language_value)

        manual_chunk_data = None
        manual_segments_used = False
        if segments_override_value:
            try:
                manual_chunk_data = _parse_manual_segments_input(segments_override_value)
                manual_segments_used = True
            except ValueError as exc:
                return JSONResponse(
                    status_code=400,
                    content={"status": "error", "message": str(exc)},
                )
        if manual_chunk_data is not None:
            gemini_chunks = manual_chunk_data
        else:
            gemini_chunks = await _gemini_transcribe_translate(
                processed_audio_bytes,
                gemini_mime_type,
                dest_language_value,
                final_prompt,
                model_name=resolved_gemini_model,
                api_key_override=gemini_api_key_value or None,
            )
        segments = _prepare_translation_segments(
            original_audio,
            gemini_chunks,
            dest_language_value,
            min_speech_duration_ms=min_speech_duration,
            max_merge_interval_ms=max_merge_interval,
        )

        audio_payload, media_type, metadata = await _synthesize_translated_audio(
            original_audio,
            segments,
            dest_language_value,
            response_format=response_format_value,
            bitrate=bitrate_value,
            input_mime_type=input_mime_type,
            clearvoice_settings={
                "enhancement": apply_enhancement,
                "super_resolution": apply_super_resolution,
            },
            backing_track_audio=backing_track_audio,
            merge_with_backing=merge_with_backing,
        )

        metadata.setdefault("backing_track", {}).update(
            {
                "requested": requested_merge_backing,
                "available": backing_track_audio is not None,
                "merged": merge_with_backing,
            }
        )
        metadata["segment_rules"] = {
            "min_speech_ms": min_speech_duration,
            "max_merge_ms": max_merge_interval,
        }
        metadata["manual_segments"] = manual_segments_used
        metadata["segment_rules"] = {
            "min_speech_ms": min_speech_duration,
            "max_merge_ms": max_merge_interval,
        }
        metadata["gemini_model"] = resolved_gemini_model

        headers = {
            "Content-Disposition": f"attachment; filename=translated_speech.{response_format_value}",
            "X-Translation-Model": resolved_gemini_model,
            "X-Translation-Segments": str(metadata.get("segment_count", len(segments))),
            "X-Translation-Speech-Segments": str(metadata.get("speech_segment_count", 0)),
            "X-Translation-Silence-Segments": str(metadata.get("silence_segment_count", 0)),
            "X-Translation-Input-Mime": input_mime_type or "",
            "X-Translation-ClearVoice": f"enhancement={str(apply_enhancement).lower()};super_resolution={str(apply_super_resolution).lower()}",
        }
        headers["X-Translation-Backing"] = (
            f"available={str(bool(backing_track_audio)).lower()};merged={str(merge_with_backing).lower()}"
        )

        print(
            f"✅ Translation complete: {metadata.get('segment_count', len(segments))} segments "
            f"({metadata.get('speech_segment_count', 0)} speech / {metadata.get('silence_segment_count', 0)} silence), "
            f"dest_language={dest_language_value}, format={response_format_value}, input_mime={input_mime_type}, "
            f"gemini_model={resolved_gemini_model}"
        )

        return Response(content=audio_payload, media_type=media_type, headers=headers)

    except RuntimeError as runtime_error:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(runtime_error)},
        )
    except Exception as exc:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Translation failed: {str(exc)}"},
        )


# API Helper Functions (matching deploy_vllm_indextts.py exactly)
async def get_audio_bytes_from_url(url: str) -> bytes:
    """Download audio from URL"""
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            if response.status_code != 200:
                raise HTTPException(status_code=400, detail="Cannot download audio from URL")
            return response.content
    except ImportError:
        # Fallback if httpx is not available - run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        
        def download_sync():
            try:
                with urllib.request.urlopen(url) as response:
                    if response.status != 200:
                        raise Exception("Cannot download audio from URL")
                    return response.read()
            except Exception as e:
                raise Exception(f"Failed to download audio: {str(e)}")
        
        try:
            return await loop.run_in_executor(executor, download_sync)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

async def load_base64_or_url(audio: str) -> BytesIO:
    """Load audio from base64 or URL"""
    if audio.startswith("http://") or audio.startswith("https://"):
        audio_bytes = await get_audio_bytes_from_url(audio)
    else:
        payload = audio.strip()
        if payload.startswith("data:"):
            try:
                _, payload = payload.split(",", 1)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid data URI audio: {str(e)}") from e
        try:
            audio_bytes = base64.b64decode(payload)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 audio data: {str(e)}")
    
    return BytesIO(audio_bytes)

async def load_audio_bytes_from_request(audio_file, audio):
    """Load audio bytes from file or reference audio string"""
    if audio_file is None:
        if audio is None:
            return None
        return await load_base64_or_url(audio)
    else:
        content = await audio_file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Reference audio file is empty")
        return BytesIO(content)

# API Compatibility Endpoints
@app.post("/add_speaker")
async def add_speaker(
    background_tasks: BackgroundTasks,
    name: str = Form(..., description="The name of the speaker"),
    audio: Optional[str] = Form(None, description="Reference audio URL or base64"),
    reference_text: Optional[str] = Form(None, description="Optional transcript"),
    audio_file: Optional[UploadFile] = File(None, description="Upload reference audio file"),
    enhance_voice: bool = Form(False, description="Apply ClearVoice MossFormer2_SE_48K enhancement"),
    super_resolution_voice: bool = Form(False, description="Apply ClearVoice MossFormer2_SR_48K super-resolution"),
):
    """API: Add a new speaker"""
    try:
        print(f"🎭 API: Adding speaker '{name}'")
        print(f"🔍 Debug: audio_file={audio_file}, audio={audio is not None}, reference_text={reference_text}")
        
        if not speaker_api:
            return JSONResponse(
                status_code=500,
                content={"success": False, "error": "Speaker manager not initialized"}
            )
        
        # Load audio from file or reference string (matching deploy_vllm_indextts.py)
        try:
            audio_io = await load_audio_bytes_from_request(audio_file, audio)
            if audio_io is None:
                print(f"❌ API: No audio provided for speaker '{name}'")
                return JSONResponse(
                    status_code=400,
                    content={"success": False, "error": "No audio provided"}
                )
        except Exception as audio_error:
            print(f"❌ API: Audio loading failed for speaker '{name}': {audio_error}")
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": f"Audio loading failed: {str(audio_error)}"}
            )
        
        # Get audio data and filename
        audio_data = audio_io.read()
        filename = audio_file.filename if audio_file else f"{name}_reference.wav"
        
        apply_enhancement = bool(enhance_voice)
        apply_super_resolution_flag = bool(super_resolution_voice)
        print(f"🎚️ API: ClearVoice options -> enhancement={apply_enhancement}, super_resolution={apply_super_resolution_flag}")
        
        if (apply_enhancement or apply_super_resolution_flag) and ClearVoice is None:
            error_msg = "ClearVoice is required for enhancement or super-resolution. Install the `clearvoice` package to enable these options."
            print(f"❌ API: {error_msg}")
            return JSONResponse(
                status_code=500,
                content={"success": False, "error": error_msg}
            )
        
        # Add speaker using SpeakerPresetManager (handles ClearVoice processing internally)
        result = await speaker_api.add_speaker(
            name,
            [audio_data],
            [filename],
            apply_enhancement=apply_enhancement,
            apply_super_resolution=apply_super_resolution_flag,
        )
        
        if result["status"] == "success":
            payload = {"success": True, "role": name}
            if result.get("clearvoice"):
                payload["clearvoice"] = result["clearvoice"]
            return JSONResponse(content=payload)
        else:
            return JSONResponse(
                status_code=500,
                content={"success": False, "error": result["message"]}
            )
            
    except Exception as e:
        error_msg = f"Failed to add speaker '{name}': {str(e)}"
        print(f"❌ API: {error_msg}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": error_msg}
        )

@app.post("/delete_speaker")
async def delete_speaker(
    background_tasks: BackgroundTasks,
    name: str = Form(..., description="The name of the speaker")
):
    """API: Delete a speaker"""
    try:
        print(f"🗑️ API: Deleting speaker '{name}'")
        
        if not speaker_api:
            return JSONResponse(
                status_code=500,
                content={"success": False, "error": "Speaker manager not initialized"}
            )
        
        result = await speaker_api.delete_speaker(name)
        
        if result["status"] == "success":
            return JSONResponse(content={"success": True, "role": name})
        else:
            return JSONResponse(
                status_code=500,
                content={"success": False, "error": result["message"]}
            )
            
    except Exception as e:
        error_msg = f"Failed to delete speaker '{name}': {str(e)}"
        print(f"❌ API: {error_msg}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": error_msg}
        )

@app.get("/audio_roles")
async def audio_roles():
    """API: List available speakers"""
    try:
        print("📋 API: Listing audio roles")
        
        if not speaker_api:
            return JSONResponse(content={"success": False, "roles": []})
        
        speakers_data = await speaker_api.list_speakers()
        
        if speakers_data["status"] == "success":
            roles = list(speakers_data["speakers"].keys())
            return JSONResponse(content={"success": True, "roles": roles})
        else:
            return JSONResponse(content={"success": False, "roles": []})
            
    except Exception as e:
        error_msg = f"Failed to list audio roles: {str(e)}"
        print(f"❌ API: {error_msg}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": error_msg}
        )

@app.post("/speak")
async def speak(req: SpeakRequest):
    """API: Generate speech using registered speaker"""
    try:
        print(f"🎭 API: Speaking with '{req.name}' - '{req.text[:50]}...'")
        
        if not req.name:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Speaker name is required"}
            )
        
        # Simple speaker validation to prevent failures
        if speaker_api and not speaker_api.speaker_exists(req.name):
            speakers_data = await speaker_api.list_speakers()
            available_roles = list(speakers_data.get("speakers", {}).keys())
            error_msg = f"'{req.name}' is not in the list of existing roles: {', '.join(available_roles)}"
            return JSONResponse(
                status_code=500,
                content={"success": False, "error": error_msg}
            )
        
        tts = tts_manager.get_tts()
        
        # Generate speech
        output_path = os.path.join("outputs", f"speak_{uuid.uuid4().hex}.wav")
        
        # Check if emotion text is provided and not empty
        use_emotion_text = req.emotion_text and req.emotion_text.strip() != ""
        
        result = await tts.infer(
            spk_audio_prompt="",
            text=req.text,
            output_path=output_path,
            speaker_preset=req.name,
            use_emo_text=use_emotion_text,
            emo_text=req.emotion_text if use_emotion_text else None,
            emo_alpha=req.emotion_weight,
            speech_length=req.speech_length,
            diffusion_steps=req.diffusion_steps,
            max_text_tokens_per_sentence=req.max_text_tokens_per_sentence,
            verbose=cmd_args.verbose
        )
        
        # Convert to requested format and return as bytes (matching deploy_vllm_indextts.py)
        if req.response_format != "wav":
            # Read audio data once
            audio_data, sample_rate = await async_audio_read(result)
            # Convert to requested format and get bytes
            audio_bytes, media_type, _ = await convert_audio_to_format(
                audio_data, sample_rate, req.response_format, "128k"
            )
        else:
            # Read WAV file as bytes
            audio_bytes = await async_read_file(result)
            media_type = "audio/wav"
        
        # Set content type (matching deploy_vllm_indextts.py)
        content_type_map = {
            "mp3": "audio/mpeg",
            "opus": "audio/opus", 
            "aac": "audio/aac",
            "flac": "audio/flac",
            "wav": "audio/wav",
            "pcm": "audio/pcm",
        }
        content_type = content_type_map.get(req.response_format, f"audio/{req.response_format}")
        
        print(f"✅ API: Generated {len(audio_bytes)} bytes of {req.response_format.upper()} audio")
        
        # Cleanup temporary file
        await async_remove_file(result)
        
        # Return Response with bytes (matching deploy_vllm_indextts.py format exactly)
        return Response(
            content=audio_bytes,
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename=speech.{req.response_format}",
                "Cache-Control": "no-cache",
            }
        )
        
    except Exception as e:
        import traceback
        error_msg = f"Voice synthesis failed: {str(e)}"
        print(f"❌ API: {error_msg}")
        print(f"🔍 Full traceback:")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": error_msg}
        )

def parse_clone_form(
    text: str = Form(...),
    reference_audio: Optional[str] = Form(None),
    reference_text: Optional[str] = Form(None),
    pitch: Optional[Literal["very_low", "low", "moderate", "high", "very_high"]] = Form(None),
    speed: Optional[Literal["very_low", "low", "moderate", "high", "very_high"]] = Form(None),
    temperature: float = Form(0.9),
    top_k: int = Form(50),
    top_p: float = Form(0.95),
    repetition_penalty: float = Form(1.0),
    max_tokens: int = Form(4096),
    length_threshold: int = Form(50),
    window_size: int = Form(50),
    stream: bool = Form(False),
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Form("mp3"),
    emotion_text: Optional[str] = Form(""),
    emotion_weight: float = Form(0.6),
    diffusion_steps: int = Form(10),
    max_text_tokens_per_sentence: int = Form(120),
):
    return CloneRequest(
        text=text, reference_audio=reference_audio, reference_text=reference_text,
        pitch=pitch, speed=speed, temperature=temperature, top_k=top_k, top_p=top_p,
        repetition_penalty=repetition_penalty, max_tokens=max_tokens,
        length_threshold=length_threshold, window_size=window_size,
        stream=stream, response_format=response_format,
        emotion_text=emotion_text, emotion_weight=emotion_weight,
        diffusion_steps=diffusion_steps, max_text_tokens_per_sentence=max_text_tokens_per_sentence
    )

@app.post("/clone_voice")
async def clone_voice(
    req: CloneRequest = Depends(parse_clone_form),
    reference_audio_file: Optional[UploadFile] = File(None),
):
    """API: Clone voice using reference audio"""
    try:
        print(f"🎵 API: Cloning voice - '{req.text[:50]}...'")
        
        # Load reference audio (matching deploy_vllm_indextts.py)
        audio_io = await load_audio_bytes_from_request(reference_audio_file, req.reference_audio)
        if audio_io is None:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "No reference audio provided"}
            )
        
        # Save reference audio to temporary file
        audio_data = audio_io.read()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = tmp_file.name
        await async_write_file(tmp_path, audio_data)
        
        try:
            tts = tts_manager.get_tts()
            
            # Generate speech using reference audio
            output_path = os.path.join("outputs", f"clone_{uuid.uuid4().hex}.wav")
            
            # Check if emotion text is provided and not empty
            use_emotion_text = req.emotion_text and req.emotion_text.strip() != ""
            
            result = await tts.infer(
                spk_audio_prompt=tmp_path,
                text=req.text,
                output_path=output_path,
                use_emo_text=use_emotion_text,
                emo_text=req.emotion_text if use_emotion_text else None,
                emo_alpha=req.emotion_weight,
                speech_length=req.speech_length,
                diffusion_steps=req.diffusion_steps,
                max_text_tokens_per_sentence=req.max_text_tokens_per_sentence,
                verbose=cmd_args.verbose
            )
            
            # Convert to requested format and return as bytes (matching deploy_vllm_indextts.py)
            if req.response_format != "wav":
                # Read audio data once
                audio_data, sample_rate = await async_audio_read(result)
                # Convert to requested format and get bytes
                audio_bytes, media_type, _ = await convert_audio_to_format(
                    audio_data, sample_rate, req.response_format, "128k"
                )
            else:
                # Read WAV file as bytes
                audio_bytes = await async_read_file(result)
                media_type = "audio/wav"
            
            # Set content type (matching deploy_vllm_indextts.py)
            content_type_map = {
                "mp3": "audio/mpeg",
                "opus": "audio/opus",
                "aac": "audio/aac", 
                "flac": "audio/flac",
                "wav": "audio/wav",
                "pcm": "audio/pcm",
            }
            content_type = content_type_map.get(req.response_format, f"audio/{req.response_format}")
            
            print(f"✅ API: Cloned voice - {len(audio_bytes)} bytes of {req.response_format.upper()}")
            
            # Cleanup temporary file
            await async_remove_file(result)
            
            return Response(
                content=audio_bytes,
                media_type=content_type,
                headers={
                    "Content-Disposition": f"attachment; filename=speech.{req.response_format}",
                    "Cache-Control": "no-cache",
                }
            )
            
        finally:
            # Cleanup
            await async_remove_file(tmp_path)
                
    except Exception as e:
        error_msg = f"Failed to clone voice: {str(e)}"
        print(f"❌ API: {error_msg}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": error_msg}
        )

@app.get("/server_info")
async def server_info():
    """API: Get server information"""
    try:
        if not speaker_api:
            roles = []
        else:
            speakers_data = await speaker_api.list_speakers()
            roles = list(speakers_data["speakers"].keys()) if speakers_data["status"] == "success" else []
        
        return JSONResponse(content={
            "success": True,
            "info": {
                "model": "IndexTTS-vLLM-v2",
                "roles": roles,
                "sample_rate": 22050,
                "engine": "vLLM v2",
                "chinese_support": True,
                "speaker_presets": True,
                "speaker_manager": "SpeakerPresetManager"
            }
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": f"Failed to get server info: {str(e)}"
            }
        )

@app.post("/speak_stream")
async def speak_stream(req: SpeakRequest):
    """API: Generate speech using registered speaker with streaming"""
    from fastapi.responses import StreamingResponse
    
    try:
        print(f"🎭 API Streaming: Speaking with '{req.name}' - '{req.text[:50]}...'")
        
        if not req.name:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Speaker name is required"}
            )
        
        # Simple speaker validation to prevent failures
        if speaker_api and not speaker_api.speaker_exists(req.name):
            speakers_data = await speaker_api.list_speakers()
            available_roles = list(speakers_data.get("speakers", {}).keys())
            error_msg = f"'{req.name}' is not in the list of existing roles: {', '.join(available_roles)}"
            return JSONResponse(
                status_code=500,
                content={"success": False, "error": error_msg}
            )
        
        tts = tts_manager.get_tts()
        
        # Check if emotion text is provided and not empty
        use_emotion_text = req.emotion_text and req.emotion_text.strip() != ""
        
        async def audio_stream_generator():
            """Generator that yields audio chunks as they are produced"""
            try:
                chunk_count = 0
                async for chunk_idx, wav_cpu, is_last in tts.infer_stream(
                    spk_audio_prompt="",
                    text=req.text,
                    speaker_preset=req.name,
                    use_emo_text=use_emotion_text,
                    emo_text=req.emotion_text if use_emotion_text else None,
                    emo_alpha=req.emotion_weight,
                    speech_length=req.speech_length,
                    diffusion_steps=req.diffusion_steps,
                    max_text_tokens_per_sentence=req.max_text_tokens_per_sentence,
                    first_chunk_max_tokens=40,  # Default first chunk size
                    verbose=cmd_args.verbose
                ):
                    chunk_count += 1
                    print(f"🎵 Streaming chunk {chunk_idx} (is_last={is_last})")
                    
                    # Convert tensor to WAV bytes
                    wav_data = wav_cpu.numpy().astype(np.int16)
                    
                    # Create WAV file in memory
                    with BytesIO() as wav_buffer:
                        sf.write(wav_buffer, wav_data.T, 22050, format='WAV')
                        wav_bytes = wav_buffer.getvalue()
                    
                    # Convert to requested format
                    if req.response_format == "wav":
                        audio_bytes = wav_bytes
                    else:
                        audio_segment = AudioSegment.from_wav(BytesIO(wav_bytes))
                        with BytesIO() as audio_buffer:
                            audio_segment.export(audio_buffer, format=req.response_format, bitrate="128k" if req.response_format == "mp3" else None)
                            audio_bytes = audio_buffer.getvalue()
                    
                    # Yield chunk with metadata header
                    header = f"CHUNK:{chunk_idx}:{len(audio_bytes)}:{'LAST' if is_last else 'MORE'}\n".encode('utf-8')
                    yield header + audio_bytes
                
                print(f"✅ Streaming complete: {chunk_count} chunks sent")
                
            except Exception as e:
                error_msg = f"ERROR:{str(e)}\n".encode('utf-8')
                print(f"❌ Streaming error: {e}")
                traceback.print_exc()
                yield error_msg
        
        return StreamingResponse(
            audio_stream_generator(),
            media_type="application/octet-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",  # Disable proxy buffering
            }
        )
        
    except Exception as e:
        import traceback
        error_msg = f"Voice synthesis streaming failed: {str(e)}"
        print(f"❌ API Streaming: {error_msg}")
        print(f"🔍 Full traceback:")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": error_msg}
        )

@app.post("/clone_voice_stream")
async def clone_voice_stream(
    req: CloneRequest = Depends(parse_clone_form),
    reference_audio_file: Optional[UploadFile] = File(None),
):
    """API: Clone voice using reference audio with streaming"""
    from fastapi.responses import StreamingResponse
    
    try:
        print(f"🎵 API Streaming: Cloning voice - '{req.text[:50]}...'")
        
        # Load reference audio (matching deploy_vllm_indextts.py)
        audio_io = await load_audio_bytes_from_request(reference_audio_file, req.reference_audio)
        if audio_io is None:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "No reference audio provided"}
            )
        
        # Save reference audio to temporary file
        audio_data = audio_io.read()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = tmp_file.name
        await async_write_file(tmp_path, audio_data)
        
        tts = tts_manager.get_tts()
        
        # Check if emotion text is provided and not empty
        use_emotion_text = req.emotion_text and req.emotion_text.strip() != ""
        
        async def audio_stream_generator():
            """Generator that yields audio chunks as they are produced"""
            try:
                chunk_count = 0
                async for chunk_idx, wav_cpu, is_last in tts.infer_stream(
                    spk_audio_prompt=tmp_path,
                    text=req.text,
                    use_emo_text=use_emotion_text,
                    emo_text=req.emotion_text if use_emotion_text else None,
                    emo_alpha=req.emotion_weight,
                    speech_length=req.speech_length,
                    diffusion_steps=req.diffusion_steps,
                    max_text_tokens_per_sentence=req.max_text_tokens_per_sentence,
                    first_chunk_max_tokens=40,  # Default first chunk size
                    verbose=cmd_args.verbose
                ):
                    chunk_count += 1
                    print(f"🎵 Streaming chunk {chunk_idx} (is_last={is_last})")
                    
                    # Convert tensor to WAV bytes
                    wav_data = wav_cpu.numpy().astype(np.int16)
                    
                    # Create WAV file in memory
                    with BytesIO() as wav_buffer:
                        sf.write(wav_buffer, wav_data.T, 22050, format='WAV')
                        wav_bytes = wav_buffer.getvalue()
                    
                    # Convert to requested format
                    if req.response_format == "wav":
                        audio_bytes = wav_bytes
                    else:
                        audio_segment = AudioSegment.from_wav(BytesIO(wav_bytes))
                        with BytesIO() as audio_buffer:
                            audio_segment.export(audio_buffer, format=req.response_format, bitrate="128k" if req.response_format == "mp3" else None)
                            audio_bytes = audio_buffer.getvalue()
                    
                    # Yield chunk with metadata header
                    header = f"CHUNK:{chunk_idx}:{len(audio_bytes)}:{'LAST' if is_last else 'MORE'}\n".encode('utf-8')
                    yield header + audio_bytes
                
                print(f"✅ Streaming complete: {chunk_count} chunks sent")
                
            except Exception as e:
                error_msg = f"ERROR:{str(e)}\n".encode('utf-8')
                print(f"❌ Streaming error: {e}")
                traceback.print_exc()
                yield error_msg
            finally:
                # Cleanup temporary file after streaming is complete
                await async_remove_file(tmp_path)
        
        return StreamingResponse(
            audio_stream_generator(),
            media_type="application/octet-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",  # Disable proxy buffering
            }
        )
                
    except Exception as e:
        error_msg = f"Failed to clone voice with streaming: {str(e)}"
        print(f"❌ API Streaming: {error_msg}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": error_msg}
        )

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting IndexTTS vLLM v2 FastAPI WebUI...")
    print(f"📁 Model directory: {cmd_args.model_dir}")
    print(f"🔧 GPU memory utilization: {cmd_args.gpu_memory_utilization}")
    print(f"🎯 FP16 mode: {cmd_args.is_fp16}")
    print(f"🌐 Server will start on {cmd_args.host}:{cmd_args.port}")
    print(f"🎯 Concurrent capacity: 100 requests (matching Modal deployment)")
    print(f"⚡ Single worker process for optimal GPU utilization")
    print(f"💡 Features:")
    print(f"   - IndexTTS vLLM v2 backend for ultra-fast inference")
    print(f"   - Speaker preset management with persistent storage")
    print(f"   - API compatibility for external integrations")
    print(f"   - Modern web interface with Chinese support")
    print(f"   - MP3 output for smaller file sizes")
    print(f"   - High concurrency support (100 concurrent connections)")
    print(f"   - Advanced translate/edit mode with segment editing")
    print(f"   - Gemini model selection (Flash vs Pro) with API key override")
    print(f"   - Per-segment generation control for efficient processing")
    
    uvicorn.run(
        app,
        host=cmd_args.host,
        port=cmd_args.port,
        log_level="info",
        workers=1,  # Single worker to match Modal's single container
        limit_concurrency=100,  # Match Modal's max_inputs=100
        limit_max_requests=None,  # No limit on total requests
        backlog=2048,  # Handle request queue efficiently
        timeout_keep_alive=300,  # Set timeout to 300 seconds
        h11_max_incomplete_event_size=16777216,  # 16MB for large audio uploads
        access_log=True  # Enable access logging for debugging
    )
