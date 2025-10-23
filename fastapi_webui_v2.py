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
"""

import os
import sys
import asyncio
import tempfile
import traceback
import uuid
from pathlib import Path
from typing import List, Dict, Optional, Literal
from contextlib import asynccontextmanager
from io import BytesIO
import base64
import urllib.request
from concurrent.futures import ThreadPoolExecutor


# Audio processing
import numpy as np
import soundfile as sf
from pydub import AudioSegment


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
    
    async def add_speaker(self, speaker_name: str, audio_files: List[bytes], filenames: List[str]) -> Dict[str, str]:
        """Add a new speaker with audio files"""
        try:
            # Check if speaker already exists
            existing_presets = self.preset_manager.list_presets()
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
            temp_dir.mkdir(exist_ok=True)
            
            temp_path = temp_dir / f"{speaker_name}_{filename}"
            await async_write_file(str(temp_path), audio_data)
            
            try:
                # Cut audio to 10 seconds if it exceeds the limit
                cut_temp_path = await async_cut_audio_to_duration(str(temp_path), max_duration=10.0)
                
                # Use SpeakerPresetManager to add the speaker
                success = self.preset_manager.add_speaker_preset(
                    preset_name=speaker_name,
                    audio_path=cut_temp_path,
                    description=f"Added via API with {len(audio_files)} audio files (auto-cut to 10s)"
                )
                
                if success:
                    return {
                        "status": "success", 
                        "message": f"Speaker '{speaker_name}' added successfully",
                        "info": "newly_added",
                        "audio_count": len(audio_files)
                    }
                else:
                    return {"status": "error", "message": f"Failed to add speaker '{speaker_name}'"}
            
            finally:
                # Clean up temporary files asynchronously
                try:
                    # Clean up original temp file (if different from cut file)
                    if temp_path.exists():
                        await async_remove_file(str(temp_path))
                    # Clean up cut file if it's different and exists
                    if cut_temp_path != str(temp_path) and os.path.exists(cut_temp_path):
                        await async_remove_file(cut_temp_path)
                except:
                    pass
            
        except Exception as e:
            return {"status": "error", "message": f"Failed to add speaker: {str(e)}"}
    
    async def delete_speaker(self, speaker_name: str) -> Dict[str, str]:
        """Delete a speaker"""
        try:
            success = self.preset_manager.delete_preset(speaker_name)
            
            if success:
                return {
                    "status": "success",
                    "message": f"Speaker '{speaker_name}' deleted successfully"
                }
            else:
                return {"status": "error", "message": f"Speaker '{speaker_name}' not found"}
            
        except Exception as e:
            return {"status": "error", "message": f"Failed to delete speaker: {str(e)}"}
    
    async def list_speakers(self) -> Dict[str, any]:
        """List all speakers with metadata"""
        try:
            presets = self.preset_manager.list_presets()
            
            speaker_info = {}
            for speaker_name, preset_data in presets.items():
                # Calculate cache file size
                cache_file = preset_data.get('cache_file', '')
                total_size = 0
                if cache_file and os.path.exists(cache_file):
                    total_size = os.path.getsize(cache_file)
                
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
            presets = self.preset_manager.list_presets()
            if speaker_name in presets:
                audio_path = presets[speaker_name].get('audio_path', '')
                if audio_path and os.path.exists(audio_path):
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

# FastAPI lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    # Startup
    print("🚀 Starting IndexTTS vLLM v2 FastAPI WebUI...")
    await tts_manager.initialize()
    yield
    # Shutdown (if needed)
    print("🔄 Shutting down IndexTTS vLLM v2...")
    # Shutdown the thread executor
    executor.shutdown(wait=True)

# Create FastAPI app
app = FastAPI(
    title="IndexTTS vLLM v2 FastAPI WebUI",
    description="Ultra-fast TTS with vLLM backend and speaker presets",
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
                <p class="subtitle">Ultra-Fast TTS with vLLM Backend / 超快速中文语音合成</p>
                <div>
                    <span class="performance-badge">⚡ vLLM v2 Backend</span>
                    <span class="performance-badge">🇨🇳 Chinese Support</span>
                    <span class="performance-badge">🎭 Speaker Presets</span>
                    <span class="performance-badge">🎵 MP3 Output</span>
                    <span class="performance-badge">🔌 API Integration</span>
                    <span class="performance-badge">😊 Emotion Text Control</span>
                    <span class="performance-badge">🌊 Streaming Mode</span>
                </div>
            </div>
            <div class="content">
                <div class="tabs">
                    <div class="tab active" onclick="switchTab('synthesis')">🎵 Speech Synthesis</div>
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
                                    <li>Audio will be automatically trimmed to 3-15 seconds at silence points</li>
                                </ul>
                            </li>
                            <li><strong>POST /delete_speaker</strong> - Remove an existing speaker preset
                                <ul style="margin-left: 20px; margin-top: 5px; color: #666;">
                                    <li>Form data: <code>name</code> (string)</li>
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
        try:
            audio_bytes = base64.b64decode(audio)
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
        
        # Save to temporary file and smart cut at silence intervals
        temp_dir = Path("speaker_presets") / "temp"
        temp_dir.mkdir(exist_ok=True)
        temp_path = temp_dir / f"speaker_{name}_{filename}"
        
        try:
            await async_write_file(str(temp_path), audio_data)
            # Smart cut: finds silence intervals and cuts at natural pauses (3-15s)
            cut_temp_path = await async_cut_audio_to_duration(str(temp_path), max_duration=10.0)
            
            # Read the cut audio data
            cut_audio_data = await async_read_file(cut_temp_path)
            
            # Add speaker using SpeakerPresetManager
            result = await speaker_api.add_speaker(name, [cut_audio_data], [filename])
            
        finally:
            # Clean up temporary files asynchronously
            try:
                if temp_path.exists():
                    await async_remove_file(str(temp_path))
                if cut_temp_path != str(temp_path) and os.path.exists(cut_temp_path):
                    await async_remove_file(cut_temp_path)
            except:
                pass
        
        if result["status"] == "success":
            return JSONResponse(content={"success": True, "role": name})
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
