#!/usr/bin/env python3
"""
FastAPI Web Interface for IndexTTS vLLM v2
A single-file FastAPI application that combines webui_with_presets.py functionality
with the API structure from deploy_vllm_indextts.py, using IndexTTS vLLM v2 as backend.

Features:
- IndexTTS vLLM v2 backend for ultra-fast inference
- Speaker preset management with persistent storage
- FlashTTS API compatibility
- Modern web interface with Chinese support
- Parallel chunk processing for long texts
- MP3 output support
"""

import os
import sys
import json
import time
import asyncio
import tempfile
import traceback
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Literal
from contextlib import asynccontextmanager
from io import BytesIO
import re
import base64
import urllib.request
from concurrent.futures import ThreadPoolExecutor
import functools

# Audio processing
import numpy as np
import soundfile as sf
from pydub import AudioSegment
import librosa
import torchaudio
import torch

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
from text_splitter import split_text
from tools.i18n.i18n import I18nAuto

# Configuration
import argparse

# Global thread executor for blocking operations
executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="fastapi_async")

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

# Initialize i18n
i18n = I18nAuto(language="Auto")

# Constants
EMO_CHOICES = [
    i18n("与音色参考音频相同"),
    i18n("使用情感参考音频"),
    i18n("使用情感向量控制"),
    i18n("使用情感描述文本控制")
]

MAX_LENGTH_TO_USE_SPEED = 70

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

async def async_audio_write(file_path: str, data, sample_rate: int, format: str = 'WAV'):
    """Async wrapper for soundfile.write()"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, sf.write, file_path, data, sample_rate, format)

async def async_audio_convert(input_path: str, output_format: str = "mp3", bitrate: str = "128k"):
    """Async wrapper for audio conversion"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, convert_audio_format, input_path, output_format, bitrate)

async def async_cut_audio_to_duration(input_path: str, max_duration: float = 10.0):
    """Async wrapper for cutting audio to specified duration"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, _cut_audio_to_duration_sync, input_path, max_duration)

def _cut_audio_to_duration_sync(input_path: str, max_duration: float = 10.0):
    """Cut audio to specified duration (in seconds)"""
    try:
        # Load audio data
        audio_data, sample_rate = sf.read(input_path)
        
        # Calculate the number of samples for the desired duration
        max_samples = int(max_duration * sample_rate)
        
        # If audio is shorter than max_duration, return original path
        if len(audio_data) <= max_samples:
            print(f"📏 Audio duration ({len(audio_data)/sample_rate:.1f}s) is within limit ({max_duration}s)")
            return input_path
        
        # Cut audio to max_duration
        cut_audio = audio_data[:max_samples]
        
        # Create output path with _cut suffix
        input_name = os.path.splitext(input_path)[0]
        output_path = f"{input_name}_cut.wav"
        
        # Save cut audio
        sf.write(output_path, cut_audio, sample_rate)
        
        original_duration = len(audio_data) / sample_rate
        cut_duration = len(cut_audio) / sample_rate
        
        print(f"✂️ Audio cut: {original_duration:.1f}s → {cut_duration:.1f}s (saved to {os.path.basename(output_path)})")
        
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


def convert_audio_format(input_path, output_format="mp3", bitrate="128k"):
    """Convert audio file to specified format"""
    if output_format.lower() == "wav":
        # Keep original WAV format
        return input_path
    
    try:
        # Load audio using pydub
        audio = AudioSegment.from_wav(input_path)
        
        # Generate output path
        input_name = os.path.splitext(input_path)[0]
        output_path = f"{input_name}.{output_format}"
        
        if output_format.lower() == "mp3":
            # Export as MP3 with specified bitrate
            audio.export(output_path, format="mp3", bitrate=bitrate)
            
            # Get file sizes for comparison
            wav_size = os.path.getsize(input_path) / (1024 * 1024)  # MB
            mp3_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            compression_ratio = wav_size / mp3_size if mp3_size > 0 else 1
            
            print(f"🔄 Audio converted: WAV ({wav_size:.1f}MB) → MP3 ({mp3_size:.1f}MB)")
            print(f"📦 Compression ratio: {compression_ratio:.1f}x smaller")
            
            # Remove original WAV file to save space
            try:
                os.remove(input_path)
                print(f"🗑️ Original WAV file removed")
            except Exception as cleanup_error:
                print(f"⚠️ Could not remove original WAV: {cleanup_error}")
                
        else:
            # Other formats (flac, ogg, etc.)
            audio.export(output_path, format=output_format)
            print(f"🔄 Audio converted to {output_format.upper()}")
            try:
                os.remove(input_path)
                print(f"🗑️ Original WAV file removed")
            except Exception as cleanup_error:
                print(f"⚠️ Could not remove original WAV: {cleanup_error}")
        
        return output_path
        
    except ImportError as import_error:
        error_msg = f"❌ Audio conversion failed: Missing dependency - {import_error}"
        print(error_msg)
        print("💡 Install required packages: pip install pydub")
        if "mp3" in output_format.lower():
            print("💡 For MP3 support: pip install pydub[mp3] or install ffmpeg")
        raise Exception(error_msg)
        
    except Exception as e:
        error_msg = f"❌ Audio conversion to {output_format.upper()} failed: {e}"
        print(error_msg)
        raise Exception(error_msg)

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
                # Clean up temporary files
                try:
                    # Clean up original temp file (if different from cut file)
                    if temp_path.exists():
                        temp_path.unlink()
                    # Clean up cut file if it's different and exists
                    if cut_temp_path != str(temp_path) and os.path.exists(cut_temp_path):
                        os.remove(cut_temp_path)
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

# Generation functions
async def generate_chunk(chunk_text, chunk_index, emo_control_method, prompt, 
                        emo_ref_path, emo_weight, vec, emo_text, emo_random,
                        use_preset, preset_name, max_text_tokens_per_sentence, **kwargs):
    """Generate audio for a single text chunk"""
    try:
        output_path = os.path.join("outputs", f"chunk_{chunk_index}_{int(time.time())}.wav")
        
        
        if use_preset and preset_name and preset_name != "None":
            output = await tts_manager.get_tts().infer(
                spk_audio_prompt="",
                text=chunk_text,
                output_path=output_path,
                emo_audio_prompt=emo_ref_path,
                emo_alpha=emo_weight,
                emo_vector=vec,
                use_emo_text=(emo_control_method==3),
                emo_text=emo_text,
                use_random=emo_random,
                verbose=cmd_args.verbose,
                max_text_tokens_per_sentence=int(max_text_tokens_per_sentence),
                speaker_preset=preset_name,
                **kwargs
            )
        else:
            output = await tts_manager.get_tts().infer(
                spk_audio_prompt=prompt,
                text=chunk_text,
                output_path=output_path,
                emo_audio_prompt=emo_ref_path,
                emo_alpha=emo_weight,
                emo_vector=vec,
                use_emo_text=(emo_control_method==3),
                emo_text=emo_text,
                use_random=emo_random,
                verbose=cmd_args.verbose,
                max_text_tokens_per_sentence=int(max_text_tokens_per_sentence),
                **kwargs
            )
        
        return chunk_index, output
    except Exception as e:
        print(f"❌ Error generating chunk {chunk_index}: {e}")
        print(f"📝 Chunk text: {chunk_text}")
        traceback.print_exc()
        return chunk_index, None

async def combine_audio_chunks(chunk_results, output_path, sample_rate=22050):
    """Combine multiple audio files into one"""
    try:
        # Sort by chunk index to maintain order
        chunk_results.sort(key=lambda x: x[0])
        
        # Load and combine audio data
        combined_audio = []
        for chunk_idx, audio_path in chunk_results:
            if audio_path and os.path.exists(audio_path):
                audio_data, sr = await async_audio_read(audio_path)
                if len(audio_data.shape) > 1:
                    audio_data = audio_data[:, 0]  # Take first channel if stereo
                combined_audio.append(audio_data)
                
                # Clean up chunk file
                await async_remove_file(audio_path)
            else:
                print(f"⚠️ Warning: Missing audio for chunk {chunk_idx}")
        
        if not combined_audio:
            raise ValueError("No valid audio chunks to combine")
        
        # Concatenate all audio with small silence gaps (100ms)
        silence_samples = int(0.1 * sample_rate)  # 100ms silence
        silence = np.zeros(silence_samples)
        
        final_audio = combined_audio[0]
        for audio_chunk in combined_audio[1:]:
            final_audio = np.concatenate([final_audio, silence, audio_chunk])
        
        # Save combined audio
        await async_audio_write(output_path, final_audio, sample_rate)
        return output_path
        
    except Exception as e:
        print(f"❌ Error combining audio chunks: {e}")
        return None

# FlashTTS API Models
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
    emotion_weight: float = Field(default=0.3, description="Emotion control weight (0.0 to 1.0)")

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
    emotion_weight: float = Field(default=0.3, description="Emotion control weight (0.0 to 1.0)")

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
                    <span class="performance-badge">🔌 FlashTTS API</span>
                    <span class="performance-badge">😊 Emotion Text Control</span>
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
                                    <label for="emotionWeight" style="color: white;">Emotion Strength / 情感强度: <span id="emotionWeightValue">0.3</span></label>
                                    <input type="range" id="emotionWeight" name="emotionWeight" 
                                           min="0.0" max="1.0" step="0.1" value="0.3"
                                           style="width: 100%; margin-bottom: 10px;"
                                           oninput="document.getElementById('emotionWeightValue').textContent = this.value">
                                </div>
                                <p style="color: #fff; font-size: 0.9em; margin: 0;">
                                    💡 输入情感描述文本可以让AI更精准地控制语音的情感表达。留空则使用默认情感。
                                </p>
                            </div>
                            
                            <button type="submit" class="btn" id="generateBtn">
                                🎵 Generate Speech
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
                                    📏 Audio will be automatically cut to 10 seconds for optimal performance
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
                        <h4>🔹 IndexTTS Native API</h4>
                        <h5>Speaker Management</h5>
                        <ul style="margin-left: 20px; line-height: 1.6;">
                            <li><strong>GET /api/speakers</strong> - List all speakers</li>
                            <li><strong>POST /api/speakers</strong> - Add new speaker</li>
                            <li><strong>DELETE /api/speakers/{speaker_name}</strong> - Delete speaker</li>
                        </ul>
                        
                        <h5>Speech Generation</h5>
                        <ul style="margin-left: 20px; line-height: 1.6;">
                            <li><strong>POST /generate</strong> - Generate with uploaded files</li>
                            <li><strong>POST /generate_speaker</strong> - Generate with speaker preset</li>
                        </ul>
                        
                        <h4>🔹 FlashTTS Compatible API</h4>
                        <h5>Speaker Management</h5>
                        <ul style="margin-left: 20px; line-height: 1.6;">
                            <li><strong>GET /audio_roles</strong> - List available speakers</li>
                            <li><strong>POST /add_speaker</strong> - Add new speaker</li>
                            <li><strong>POST /delete_speaker</strong> - Delete speaker</li>
                            <li><strong>GET /server_info</strong> - Get server information</li>
                        </ul>
                        
                        <h5>Speech Generation</h5>
                        <ul style="margin-left: 20px; line-height: 1.6;">
                            <li><strong>POST /speak</strong> - Generate speech using speaker
                                <ul style="margin-left: 20px; margin-top: 5px; color: #666;">
                                    <li>Supports emotion text control via <code>emotion_text</code> and <code>emotion_weight</code> parameters</li>
                                </ul>
                            </li>
                            <li><strong>POST /clone_voice</strong> - Clone voice using reference audio
                                <ul style="margin-left: 20px; margin-top: 5px; color: #666;">
                                    <li>Supports emotion text control via <code>emotion_text</code> and <code>emotion_weight</code> parameters</li>
                                </ul>
                            </li>
                        </ul>
                        
                        <h5>🆕 Emotion Text Control</h5>
                        <ul style="margin-left: 20px; line-height: 1.6;">
                            <li><strong>emotion_text</strong> (optional): Emotion description text (e.g., "happy and excited", "sad and melancholic")</li>
                            <li><strong>emotion_weight</strong> (optional): Emotion strength from 0.0 to 1.0 (default: 0.3)</li>
                            <li>When <code>emotion_text</code> is provided, the system uses IndexTTS2's advanced emotion text control</li>
                            <li>Example: <code>{"text": "Hello world", "emotion_text": "cheerful and energetic", "emotion_weight": 0.7}</code></li>
                        </ul>
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
                    const response = await fetch('/api/speakers');
                    const data = await response.json();
                    const select = document.getElementById('speaker');
                    
                    // Clear existing options except first
                    select.innerHTML = '<option value="">Select a speaker...</option>';
                    
                    if (data.status === 'success') {
                        Object.keys(data.speakers).forEach(speaker => {
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
                    const response = await fetch('/api/speakers');
                    const data = await response.json();
                    const listDiv = document.getElementById('speakerList');
                    
                    if (data.status === 'success') {
                        const speakers = data.speakers;
                        let html = `<h4>📊 ${data.total_speakers} Speakers Available</h4>`;
                        
                        for (const [name, info] of Object.entries(speakers)) {
                            html += `
                                <div class="speaker-item">
                                    <div class="speaker-info">
                                        <h4>🎭 ${name}</h4>
                                        <small>📁 ${info.audio_count} files • 💾 ${info.total_size_mb.toFixed(2)} MB</small>
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
                    const response = await fetch(`/api/speakers/${speakerName}`, {
                        method: 'DELETE'
                    });
                    
                    const result = await response.json();
                    showStatus(result.message, result.status === 'success' ? 'success' : 'error', 'speakerStatus');
                    
                    if (result.status === 'success') {
                        loadSpeakerList();
                        loadSpeakers(); // Refresh dropdown
                    }
                } catch (error) {
                    showStatus(`Error deleting speaker: ${error.message}`, 'error', 'speakerStatus');
                }
            }

            // Add Speaker Form
            document.getElementById('addSpeakerForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const formData = new FormData(this);
                
                try {
                    showStatus('Adding speaker...', 'success', 'speakerStatus');
                    
                    const response = await fetch('/api/speakers', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    let displayMessage = result.message;
                    if (result.status === 'success' && result.info === 'already_exists') {
                        displayMessage += ` (${result.audio_count} existing audio files)`;
                    } else if (result.status === 'success' && result.info === 'newly_added') {
                        displayMessage += ` (${result.audio_count} audio files uploaded)`;
                    }
                    
                    showStatus(displayMessage, result.status === 'success' ? 'success' : 'error', 'speakerStatus');
                    
                    if (result.status === 'success') {
                        this.reset();
                        loadSpeakerList();
                        loadSpeakers(); // Refresh dropdown
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
                
                if (!text.trim()) {
                    showStatus('Please enter some text to synthesize.', 'error');
                    return;
                }
                
                try {
                    const startTime = performance.now();
                    let response;
                    
                    if (speaker) {
                        // Use JSON API for speaker preset with emotion support
                        const requestData = {
                            text: text, 
                            speaker: speaker,
                            emotion_text: emotionText || "",
                            emotion_weight: emotionWeight
                        };
                        
                        response = await fetch('/generate_speaker', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify(requestData)
                        });
                    } else {
                        // Add emotion parameters to form data for file upload
                        formData.append('emotion_text', emotionText || "");
                        formData.append('emotion_weight', emotionWeight.toString());
                        
                        response = await fetch('/generate', {
                            method: 'POST',
                            body: formData
                        });
                    }
                    
                    if (response.ok) {
                        const endTime = performance.now();
                        const duration = ((endTime - startTime) / 1000).toFixed(2);
                        
                        const blob = await response.blob();
                        const audioUrl = URL.createObjectURL(blob);
                        
                        document.getElementById('audioResult').innerHTML = `
                            <h3>🎵 Generated Speech (${duration}s)</h3>
                            <audio controls style="width: 100%; margin: 10px 0;">
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
                } catch (error) {
                    showStatus(`Network error: ${error.message}`, 'error');
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
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if tts_manager.is_ready() and speaker_api:
        speakers_data = await speaker_api.list_speakers()
        speaker_count = speakers_data.get("total_speakers", 0) if speakers_data["status"] == "success" else 0
        return {
            "status": "healthy",
            "message": "IndexTTS vLLM v2 FastAPI WebUI",
            "engine": "vLLM v2",
            "concurrent_capacity": "100",
            "available_speakers": speaker_count,
            "chinese_support": "✅ Full Chinese text normalization",
            "audio_format": "🎵 MP3 (128k) for smaller files",
            "speaker_presets": "✅ Using SpeakerPresetManager"
        }
    else:
        return {
            "status": "initializing",
            "message": "IndexTTS vLLM v2 is starting up..."
        }

# Speaker Management API Endpoints
@app.get("/api/speakers")
async def api_list_speakers():
    """API: List all speakers"""
    if not speaker_api:
        return {"status": "error", "message": "Speaker manager not initialized"}
    return await speaker_api.list_speakers()

@app.post("/api/speakers")
async def api_add_speaker(
    speakerName: str = Form(...),
    speakerAudioFiles: List[UploadFile] = File(...)
):
    """API: Add a new speaker"""
    try:
        if not speaker_api:
            return {"status": "error", "message": "Speaker manager not initialized"}
        
        if not speakerAudioFiles:
            return {"status": "error", "message": "No audio files provided"}
        
        # Read audio files
        audio_data = []
        filenames = []
        
        for file in speakerAudioFiles:
            if file.filename:
                content = await file.read()
                audio_data.append(content)
                filenames.append(file.filename)
        
        # Add speaker using SpeakerPresetManager
        result = await speaker_api.add_speaker(speakerName, audio_data, filenames)
        return result
        
    except Exception as e:
        return {"status": "error", "message": f"Failed to add speaker: {str(e)}"}

@app.delete("/api/speakers/{speaker_name}")
async def api_delete_speaker(speaker_name: str):
    """API: Delete a speaker"""
    try:
        if not speaker_api:
            return {"status": "error", "message": "Speaker manager not initialized"}
        
        result = await speaker_api.delete_speaker(speaker_name)
        return result
    except Exception as e:
        return {"status": "error", "message": f"Failed to delete speaker: {str(e)}"}

# Speech Generation Endpoints
@app.post("/generate")
async def generate_speech(
    text: str = Form(...),
    voice_files: List[UploadFile] = File(None),
    emotion_text: Optional[str] = Form(""),
    emotion_weight: float = Form(0.3)
):
    """Generate speech with uploaded voice files"""
    try:
        tts = tts_manager.get_tts()
        
        # Handle multiple reference voice files
        audio_paths = []
        if voice_files:
            for voice_file in voice_files:
                if voice_file.filename:
                    voice_content = await voice_file.read()
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_voice:
                        tmp_path = tmp_voice.name
                    await async_write_file(tmp_path, voice_content)
                    audio_paths.append(tmp_path)
        
        # Generate speech with vLLM v2
        print(f"🎵 Generating speech: {text[:50]}...")
        
        # Check if emotion text is provided and not empty
        use_emotion_text = emotion_text and emotion_text.strip() != ""
        
        output_path = os.path.join("outputs", f"gen_{int(time.time())}.wav")
        result = await tts.infer(
            spk_audio_prompt=audio_paths[0] if audio_paths else "",
            text=text,
            output_path=output_path,
            use_emo_text=use_emotion_text,
            emo_text=emotion_text if use_emotion_text else None,
            emo_alpha=emotion_weight,
            verbose=cmd_args.verbose
        )
        
        # Convert to MP3 format
        mp3_path = await async_audio_convert(result, "mp3", "128k")
        
        # Cleanup
        cleanup_tasks = [async_remove_file(path) for path in audio_paths]
        await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        return FileResponse(
            mp3_path,
            media_type="audio/mpeg",
            filename="speech.mp3"
        )
        
    except Exception as e:
        print(f"❌ Error generating speech: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_speaker")
async def generate_speech_speaker(request: Request):
    """Generate speech using speaker preset"""
    try:
        data = await request.json()
        text = data["text"]
        speaker = data["speaker"]
        emotion_text = data.get("emotion_text", "")
        emotion_weight = data.get("emotion_weight", 0.3)
        
        print(f"🎭 Request received - Speaker: '{speaker}', Text: '{text[:50]}...'")
        
        # Simple speaker validation to prevent failures
        if speaker_api and not speaker_api.speaker_exists(speaker):
            return Response(
                content=f"Speaker '{speaker}' not found".encode('utf-8'),
                status_code=404,
                media_type="text/plain"
            )
        
        tts = tts_manager.get_tts()
        
        # Use speaker preset
        output_path = os.path.join("outputs", f"spk_{int(time.time())}.wav")
        
        # Check if emotion text is provided and not empty
        use_emotion_text = emotion_text and emotion_text.strip() != ""
        
        result = await tts.infer(
            spk_audio_prompt="",
            text=text,
            output_path=output_path,
            speaker_preset=speaker,
            use_emo_text=use_emotion_text,
            emo_text=emotion_text if use_emotion_text else None,
            emo_alpha=emotion_weight,
            verbose=cmd_args.verbose
        )
        
        # Convert to MP3 format
        mp3_path = await async_audio_convert(result, "mp3", "128k")
        
        return FileResponse(
            mp3_path,
            media_type="audio/mpeg",
            filename="speech.mp3"
        )
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        print(f"❌ Error generating speech with speaker '{speaker}': {error_msg}")
        print(f"🔍 Full traceback:")
        traceback.print_exc()
        
        # Return simple error response matching deploy_vllm_indextts.py format
        return Response(
            content=f"Error: {error_msg}".encode('utf-8'),
            status_code=500,
            media_type="text/plain; charset=utf-8"
        )

# FlashTTS API Helper Functions (matching deploy_vllm_indextts.py exactly)
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
        # Fallback if httpx is not available
        try:
            with urllib.request.urlopen(url) as response:
                if response.status != 200:
                    raise HTTPException(status_code=400, detail="Cannot download audio from URL")
                return response.read()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to download audio: {str(e)}")

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

async def load_audio_bytes_flashtts(audio_file, audio):
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

# FlashTTS API Compatibility Endpoints
@app.post("/add_speaker")
async def flashtts_add_speaker(
    background_tasks: BackgroundTasks,
    name: str = Form(..., description="The name of the speaker"),
    audio: Optional[str] = Form(None, description="Reference audio URL or base64"),
    reference_text: Optional[str] = Form(None, description="Optional transcript"),
    audio_file: Optional[UploadFile] = File(None, description="Upload reference audio file"),
):
    """FlashTTS API: Add a new speaker"""
    try:
        print(f"🎭 FlashTTS API: Adding speaker '{name}'")
        print(f"🔍 Debug: audio_file={audio_file}, audio={audio is not None}, reference_text={reference_text}")
        
        if not speaker_api:
            return JSONResponse(
                status_code=500,
                content={"success": False, "error": "Speaker manager not initialized"}
            )
        
        # Load audio from file or reference string (matching deploy_vllm_indextts.py)
        try:
            audio_io = await load_audio_bytes_flashtts(audio_file, audio)
            if audio_io is None:
                print(f"❌ FlashTTS API: No audio provided for speaker '{name}'")
                return JSONResponse(
                    status_code=400,
                    content={"success": False, "error": "No audio provided"}
                )
        except Exception as audio_error:
            print(f"❌ FlashTTS API: Audio loading failed for speaker '{name}': {audio_error}")
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": f"Audio loading failed: {str(audio_error)}"}
            )
        
        # Get audio data and filename
        audio_data = audio_io.read()
        filename = audio_file.filename if audio_file else f"{name}_reference.wav"
        
        # Save to temporary file and cut to 10 seconds
        temp_dir = Path("speaker_presets") / "temp"
        temp_dir.mkdir(exist_ok=True)
        temp_path = temp_dir / f"flashtts_{name}_{filename}"
        
        try:
            await async_write_file(str(temp_path), audio_data)
            # Cut audio to 10 seconds if it exceeds the limit
            cut_temp_path = await async_cut_audio_to_duration(str(temp_path), max_duration=10.0)
            
            # Read the cut audio data
            cut_audio_data = await async_read_file(cut_temp_path)
            
            # Add speaker using SpeakerPresetManager
            result = await speaker_api.add_speaker(name, [cut_audio_data], [filename])
            
        finally:
            # Clean up temporary files
            try:
                if temp_path.exists():
                    temp_path.unlink()
                if cut_temp_path != str(temp_path) and os.path.exists(cut_temp_path):
                    os.remove(cut_temp_path)
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
        print(f"❌ FlashTTS API: {error_msg}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": error_msg}
        )

@app.post("/delete_speaker")
async def flashtts_delete_speaker(
    background_tasks: BackgroundTasks,
    name: str = Form(..., description="The name of the speaker")
):
    """FlashTTS API: Delete a speaker"""
    try:
        print(f"🗑️ FlashTTS API: Deleting speaker '{name}'")
        
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
        print(f"❌ FlashTTS API: {error_msg}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": error_msg}
        )

@app.get("/audio_roles")
async def flashtts_audio_roles():
    """FlashTTS API: List available speakers"""
    try:
        print("📋 FlashTTS API: Listing audio roles")
        
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
        print(f"❌ FlashTTS API: {error_msg}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": error_msg}
        )

@app.post("/speak")
async def flashtts_speak(req: SpeakRequest):
    """FlashTTS API: Generate speech using registered speaker"""
    try:
        print(f"🎭 FlashTTS API: Speaking with '{req.name}' - '{req.text[:50]}...'")
        
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
        output_path = os.path.join("outputs", f"speak_{int(time.time())}.wav")
        
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
        
        print(f"✅ FlashTTS API: Generated {len(audio_bytes)} bytes of {req.response_format.upper()} audio")
        
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
        print(f"❌ FlashTTS API: {error_msg}")
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
    emotion_weight: float = Form(0.3),
):
    return CloneRequest(
        text=text, reference_audio=reference_audio, reference_text=reference_text,
        pitch=pitch, speed=speed, temperature=temperature, top_k=top_k, top_p=top_p,
        repetition_penalty=repetition_penalty, max_tokens=max_tokens,
        length_threshold=length_threshold, window_size=window_size,
        stream=stream, response_format=response_format,
        emotion_text=emotion_text, emotion_weight=emotion_weight
    )

@app.post("/clone_voice")
async def flashtts_clone_voice(
    req: CloneRequest = Depends(parse_clone_form),
    reference_audio_file: Optional[UploadFile] = File(None),
):
    """FlashTTS API: Clone voice using reference audio"""
    try:
        print(f"🎵 FlashTTS API: Cloning voice - '{req.text[:50]}...'")
        
        # Load reference audio (matching deploy_vllm_indextts.py)
        audio_io = await load_audio_bytes_flashtts(reference_audio_file, req.reference_audio)
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
            output_path = os.path.join("outputs", f"clone_{int(time.time())}.wav")
            
            # Check if emotion text is provided and not empty
            use_emotion_text = req.emotion_text and req.emotion_text.strip() != ""
            
            result = await tts.infer(
                spk_audio_prompt=tmp_path,
                text=req.text,
                output_path=output_path,
                use_emo_text=use_emotion_text,
                emo_text=req.emotion_text if use_emotion_text else None,
                emo_alpha=req.emotion_weight,
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
            
            print(f"✅ FlashTTS API: Cloned voice - {len(audio_bytes)} bytes of {req.response_format.upper()}")
            
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
        print(f"❌ FlashTTS API: {error_msg}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": error_msg}
        )

@app.get("/server_info")
async def flashtts_server_info():
    """FlashTTS API: Get server information"""
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
    print(f"   - FlashTTS API compatibility")
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
