import json
import logging
import os
import re
import sys
import threading
import time

import warnings
import pandas as pd

# Suppress pandas future warning that can interfere with concurrent processing  
warnings.filterwarnings("ignore", message=".*pandas.*future.*", category=FutureWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

import argparse
parser = argparse.ArgumentParser(description="IndexTTS WebUI with Speaker Presets")
parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose mode")
parser.add_argument("--port", type=int, default=6006, help="Port to run the web UI on")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the web UI on")
parser.add_argument("--model_dir", type=str, default="checkpoints", help="Model checkpoints directory")
parser.add_argument("--is_fp16", action="store_true", default=False, help="Fp16 infer")
parser.add_argument("--gpu_memory_utilization", type=float, default=0.25, help="GPU memory utilization")
cmd_args = parser.parse_args()

if not os.path.exists(cmd_args.model_dir):
    print(f"Model directory {cmd_args.model_dir} does not exist. Please download the model first.")
    sys.exit(1)

for file in [
    "bpe.model",
    "gpt.pth",
    "config.yaml",
    "s2mel.pth",
    "wav2vec2bert_stats.pt"
]:
    file_path = os.path.join(cmd_args.model_dir, file)
    if not os.path.exists(file_path):
        print(f"Required file {file_path} does not exist. Please download it.")
        sys.exit(1)

import asyncio
import numpy as np
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor
from pydub import AudioSegment
from pydub.utils import which

import gradio as gr
from indextts.infer_vllm_v2 import IndexTTS2
from speaker_preset_manager import SpeakerPresetManager, initialize_preset_manager
from text_splitter import split_text
from tools.i18n.i18n import I18nAuto

i18n = I18nAuto(language="Auto")

# 支持的语言列表
LANGUAGES = {
    "中文": "zh_CN",
    "English": "en_US"
}
EMO_CHOICES = [i18n("与音色参考音频相同"),
                i18n("使用情感参考音频"),
                i18n("使用情感向量控制"),
                i18n("使用情感描述文本控制")]
os.makedirs("outputs/tasks", exist_ok=True)
os.makedirs("prompts", exist_ok=True)
os.makedirs("speaker_presets", exist_ok=True)
MAX_LENGTH_TO_USE_SPEED = 70


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


# Load example cases
with open("examples/cases.jsonl", "r", encoding="utf-8") as f:
    example_cases = []
    for line in f:
        line = line.strip()
        if not line:
            continue
        example = json.loads(line)
        if example.get("emo_audio", None):
            emo_audio_path = os.path.join("examples", example["emo_audio"])
        else:
            emo_audio_path = None
        example_cases.append([os.path.join("examples", example.get("prompt_audio", "sample_prompt.wav")),
                            EMO_CHOICES[example.get("emo_mode", 0)],
                            example.get("text"),
                            emo_audio_path,
                            example.get("emo_weight", 1.0),
                            example.get("emo_text", ""),
                            example.get("emo_vec_1", 0),
                            example.get("emo_vec_2", 0),
                            example.get("emo_vec_3", 0),
                            example.get("emo_vec_4", 0),
                            example.get("emo_vec_5", 0),
                            example.get("emo_vec_6", 0),
                            example.get("emo_vec_7", 0),
                            example.get("emo_vec_8", 0)]
                            )


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


async def generate_chunk(chunk_text, chunk_index, emo_control_method, prompt, 
                        emo_ref_path, emo_weight, vec, emo_text, emo_random,
                        use_preset, preset_name, max_text_tokens_per_sentence, speech_length=0, diffusion_steps=10, **kwargs):
    """Generate audio for a single text chunk"""
    try:
        output_path = os.path.join("outputs", f"chunk_{chunk_index}_{int(time.time())}.wav")
        
        
        if use_preset and preset_name and preset_name != "None":
            output = await tts.infer(
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
                speaker_preset=preset_name,  # NEW: Use speaker_preset parameter
                speech_length=speech_length,
                diffusion_steps=diffusion_steps,
                **kwargs
            )
        else:
            output = await tts.infer(
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
                speech_length=speech_length,
                diffusion_steps=diffusion_steps,
                **kwargs
            )
        
        return chunk_index, output
    except Exception as e:
        print(f"❌ Error generating chunk {chunk_index}: {e}")
        print(f"📝 Chunk text: {chunk_text}")
        import traceback
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
                audio_data, sr = sf.read(audio_path)
                if len(audio_data.shape) > 1:
                    audio_data = audio_data[:, 0]  # Take first channel if stereo
                combined_audio.append(audio_data)
                
                # Clean up chunk file
                try:
                    os.remove(audio_path)
                except:
                    pass
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
        sf.write(output_path, final_audio, sample_rate)
        return output_path
        
    except Exception as e:
        print(f"❌ Error combining audio chunks: {e}")
        return None
async def gen_parallel(emo_control_method, prompt, text,
                      emo_ref_path, emo_weight,
                      vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
                      emo_text, emo_random,
                      max_text_tokens_per_sentence=120,
                      use_preset=False, preset_name="",
                     max_parallel_chunks=5,  # NEW: limit parallel processing
                     chunk_length_zh=100,   # NEW: chunk length for Chinese
                     chunk_length_en=200,   # NEW: chunk length for English
                     output_format="mp3",  # NEW: output format
                     speech_length=0,  # NEW: target duration control
                     diffusion_steps=10,  # NEW: diffusion steps control
                     *args, progress=gr.Progress()):
    """Generate speech for long text using parallel chunk processing"""
    
    try:
        # Start timing for RTF calculation
        start_time = time.perf_counter()
        output_path = os.path.join("outputs", f"parallel_{int(time.time())}.wav")
        
        do_sample, top_p, top_k, temperature, \
            length_penalty, num_beams, repetition_penalty, max_mel_tokens = args
        kwargs = {
            "do_sample": bool(do_sample),
            "top_p": float(top_p),
            "top_k": int(top_k) if int(top_k) > 0 else None,
            "temperature": float(temperature),
            "length_penalty": float(length_penalty),
            "num_beams": num_beams,
            "repetition_penalty": float(repetition_penalty),
            "max_mel_tokens": int(max_mel_tokens),
        }
        
        if type(emo_control_method) is not int:
            emo_control_method = emo_control_method.value
        
        if emo_control_method == 0:
            emo_ref_path = None
            # For "same as speaker audio", emo_weight doesn't matter since emo_cond_emb == spk_cond_emb
            # The merge_emovec result will be the same regardless of alpha value
            vec = None
        elif emo_control_method == 1:
            emo_weight = emo_weight
            vec = None
        elif emo_control_method == 2:
            vec = [vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
            vec_sum = sum([vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8])
            if vec_sum > 1.5:
                gr.Warning("情感向量之和不能超过1.5，请调整后重试。")
                return
        else:
            vec = None
        print(f"🔄 Starting parallel generation for text length: {len(text)}")
        print(f"📝 Using preset: {use_preset}, preset_name: {preset_name}")
        
        # Split text into chunks
        settings = {
            "max_chunk_length_zh": chunk_length_zh,
            "max_chunk_length_en": chunk_length_en
        }
        
        chunks = split_text(text, settings)
        print(f"📦 Split text into {len(chunks)} chunks")
        
        if len(chunks) <= 1:
            print("📝 Text is short, using single generation")
            return await gen_single(emo_control_method, prompt, text,
                                  emo_ref_path, emo_weight,
                                  vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
                                  emo_text, emo_random, max_text_tokens_per_sentence,
                                  use_preset, preset_name, output_format, speech_length, *args, progress=progress)
        
        # Process chunks in parallel with concurrency limit
        progress(0, desc="Generating audio chunks...")
        
        chunk_tasks = []
        chunk_results = []
        
        # Process chunks in batches to limit concurrency
        for i in range(0, len(chunks), max_parallel_chunks):
            batch = chunks[i:i + max_parallel_chunks]
            batch_tasks = []
            
            for j, chunk in enumerate(batch):
                chunk_index = i + j
                task = generate_chunk(
                    chunk, chunk_index, emo_control_method, prompt,
                    emo_ref_path, emo_weight, vec, emo_text, emo_random,
                    use_preset, preset_name, max_text_tokens_per_sentence,
                    speech_length=speech_length,
                    diffusion_steps=diffusion_steps,
                    **kwargs
                )
                batch_tasks.append(task)
            
            # Wait for batch to complete
            print(f"🚀 Processing batch {i//max_parallel_chunks + 1}/{(len(chunks)-1)//max_parallel_chunks + 1} ({len(batch_tasks)} chunks)")
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Collect successful results
            for result in batch_results:
                if isinstance(result, tuple) and result[1] is not None:
                    chunk_results.append(result)
                elif isinstance(result, Exception):
                    print(f"❌ Chunk generation failed: {result}")
            
            # Update progress
            progress((i + len(batch)) / len(chunks), desc=f"Generated {len(chunk_results)}/{len(chunks)} chunks")
        
        if not chunk_results:
            raise ValueError("All chunk generations failed")
        
        print(f"✅ Successfully generated {len(chunk_results)}/{len(chunks)} chunks")
        
        # Combine audio chunks
        progress(0.9, desc="Combining audio chunks...")
        combined_path = await combine_audio_chunks(chunk_results, output_path)
        
        if combined_path:
            # Calculate total time and RTF
            total_time = time.perf_counter() - start_time
            
            # Get audio duration for RTF calculation
            try:
                import soundfile as sf
                audio_data, sample_rate = sf.read(combined_path)
                audio_duration = len(audio_data) / sample_rate
                rtf = total_time / audio_duration
                
                print(f"🎵 Combined audio saved to: {combined_path}")
                print(f"⚡ PARALLEL GENERATION STATS:")
                print(f"   📊 Total chunks: {len(chunks)}")
                print(f"   ⏱️  Total time: {total_time:.2f}s")
                print(f"   🎧 Audio length: {audio_duration:.2f}s") 
                print(f"   🚀 Parallel RTF: {rtf:.4f}")
                print(f"   💾 Max parallel chunks: {max_parallel_chunks}")
                
                # Calculate estimated sequential time for comparison
                estimated_sequential = len(chunks) * (total_time / (len(chunks) / max_parallel_chunks))
                speedup = estimated_sequential / total_time if total_time > 0 else 1.0
                print(f"   📈 Estimated speedup: {speedup:.1f}x")
                
            except Exception as e:
                print(f"🎵 Combined audio saved to: {combined_path}")
                print(f"⚡ Parallel generation completed in {total_time:.2f}s")
                print(f"⚠️  Could not calculate RTF: {e}")
            
            # Convert output format if needed
            try:
                if output_format != "wav":
                    combined_path = convert_audio_format(combined_path, output_format)
            except Exception as conversion_error:
                print(f"❌ Format conversion failed: {conversion_error}")
                gr.Error(f"Audio conversion to {output_format.upper()} failed: {str(conversion_error)}")
                # Continue with WAV file
            
            progress(1.0, desc="Parallel generation complete!")
            return gr.update(value=combined_path, visible=True)
        else:
            raise ValueError("Failed to combine audio chunks")
            
    except Exception as e:
        print(f"❌ Parallel generation failed: {e}")
        import traceback
        traceback.print_exc()
        gr.Error(f"Parallel generation failed: {str(e)}")
        return gr.update(value=None, visible=True)
async def gen_single(emo_control_method, prompt, text,
            emo_ref_path, emo_weight,
            vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
            emo_text, emo_random,
            max_text_tokens_per_sentence=120,
            use_preset=False, preset_name="",  # NEW: preset parameters
            output_format="mp3",  # NEW: output format
            speech_length=0,  # NEW: target duration control
            diffusion_steps=10,  # NEW: diffusion steps control
                *args, progress=gr.Progress()):
    output_path = None
    if not output_path:
        output_path = os.path.join("outputs", f"spk_{int(time.time())}.wav")
    
    do_sample, top_p, top_k, temperature, \
        length_penalty, num_beams, repetition_penalty, max_mel_tokens = args
    kwargs = {
        "do_sample": bool(do_sample),
        "top_p": float(top_p),
        "top_k": int(top_k) if int(top_k) > 0 else None,
        "temperature": float(temperature),
        "length_penalty": float(length_penalty),
        "num_beams": num_beams,
        "repetition_penalty": float(repetition_penalty),
        "max_mel_tokens": int(max_mel_tokens),
    }
    
    if type(emo_control_method) is not int:
        emo_control_method = emo_control_method.value
    if emo_control_method == 0:
        emo_ref_path = None
        # For "same as speaker audio", emo_weight doesn't matter since emo_cond_emb == spk_cond_emb
    if emo_control_method == 1:
        emo_weight = emo_weight
    if emo_control_method == 2:
        vec = [vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
        vec_sum = sum([vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8])
        if vec_sum > 1.5:
            gr.Warning(i18n("情感向量之和不能超过1.5，请调整后重试。"))
            return
    else:
        vec = None
    print(f"Emo control mode:{emo_control_method}, vec:{vec}")
    print(f"Using preset: {use_preset}, preset_name: {preset_name}")
    
    
    # Use preset if specified, otherwise use uploaded audio
    if use_preset and preset_name and preset_name != "None":
        # Use preset - audio prompt will be ignored
        output = await tts.infer(
            spk_audio_prompt="",  # Not used when preset is specified
            text=text,
            output_path=output_path,
            emo_audio_prompt=emo_ref_path, 
            emo_alpha=emo_weight,
            emo_vector=vec,
            use_emo_text=(emo_control_method==3), 
            emo_text=emo_text,
            use_random=emo_random,
            verbose=cmd_args.verbose,
            max_text_tokens_per_sentence=int(max_text_tokens_per_sentence),
            speaker_preset=preset_name,  # NEW: Use speaker_preset parameter
            speech_length=speech_length,
            diffusion_steps=diffusion_steps,
            **kwargs
        )
    else:
        # Use uploaded audio (original behavior)
        output = await tts.infer(
            spk_audio_prompt=prompt, 
            text=text,
            output_path=output_path,
            emo_audio_prompt=emo_ref_path, 
            emo_alpha=emo_weight,
            emo_vector=vec,
            use_emo_text=(emo_control_method==3), 
            emo_text=emo_text,
            use_random=emo_random,
            verbose=cmd_args.verbose,
            max_text_tokens_per_sentence=int(max_text_tokens_per_sentence),
            speech_length=speech_length,
            diffusion_steps=diffusion_steps,
            **kwargs
        )
    
    # Convert output format if needed
    try:
        if output_format != "wav":
            output = convert_audio_format(output, output_format)
        return gr.update(value=output, visible=True)
    except Exception as conversion_error:
        print(f"❌ Format conversion failed: {conversion_error}")
        gr.Error(f"Audio conversion to {output_format.upper()} failed: {str(conversion_error)}")
        return gr.update(value=output, visible=True)  # Return original WAV file
def add_speaker_preset(preset_name, audio_file, description):
    """Add a new speaker preset"""
    if not preset_name:
        return "❌ Please enter a preset name", gr.update()
    
    if not audio_file:
        return "❌ Please upload an audio file", gr.update()
    
    try:
        success = tts.preset_manager.add_speaker_preset(
            preset_name=preset_name,
            audio_path=audio_file,
            description=description or ""
        )
        
        if success:
            # Update preset dropdown
            preset_choices = ["None"] + list(tts.preset_manager.list_presets().keys())
            return f"✅ Speaker preset '{preset_name}' added successfully!", gr.update(choices=preset_choices)
        else:
            return "❌ Failed to add speaker preset", gr.update()
            
    except Exception as e:
        return f"❌ Error: {str(e)}", gr.update()
def delete_speaker_preset(preset_name):
    """Delete a speaker preset"""
    if not preset_name or preset_name == "None":
        return "❌ Please select a preset to delete", gr.update()
    
    try:
        success = tts.preset_manager.delete_preset(preset_name)
        if success:
            preset_choices = ["None"] + list(tts.preset_manager.list_presets().keys())
            return f"✅ Speaker preset '{preset_name}' deleted successfully!", gr.update(choices=preset_choices, value="None")
        else:
            return "❌ Failed to delete speaker preset", gr.update()
    except Exception as e:
        return f"❌ Error: {str(e)}", gr.update()
def get_preset_info():
    """Get information about all presets"""
    presets = tts.preset_manager.list_presets()
    stats = tts.preset_manager.get_cache_stats()
    
    info = f"📊 **Preset Statistics:**\n"
    info += f"- Total presets: {stats['total_presets']}\n"
    info += f"- Cache size: {stats['total_cache_size_mb']:.1f} MB\n"
    info += f"- Cache directory: `{stats['cache_directory']}`\n\n"
    
    if presets:
        info += "📋 **Available Presets:**\n"
        for name, data in presets.items():
            created = time.strftime("%Y-%m-%d %H:%M", time.localtime(data.get('created_at', 0)))
            last_used = time.strftime("%Y-%m-%d %H:%M", time.localtime(data.get('last_used', 0)))
            desc = data.get('description', 'No description')
            info += f"- **{name}**: {desc}\n"
            info += f"  - Created: {created}\n"
            info += f"  - Last used: {last_used}\n\n"
    else:
        info += "No presets available.\n"
    
    return info
def update_prompt_audio():
    return gr.update(interactive=True)
def on_preset_toggle(use_preset):
    """Handle preset toggle to show/hide relevant components"""
    if use_preset:
        return (
            gr.update(visible=False),  # Hide audio upload
            gr.update(visible=True),   # Show preset dropdown
        )
    else:
        return (
            gr.update(visible=True),   # Show audio upload
            gr.update(visible=False),  # Hide preset dropdown
        )
if __name__ == "__main__":
    print("🚀 Starting IndexTTS2 WebUI with Speaker Presets...")
    print(f"📁 Model directory: {cmd_args.model_dir}")
    print(f"🔧 GPU memory utilization: {cmd_args.gpu_memory_utilization}")
    print(f"🎯 FP16 mode: {cmd_args.is_fp16}")
    
    try:
        print("🔄 Initializing IndexTTS2...")
        tts = IndexTTS2(model_dir=cmd_args.model_dir, is_fp16=cmd_args.is_fp16, gpu_memory_utilization=cmd_args.gpu_memory_utilization)
        print("✅ IndexTTS2 initialized successfully!")
        
        print("🔄 Setting up speaker preset manager...")
        preset_manager = initialize_preset_manager(tts)
        print("✅ Speaker preset manager initialized!")
        
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    with gr.Blocks(title="IndexTTS Demo with Speaker Presets") as demo:
        gr.HTML('''
        <h2><center>IndexTTS2: Speaker Presets + Parallel Generation + MP3 Output</h2>
        <p align="center">
        <a href='https://arxiv.org/abs/2506.21619'><img src='https://img.shields.io/badge/ArXiv-2506.21619-red'></a>
        </p>
        <p align="center"><strong>🚀 Pre-process speaker audio once, use instantly forever!</strong></p>
        <p align="center"><strong>⚡ Parallel generation for ultra-fast long text synthesis!</strong></p>
        <p align="center"><strong>🎵 MP3 output for 10x smaller files!</strong></p>
        ''')
        
        with gr.Tab("🎙️ Audio Generation"):
            with gr.Row():
                with gr.Column(scale=2):
                    # Speaker selection section
                    with gr.Group():
                        gr.Markdown("### 👤 Speaker Selection")
                        use_preset = gr.Checkbox(label="Use Speaker Preset", value=False, 
                                            info="Enable to use pre-processed speaker presets for faster inference")
                        
                        # Audio upload (shown when not using preset)
                        with gr.Group(visible=True) as audio_upload_group:
                            prompt_audio = gr.Audio(label="Upload Speaker Audio", 
                                                sources=["upload", "microphone"], 
                                                type="filepath")
                        
                        # Preset selection (hidden by default)
                        with gr.Group(visible=False) as preset_selection_group:
                            preset_choices = ["None"] + list(preset_manager.list_presets().keys())
                            preset_dropdown = gr.Dropdown(
                                choices=preset_choices,
                                value="None",
                                label="Select Speaker Preset",
                                info="Choose a pre-processed speaker preset"
                            )
                    
                    # Text input
                    with gr.Group():
                        gr.Markdown("### 📝 Text Input") 
                        input_text_single = gr.TextArea(
                            label="Text to Generate", 
                            placeholder="Enter the text you want to synthesize...",
                            info="Current model version: 2.0"
                        )
                        with gr.Row():
                            gen_button = gr.Button("🎵 Generate Speech", variant="primary", size="lg", scale=2)
                            parallel_gen_button = gr.Button("⚡ Parallel Generate", variant="stop", size="lg", scale=2)
                        
                        gr.Markdown("💡 **Tip:** Use *Parallel Generate* for long text (>200 chars) to get 2-50x faster synthesis! Output defaults to MP3 for smaller file sizes. RTF stats shown in console.")
                        
                        # Parallel generation settings (collapsed by default)
                        with gr.Accordion("⚡ Parallel Generation Settings", open=False):
                            with gr.Row():
                                max_parallel_chunks = gr.Slider(
                                    label="Max Parallel Chunks", 
                                    minimum=2, maximum=50, value=20, step=1,
                                    info="Number of chunks to process simultaneously (higher = faster but more memory)"
                                )
                                chunk_length_zh = gr.Slider(
                                    label="Chinese Chunk Length", 
                                    minimum=50, maximum=300, value=100, step=10,
                                    info="Characters per chunk for Chinese text"
                                )
                                chunk_length_en = gr.Slider(
                                    label="English Chunk Length", 
                                    minimum=100, maximum=500, value=200, step=20,
                                    info="Characters per chunk for English text"
                                )
                
                with gr.Column(scale=1):
                    output_audio = gr.Audio(label="🔊 Generated Audio", visible=True)
            
            # Advanced settings (collapsed by default)
            with gr.Accordion("⚙️ Advanced Settings", open=False):
                # Emotion control
                with gr.Row():
                    emo_control_method = gr.Radio(
                        choices=EMO_CHOICES,
                        type="index",
                        value=EMO_CHOICES[0], 
                        label="Emotion Control Method"
                    )
                
                # Emotion reference audio (hidden by default)
                with gr.Group(visible=False) as emotion_reference_group:
                    emo_upload = gr.Audio(label="Upload Emotion Reference Audio", type="filepath")
                
                # Emotion weight control (available for all methods except "与音色参考音频相同")
                with gr.Group(visible=False) as emotion_weight_group:
                    emo_weight = gr.Slider(label="Emotion Weight", minimum=0.0, maximum=1.0, value=0.6, step=0.01)
                
                # Emotion random sampling
                emo_random = gr.Checkbox(label="Random Emotion Sampling", value=False, visible=False)
                
                # Emotion vector control (hidden by default) 
                with gr.Group(visible=False) as emotion_vector_group:
                    with gr.Row():
                        with gr.Column():
                            vec1 = gr.Slider(label="Joy", minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                            vec2 = gr.Slider(label="Anger", minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                            vec3 = gr.Slider(label="Sadness", minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                            vec4 = gr.Slider(label="Fear", minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                        with gr.Column():
                            vec5 = gr.Slider(label="Disgust", minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                            vec6 = gr.Slider(label="Low", minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                            vec7 = gr.Slider(label="Surprise", minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                            vec8 = gr.Slider(label="Calm", minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                
                # Emotion text control (hidden by default)
                with gr.Group(visible=False) as emo_text_group:
                    emo_text = gr.Textbox(
                        label="Emotion Description Text", 
                        placeholder="Enter emotion description...", 
                        value="", 
                        info="e.g., happy, angry, sad"
                    )
                
                # Generation parameters
                with gr.Accordion("🎛️ Generation Parameters", open=False):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("**GPT2 Sampling Settings**")
                            do_sample = gr.Checkbox(label="do_sample", value=True)
                            temperature = gr.Slider(label="temperature", minimum=0.1, maximum=2.0, value=0.8, step=0.1)
                            top_p = gr.Slider(label="top_p", minimum=0.0, maximum=1.0, value=0.8, step=0.01)
                            top_k = gr.Slider(label="top_k", minimum=0, maximum=100, value=30, step=1)
                            num_beams = gr.Slider(label="num_beams", value=3, minimum=1, maximum=10, step=1)
                            repetition_penalty = gr.Number(label="repetition_penalty", value=10.0, minimum=0.1, maximum=20.0, step=0.1)
                            length_penalty = gr.Number(label="length_penalty", value=0.0, minimum=-2.0, maximum=2.0, step=0.1)
                            max_mel_tokens = gr.Slider(label="max_mel_tokens", value=1500, minimum=50, maximum=2000, step=10)
                        
                        with gr.Column():
                            gr.Markdown("**Text Processing Settings**")
                            max_text_tokens_per_sentence = gr.Slider(
                                label="Max Tokens per Sentence", 
                                value=120, 
                                minimum=20, 
                                maximum=500, 
                                step=2,
                                info="Recommended: 80-200. Higher = longer sentences, Lower = more fragmented"
                            )
                            
                            gr.Markdown("**Duration Control** ⏱️")
                            with gr.Row():
                                speech_length = gr.Number(
                                    label="Target Duration (ms)",
                                    value=0,
                                    minimum=0,
                                    maximum=6000000,
                                    step=100,
                                    info="0 = auto duration. Set specific duration for dubbing/timing control."
                                )
                                estimate_button = gr.Button("📊 Estimate", size="sm")
                            
                            estimated_duration = gr.Textbox(
                                label="Estimated Duration",
                                value="",
                                interactive=False,
                                visible=False
                            )
                            
                            gr.Markdown("**Quality Control** 🎨")
                            with gr.Row():
                                diffusion_steps = gr.Slider(
                                    label="Diffusion Steps",
                                    value=10,
                                    minimum=1,
                                    maximum=50,
                                    step=1,
                                    info="Higher steps improve quality but increase latency. Fast=5, Default=10, High-quality=20-30"
                                )
                            
                            gr.Markdown("**Output Settings**")
                            output_format = gr.Radio(
                                choices=["mp3", "wav"],
                                value="mp3",
                                label="Output Format",
                                info="MP3: Smaller file size (~10x compression), WAV: Original quality"
                            )
                            
                            # Sentence preview
                            with gr.Accordion("📋 Sentence Preview", open=False):
                                sentences_preview = gr.Dataframe(
                                    headers=["#", "Sentence", "Tokens"],
                                    wrap=True,
                                )
                
                advanced_params = [
                    do_sample, top_p, top_k, temperature,
                    length_penalty, num_beams, repetition_penalty, max_mel_tokens,
                ]
            
            # Examples
            if len(example_cases) > 0:
                with gr.Accordion("📚 Examples", open=False):
                    gr.Examples(
                        examples=example_cases,
                        examples_per_page=10,
                        inputs=[prompt_audio, emo_control_method, input_text_single, emo_upload, emo_weight, emo_text,
                            vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
                    )
        
        with gr.Tab("👥 Speaker Preset Manager"):
            gr.Markdown("## 🎭 Manage Speaker Presets")
            gr.Markdown("""
            Speaker presets allow you to **pre-process speaker audio once** and reuse it instantly for faster inference.
            This eliminates the ~200-500ms audio processing overhead on every generation.
            """)
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### ➕ Add New Preset")
                    new_preset_name = gr.Textbox(label="Preset Name", placeholder="e.g., John_Professional")
                    new_preset_audio = gr.Audio(label="Speaker Audio File", type="filepath", sources=["upload"])
                    new_preset_desc = gr.Textbox(label="Description (Optional)", placeholder="e.g., Professional male voice")
                    add_preset_btn = gr.Button("Add Speaker Preset", variant="primary")
                    add_status = gr.Textbox(label="Status", interactive=False)
                
                with gr.Column():
                    gr.Markdown("### 🗑️ Delete Preset")
                    delete_preset_dropdown = gr.Dropdown(
                        choices=preset_choices,
                        value="None",
                        label="Select Preset to Delete"
                    )
                    delete_preset_btn = gr.Button("Delete Selected Preset", variant="stop")
                    delete_status = gr.Textbox(label="Status", interactive=False)
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📊 Preset Information")
                    info_button = gr.Button("Refresh Preset Info")
                    preset_info = gr.Markdown(value=get_preset_info())
        # Event handlers
        def on_estimate_duration(text):
            """Estimate speech duration from text"""
            if not text or not text.strip():
                return gr.update(value="⚠️ No text entered", visible=True)
            
            duration_ms = estimate_speech_duration(text, language="auto")
            duration_s = duration_ms / 1000.0
            
            # Detect language for display
            chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
            lang = "Chinese" if chinese_chars / max(len(text), 1) > 0.3 else "English"
            
            return gr.update(
                value=f"📊 Estimated: {duration_s:.1f}s ({duration_ms}ms) | Language: {lang} | {len(text)} chars",
                visible=True
            )
        
        def on_input_text_change(text, max_tokens_per_sentence):
            if text and len(text) > 0:
                text_tokens_list = tts.tokenizer.tokenize(text)
                sentences = tts.tokenizer.split_sentences(text_tokens_list, max_tokens_per_sentence=int(max_tokens_per_sentence))
                data = []
                for i, s in enumerate(sentences):
                    sentence_str = ''.join(s)
                    tokens_count = len(s)
                    data.append([i+1, sentence_str, tokens_count])
                return gr.update(value=data, visible=True)
            else:
                return gr.update(value=[], visible=False)
        
        def on_method_select(emo_control_method):
            if emo_control_method == 1:
                # emotion_reference_group, emo_random, emotion_weight_group, emotion_vector_group, emo_text_group
                return (gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False))
            elif emo_control_method == 2:
                # emotion_reference_group, emo_random, emotion_weight_group, emotion_vector_group, emo_text_group
                return (gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=False))
            elif emo_control_method == 3:
                # emotion_reference_group, emo_random, emotion_weight_group, emotion_vector_group, emo_text_group
                return (gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), gr.update(visible=True))
            else:
                # emotion_reference_group, emo_random, emotion_weight_group, emotion_vector_group, emo_text_group
                return (gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False))
        
        # Connect event handlers
        use_preset.change(
            on_preset_toggle,
            inputs=[use_preset],
            outputs=[audio_upload_group, preset_selection_group]
        )
        
        emo_control_method.select(
            on_method_select,
            inputs=[emo_control_method],
            outputs=[emotion_reference_group, emo_random, emotion_weight_group, emotion_vector_group, emo_text_group]
        )
        
        input_text_single.change(
            on_input_text_change,
            inputs=[input_text_single, max_text_tokens_per_sentence],
            outputs=[sentences_preview]
        )
        
        max_text_tokens_per_sentence.change(
            on_input_text_change,
            inputs=[input_text_single, max_text_tokens_per_sentence],
            outputs=[sentences_preview]
        )
        
        prompt_audio.upload(update_prompt_audio, outputs=[gen_button])
        
        estimate_button.click(
            on_estimate_duration,
            inputs=[input_text_single],
            outputs=[estimated_duration]
        )
        
        gen_button.click(
            gen_single,
            inputs=[emo_control_method, prompt_audio, input_text_single, emo_upload, emo_weight,
                   vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
                   emo_text, emo_random, max_text_tokens_per_sentence,
                   use_preset, preset_dropdown,  # NEW: preset inputs
                   output_format,  # NEW: output format
                   speech_length,  # NEW: duration control
                   diffusion_steps,  # NEW: diffusion steps control
                   *advanced_params],
            outputs=[output_audio]
        )
        
        parallel_gen_button.click(
            gen_parallel,
            inputs=[emo_control_method, prompt_audio, input_text_single, emo_upload, emo_weight,
                   vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
                   emo_text, emo_random, max_text_tokens_per_sentence,
                   use_preset, preset_dropdown,  # NEW: preset inputs
                   max_parallel_chunks, chunk_length_zh, chunk_length_en,  # NEW: parallel settings
                   output_format,  # NEW: output format
                   speech_length,  # NEW: duration control
                   diffusion_steps,  # NEW: diffusion steps control
                   *advanced_params],
            outputs=[output_audio]
        )
        
        # Preset management handlers
        add_preset_btn.click(
            add_speaker_preset,
            inputs=[new_preset_name, new_preset_audio, new_preset_desc],
            outputs=[add_status, preset_dropdown]
        )
        
        delete_preset_btn.click(
            delete_speaker_preset,
            inputs=[delete_preset_dropdown],
            outputs=[delete_status, delete_preset_dropdown]
        )
        
        info_button.click(
            get_preset_info,
            outputs=[preset_info]
        )
    # Enable high concurrency in Gradio queue to support multiple simultaneous requests
    print("⚙️ Setting up Gradio queue with concurrency limit: 10")
    demo.queue(default_concurrency_limit=10)
    
    print(f"🌐 Launching webui on {cmd_args.host}:{cmd_args.port}")
    try:
        demo.launch(server_name=cmd_args.host, server_port=cmd_args.port, share=True)
    except Exception as e:
        print(f"❌ Failed to launch Gradio interface: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
