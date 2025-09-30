#!/usr/bin/env python3
"""
Speaker Preset Manager for IndexTTS2
Handles persistent caching of speaker embeddings and audio processing results.
"""

import os
import json
import hashlib
import time
from typing import Dict, Optional, Tuple, Any
import pickle
import torch
import librosa
import torchaudio
from pathlib import Path
import logging

class SpeakerPresetManager:
    """
    Manages speaker presets with persistent disk caching for IndexTTS2.
    
    This dramatically speeds up inference by pre-computing and caching:
    - Speaker condition embeddings (spk_cond_emb)
    - Semantic references (S_ref) 
    - Reference mel spectrograms (ref_mel)
    - Speaker styles (style)
    - Prompt conditions (prompt_condition)
    
    Optimization: All speaker data is kept in memory (GPU) after first load.
    """
    
    def __init__(self, cache_dir: str = "speaker_presets", tts_model=None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.presets_file = self.cache_dir / "presets.json"
        self.tts_model = tts_model
        
        # Load existing presets metadata
        self.presets = self._load_presets()
        
        # In-memory cache for loaded speaker data (kept on GPU)
        # This prevents repeated disk I/O and GPU transfers
        self._memory_cache: Dict[str, Dict[str, torch.Tensor]] = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def _load_presets(self) -> Dict[str, Dict[str, Any]]:
        """Load speaker presets from disk"""
        if self.presets_file.exists():
            try:
                with open(self.presets_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load presets: {e}")
                return {}
        return {}
    
    def _save_presets(self):
        """Save speaker presets to disk"""
        try:
            with open(self.presets_file, 'w', encoding='utf-8') as f:
                json.dump(self.presets, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Failed to save presets: {e}")
    
    def _get_audio_hash(self, audio_path: str) -> str:
        """Generate hash for audio file to detect changes"""
        with open(audio_path, 'rb') as f:
            content = f.read()
        return hashlib.md5(content).hexdigest()
    
    def _get_cache_path(self, preset_name: str) -> Path:
        """Get cache file path for a preset"""
        safe_name = "".join(c for c in preset_name if c.isalnum() or c in (' ', '-', '_')).strip()
        return self.cache_dir / f"{safe_name}.cache"
    
    def add_speaker_preset(self, preset_name: str, audio_path: str, description: str = "") -> bool:
        """
        Add a new speaker preset by processing audio and caching all embeddings.
        
        Args:
            preset_name: Unique name for the speaker preset
            audio_path: Path to the speaker's reference audio file
            description: Optional description of the speaker
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not os.path.exists(audio_path):
            self.logger.error(f"Audio file not found: {audio_path}")
            return False
            
        if self.tts_model is None:
            self.logger.error("TTS model not provided")
            return False
            
        try:
            self.logger.info(f"Processing speaker preset: {preset_name}")
            start_time = time.time()
            
            # Generate audio hash for change detection
            audio_hash = self._get_audio_hash(audio_path)
            
            # Check if preset already exists and hasn't changed
            if preset_name in self.presets:
                if self.presets[preset_name].get('audio_hash') == audio_hash:
                    self.logger.info(f"Preset {preset_name} already exists and unchanged")
                    # Ensure it's loaded in memory cache
                    if preset_name not in self._memory_cache:
                        self._load_to_memory(preset_name)
                    return True
            
            # Process audio through the complete speaker pipeline
            processed_data = self._process_speaker_audio(audio_path)
            
            if processed_data is None:
                return False
            
            # Save CPU version to disk for persistence
            cpu_data = {k: v.cpu() if isinstance(v, torch.Tensor) else v 
                       for k, v in processed_data.items()}
            cache_path = self._get_cache_path(preset_name)
            with open(cache_path, 'wb') as f:
                pickle.dump(cpu_data, f)
            
            # Update presets metadata
            self.presets[preset_name] = {
                'audio_path': audio_path,
                'audio_hash': audio_hash,
                'description': description,
                'cache_file': str(cache_path),
                'created_at': time.time(),
                'last_used': time.time()
            }
            
            self._save_presets()
            
            # Store GPU version directly in memory cache (processed_data is already on GPU)
            self._memory_cache[preset_name] = processed_data
            
            processing_time = time.time() - start_time
            self.logger.info(f"Speaker preset '{preset_name}' created and cached in memory in {processing_time:.2f}s")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create speaker preset: {e}")
            return False
    
    def _process_speaker_audio(self, audio_path: str) -> Optional[Dict[str, torch.Tensor]]:
        """
        Process speaker audio through the complete IndexTTS2 pipeline.
        This replicates the expensive computations from infer_vllm_v2.py lines 286-318
        """
        try:
            # Step 1: Load and resample audio (same as lines 287-290)
            audio, sr = librosa.load(audio_path)
            audio = torch.tensor(audio).unsqueeze(0)
            audio_22k = torchaudio.transforms.Resample(sr, 22050)(audio)
            audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio)
            
            # Step 2: Extract W2V-BERT features (same as lines 292-297)
            inputs = self.tts_model.extract_features(audio_16k, sampling_rate=16000, return_tensors="pt")
            input_features = inputs["input_features"]
            attention_mask = inputs["attention_mask"]
            input_features = input_features.to(self.tts_model.device)
            attention_mask = attention_mask.to(self.tts_model.device)
            spk_cond_emb = self.tts_model.get_emb(input_features, attention_mask)
            
            # Step 3: Semantic codec quantization (same as line 299)
            _, S_ref = self.tts_model.semantic_codec.quantize(spk_cond_emb)
            
            # Step 4: Generate mel spectrogram (same as line 300)
            ref_mel = self.tts_model.mel_fn(audio_22k.to(spk_cond_emb.device).float())
            ref_target_lengths = torch.LongTensor([ref_mel.size(2)]).to(ref_mel.device)
            
            # Step 5: Extract speaker style using CAMPPlus (same as lines 302-307)
            feat = torchaudio.compliance.kaldi.fbank(audio_16k.to(ref_mel.device),
                                                   num_mel_bins=80,
                                                   dither=0,
                                                   sample_frequency=16000)
            feat = feat - feat.mean(dim=0, keepdim=True)
            style = self.tts_model.campplus_model(feat.unsqueeze(0))
            
            # Step 6: Generate prompt condition (same as lines 309-312)
            prompt_condition = self.tts_model.s2mel.models['length_regulator'](S_ref,
                                                                              ylens=ref_target_lengths,
                                                                              n_quantizers=3,
                                                                              f0=None)[0]
            
            # Return all processed data as GPU tensors (keep on GPU for memory cache)
            return {
                'spk_cond_emb': spk_cond_emb,
                'S_ref': S_ref,
                'ref_mel': ref_mel,
                'ref_target_lengths': ref_target_lengths,
                'style': style,
                'prompt_condition': prompt_condition,
                'audio_22k': audio_22k,  # Keep for potential future use
                'audio_16k': audio_16k   # Keep for potential future use
            }
            
        except Exception as e:
            self.logger.error(f"Failed to process speaker audio: {e}")
            return None
    
    def _load_to_memory(self, preset_name: str) -> bool:
        """
        Load a speaker preset from disk into memory cache.
        Internal helper method for lazy loading.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if preset_name in self._memory_cache:
            return True  # Already in cache
            
        if preset_name not in self.presets:
            return False
            
        try:
            cache_path = Path(self.presets[preset_name]['cache_file'])
            if not cache_path.exists():
                self.logger.warning(f"Cache file not found for preset '{preset_name}'")
                return False
            
            # Load from disk
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            
            # Move tensors to GPU
            if self.tts_model and hasattr(self.tts_model, 'device'):
                for key, tensor in data.items():
                    if isinstance(tensor, torch.Tensor):
                        data[key] = tensor.to(self.tts_model.device)
            
            # Store in memory cache
            self._memory_cache[preset_name] = data
            self.logger.info(f"Loaded speaker preset '{preset_name}' into memory cache")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load preset '{preset_name}' to memory: {e}")
            return False
    
    def get_speaker_preset(self, preset_name: str) -> Optional[Dict[str, torch.Tensor]]:
        """
        Get speaker preset from memory cache (extremely fast, no disk I/O).
        Lazy loads from disk if not in memory.
        
        Args:
            preset_name: Name of the speaker preset
            
        Returns:
            Dict containing all processed speaker data (on GPU), or None if not found
        """
        if preset_name not in self.presets:
            self.logger.warning(f"Speaker preset '{preset_name}' not found")
            return None
        
        # Check memory cache first (fast path)
        if preset_name in self._memory_cache:
            # Update last used time (lightweight operation)
            self.presets[preset_name]['last_used'] = time.time()
            return self._memory_cache[preset_name]
        
        # Lazy load from disk if not in memory (slow path, happens only once)
        if self._load_to_memory(preset_name):
            self.presets[preset_name]['last_used'] = time.time()
            self._save_presets()  # Save updated timestamp
            return self._memory_cache[preset_name]
        
        return None
    
    def list_presets(self) -> Dict[str, Dict[str, Any]]:
        """Get list of all available speaker presets"""
        return self.presets.copy()
    
    def delete_preset(self, preset_name: str) -> bool:
        """Delete a speaker preset from disk and memory cache"""
        if preset_name not in self.presets:
            return False
            
        try:
            # Delete cache file
            cache_path = Path(self.presets[preset_name]['cache_file'])
            if cache_path.exists():
                cache_path.unlink()
            
            # Remove from memory cache
            if preset_name in self._memory_cache:
                del self._memory_cache[preset_name]
            
            # Remove from presets
            del self.presets[preset_name]
            self._save_presets()
            
            self.logger.info(f"Deleted speaker preset: {preset_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete preset '{preset_name}': {e}")
            return False
    
    def cleanup_unused_presets(self, days_unused: int = 30):
        """Remove presets that haven't been used for a specified number of days"""
        cutoff_time = time.time() - (days_unused * 24 * 60 * 60)
        
        to_delete = []
        for preset_name, preset_data in self.presets.items():
            if preset_data.get('last_used', 0) < cutoff_time:
                to_delete.append(preset_name)
        
        for preset_name in to_delete:
            self.delete_preset(preset_name)
            
        if to_delete:
            self.logger.info(f"Cleaned up {len(to_delete)} unused presets")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the preset cache"""
        total_presets = len(self.presets)
        total_size = 0
        
        for preset_data in self.presets.values():
            cache_path = Path(preset_data['cache_file'])
            if cache_path.exists():
                total_size += cache_path.stat().st_size
        
        return {
            'total_presets': total_presets,
            'presets_in_memory': len(self._memory_cache),
            'total_cache_size_mb': total_size / (1024 * 1024),
            'cache_directory': str(self.cache_dir),
            'memory_cache_hit_rate': f"{len(self._memory_cache)}/{total_presets}" if total_presets > 0 else "0/0"
        }
    
    def preload_all_speakers(self) -> Dict[str, bool]:
        """
        Preload all speaker presets into memory cache at startup.
        This eliminates first-time loading delays during inference.
        
        Returns:
            Dict mapping speaker names to load success status
        """
        results = {}
        self.logger.info(f"Preloading {len(self.presets)} speaker presets into memory...")
        start_time = time.time()
        
        for preset_name in self.presets.keys():
            if preset_name not in self._memory_cache:
                results[preset_name] = self._load_to_memory(preset_name)
            else:
                results[preset_name] = True  # Already loaded
        
        successful = sum(1 for v in results.values() if v)
        elapsed = time.time() - start_time
        
        self.logger.info(f"Preloaded {successful}/{len(self.presets)} speaker presets in {elapsed:.2f}s")
        return results
    
    def is_loaded(self, preset_name: str) -> bool:
        """Check if a speaker preset is currently loaded in memory"""
        return preset_name in self._memory_cache
    
    def unload_speaker(self, preset_name: str) -> bool:
        """
        Unload a speaker from memory cache (keeps it on disk).
        Useful for memory management if you have many speakers.
        
        Returns:
            bool: True if unloaded, False if wasn't loaded
        """
        if preset_name in self._memory_cache:
            del self._memory_cache[preset_name]
            self.logger.info(f"Unloaded speaker preset '{preset_name}' from memory")
            return True
        return False
    
    def clear_memory_cache(self):
        """Clear all speakers from memory cache (keeps them on disk)"""
        count = len(self._memory_cache)
        self._memory_cache.clear()
        self.logger.info(f"Cleared {count} speaker presets from memory cache")


def initialize_preset_manager(tts_model, cache_dir: str = "speaker_presets", preload_all: bool = True):
    """
    Initialize speaker preset manager for IndexTTS2 model.
    This is thread-safe and doesn't modify the inference pipeline.
    
    Args:
        tts_model: The IndexTTS2 model instance
        cache_dir: Directory to store speaker presets
        preload_all: If True, preload all speakers into memory at startup (recommended)
        
    Returns:
        SpeakerPresetManager: Initialized preset manager
    """
    preset_manager = SpeakerPresetManager(cache_dir=cache_dir, tts_model=tts_model)
    tts_model.preset_manager = preset_manager
    
    # Preload all speakers for zero-latency access
    if preload_all and len(preset_manager.presets) > 0:
        preset_manager.preload_all_speakers()
    
    return preset_manager
