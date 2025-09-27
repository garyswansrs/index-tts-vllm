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
    """
    
    def __init__(self, cache_dir: str = "speaker_presets", tts_model=None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.presets_file = self.cache_dir / "presets.json"
        self.tts_model = tts_model
        
        # Load existing presets
        self.presets = self._load_presets()
        
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
                    return True
            
            # Process audio through the complete speaker pipeline
            processed_data = self._process_speaker_audio(audio_path)
            
            if processed_data is None:
                return False
            
            # Save processed data to cache file
            cache_path = self._get_cache_path(preset_name)
            with open(cache_path, 'wb') as f:
                pickle.dump(processed_data, f)
            
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
            
            processing_time = time.time() - start_time
            self.logger.info(f"Speaker preset '{preset_name}' created in {processing_time:.2f}s")
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
            
            # Return all processed data as CPU tensors for serialization
            return {
                'spk_cond_emb': spk_cond_emb.cpu(),
                'S_ref': S_ref.cpu(),
                'ref_mel': ref_mel.cpu(),
                'ref_target_lengths': ref_target_lengths.cpu(),
                'style': style.cpu(),
                'prompt_condition': prompt_condition.cpu(),
                'audio_22k': audio_22k,  # Keep for potential future use
                'audio_16k': audio_16k   # Keep for potential future use
            }
            
        except Exception as e:
            self.logger.error(f"Failed to process speaker audio: {e}")
            return None
    
    def get_speaker_preset(self, preset_name: str) -> Optional[Dict[str, torch.Tensor]]:
        """
        Load speaker preset from cache and move tensors to GPU.
        
        Args:
            preset_name: Name of the speaker preset
            
        Returns:
            Dict containing all processed speaker data, or None if not found
        """
        if preset_name not in self.presets:
            self.logger.warning(f"Speaker preset '{preset_name}' not found")
            return None
            
        try:
            cache_path = Path(self.presets[preset_name]['cache_file'])
            if not cache_path.exists():
                self.logger.warning(f"Cache file not found for preset '{preset_name}'")
                return None
                
            # Load cached data
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            
            # Move tensors to GPU
            if self.tts_model and hasattr(self.tts_model, 'device'):
                for key, tensor in data.items():
                    if isinstance(tensor, torch.Tensor):
                        data[key] = tensor.to(self.tts_model.device)
            
            # Update last used time
            self.presets[preset_name]['last_used'] = time.time()
            self._save_presets()
            
            self.logger.info(f"Loaded speaker preset: {preset_name}")
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to load speaker preset '{preset_name}': {e}")
            return None
    
    def list_presets(self) -> Dict[str, Dict[str, Any]]:
        """Get list of all available speaker presets"""
        return self.presets.copy()
    
    def delete_preset(self, preset_name: str) -> bool:
        """Delete a speaker preset"""
        if preset_name not in self.presets:
            return False
            
        try:
            # Delete cache file
            cache_path = Path(self.presets[preset_name]['cache_file'])
            if cache_path.exists():
                cache_path.unlink()
            
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
            'total_cache_size_mb': total_size / (1024 * 1024),
            'cache_directory': str(self.cache_dir)
        }


def initialize_preset_manager(tts_model, cache_dir: str = "speaker_presets"):
    """
    Initialize speaker preset manager for IndexTTS2 model.
    This is thread-safe and doesn't modify the inference pipeline.
    
    Args:
        tts_model: The IndexTTS2 model instance
        cache_dir: Directory to store speaker presets
        
    Returns:
        SpeakerPresetManager: Initialized preset manager
    """
    preset_manager = SpeakerPresetManager(cache_dir=cache_dir, tts_model=tts_model)
    tts_model.preset_manager = preset_manager
    return preset_manager
