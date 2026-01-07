import logging
import threading
import wave
from typing import List, Optional

import numpy as np
import torch

from whisperlivekit.timed_objects import SpeakerSegment

logger = logging.getLogger(__name__)

try:
    from nemo.collections.asr.models import SortformerEncLabelModel
    from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor
except ImportError:
    raise SystemExit(
        """Please use `pip install "git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]"` to use the Sortformer diarization"""
    )


class StreamingSortformerState:
    """
    This class creates a class instance that will be used to store the state of the
    streaming Sortformer model.

    Attributes:
        spkcache (torch.Tensor): Speaker cache to store embeddings from start
        spkcache_lengths (torch.Tensor): Lengths of the speaker cache
        spkcache_preds (torch.Tensor): The speaker predictions for the speaker cache parts
        fifo (torch.Tensor): FIFO queue to save the embedding from the latest chunks
        fifo_lengths (torch.Tensor): Lengths of the FIFO queue
        fifo_preds (torch.Tensor): The speaker predictions for the FIFO queue parts
        spk_perm (torch.Tensor): Speaker permutation information for the speaker cache
        mean_sil_emb (torch.Tensor): Mean silence embedding
        n_sil_frames (torch.Tensor): Number of silence frames
    """

    def __init__(self):
        self.spkcache = None  # Speaker cache to store embeddings from start
        self.spkcache_lengths = None
        self.spkcache_preds = None  # speaker cache predictions
        self.fifo = None  # to save the embedding from the latest chunks
        self.fifo_lengths = None
        self.fifo_preds = None
        self.spk_perm = None
        self.mean_sil_emb = None
        self.n_sil_frames = None


class SortformerDiarization:
    def __init__(self, model_name: str = "nvidia/diar_streaming_sortformer_4spk-v2"):
        """
        Stores the shared streaming Sortformer diarization model. Used when a new online_diarization is initialized.
        """
        self.model_name = model_name
        self._load_model(model_name)

    def _load_model(self, model_name: str):
        """Load and configure the Sortformer model for streaming."""
        try:
            self.diar_model = SortformerEncLabelModel.from_pretrained(model_name)
            self.diar_model.eval()

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.diar_model.to(device)

            ## to test
            # for name, param in self.diar_model.named_parameters():
            #     if param.device != device:
            #         raise RuntimeError(f"Parameter {name} is on {param.device} but should be on {device}")

            logger.info(f"Using {device.type.upper()} for Sortformer model")

            self.diar_model.sortformer_modules.chunk_len = 10
            self.diar_model.sortformer_modules.subsampling_factor = 10
            self.diar_model.sortformer_modules.chunk_right_context = 0
            self.diar_model.sortformer_modules.chunk_left_context = 10
            self.diar_model.sortformer_modules.spkcache_len = 188
            self.diar_model.sortformer_modules.fifo_len = 188
            self.diar_model.sortformer_modules.spkcache_update_period = 144
            self.diar_model.sortformer_modules.log = False
            self.diar_model.sortformer_modules._check_streaming_parameters()

        except Exception as e:
            logger.error(f"Failed to load Sortformer model: {e}")
            raise


class SortformerDiarizationOnline:
    def __init__(self, shared_model, sample_rate: int = 16000):
        """
        Initialize the streaming Sortformer diarization system.

        Args:
            sample_rate: Audio sample rate (default: 16000)
            model_name: Pre-trained model name (default: "nvidia/diar_streaming_sortformer_4spk-v2")
        """
        self.sample_rate = sample_rate
        self.diarization_segments = []
        self.buffer_audio = np.array([], dtype=np.float32)
        self.segment_lock = threading.Lock()
        self.global_time_offset = 0.0
        self.debug = False

        self.diar_model = shared_model.diar_model
        self.model_name = getattr(shared_model, "model_name", None)

        self.audio2mel = AudioToMelSpectrogramPreprocessor(
            window_size=0.025, normalize="NA", n_fft=512, features=128, pad_to=0
        )
        self.audio2mel.to(self.diar_model.device)

        self.chunk_duration_seconds = (
            self.diar_model.sortformer_modules.chunk_len
            * self.diar_model.sortformer_modules.subsampling_factor
            * self.diar_model.preprocessor._cfg.window_stride
        )

        self._init_streaming_state()

        self._previous_chunk_features = None
        self._chunk_index = 0
        self._len_prediction = None

        # Audio buffer to store PCM chunks for debugging
        self.audio_buffer = []

        logger.info("SortformerDiarization initialized successfully")

    def _init_streaming_state(self):
        """Initialize the streaming state for the model."""
        batch_size = 1
        device = self.diar_model.device
        async_streaming = getattr(self.diar_model, "async_streaming", False)
        self.streaming_state = self.diar_model.sortformer_modules.init_streaming_state(
            batch_size=batch_size, async_streaming=async_streaming, device=device
        )
        self.total_preds = torch.zeros(
            (batch_size, 0, self.diar_model.sortformer_modules.n_spk), device=device
        )

    def insert_silence(self, silence_duration: Optional[float]):
        """
        Insert silence period by adjusting the global time offset.

        Args:
            silence_duration: Duration of silence in seconds
        """
        with self.segment_lock:
            self.global_time_offset += silence_duration
        logger.debug(
            f"Inserted silence of {silence_duration:.2f}s, new offset: {self.global_time_offset:.2f}s"
        )

    def insert_audio_chunk(self, pcm_array: np.ndarray):
        if self.debug:
            self.audio_buffer.append(pcm_array.copy())
        self.buffer_audio = np.concatenate([self.buffer_audio, pcm_array.copy()])

    async def diarize(self, pcm_array: Optional[np.ndarray] = None):
        """
        Process audio data for diarization in streaming fashion.

        Args:
            pcm_array: Audio data as numpy array
        """
        if pcm_array is not None:
            self.insert_audio_chunk(pcm_array)

        threshold = int(self.chunk_duration_seconds * self.sample_rate)

        if not len(self.buffer_audio) >= threshold:
            return []

        audio = self.buffer_audio[:threshold]
        self.buffer_audio = self.buffer_audio[threshold:]

        device = self.diar_model.device
        audio_signal_chunk = torch.tensor(audio, device=device).unsqueeze(0)
        audio_signal_length_chunk = torch.tensor(
            [audio_signal_chunk.shape[1]], device=device
        )

        processed_signal_chunk, processed_signal_length_chunk = (
            self.audio2mel.get_features(audio_signal_chunk, audio_signal_length_chunk)
        )
        processed_signal_chunk = processed_signal_chunk.to(device)
        processed_signal_length_chunk = processed_signal_length_chunk.to(device)

        if self._previous_chunk_features is not None:
            to_add = self._previous_chunk_features[:, :, -99:].to(device)
            total_features = torch.concat([to_add, processed_signal_chunk], dim=2).to(
                device
            )
        else:
            total_features = processed_signal_chunk.to(device)

        self._previous_chunk_features = processed_signal_chunk.to(device)

        chunk_feat_seq_t = torch.transpose(total_features, 1, 2).to(device)

        with torch.inference_mode():
            left_offset = 8 if self._chunk_index > 0 else 0
            right_offset = 8

            self.streaming_state, self.total_preds = (
                self.diar_model.forward_streaming_step(
                    processed_signal=chunk_feat_seq_t,
                    processed_signal_length=torch.tensor(
                        [chunk_feat_seq_t.shape[1]]
                    ).to(device),
                    streaming_state=self.streaming_state,
                    total_preds=self.total_preds,
                    left_offset=left_offset,
                    right_offset=right_offset,
                )
            )
        new_segments = self._process_predictions()
        if new_segments:
            with self.segment_lock:
                self.diarization_segments.extend(new_segments)

        self._chunk_index += 1
        return new_segments

    def flush_remaining(self):
        """Process any remaining buffered audio by zero-padding to the threshold.

        This ensures at least one final streaming step updates embeddings caches
        before end-of-stream. No-op when there is no remaining audio.
        """
        threshold = int(self.chunk_duration_seconds * self.sample_rate)
        remaining = len(self.buffer_audio)
        if remaining == 0:
            return False

        # Build a padded chunk to reach threshold
        if remaining < threshold:
            padded = np.zeros(threshold, dtype=np.float32)
            padded[:remaining] = self.buffer_audio
            self.buffer_audio = np.array([], dtype=np.float32)
            audio = padded
        else:
            audio = self.buffer_audio[:threshold]
            self.buffer_audio = self.buffer_audio[threshold:]

        device = self.diar_model.device
        audio_signal_chunk = torch.tensor(audio, device=device).unsqueeze(0)
        audio_signal_length_chunk = torch.tensor(
            [audio_signal_chunk.shape[1]], device=device
        )

        processed_signal_chunk, processed_signal_length_chunk = (
            self.audio2mel.get_features(audio_signal_chunk, audio_signal_length_chunk)
        )
        processed_signal_chunk = processed_signal_chunk.to(device)
        processed_signal_length_chunk = processed_signal_length_chunk.to(device)

        if self._previous_chunk_features is not None:
            to_add = self._previous_chunk_features[:, :, -99:].to(device)
            total_features = torch.concat([to_add, processed_signal_chunk], dim=2).to(
                device
            )
        else:
            total_features = processed_signal_chunk.to(device)

        self._previous_chunk_features = processed_signal_chunk.to(device)
        chunk_feat_seq_t = torch.transpose(total_features, 1, 2).to(device)

        with torch.inference_mode():
            left_offset = 8 if self._chunk_index > 0 else 0
            right_offset = 8
            self.streaming_state, self.total_preds = (
                self.diar_model.forward_streaming_step(
                    processed_signal=chunk_feat_seq_t,
                    processed_signal_length=torch.tensor(
                        [chunk_feat_seq_t.shape[1]]
                    ).to(device),
                    streaming_state=self.streaming_state,
                    total_preds=self.total_preds,
                    left_offset=left_offset,
                    right_offset=right_offset,
                )
            )
        new_segments = self._process_predictions()
        if new_segments:
            with self.segment_lock:
                self.diarization_segments.extend(new_segments)
        self._chunk_index += 1
        return True

    def _process_predictions(self):
        """Process model predictions and convert to speaker segments."""
        preds_np = self.total_preds[0].cpu().numpy()
        if preds_np.size == 0:
            return []
        active_speakers = np.argmax(preds_np, axis=1)

        if self._len_prediction is None:
            self._len_prediction = len(active_speakers)  # 12

        frame_duration = self.chunk_duration_seconds / self._len_prediction
        current_chunk_preds = active_speakers[-self._len_prediction :]

        new_segments = []

        with self.segment_lock:
            base_time = (
                self._chunk_index * self.chunk_duration_seconds
                + self.global_time_offset
            )
            current_spk = current_chunk_preds[0]
            start_time = round(base_time, 2)
            for idx, spk in enumerate(current_chunk_preds):
                current_time = round(base_time + idx * frame_duration, 2)
                if spk != current_spk:
                    new_segments.append(
                        SpeakerSegment(
                            speaker=current_spk, start=start_time, end=current_time
                        )
                    )
                    start_time = current_time
                    current_spk = spk
            new_segments.append(
                SpeakerSegment(speaker=current_spk, start=start_time, end=current_time)
            )
        return new_segments

    def get_segments(self) -> List[SpeakerSegment]:
        """Get a copy of the current speaker segments."""
        with self.segment_lock:
            return self.diarization_segments.copy()

    def get_speaker_embeddings(self):
        """Aggregate per-speaker embeddings from streaming state (cache + fifo).

        Returns a dict mapping speaker_id (int) -> embedding (list[float]).
        """
        parts = []

        # Validate streaming state presence
        if self.streaming_state is None:
            raise RuntimeError("Streaming state is not initialized.")

        # Speaker cache
        if getattr(self.streaming_state, "spkcache", None) is not None:
            k = None
            if getattr(self.streaming_state, "spkcache_lengths", None) is not None:
                k = int(self.streaming_state.spkcache_lengths[0].item())
            else:
                k = int(self.streaming_state.spkcache.shape[1])
            if k > 0:
                E = self.streaming_state.spkcache[0, :k, :].detach().cpu().numpy()
                P = None
                if getattr(self.streaming_state, "spkcache_preds", None) is not None:
                    P = (
                        self.streaming_state.spkcache_preds[0, :k, :]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                parts.append((E, P))

        # FIFO queue
        if getattr(self.streaming_state, "fifo", None) is not None:
            kf = None
            if getattr(self.streaming_state, "fifo_lengths", None) is not None:
                kf = int(self.streaming_state.fifo_lengths[0].item())
            else:
                kf = int(self.streaming_state.fifo.shape[1])
            if kf > 0:
                Ef = self.streaming_state.fifo[0, :kf, :].detach().cpu().numpy()
                Pf = None
                if getattr(self.streaming_state, "fifo_preds", None) is not None:
                    Pf = (
                        self.streaming_state.fifo_preds[0, :kf, :]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                parts.append((Ef, Pf))

        if not parts:
            # No embeddings collected yet; return empty dict to avoid failing the stream.
            return {}

        n_spk = int(self.diar_model.sortformer_modules.n_spk)
        out = {}
        for s in range(n_spk):
            vecs = []
            for E, P in parts:
                if P is None:
                    continue
                idx = np.argmax(P, axis=1) == s
                if np.any(idx):
                    vecs.append(E[idx])
            if vecs:
                V = np.concatenate(vecs, axis=0)
                out[int(s)] = V.mean(axis=0).astype(float).tolist()

        return out

    def close(self):
        """Close the diarization system and clean up resources."""
        logger.info("Closing SortformerDiarization")
        with self.segment_lock:
            self.diarization_segments.clear()

        if self.debug:
            concatenated_audio = np.concatenate(self.audio_buffer)
            audio_data_int16 = (concatenated_audio * 32767).astype(np.int16)
            with wave.open("diarization_audio.wav", "wb") as wav_file:
                wav_file.setnchannels(1)  # mono audio
                wav_file.setsampwidth(2)  # 2 bytes per sample (int16)
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_data_int16.tobytes())
            logger.info(
                f"Saved {len(concatenated_audio)} samples to diarization_audio.wav"
            )


def extract_number(s: str) -> int:
    """Extract number from speaker string (compatibility function)."""
    import re

    m = re.search(r"\d+", s)
    return int(m.group()) if m else 0


if __name__ == "__main__":
    import asyncio

    import librosa

    async def main():
        """TEST ONLY."""
        an4_audio = "diarization_audio.wav"
        signal, sr = librosa.load(an4_audio, sr=16000)
        signal = signal[: 16000 * 30]

        print("\n" + "=" * 50)
        print("ground truth:")
        print("Speaker 0: 0:00 - 0:09")
        print("Speaker 1: 0:09 - 0:19")
        print("Speaker 2: 0:19 - 0:25")
        print("Speaker 0: 0:25 - 0:30")
        print("=" * 50)

        diarization_backend = SortformerDiarization()
        diarization = SortformerDiarizationOnline(shared_model=diarization_backend)
        chunk_size = 1600

        for i in range(0, len(signal), chunk_size):
            chunk = signal[i : i + chunk_size]
            new_segments = await diarization.diarize(chunk)
            print(f"Processed chunk {i // chunk_size + 1}")
            print(new_segments)

        segments = diarization.get_segments()
        print("\nDiarization results:")
        for segment in segments:
            print(
                f"Speaker {segment.speaker}: {segment.start:.2f}s - {segment.end:.2f}s"
            )

    asyncio.run(main())
