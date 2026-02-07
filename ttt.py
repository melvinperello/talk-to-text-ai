from io import BytesIO
from pydub import AudioSegment
import soundfile as sf
import numpy as np
import librosa
import torch
import whisper
import librosa
from resemblyzer import VoiceEncoder, preprocess_wav
from sklearn.cluster import AgglomerativeClustering
import whisper


_TARGET_SR = 16000
_PADDING_MS = 150
_PADDING_SAMPLES = int(_PADDING_MS * _TARGET_SR / 1000)


def load_audio(file_path):
    # AudioSegment.from_file() (pydub):
    # Uses ffmpeg backend — handles many formats (m4a, mp3, ogg, etc.)
    # Returns a pydub.AudioSegment object (not NumPy arrays)
    # Good for format conversion and audio manipulation (channels, frame rate)
    # Slower for numerical processing
    # Example: AudioSegment.from_file("file.m4a", format="m4a")
    audio_file = AudioSegment.from_file(file=file_path, format="m4a")
    # Set mono and 16khz
    audio_file = audio_file.set_channels(1).set_frame_rate(_TARGET_SR)
    # -----------------------------------
    # sf.read() (soundfile):
    # Direct NumPy array output — optimized for numerical processing
    # Supports fewer formats than pydub (wav, flac, ogg, etc.)
    # Fast and efficient for ML/DSP workflows
    # Returns (data, sample_rate) tuple immediately usable for model inference
    # Example: audio, sr = sf.read("file.wav")
    wav_buffer = BytesIO()
    audio_file.export(wav_buffer, format="wav")
    wav_buffer.seek(0)
    audio, original_sampling_rate = sf.read(wav_buffer)

    # convert to stero to mono.
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Variable input types — soundfile.read() returns different dtypes depending on the audio file format
    # (int16, int32, float64, etc.). Converting to float32 standardizes this.
    audio = audio.astype(np.float32)

    # Standard audio format — Most audio processing models (including Whisper) expect audio samples normalized to [-1, 1]
    # Prevents clipping — If soundfile loads audio with peak values > 1, dividing by the max normalizes it without losing information
    m = np.max(np.abs(audio))
    if m > 1:
        audio = audio / m

    # Model training data — Whisper was trained on 16 kHz audio. Feeding it a different sample rate can degrade accuracy or require retraining
    # Speech frequency content — Human speech important frequencies are captured well at 16 kHz (Nyquist theorem: up to ~8 kHz after resampling). This is the standard for ASR models
    # Set 16khz
    resample_audio = librosa.resample(
        audio, orig_sr=original_sampling_rate, target_sr=_TARGET_SR
    )

    return resample_audio


def vad(audio_samples):
    audio_samples = audio_samples.astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad", model="silero_vad", verbose=False
    )
    model.to(device)
    # utils is a TUPLE → use indexing
    get_speech_ts = utils[0]
    # save_audio       = utils[1]
    # read_audio       = utils[2]
    # VADIterator      = utils[3]
    # collect_chunks   = utils[4]
    audio_tensor = torch.tensor(audio_samples, device=device)
    speech_ts = get_speech_ts(audio_tensor, model, sampling_rate=_TARGET_SR)

    # Apply padding to each segment
    for seg in speech_ts:
        seg["start"] = max(0, seg["start"] - _PADDING_SAMPLES)
        seg["end"] = min(len(audio_samples), seg["end"] + _PADDING_SAMPLES)
    return speech_ts


def transcribe(audio_samples, speech_segments):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = whisper.load_model("small", device=device)
    for i, seg in enumerate(speech_segments):
        start_sample = int(seg["start"])
        end_sample = int(seg["end"])
        print(
            f"Segment [{i + 1}/{len(speech_segments)}] starting at {start_sample} ending at {end_sample}"
        )
        segment_audio = audio_samples[start_sample:end_sample]
        result = model.transcribe(
            segment_audio, fp16=False, language="en"
        )  # IMPORTANT FOR GTX 1650 SUPER
        seg["text"] = result["text"]


def diarization(audio_samples, speech_segments):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = VoiceEncoder(device=device)

    embeddings = []
    for seg in speech_segments:
        start_sample = int(seg["start"])
        end_sample = int(seg["end"])
        segment_audio = audio_samples[start_sample:end_sample]

        # if len(segment_audio) < 16000:  # skip too short segments (<1s)
        #     continue

        emb = encoder.embed_utterance(segment_audio)
        embeddings.append((seg, emb))

    segs = [x[0] for x in embeddings]
    embs = np.array([x[1] for x in embeddings])

    clustering = AgglomerativeClustering(n_clusters=2)
    labels = clustering.fit_predict(embs)

    diarized = []
    for seg, label in zip(segs, labels):
        diarized.append(
            {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg.get("text", ""),
                "speaker": f"SPEAKER_{label + 1}",
            }
        )
    return diarized


def format_speech_segments(speech_segments):
    """Convert sample indices to timestamps and format for display."""
    formatted = []
    for seg in speech_segments:
        start_seconds = seg["start"] / _TARGET_SR
        minutes = int(start_seconds // 60)
        seconds = int(start_seconds % 60)

        timestamp = f"[{minutes:02d}:{seconds:02d}]"
        speaker = seg["speaker"]
        text = seg["text"].strip()

        formatted_line = f"{timestamp} {speaker}: {text}"
        formatted.append(formatted_line)

    return "\n".join(formatted)


def main():
    FILE = "Voice 251208_225815.m4a"
    audio_data = load_audio(FILE)
    speech_segments = vad(audio_data)
    transcribe(audio_data, speech_segments)
    diarized_segments = diarization(audio_data, speech_segments)
    formatted_output = format_speech_segments(diarized_segments)
    with open(f"{FILE}.txt", "w", encoding="utf-8") as f:
        f.write(formatted_output)


if __name__ == "__main__":
    main()
