import torch
import numpy as np
from pydub import AudioSegment
from pathlib import Path

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
TARGET_SR = 16000
PADDING_MS = 150  # padding around speech
# The constant (32768.0) is the int16 scaling factor used to convert 16‑bit PCM samples to float in roughly the [-1, 1] range.
# Pydub returns signed 16‑bit integers; dividing by 32768.0 maps -32768 → -1.0 and 32767 → ~0.99997.
_INT16_SCALE = 32768.0
# ----------------------------------------------------------------------


def _load_m4a(path: str, target_sr: int) -> tuple[AudioSegment, np.ndarray]:
    """
    Load audio using pydub and resample to 16k mono.
    """
    audio = AudioSegment.from_file(path)
    audio = audio.set_channels(1).set_frame_rate(target_sr)
    samples = (
        np.array(audio.get_array_of_samples()).astype("float32") / _INT16_SCALE
    )
    return audio, samples


def trim_audio(input_file: str, output_file: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    print("Loading Silero VAD...")
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

    print(f"Loading audio: {input_file}")
    audio, samples = _load_m4a(input_file, TARGET_SR)

    # Convert to tensor
    audio_tensor = torch.tensor(samples, device=device)

    print("Running VAD...")
    speech_ts = get_speech_ts(audio_tensor, model, sampling_rate=TARGET_SR)

    if not speech_ts:
        print("No speech detected — outputting empty file.")
        AudioSegment.silent(duration=100).export(output_file, format="wav")
        return

    print(f"Detected {len(speech_ts)} speech segments")

    # Build merged audio
    result = AudioSegment.empty()

    for seg in speech_ts:
        start_ms = int(seg["start"] * 1000 / TARGET_SR) - PADDING_MS
        end_ms = int(seg["end"] * 1000 / TARGET_SR) + PADDING_MS

        start_ms = max(start_ms, 0)
        end_ms = max(end_ms, 0)

        result += audio[start_ms:end_ms]

    print(f"Saving: {output_file}")
    result.export(output_file, format="wav")
    print(f"Done! {output_file} created.")


def main():
    input_dir = Path("01_recordings")
    output_dir = Path("02_trimmed")

    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)

    # Find all m4a files
    m4a_files = sorted(input_dir.glob("*.m4a"))

    if not m4a_files:
        print(f"No m4a files found in {input_dir}")
        return

    print(f"Found {len(m4a_files)} m4a file(s):")
    for file in m4a_files:
        print(f"  - {file.name}")

    # Process each file
    for input_file in m4a_files:
        print(f"CUDA: {torch.cuda.is_available()}")
        print(f"CUDA DEVICE: {torch.cuda.get_device_name(0)}")
        output_file = output_dir / f"{input_file.stem}_trimmed.wav"
        print(f"\nProcessing: {input_file.name}")
        try:
            trim_audio(input_file=str(input_file), output_file=str(output_file))
        except Exception as e:
            print(f"Error processing {input_file.name}: {e}")


if __name__ == "__main__":
    main()
