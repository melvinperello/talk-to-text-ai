import torch

# https://github.com/openai/whisper
import whisper
import soundfile as sf
import numpy as np
import datetime
import librosa
from pathlib import Path


def load_audio_correct(path):
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)  # convert to mono

    # Convert to float32
    audio = audio.astype(np.float32)

    # Normalize to [-1, 1]
    m = np.max(np.abs(audio))
    if m > 1:
        audio = audio / m

    # Resample to 16k
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    return audio


def transcribe(input_file, output_file):
    # Load model
    model = whisper.load_model("small", device="cuda")

    # Load audio safely
    audio = load_audio_correct(input_file)

    # Audio duration (audio is resampled to 16 kHz in load_audio_correct)
    sr = 16000
    audio_duration = audio.shape[0] / sr

    # Whisper expects numpy float32
    start_time = datetime.datetime.now()
    print("Transcription start:", start_time.isoformat())

    print("Running transcription...")
    result = model.transcribe(audio, fp16=False)  # IMPORTANT FOR GTX 1650 SUPER

    end_time = datetime.datetime.now()
    print("Transcription end:  ", end_time.isoformat())

    elapsed = end_time - start_time
    print(f"Transcription elapsed: {elapsed.total_seconds():.2f} seconds")

    print(
        f"Audio start: 0.00s, end: {audio_duration:.2f}s, length: {audio_duration:.2f}s"
    )

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(result["text"])


def main():
    input_dir = Path("02_trimmed")
    output_dir = Path("03_transcripts")

    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)

    # Find all wav files
    wav_files = sorted(input_dir.glob("*.wav"))

    if not wav_files:
        print(f"No wav files found in {input_dir}")
        return

    print(f"Found {len(wav_files)} wav file(s):")
    for file in wav_files:
        print(f"  - {file.name}")

    # Process each file
    for input_file in wav_files:
        print(f"CUDA: {torch.cuda.is_available()}")
        print(f"CUDA DEVICE: {torch.cuda.get_device_name(0)}")
        output_file = output_dir / f"{input_file.stem}.txt"
        print(f"\nProcessing: {input_file.name}")
        try:
            transcribe(input_file=str(input_file), output_file=str(output_file))
        except Exception as e:
            print(f"Error processing {input_file.name}: {e}")


if __name__ == "__main__":
    main()
