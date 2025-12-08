import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
from pydub import AudioSegment
import tempfile
import os
import numpy as np




def load_audio(path):
    """Load audio using pysoundfile; if that fails, decode via pydub/ffmpeg."""
    try:
        data, sr = sf.read(path, always_2d=False)
        # ensure mono (librosa.load default behavior)
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        return data, sr
    except Exception:
        # fallback: use pydub to convert to wav then read
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_name = tmp.name
        try:
            AudioSegment.from_file(path).export(tmp_name, format="wav")
            data, sr = sf.read(tmp_name)
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            return data, sr
        finally:
            try:
                os.remove(tmp_name)
            except OSError:
                pass

def visualize(input_file, output_file):
    # y_orig, sr_orig = librosa.load(orig_path, sr=None)
    y_orig, sr_orig = load_audio(input_file)
    # y_vad, sr_vad = librosa.load(vad_path, sr=None)
    y_vad, sr_vad = load_audio(output_file)

    # Calculate percentage trimmed
    orig_duration = len(y_orig) / sr_orig
    output_duration = len(y_vad) / sr_vad
    percent_trimmed = ((orig_duration - output_duration) / orig_duration) * 100

    plt.figure(figsize=(15, 8))
    title = f"{os.path.basename(input_file)} - {percent_trimmed:.1f}% reduced"
    plt.suptitle(title, fontsize=14, fontweight='bold')

    # Original
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(y_orig, sr=sr_orig)
    plt.title("Original Audio Waveform")

    # # VAD-processed

    plt.subplot(2, 1, 2)
    librosa.display.waveshow(y_vad, sr=sr_vad, color='orange')
    plt.title("VAD-Filtered Audio Waveform")

    plt.tight_layout()
    
    # Save image with base filename (no extension)
    if input_file:
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_path = os.path.join("00_visualize", base_name + ".png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
    
    plt.close()


def main():
    # Create output directory
    os.makedirs("00_visualize", exist_ok=True)
    
    # Loop through all files in 01_recordings
    recordings_dir = "01_recordings"
    if not os.path.exists(recordings_dir):
        print(f"Error: {recordings_dir} directory not found")
        return
    
    for filename in os.listdir(recordings_dir):
        orig_path = os.path.join(recordings_dir, filename)
        
        # Skip directories
        if os.path.isdir(orig_path):
            continue
        
        # Get base name without extension
        base_name = os.path.splitext(filename)[0]
        
        # Construct corresponding trimmed file path
        vad_path = os.path.join("02_trimmed", base_name + "_trimmed.wav")
        
        # Check if trimmed file exists
        if not os.path.exists(vad_path):
            print(f"Warning: Trimmed file not found for {filename}, skipping...")
            continue
        
        print(f"Processing: {filename}")
        visualize(input_file=orig_path, output_file=vad_path)

if __name__ == "__main__":
    main()
