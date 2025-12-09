# talk-to-text

In a small office with only a handful of employees, keeping accurate meeting notes proved difficult. The team found a practical solution: placing a phone in the middle of the table to record the entire discussion. Later, they would replay the recording to create clear and comprehensive meeting notes.

Recording meetings solved the challenge of note-taking, but when we explored speech-to-text providers, the costs became a concern.

| Provider                    | Price Per Hour (USD) | Price Per Hour (PHP) |
| --------------------------- | -------------------- | -------------------- |
| AWS Transcribe              | $1.44                | P84.91               |
| Google Cloud Speech to Text | $0.96                | P56.61               |

At first glance, these rates may not seem high. However, in the Philippines—particularly in my region—the minimum wage is about ₱500.00 per day, as outlined in [Wage Order RBIII-25](https://bir-cdn.bir.gov.ph/BIR/pdf/01.-Wage-Order-No.-RBIII-25.pdf). This means that even a few hours of transcription can quickly add up to a significant portion of a worker’s daily income, making these services relatively expensive in our local context.

**Sources** (Details from December 07, 2025.)

-   [AWS Transcribe](https://aws.amazon.com/transcribe/pricing/)
-   [Google Cloud Speech to Text](https://cloud.google.com/speech-to-text?hl=en#pricing)

## Solution

To reduce costs, the team shifted from cloud-based speech-to-text services to consumer-grade hardware capable of running transcription locally, using only office resources.
The process involved:

-   Silero VAD (Voice Activity Detection): was used to filter out silences, ensuring that only speech segments were processed.
    This significantly reduced the transcription workload and shortened the overall processing time.
    https://github.com/snakers4/silero-vad
-   OpenAI Whisper: Applied to transcribe the detected speech into text with high accuracy.
    This approach eliminated recurring provider fees while keeping transcription fully in-house, making it more sustainable and cost-effective for a small office environment.
    https://github.com/openai/whisper

```mermaid
flowchart LR
    Meeting([Meeting]) --> Recording([Recording])
    Recording --> VAD([Silero VAD])
    VAD --> Whisper([OpenAI Whisper])
    Whisper --> Notes([Meeting Notes])
```

## System Information

### Operating System

```bat
wmic os get Caption,OSArchitecture,Version
```

```text
Caption                   OSArchitecture  Version
Microsoft Windows 11 Pro  64-bit          10.0.26200
```

### CPU

```bat
wmic cpu get Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed
```

```text
MaxClockSpeed  Name                               NumberOfCores  NumberOfLogicalProcessors
3600           AMD Ryzen 3 3100 4-Core Processor  4              8
```

### Memory

```bat
wmic memorychip get Capacity,Speed,PartNumber
```

```text
Capacity    PartNumber          Speed
8589934592  TEAMGROUP-UD4-3200  2400
8589934592  TEAMGROUP-UD4-3200  2400
```

### CUDA

```bat
nvcc --version
```

```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Tue_May_27_02:24:01_Pacific_Daylight_Time_2025
Cuda compilation tools, release 12.9, V12.9.86
Build cuda_12.9.r12.9/compiler.36037853_0
```

**Download Driver:** https://developer.nvidia.com/cuda-12-9-1-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local

**Download File Name:** cuda_12.9.1_576.57_windows.exe

### Driver

```bat
nvidia-smi
```

```text
Sat Dec  6 18:42:34 2025
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 581.80                 Driver Version: 581.80         CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce GTX 1650 ...  WDDM  |   00000000:06:00.0  On |                  N/A |
| 26%   38C    P8             12W /  100W |     986MiB /   4096MiB |     25%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
```

### FFMPEG

```bat
ffmpeg -version
```

```text
ffmpeg version 7.0-full_build-www.gyan.dev Copyright (c) 2000-2024 the FFmpeg developers
built with gcc 13.2.0 (Rev5, Built by MSYS2 project)
configuration: --enable-gpl --enable-version3 --enable-static --disable-w32threads --disable-autodetect --enable-fontconfig --enable-iconv --enable-gnutls --enable-libxml2 --enable-gmp --enable-bzlib --enable-lzma --enable-libsnappy --enable-zlib --enable-librist --enable-libsrt --enable-libssh --enable-libzmq --enable-avisynth --enable-libbluray --enable-libcaca --enable-sdl2 --enable-libaribb24 --enable-libaribcaption --enable-libdav1d --enable-libdavs2 --enable-libuavs3d --enable-libxevd --enable-libzvbi --enable-librav1e --enable-libsvtav1 --enable-libwebp --enable-libx264 --enable-libx265 --enable-libxavs2 --enable-libxeve --enable-libxvid --enable-libaom --enable-libjxl --enable-libopenjpeg --enable-libvpx --enable-mediafoundation --enable-libass --enable-frei0r --enable-libfreetype --enable-libfribidi --enable-libharfbuzz --enable-liblensfun --enable-libvidstab --enable-libvmaf --enable-libzimg --enable-amf --enable-cuda-llvm --enable-cuvid --enable-dxva2 --enable-d3d11va --enable-d3d12va --enable-ffnvcodec --enable-libvpl --enable-nvdec --enable-nvenc --enable-vaapi --enable-libshaderc --enable-vulkan --enable-libplacebo --enable-opencl --enable-libcdio --enable-libgme --enable-libmodplug --enable-libopenmpt --enable-libopencore-amrwb --enable-libmp3lame --enable-libshine --enable-libtheora --enable-libtwolame --enable-libvo-amrwbenc --enable-libcodec2 --enable-libilbc --enable-libgsm --enable-libopencore-amrnb --enable-libopus --enable-libspeex --enable-libvorbis --enable-ladspa --enable-libbs2b --enable-libflite --enable-libmysofa --enable-librubberband --enable-libsoxr --enable-chromaprint
libavutil      59.  8.100 / 59.  8.100
libavcodec     61.  3.100 / 61.  3.100
libavformat    61.  1.100 / 61.  1.100
libavdevice    61.  1.100 / 61.  1.100
libavfilter    10.  1.100 / 10.  1.100
libswscale      8.  1.100 /  8.  1.100
libswresample   5.  1.100 /  5.  1.100
libpostproc    58.  1.100 / 58.  1.100
```

**Download Release:** https://github.com/GyanD/codexffmpeg/releases/tag/7.0

**Download File Name:** ffmpeg-7.0-full_build.zip

### Python 3.12.10

```bat
python --version
```

```text
Python 3.12.10
```

### Requirements

```bat
python -m venv env
env\Scripts\activate.bat

# Silero VAD
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu129
pip install numpy
pip install pydub
pip install packaging

# Visualize
pip install librosa
pip install matplotlib

# Open AI Whispher
pip install openai-whisper
```

### How To Run?

Create a folder named "01_recordings" and put all the m4a (recording files) in the folder.

```bash
# Trim the audio this will create a folder "02_trimmed"
python 02_trim.py

# This will create visuals of the audio after trimming and will create a folder "00_visualize"
python 00_visualize.py

# This will create a folder "03_transcripts" and will contain all the transcription.
python 03_transcribe.py
```
