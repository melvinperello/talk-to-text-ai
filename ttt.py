from io import BytesIO
from pydub import AudioSegment
import soundfile as sf
import numpy as np
import librosa
import torch
import whisper
import librosa
from resemblyzer import VoiceEncoder
from sklearn.cluster import AgglomerativeClustering
import whisper
import librosa.display
import matplotlib.pyplot as plt
from openai import OpenAI
from urllib.parse import quote
import json
import time
import os

client = OpenAI()


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


def vad(audio_samples, output_file=None):
    file_cache = f"{output_file}.vad.json"
    if output_file and os.path.exists(file_cache):
        with open(file_cache, "r", encoding="utf-8") as f:
            json_data = json.load(f)
            print(f"using cache {file_cache}")
            return json_data.get("speech_segments", [])

    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name()
    vad_model_name = "silero_vad"
    vad_model_source = "snakers4/silero-vad"
    audio_samples = audio_samples.astype(np.float32)

    model, utils = torch.hub.load(
        repo_or_dir=vad_model_source, model=vad_model_name, verbose=False
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

    end_time = time.time()
    json_data = {
        "device_name": device_name,
        "original_duration_sec": len(audio_samples) / _TARGET_SR,
        "num_speech_segments": len(speech_ts),
        "vad_model_name": vad_model_name,
        "vad_model_source": vad_model_source,
        "runtime_sec": end_time - start_time,
        "speech_segments": speech_ts,
    }
    if output_file:
        with open(file_cache, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2)
    return speech_ts


def transcribe(audio_samples, speech_segments, output_file=None):
    file_cache = f"{output_file}.whisper.json"
    if output_file and os.path.exists(file_cache):
        with open(file_cache, "r", encoding="utf-8") as f:
            json_data = json.load(f)
            print(f"using cache {file_cache}")
            return json_data.get("speech_segments", [])

    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name()
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
    end_time = time.time()
    json_data = {
        "device_name": device_name,
        "original_duration_sec": len(audio_samples) / _TARGET_SR,
        "num_speech_segments": len(speech_segments),
        "whisper_model_name": "small",
        "runtime_sec": end_time - start_time,
        "speech_segments": speech_segments,
    }
    if output_file:
        with open(file_cache, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2)
    return speech_segments


def visualize(audio_samples, speech_segments, output_file=None):
    result_segments = []
    for seg in speech_segments:
        start_sample = int(seg["start"])
        end_sample = int(seg["end"])
        start_sample = max(start_sample, 0)
        end_sample = max(end_sample, 0)
        result_segments.append(audio_samples[start_sample:end_sample])
    result = (
        np.concatenate(result_segments) if result_segments else np.array([])
    )

    # Calculate percentage trimmed
    orig_duration = len(audio_samples) / _TARGET_SR
    output_duration = len(result) / _TARGET_SR
    percent_trimmed = ((orig_duration - output_duration) / orig_duration) * 100

    plt.figure(figsize=(15, 8))
    title = f"Silence Percentage - {percent_trimmed:.1f}%"
    plt.suptitle(title, fontsize=14, fontweight="bold")

    # Original
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(audio_samples, sr=_TARGET_SR, color="blue")
    plt.title("Original Audio")

    # # VAD-processed

    plt.subplot(2, 1, 2)
    librosa.display.waveshow(result, sr=_TARGET_SR, color="green")
    plt.title("Voice Activity Detected")

    plt.tight_layout()

    # Save image with base filename (no extension)

    plt.savefig(f"{output_file}.png", dpi=150, bbox_inches="tight")

    plt.close()


def recreate_audio(audio_samples, speech_segments, output_file=None):
    if not speech_segments:
        return

    result_segments = []
    for seg in speech_segments:
        start_sample = int(seg["start"])
        end_sample = int(seg["end"])
        start_sample = max(start_sample, 0)
        end_sample = max(end_sample, 0)
        result_segments.append(audio_samples[start_sample:end_sample])
    result = (
        np.concatenate(result_segments) if result_segments else np.array([])
    )

    sf.write(f"{output_file}.wav", result, _TARGET_SR)


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


def format_speech_segment_for_summary(speech_segments) -> str:
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


def format_speech_segments_for_md(speech_segments) -> str:
    """Convert sample indices to timestamps and format as a Markdown table.

    Produces a table with columns: No., Time, Speaker, Text
    """
    lines = []
    # Header
    lines.append("| No. | Text |")
    lines.append("|----:|:-----|")

    for i, seg in enumerate(speech_segments, start=1):
        start_seconds = seg.get("start", 0) / _TARGET_SR
        minutes = int(start_seconds // 60)
        seconds = int(start_seconds % 60)

        timestamp = f"{minutes:02d}:{seconds:02d}"
        speaker = seg.get("speaker", "")
        text = seg.get("text", "").strip()

        # Escape pipe characters in text to avoid breaking the table
        text = text.replace("|", "\\|")

        # Build row
        row = f"| {i} | [{timestamp}] **{speaker}:**  {text} |"
        lines.append(row)

    return "\n".join(lines)


def summarize(transcript_text):

    summarization_prompt = """Summarize the provided meeting transcript by identifying and condensing the most important discussion points, decisions made, action items assigned (including to whom and due dates if specified), and any unresolved issues or follow-up required. Pay special attention to the following:

- Speaker labels such as "Speaker 1", "Speaker 2" may refer to different people; do not assume consistent identity or specific names unless the transcript provides clarification.
- If possible, infer roles or identities for speakers only if the transcript offers clear context, but otherwise simply refer to the provided labels.
- The transcription may contain errors or unclear sections—note any ambiguities, missing, or questionable content in your summary.

Your summary must be provided in well-structured markdown format, using clear section headers and bullet points or short paragraphs as appropriate.

### Steps

1. Read the entire meeting transcript thoroughly.
2. Organize your response with the following markdown section headers:
- **Key Discussion Points** (summarize main topics discussed)
- **Decisions Made** (list any concrete decisions or agreements)
- **Action Items** (include assignee label, e.g., "Speaker 2", and due dates if available)
- **Open Questions / Follow-ups** (note unresolved matters or unclear portions)
3. If any part of the transcript is unclear, ambiguous, or missing, explicitly note it in the relevant section, indicating the nature of the uncertainty.
4. Be concise but ensure all major points and context are represented.
5. Do not omit crucial context or bias the summary toward any participant.

# Output Format

- Respond in markdown format only.
- Use the specified section headers.
- Each section may use bullet points or short paragraphs.
- Refer to speakers as labeled in the transcript, and do not substitute names unless provided.
- Keep the entire summary between 150 and 300 words, or match summary length proportionally if the transcript is very brief.

# Example

**Input Transcript (Excerpt):**
```
Speaker 1: Let's review the quarterly numbers. Looks like we're over budget on project A.
Speaker 2: Should we adjust the timeline or scope?
Speaker 1: Let's ask Speaker 3 to reforecast.
Speaker 3: I'll prepare the new projections by Friday.
Speaker 1: Great, thanks. Next, the marketing update…
```

**Output Summary (Markdown):**

## Key Discussion Points
- Reviewed quarterly financials, noting that project A is over budget.
- Discussed the possibility of adjusting the timeline or scope for project A.
- Planned for Speaker 3 to provide a reforecast.
- Began discussion of marketing updates. *(Further details may be available in full transcript.)*

## Decisions Made
- Speaker 3 will prepare new project A projections.

## Action Items
- **Speaker 3**: Prepare reforecast of project A budget by Friday.

## Open Questions / Follow-ups
- Whether the timeline or scope for project A will be adjusted remains undecided.
- Speaker identities are listed per label; actual names or roles are not provided in the transcript.
- No further details provided about the marketing update at this time.

*(In real outputs, replace "Speaker 1", "Speaker 2", etc. with the exact labels used in your transcript. If any transcription seems inaccurate or context is missing, explicitly note those uncertainties in this section.)*

# Notes

- Always reference speakers as labeled. If the speaker identity becomes ambiguous or changes, note this explicitly.
- Highlight any sections of the transcript that appear unclear or likely contain transcription errors.
- If required, expand or condense the summary to remain within word limits and maintain all major context.

**Reminder: Output the summary in markdown format using the specified section headers, and pay close attention to the labeling and clarity of speaker identities and any transcription ambiguities.**"""
    response = client.responses.create(
        model="gpt-5.2",
        input=[
            {
                "role": "developer",
                "content": [
                    {
                        "type": "input_text",
                        "text": summarization_prompt,
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": transcript_text,
                    }
                ],
            },
        ],
        text={"format": {"type": "text"}, "verbosity": "medium"},
        reasoning={"effort": "medium", "summary": "auto"},
        tools=[],
        store=False,
        include=[
            "reasoning.encrypted_content",
            "web_search_call.action.sources",
        ],
    )

    # Find the output message (skip reasoning items)
    for item in response.output:
        if hasattr(item, "content") and item.content and len(item.content) > 0:
            return item.content[0].text

    return ""  # Fallback if no content found


def main():
    FILE = "Voice 260113_224416.m4a"
    audio_data = load_audio(FILE)
    speech_segments = vad(audio_data, FILE)
    visualize(audio_data, speech_segments, FILE)
    recreate_audio(audio_data, speech_segments, FILE)
    speech_segments = transcribe(audio_data, speech_segments, FILE)
    diarized_segments = diarization(audio_data, speech_segments)
    txt_table = format_speech_segment_for_summary(diarized_segments)
    summary_md = summarize(txt_table)
    md_table = format_speech_segments_for_md(diarized_segments)

    # URL-encode filename (spaces become %20)
    encoded_file = quote(FILE, safe="")
    final_md = f"""{summary_md}

# Visualization

![VAD Visualization]({encoded_file}.png)

# Full Transcript

{md_table}"""

    with open(f"{FILE}.md", "w", encoding="utf-8") as f:
        f.write(final_md)


if __name__ == "__main__":
    main()
