import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from collections import defaultdict
import io
from audiorecorder import audiorecorder

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(layout="wide")
st.title("FFT-based Automatic Chord Recognition")
st.markdown("### Identify Chords from Audio Signals by Analyzing the Fourier Transform.")

# -----------------------------
# Utility: Frequency → MIDI
# -----------------------------
def freq_to_midi(frequency):
    if frequency <= 0:
        return -1
    midi_note = 69 + 12 * np.log2(frequency / 440.0)
    return int(round(midi_note))

# -----------------------------
# 1) Audio Recording Section
# -----------------------------
st.subheader("🎤 Record Audio")

audio = audiorecorder("Start Recording", "Stop Recording")
recorded_file = None

if len(audio) > 0:
    st.success("Recording complete!")

    # Convert AudioSegment → WAV bytes
    wav_buffer = io.BytesIO()
    audio.export(wav_buffer, format="wav")
    wav_bytes = wav_buffer.getvalue()

    # Playback
    st.audio(wav_bytes, format="audio/wav")

    # Save to variable for main analysis
    recorded_file = io.BytesIO(wav_bytes)
    recorded_file.name = "recorded_audio.wav"

# -----------------------------
# 2) File Uploader Section
# -----------------------------
st.subheader("📁 Or Upload an Audio File")
uploaded_file = st.file_uploader("Upload WAV/MP3", type=['wav', 'mp3'])

# If recorded audio exists → override uploaded_file
if recorded_file is not None:
    uploaded_file = recorded_file

# -----------------------------
# No file yet
# -----------------------------
if uploaded_file is None:
    st.info("Record audio or upload a file to analyze.")
    st.stop()

# -----------------------------
# Begin Analysis
# -----------------------------
try:
    y, sr = librosa.load(uploaded_file, sr=None)

    st.success("Audio loaded!")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Sampling Rate", f"{sr} Hz")
    with col2:
        st.metric("Duration", f"{len(y)/sr:.2f} seconds")

    # -----------------------------
    # FFT Calculation
    # -----------------------------
    N = len(y)
    yf = fft(y)
    xf = fftfreq(N, 1/sr)

    half_n = N // 2
    xf_positive = xf[:half_n]
    yf_positive = np.abs(yf[:half_n])

    # -----------------------------
    # Show Spectrum
    # -----------------------------
    st.subheader("Frequency Spectrum")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(xf_positive, yf_positive)
    ax.set_title("Frequency Spectrum")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_xlim([20, 2000])
    ax.grid(True)
    st.pyplot(fig)

    # -----------------------------
    # Peak Identification
    # -----------------------------
    magnitude_threshold = np.max(yf_positive) * 0.05
    frequency_resolution = sr / N
    min_freq_separation_hz = 10
    distance_bins = int(min_freq_separation_hz / frequency_resolution)

    peak_indices, _ = find_peaks(
        yf_positive,
        height=magnitude_threshold,
        distance=distance_bins
    )

    peak_frequencies = xf_positive[peak_indices]
    peak_magnitudes = yf_positive[peak_indices]

    # -----------------------------
    # Harmonic Filtering
    # -----------------------------
    initial_sorted_peaks = sorted(
        zip(peak_magnitudes, peak_frequencies),
        key=lambda x: x[0],
        reverse=True
    )

    filtered_fundamentals = []
    tolerance = 0.015  # 1.5%

    for mag, freq in initial_sorted_peaks:
        is_harmonic = False
        for fundamental_freq, _ in filtered_fundamentals:
            for n in range(2, 6):
                expected = fundamental_freq * n
                if abs(freq - expected) / expected < tolerance:
                    is_harmonic = True
                    break
            if is_harmonic:
                break
        if not is_harmonic:
            filtered_fundamentals.append((freq, mag))

    filtered_fundamentals.sort(key=lambda x: x[0])
    fundamental_frequencies = [f for f, m in filtered_fundamentals]
    fundamental_midi_notes = [
        freq_to_midi(f) for f in fundamental_frequencies if f > 50
    ]

    st.subheader("Fundamental Frequencies")
    st.write(np.round(fundamental_frequencies, 2))

    # -----------------------------
    # Chord Detection
    # -----------------------------
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F',
                  'F#', 'G', 'G#', 'A', 'A#', 'B']
    chord_templates = {
        'Major': [0, 4, 7],
        'Minor': [0, 3, 7],
        'Dominant 7th': [0, 4, 7, 10],
        'Major 7th': [0, 4, 7, 11],
        'Minor 7th': [0, 3, 7, 10]
    }

    best_match_score = -1
    best_root_midi = -1
    best_chord_type = "Unknown"

    unique_fundamental_midi_notes = sorted(list(set(fundamental_midi_notes)))

    for root_midi in unique_fundamental_midi_notes:
        intervals = set((n - root_midi) % 12 for n in fundamental_midi_notes)

        for chord_type, template in chord_templates.items():
            score = sum(1 for t in template if t in intervals)
            if score >= 2 and score > best_match_score:
                best_match_score = score
                best_root_midi = root_midi
                best_chord_type = chord_type

    if best_root_midi != -1 and best_match_score >= 2:
        root_note = note_names[best_root_midi % 12]
        chord = f"{root_note} {best_chord_type}"
    else:
        chord = "No chord identified"

    st.subheader("🎵 Final Identified Chord")
    st.markdown(f"### **{chord}**")
    st.info(f"Match Score: {best_match_score}")

except Exception as e:
    st.error(f"Error analyzing audio: {e}")
