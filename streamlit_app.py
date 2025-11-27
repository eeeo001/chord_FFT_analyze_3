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
    # Load audio file
    y, sr = librosa.load(uploaded_file, sr=None)

    st.success("Audio loaded!")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Sampling Rate", f"{sr} Hz")
    with col2:
        st.metric("Duration", f"{len(y)/sr:.2f} sec")

    # -----------------------------
    # FFT
    # -----------------------------
    N = len(y)
    yf = fft(y)
    xf = fftfreq(N, 1/sr)

    half = N // 2
    xf_pos = xf[:half]
    yf_pos = np.abs(yf[:half])

    # Plot
    st.subheader("Frequency Spectrum")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(xf_pos, yf_pos)
    ax.set_xlim([20, 2000])
    st.pyplot(fig)

    # -----------------------------
    # Peak Detection
    # -----------------------------
    threshold = np.max(yf_pos) * 0.05
    freq_resolution = sr / N
    distance_bins = int(10 / freq_resolution)

    peak_idx, _ = find_peaks(yf_pos, height=threshold, distance=distance_bins)
    peak_freqs = xf_pos[peak_idx]
    peak_mags = yf_pos[peak_idx]

    # -----------------------------
    # Harmonic Filtering → Fundamental Frequencies
    # -----------------------------
    peaks_sorted = sorted(zip(peak_mags, peak_freqs), reverse=True)
    fund_freqs = []
    tolerance = 0.015

    for mag, f in peaks_sorted:
        harmonic = False
        for (ff, _) in fund_freqs:
            for n in range(2, 6):
                if abs(f - ff * n) / (ff * n) < tolerance:
                    harmonic = True
                    break
            if harmonic:
                break
        if not harmonic:
            fund_freqs.append((f, mag))

    fund_freqs = [f for f, m in sorted(fund_freqs)]
    fund_midi = [freq_to_midi(f) for f in fund_freqs if f > 50]

    st.subheader("Detected Fundamental Frequencies")
    st.write(fund_freqs)

    # -----------------------------
    # Chord Matching
    # -----------------------------
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F',
                  'F#', 'G', 'G#', 'A', 'A#', 'B']

    chord_templates = {
        "Major": [0, 4, 7],
        "Minor": [0, 3, 7],
        "Dominant7": [0, 4, 7, 10],
        "Major7": [0, 4, 7, 11],
        "Minor7": [0, 3, 7, 10]
    }

    unique_midi = sorted(list(set(fund_midi)))

    best_score = -1
    best_root = None
    best_type = None

    for root in unique_midi:
        intervals = set((m - root) % 12 for m in fund_midi)

        for chord_type, pattern in chord_templates.items():
            score = sum(1 for p in pattern if p in intervals)

            if score > best_score and score >= 2:
                best_score = score
                best_root = root
                best_type = chord_type

    if best_root is None:
        final_chord = "No chord identified"
    else:
        final_chord = f"{note_names[best_root % 12]} {best_type}"

    # -----------------------------
    # 🎵 Final Output + 구성음 + 추천코드
    # -----------------------------

    CHORDS = {
        "C Major": ["C", "E", "G"],
        "C Minor": ["C", "Eb", "G"],
        "C Dominant7": ["C", "E", "G", "Bb"],
        "C Major7": ["C", "E", "G", "B"],
        "C Minor7": ["C", "Eb", "G", "Bb"],
        # 필요 시 더 추가 가능
    }

    RECOMMENDED = {
        "C Major": ["F Major", "G Major", "A Minor"],
        "A Minor": ["D Minor", "E Major", "G Major"],
        "G Major": ["C Major", "D Major", "E Minor"],
        "E Minor": ["G Major", "A Minor", "C Major"],
        "F Major": ["Bb Major", "C Major", "D Minor"]
    }

    st.header("🎶 Final Identified Chord")
    st.write(f"### {final_chord}")
    st.write(f"Match Score: {best_score}")

    # 구성음
    if final_chord in CHORDS:
        st.write("**구성음:** " + ", ".join(CHORDS[final_chord]))
    else:
        st.write("구성음 데이터 없음.")

    # 추천 코드
    if final_chord in RECOMMENDED:
        rec = RECOMMENDED[final_chord]
        st.write("**추천 코드 진행:** " + " → ".join(rec))
    else:
        st.write("추천 코드 데이터 없음.")

except Exception as e:
    st.error(f"Error analyzing audio: {e}")
