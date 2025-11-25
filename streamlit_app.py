import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from collections import defaultdict
import io
from pydub import AudioSegment
# 녹음 기능을 위한 라이브러리 추가
from audiorecorder import audiorecorder


# Define constants
TOLERANCE = 0.015 # 1.5% Tolerance for harmonic check
MIN_FREQ_HZ = 50 # Filter out noise below 50Hz
MAX_FREQ_HZ = 2000 # Max frequency for spectrum visualization
MAX_HARMONIC_N = 8 # Maximum harmonic check range

# Note names for output
note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Chord Templates (Optimized for up to 7 Notes)
chord_templates = {
    # 7음 화음 (7 Notes) - 13th: [1, 3, 5, 7, 9, 11, 13]
    'Major 13th': [0, 4, 7, 11, 2, 5, 9],
    'Minor 13th': [0, 3, 7, 10, 2, 5, 9],
    'Dominant 13th': [0, 4, 7, 10, 2, 5, 9],

    # 6음 화음 (6 Notes) - 11th: [1, 3, 5, 7, 9, 11]
    'Major 11th': [0, 4, 7, 11, 2, 5],
    'Minor 11th': [0, 3, 7, 10, 2, 5],
    'Dominant 11th': [0, 4, 7, 10, 2, 5],

    # 5음 화음 (5 Notes) - 9th: [1, 3, 5, 7, 9]
    'Major 9th': [0, 4, 7, 11, 2],
    'Minor 9th': [0, 3, 7, 10, 2],
    'Dominant 9th': [0, 4, 7, 10, 2],

    # 4음 화음 (4 Notes) - 7th: [1, 3, 5, 7]
    'Major 7th': [0, 4, 7, 11],
    'Minor 7th': [0, 3, 7, 10],
    'Dominant 7th': [0, 4, 7, 10],

    # 3음 화음 (3 Notes) - Triads: [1, 3, 5]
    'Major': [0, 4, 7],
    'Minor': [0, 3, 7]
}

# ----------------------------------------------------------------------
# --- 함수 정의 시작 ---
# ----------------------------------------------------------------------

# (1) frequency to MIDI note
def freq_to_midi(frequency):
    """
    frquency(Hz) to MIDI note number (A4=440Hz, MIDI 69).
    """
    if frequency <= 0:
        return -1
    midi_note = 69 + 12 * np.log2(frequency / 440.0)
    return int(max(0, min(127, round(midi_note))))

# (2) Helper function to get notes of a chord (유지)
def get_notes_string(root_index, chord_type):
    """주어진 루트와 화음 타입에 대한 구성 음정(노트) 문자열을 반환합니다."""
    # 화음 템플릿이 없는 경우, 가장 가까운 Triad(Major, Minor) 템플릿을 사용
    # V7은 Dominant 7th 템플릿 사용
    if 'Dominant' in chord_type:
        type_key = 'Dominant 7th'
    elif 'Major' in chord_type:
        type_key = 'Major'
    elif 'Minor' in chord_type:
        type_key = 'Minor'
    else:
        # 안전장치
        return "" 

    if type_key in chord_templates:
        template = chord_templates[type_key]
        notes_indices = [(root_index + interval) % 12 for interval in template]
        notes_names = [note_names[idx] for idx in notes_indices]
        return f" (`{', '.join(notes_names)}`)"
    return ""


# (3) Chord Recommendation Logic (추천 코드 로직 - 유지)
def get_recommended_chords(root_midi_index, chord_type):
    recommended = []
    
    # 1. Tonic (I/i) 코드일 때 
    if 'Major' in chord_type or 'Minor' in chord_type:
        
        dominant_root = (root_midi_index + 7) % 12
        dominant_name = note_names[dominant_root]
        notes = get_notes_string(dominant_root, 'Dominant 7th')
        recommended.append(f"{dominant_name} Dominant 7th (V7){notes}")
        
        subdominant_root = (root_midi_index + 5) % 12
        subdominant_name = note_names[subdominant_root]
        notes = get_notes_string(subdominant_root, 'Major')
        recommended.append(f"{subdominant_name} Major (IV){notes}")
        
        relative_minor_root = (root_midi_index + 9) % 12
        relative_minor_name = note_names[relative_minor_root]
        notes = get_notes_string(relative_minor_root, 'Minor')
        recommended.append(f"{relative_minor_name} Minor (vi){notes}")

    # 2. Dominant (V7, V9, V11, V13) 코드일 때
    elif 'Dominant' in chord_type:
        tonic_root = (root_midi_index - 7 + 12) % 12
        tonic_name = note_names[tonic_root]
        notes = get_notes_string(tonic_root, 'Major')
        recommended.append(f"{tonic_name} Major (I){notes}")
        
        subdominant_root = (tonic_root + 2) % 12
        subdominant_name = note_names[subdominant_root]
        notes = get_notes_string(subdominant_root, 'Minor')
        recommended.append(f"{subdominant_name} Minor (ii){notes}")
    
    return list(set(recommended))[:4]


# (4) Core Analysis Function (분석 로직은 그대로, UI 요소만 제거)
def run_analysis(y, sr, source_name="Uploaded Audio"):
    
    # ----------------------------------------------------------------------
    # --- 4. Perform FFT and Calculate Spectrum ---
    # ----------------------------------------------------------------------
    N = len(y)
    yf = fft(y)
    xf = fftfreq(N, 1/sr)
    
    half_n = N // 2
    xf_positive = xf[:half_n] # Positive Frequencies
    yf_positive = np.abs(yf[:half_n]) # Magnitude (Amplitude Spectrum)
    
    st.subheader("Frequency spectruem visualization")
    
    # --- 5. Visualize Spectrum ---
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(xf_positive, yf_positive)
    ax.set_title(f'Frequency Spectrum: {source_name}')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude')
    ax.set_xlim([MIN_FREQ_HZ, MAX_FREQ_HZ]) # Musical Frequency Range
    ax.grid(True)
    st.pyplot(fig)

    # ----------------------------------------------------------------------
    # --- 6. Peak Identification and Harmonic Filtering (Core Logic) ---
    # ----------------------------------------------------------------------
    
    # 6-1. Initial Peak Identification
    magnitude_threshold = np.max(yf_positive) * 0.05
    
    peak_indices, properties = find_peaks(yf_positive, 
                                        height=magnitude_threshold, 
                                        prominence=magnitude_threshold*0.2)
    
    # Filter peaks to the musical range
    valid_indices = [i for i in peak_indices if MIN_FREQ_HZ <= xf_positive[i] <= MAX_FREQ_HZ]
    peak_frequencies = xf_positive[valid_indices]
    peak_magnitudes = yf_positive[valid_indices]

    # 6-2. Harmonic Filtering
    initial_sorted_peaks = sorted(zip(peak_frequencies, peak_magnitudes), key=lambda x: x[1], reverse=True)
    filtered_fundamentals = []
    
    for freq, mag in initial_sorted_peaks:
        is_harmonic = False
        
        for existing_freq, _ in filtered_fundamentals:
            for n in range(2, MAX_HARMONIC_N + 1):
                expected_harmonic_freq = existing_freq * n
                
                if abs(freq - expected_harmonic_freq) / expected_harmonic_freq < TOLERANCE:
                    is_harmonic = True
                    break
            if is_harmonic:
                break
        
        if not is_harmonic:
            filtered_fundamentals.append((freq, mag))

    # --- Final Fundamental List Creation ---
    filtered_fundamentals.sort(key=lambda x: x[0])
    fundamental_frequencies = [f for f, m in filtered_fundamentals]
    fundamental_midi_notes = [freq_to_midi(f) for f in fundamental_frequencies if f >= MIN_FREQ_HZ]

    st.subheader("Fundamental frequency analysis")
    st.markdown(f"**Identified fundamental frequencies (Hz):** `{np.round(fundamental_frequencies, 2)}`")
    
    # --- 7. Chord Identification ---
    best_match_score = -1
    best_root_midi = -1
    best_chord_type = ""
    
    unique_fundamental_midi_notes = sorted(list(set(note % 12 for note in fundamental_midi_notes)))
    
    for root_midi_interval in unique_fundamental_midi_notes:
        observed_intervals = set(unique_fundamental_midi_notes)

        for chord_type, template_intervals in chord_templates.items():
            
            expected_notes = set((root_midi_interval + interval) % 12 for interval in template_intervals)
            match_score = sum(1 for note in expected_notes if note in observed_intervals)

            if match_score >= 2 and match_score > best_match_score:
                best_match_score = match_score
                best_root_midi = root_midi_interval
                best_chord_type = chord_type

    # Final Results
    if best_root_midi != -1 and best_match_score >= 2:
        root_name = note_names[best_root_midi]
        identified_chord = f"**{root_name} {best_chord_type}**"
    
        st.markdown(f"## Identified chord: {identified_chord}")
        st.info(f"Chord confidence (maximum {len(chord_templates['Major 13th'])}점): {best_match_score}")
        
        # 코드 추천 기능 호출 및 출력
        recommended_chords = get_recommended_chords(best_root_midi, best_chord_type)
        
        if recommended_chords:
            st.subheader("Suggested compatible chords
            # 요청하신 대로 줄 바꿈 출력 포맷 유지
            formatted_list = "\n".join([f"* {chord}" for chord in recommended_chords])
            st.markdown(formatted_list)

    else:
        st.error("Unable to identify the chord. Please play a single chord and try again.")


# ----------------------------------------------------------------------
# --- Streamlit 웹 페이지 레이아웃 시작 ---
# ----------------------------------------------------------------------

# (5) Streamlit web page settings
st.set_page_config(layout="wide")
st.title("FFT-based Chord Analyzer")
st.markdown("Analyze chords using the Fourier Transform.")

# ----------------------------------------------------------------------
# 1. 마이크 녹음 섹션
# ----------------------------------------------------------------------
st.header("1. Analyze with Microphone")

# 녹음 컴포넌트 생성.
wav_audio_data = audiorecorder("record", "stop")

# 데이터가 존재하고 길이도 0이 아닐 때만 처리
if wav_audio_data is not None and len(wav_audio_data) > 5000:

    st.info("Audio detected. Starting analysis...")

    try:
        # 1. WAV 바이트 데이터를 pydub의 AudioSegment로 변환
        audio_segment = AudioSegment.from_wav(io.BytesIO(wav_audio_data))

        # 2. AudioSegment를 numpy 배열로 변환
        y = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
        sr = audio_segment.frame_rate  # 녹음된 파일의 샘플링 레이트 사용

        # 오디오 신호의 볼륨 정규화 (선택 사항)
        if np.max(np.abs(y)) > 0:
            y /= np.max(np.abs(y))

        # 3. 분석 실행
        run_analysis(y, sr, "recorded audio")

    except Exception as e:
        st.error(f"Failed to process the recorded audio.: {e}")

else:
    st.write("No audio has been recorded yet.")


# ----------------------------------------------------------------------
# 2. 파일 업로드 섹션
# ----------------------------------------------------------------------
st.header("2. Analyze from File Upload")

# (6) file uploader widget
uploaded_file = st.file_uploader("Select an Audio File (WAV, MP3 recommended)", type=['wav', 'mp3'], key='uploader')

if uploaded_file is not None:
    st.info("File detected. Starting analysis...")

    # Run analysis logic only if the file is successfully loaded.
    try:
        y, sr = librosa.load(uploaded_file, sr=None)
        
        # 분석 실행
        run_analysis(y, sr, uploaded_file.name)

    except Exception as e:
        # When the audio file is corrupted or incorrectly formatted
        st.error(f"Error: Failed to analyze audio file.: {e}")
        st.info("Please check if the file is a supported format (WAV or MP3) and retry.")

else:
    # File Upload Pending (only show if no recording is done)
    if wav_audio_data is None or len(wav_audio_data) < 5000:
        st.info("Record audio or upload a file to start the analysis.")
