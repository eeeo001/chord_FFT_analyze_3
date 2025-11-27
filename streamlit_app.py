import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from collections import defaultdict
import io
from pydub import AudioSegment
# 녹음 기능을 위한 라이브러리 추가 (pip install audiorecorder pydub)
from audiorecorder import audiorecorder 
# Note: pydub requires FFmpeg to be installed on the system.


# Define constants
TOLERANCE = 0.015 # 1.5% Tolerance for harmonic check
MIN_FREQ_HZ = 50 # Filter out noise below 50Hz
MAX_FREQ_HZ = 2000 # Max frequency for spectrum visualization
MAX_HARMONIC_N = 8 # Maximum harmonic check range

# Note names for output
note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Chord Templates (Optimized for up to 7 Notes)
# 템플릿의 길이가 길수록, 실제 관찰된 음이 부족할 때 정규화 점수가 낮아짐.
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

# (1-2) 화음의 구성음을 문자열로 반환하는 함수
def get_chord_interval_string(root_index, chord_type):
    template = chord_templates.get(chord_type)
    if not template:
        return ""
    
    notes = [note_names[(root_index + interval) % 12] for interval in template]
    return f"({', '.join(notes)})"


# (2) Chord Recommendation Logic (추천 코드 로직)
def get_recommended_chords(root_midi_index, chord_type):
    recommended = []
    
    # 1. Tonic (I/i) 코드일 때 (Major, Minor, 7th, 9th, 11th, 13th 모두 해당)
    if 'Major' in chord_type or 'Minor' in chord_type:
        
        # V7 (Dominant) 코드 추천: 루트에서 +7
        dominant_root = (root_midi_index + 7) % 12
        dominant_name = note_names[dominant_root]
        recommended.append(f"{dominant_name} Dominant 7th (V7)")
        
        # IV (Subdominant) 코드 추천: 루트에서 +5
        subdominant_root = (root_midi_index + 5) % 12
        subdominant_name = note_names[subdominant_root]
        recommended.append(f"{subdominant_name} Major (IV)")
        
        # vi (Relative Minor/Tonic Substitute) 코드 추천: 루트에서 +9
        relative_minor_root = (root_midi_index + 9) % 12
        relative_minor_name = note_names[relative_minor_root]
        recommended.append(f"{relative_minor_name} Minor (vi)")

    # 2. Dominant (V7, V9, V11, V13) 코드일 때
    elif 'Dominant' in chord_type:
        # I (Tonic) 코드 추천: 루트에서 -7 (해결)
        tonic_root = (root_midi_index - 7 + 12) % 12
        tonic_name = note_names[tonic_root]
        recommended.append(f"{tonic_name} Major (I)")
        
        # ii (Minor Subdominant) 코드 추천: I에서 +2
        subdominant_root = (tonic_root + 2) % 12
        subdominant_name = note_names[subdominant_root]
        recommended.append(f"{subdominant_name} Minor (ii)")
    
    return list(set(recommended))[:4]


# (3) Core Analysis Function (기존 코드를 함수로 통합)
def run_analysis(y, sr, source_name="Uploaded Audio"):
    # --- Display File Information ---
    st.success("File successfully loaded!")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Sampling Rate (sr)", f"{sr} Hz")
    with col2:
        st.metric("Duration", f"{len(y)/sr:.2f} seconds")

    # ----------------------------------------------------------------------
    # --- 4. Perform FFT and Calculate Spectrum ---
    # ----------------------------------------------------------------------
    N = len(y)
    yf = fft(y)
    xf = fftfreq(N, 1/sr)
    
    half_n = N // 2
    xf_positive = xf[:half_n] # Positive Frequencies
    yf_positive = np.abs(yf[:half_n]) # Magnitude (Amplitude Spectrum)
    
    st.subheader("Frequency Spectrum Visualization")
    
    # --- 5. Visualize Spectrum ---
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(xf_positive, yf_positive)
    ax.set_title(f'Frequency Spectrum: {source_name}')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude')
    ax.set_xlim([MIN_FREQ_HZ, MAX_FREQ_HZ]) # Musical Frequency Range
    ax.grid(True)
    st.pyplot(fig)
    # 

    # ----------------------------------------------------------------------
    # --- 6. Peak Identification and Harmonic Filtering (Core Logic) ---
    # ----------------------------------------------------------------------
    
    # 6-1. Initial Peak Identification
    magnitude_threshold = np.max(yf_positive) * 0.05
    
    # Prominence 조건 강화 (음성 신호의 노이즈/고조파 필터링 강화)
    peak_indices, properties = find_peaks(yf_positive, 
                                         height=magnitude_threshold, 
                                         prominence=magnitude_threshold * 0.3) # Prominence 임계값 상향 조정
    
    # Filter peaks to the musical range
    valid_indices = [i for i in peak_indices if MIN_FREQ_HZ <= xf_positive[i] <= MAX_FREQ_HZ]
    peak_frequencies = xf_positive[valid_indices]
    peak_magnitudes = yf_positive[valid_indices]

    # 6-2. Harmonic Filtering (Correcting Fundamental vs. Harmonic confusion)
    # 진폭이 큰 순서대로 정렬 (가장 큰 피크부터 근음 후보로 검사)
    initial_sorted_peaks = sorted(zip(peak_frequencies, peak_magnitudes), key=lambda x: x[1], reverse=True)
    filtered_fundamentals = []
    
    for freq, mag in initial_sorted_peaks:
        is_harmonic = False
        
        # 이미 식별된 근음의 하모닉인지 확인
        for existing_freq, _ in filtered_fundamentals:
            # 2차부터 MAX_HARMONIC_N차까지 하모닉 체크
            for n in range(2, MAX_HARMONIC_N + 1):
                expected_harmonic_freq = existing_freq * n
                
                # 상대적 오차 TOLERANCE 내에 있는지 확인
                if abs(freq - expected_harmonic_freq) / expected_harmonic_freq < TOLERANCE:
                    is_harmonic = True
                    break
            if is_harmonic:
                break
        
        if not is_harmonic:
            # 하모닉이 아니면 근음으로 추가
            filtered_fundamentals.append((freq, mag))

    # --- Final Fundamental List Creation ---
    filtered_fundamentals.sort(key=lambda x: x[0])
    fundamental_frequencies = [f for f, m in filtered_fundamentals]
    fundamental_midi_notes = [freq_to_midi(f) for f in fundamental_frequencies if f >= MIN_FREQ_HZ]

    st.subheader("Fundamental Frequency Analysis")
    st.markdown(f"**Detected Fundamental Frequencies (Hz):** `{np.round(fundamental_frequencies, 2)}`")
    
    # ----------------------------------------------------------------------
    # --- 7. Chord Identification (Normalized Score) ---
    # ----------------------------------------------------------------------
    
    # best_match_score를 정규화된 비율(0.0 ~ 1.0)로 초기화
    best_match_score = -1.0 
    best_root_midi = -1
    best_chord_type = ""
    
    # Unique Note Classes (0~11)
    unique_fundamental_midi_notes = sorted(list(set(note % 12 for note in fundamental_midi_notes)))
    
    # Match the identified notes to chord templates based on music theory.
    for root_midi_interval in unique_fundamental_midi_notes:
        observed_intervals = set(unique_fundamental_midi_notes)

        for chord_type, template_intervals in chord_templates.items():
            
            expected_notes = set((root_midi_interval + interval) % 12 for interval in template_intervals)
            
            # 1. 일치하는 음의 개수
            match_count = sum(1 for note in expected_notes if note in observed_intervals)
            
            # 2. 정규화 점수 (일치율) 계산
            template_length = len(template_intervals)
            normalized_score = match_count / template_length
            
            # 3. 최고 점수 업데이트: 일치 개수가 최소 2개 이상이고, 일치율이 더 높을 때 업데이트
            if match_count >= 2 and normalized_score > best_match_score:
                best_match_score = normalized_score
                best_root_midi = root_midi_interval
                best_chord_type = chord_type

    # Final Results
    if best_root_midi != -1 and best_match_score >= 0.5: # 최소 일치율 50% 이상으로 기준 설정
        root_name = note_names[best_root_midi]
        identified_chord = f"**{root_name} {best_chord_type}**"
        
        st.markdown(f"## 최종 식별 화음: {identified_chord}")
        st.info(f"화음 일치율: **{best_match_score:.2f}** (최소 0.50 이상 필요)")
        
        # 코드 추천 기능 호출 및 출력
        recommended_chords = get_recommended_chords(best_root_midi, best_chord_type)
        
        if recommended_chords:
            st.subheader("Recommended Chords (음악 이론 기반)")

            formatted_list = []
            for chord in recommended_chords:
                chord_name = chord.split("(")[0].strip()
                # Chord Type 추출: 예) Dominant 7th
                chord_type_parts = chord_name.split(" ")[1:]
                chord_type = " ".join(chord_type_parts)
                
                # Root Name 추출: 예) G
                root_name = chord_name.split(" ")[0]
                root_index = note_names.index(root_name)

                interval_string = get_chord_interval_string(root_index, chord_type)
                formatted_list.append(f"* **{chord}** {interval_string}")
            
            st.markdown("\n".join(formatted_list))

    else:
        st.error("Chord identification failed. (일치율 50% 미만) Please try again with a single, clear chord.")


# ----------------------------------------------------------------------
# --- Streamlit 웹 페이지 레이아웃 시작 ---
# ----------------------------------------------------------------------

# (2) Streamlit web page settings
st.set_page_config(layout="wide")
st.title("FFT-based Chord Analyzer (화음 일치율 정규화 적용)")
st.markdown("라이브 녹음 또는 파일 업로드를 통해 화음을 분석합니다.")

# ----------------------------------------------------------------------
# 1. 마이크 녹음 섹션
# ----------------------------------------------------------------------
st.header("1. Analyze with Microphone 🎙️")
st.caption("녹음 시작 버튼을 누르고 명료하게 화음을 연주해주세요.")

# 녹음 컴포넌트 생성. 녹음된 wav 바이트 데이터를 반환합니다.
wav_audio_data = audiorecorder("녹음 시작", "녹음 중지")

# 데이터가 존재하고 길이도 0이 아닐 때만 처리 (녹음된 데이터는 5KB보다 커야 유효하다고 가정)
if wav_audio_data is not None and len(wav_audio_data) > 5000:

    st.info("Audio detected. Starting analysis...")

    try:
        # 1. WAV 바이트 데이터를 pydub의 AudioSegment로 변환
        audio_segment = AudioSegment.from_wav(io.BytesIO(wav_audio_data))

        # 2. AudioSegment를 numpy 배열로 변환
        # 데이터 타입 조정 (pydub은 보통 16비트 정수)
        y = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
        sr = audio_segment.frame_rate  # 녹음된 파일의 샘플링 레이트 사용

        # 오디오 신호의 볼륨 정규화 (필수)
        if np.max(np.abs(y)) > 0:
            y /= np.max(np.abs(y))

        # 3. 분석 실행
        run_analysis(y, sr, "Recorded Audio")

    except Exception as e:
        st.error(f"Failed to process the recorded audio: {e}")
        st.caption("Pydub 오류일 수 있습니다. 시스템에 FFmpeg이 설치되어 있는지 확인해주세요.")

else:
    st.write("No audio has been recorded yet.")


# ----------------------------------------------------------------------
# 2. 파일 업로드 섹션
# ----------------------------------------------------------------------
st.header("---")
st.header("2. Analyze from File Upload 📁")

# (3) file uploader widget
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
