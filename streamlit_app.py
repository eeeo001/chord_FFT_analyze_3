import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from collections import defaultdict
import io
from audiorecorder import audiorecorder
import pandas as pd # pandas import 추가

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(layout="wide")
st.title("FFT 기반 자동 화음 인식 (Automatic Chord Recognition)")
st.markdown("### 푸리에 변환 분석을 통해 오디오 신호에서 화음을 식별합니다.")

# -----------------------------
# Utility: Frequency → MIDI
# -----------------------------
def freq_to_midi(frequency):
    if frequency <= 0:
        return -1
    # 440 Hz (A4) is MIDI note 69
    midi_note = 69 + 12 * np.log2(frequency / 440.0)
    return int(round(midi_note))

# -----------------------------
# Utility: Harmonic Recommendation (화성적 추천)
# -----------------------------
# 식별된 화음과 화성적으로 잘 어울리는 코드 3개를 추천합니다. (다이아토닉 기반)
def get_harmonic_recommendations(root_midi, chord_type, note_names):
    recommendations = []
    root_idx = root_midi % 12

    # I: Major, ii: Minor, iii: Minor, IV: Major, V: Dom7, vi: Minor (다이아토닉 화음 기준)
    
    if 'Major' in chord_type: # I (으뜸 화음)으로 간주
        # 추천 1: IV Chord (버금딸림 화음)
        iv_root_idx = (root_idx + 5) % 12
        recommendations.append({'root_midi': iv_root_idx, 'chord_type': 'Major'})
        
        # 추천 2: V7 Chord (딸림 화음 - Dominant 7th로 사용)
        v_root_idx = (root_idx + 7) % 12
        recommendations.append({'root_midi': v_root_idx, 'chord_type': 'Dominant 7th'})
        
        # 추천 3: vi Minor Chord (나란한 조의 으뜸 화음 - Submediant)
        vi_root_idx = (root_idx + 9) % 12
        recommendations.append({'root_midi': vi_root_idx, 'chord_type': 'Minor'})

    elif 'Minor' in chord_type: # i (단조 으뜸 화음)으로 간주
        # i, iv, V7, III (단조 화음 진행 기준)
        
        # 추천 1: iv Chord (버금딸림 단조 화음)
        iv_root_idx = (root_idx + 5) % 12
        recommendations.append({'root_midi': iv_root_idx, 'chord_type': 'Minor'})
        
        # 추천 2: V7 Chord (딸림 화음 - 단조에서 주로 V7 사용)
        v_root_idx = (root_idx + 7) % 12
        recommendations.append({'root_midi': v_root_idx, 'chord_type': 'Dominant 7th'})
        
        # 추천 3: III Chord (나란한 조의 으뜸 화음 - Relative Major)
        III_root_idx = (root_idx + 3) % 12
        recommendations.append({'root_midi': III_root_idx, 'chord_type': 'Major'})
    
    else: # 7th, 9th 등 기타 화음 (Major Key Context 기준으로 추천)
        # 추천 1: IV Chord
        iv_root_idx = (root_idx + 5) % 12
        recommendations.append({'root_midi': iv_root_idx, 'chord_type': 'Major'})
        
        # 추천 2: V Chord
        v_root_idx = (root_idx + 7) % 12
        recommendations.append({'root_midi': v_root_idx, 'chord_type': 'Dominant 7th'})
        
        # 추천 3: vi Chord
        vi_root_idx = (root_idx + 9) % 12
        recommendations.append({'root_midi': vi_root_idx, 'chord_type': 'Minor'})
        
    return recommendations[:3] # 항상 3개만 반환

# -----------------------------
# 1) Audio Recording Section
# -----------------------------
st.subheader("🎤 오디오 녹음")

audio = audiorecorder("녹음 시작", "녹음 중지")
recorded_file = None

if len(audio) > 0:
    st.success("녹음 완료!")

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
st.subheader("📁 또는 오디오 파일 업로드")
uploaded_file = st.file_uploader("WAV/MP3 파일 업로드", type=['wav', 'mp3'])

# --- 파일 분석 소스 결정 (수정된 로직) ---
file_to_analyze = None

if uploaded_file is not None:
    # Priority 1: 사용자가 새로 업로드한 파일이 있다면 그것을 사용
    file_to_analyze = uploaded_file
elif recorded_file is not None:
    # Priority 2: 업로드된 파일이 없고 녹음 파일이 있다면 그것을 사용
    file_to_analyze = recorded_file
    
# -----------------------------
# No file yet
# -----------------------------
if file_to_analyze is None:
    st.info("분석을 위해 오디오를 녹음하거나 파일을 업로드하세요.")
    st.stop()

# -----------------------------
# Begin Analysis
# -----------------------------
try:
    # Load audio data (file_to_analyze 사용)
    y, sr = librosa.load(file_to_analyze, sr=None)

    st.success("오디오 로드 완료!")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("샘플링 속도", f"{sr} Hz")
    with col2:
        st.metric("길이", f"{len(y)/sr:.2f} 초")

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
    st.subheader("주파수 스펙트럼")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(xf_positive, yf_positive)
    ax.set_title("Frequency Spectrum")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_xlim([20, 2000]) # Display range for typical musical notes
    ax.grid(True)
    st.pyplot(fig)

    # -----------------------------
    # Peak Identification
    # -----------------------------
    # Filtering parameters
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
    # Harmonic Filtering (Fundamental Frequency Isolation)
    # -----------------------------
    initial_sorted_peaks = sorted(
        zip(peak_magnitudes, peak_frequencies),
        key=lambda x: x[0],
        reverse=True
    )

    filtered_fundamentals = []
    tolerance = 0.015  # 1.5% for harmonic detection

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
    
    # 근음 주파수와 그 강도를 매핑하여 사용
    fundamental_data = {
        freq_to_midi(f): m for f, m in filtered_fundamentals if f > 50
    }
    fundamental_midi_notes = list(fundamental_data.keys())


    # -----------------------------
    # Chord Detection (Collect all matching candidates)
    # -----------------------------
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F',
                  'F#', 'G', 'G#', 'A', 'A#', 'B']
    chord_templates = {
        'Major': [0, 4, 7],
        'Minor': [0, 3, 7],
        'Dominant 7th': [0, 4, 7, 10],
        'Major 7th': [0, 4, 7, 11],
        'Minor 7th': [0, 3, 7, 10],
        'Dominant 9th': [0, 4, 7, 10, 2],
        'Major 9th': [0, 4, 7, 11, 2],
        'Minor 9th': [0, 3, 7, 10, 2],
    }

    all_matches = []
    unique_fundamental_midi_notes = sorted(list(set(fundamental_midi_notes)))

    # Iterate through all fundamental notes as potential roots
    for root_midi in unique_fundamental_midi_notes:
        # Calculate intervals (0-11) relative to the current root_midi
        intervals = set((n - root_midi) % 12 for n in fundamental_midi_notes)
        
        # 근음의 강도 (Magnitude of the root note)
        root_magnitude = fundamental_data.get(root_midi, 0)
        # 최대 강도 (Max magnitude for normalization)
        max_magnitude = max(fundamental_data.values()) if fundamental_data else 1
        
        # 근음 강도를 0.0 ~ 1.0 사이로 정규화 (normalization)
        normalized_root_mag = root_magnitude / max_magnitude if max_magnitude > 0 else 0
        
        for chord_type, template in chord_templates.items():
            # 1. 기본적인 일치 점수: 템플릿의 몇 개 음이 오디오에 존재하는가?
            match_count = sum(1 for t in template if t in intervals)
            
            # 2. 근음 가중치 부여 (Weighting the score by root magnitude):
            # 근음 강도가 높을수록 최종 점수에 가중치를 높게 부여
            # *** 여기서 가중치를 1.5에서 2.0으로 상향 조정했습니다. ***
            weighted_score = match_count + (normalized_root_mag * 2.0) 
            
            # Store all valid matches (match_count 2 또는 weighted_score 2 이상)
            if weighted_score >= 2.0:
                all_matches.append({
                    'score': weighted_score, # 가중치가 적용된 점수를 사용
                    'root_midi': root_midi,
                    'chord_type': chord_type,
                    'template_len': len(template) # Used for tie-breaking: prefer shorter templates
                })

    # Sort matches: 1. By weighted_score (highest first), 2. By template length (shorter/simpler first), 3. By root MIDI (deterministic)
    # Note the negative sign on template_len for ascending length preference
    all_matches.sort(key=lambda x: (x['score'], -x['template_len'], -x['root_midi']), reverse=True)
    
    # Remove duplicates where root_midi and chord_type are the same, prioritizing the best score
    unique_matches = []
    seen = set()
    for match in all_matches:
        identifier = (match['root_midi'], match['chord_type'])
        if identifier not in seen:
            unique_matches.append(match)
            seen.add(identifier)
            
    best_match = unique_matches[0] if unique_matches else None

    # 화성적으로 어울리는 코드 3개 추천
    if best_match:
        root_midi = best_match['root_midi']
        chord_type = best_match['chord_type']
        recommended_matches = get_harmonic_recommendations(root_midi, chord_type, note_names)
    else:
        recommended_matches = []


    # -----------------------------
    # Final Output Generation (with constituent notes and recommendations)
    # -----------------------------
    st.subheader("🎵 최종 식별 결과")

    if best_match:
        best_root_midi = best_match['root_midi']
        best_chord_type = best_match['chord_type']
        
        root_note = note_names[best_root_midi % 12]
        chord = f"{root_note} {best_chord_type}"
        
        # Calculate Chord Notes for Best Match
        template = chord_templates[best_chord_type]
        chord_note_indices = [((best_root_midi % 12) + interval) % 12 for interval in template]
        unique_chord_notes_names = [note_names[i] for i in sorted(list(set(chord_note_indices)))]
        notes_output = " - ".join(unique_chord_notes_names)
        
        st.markdown(f"### **✅ 최종 식별 화음:** {chord}")
        st.metric(label="구성 음정 (Constituent Notes)", value=notes_output)
        
        # Display Recommendations
        if recommended_matches:
            st.markdown("---")
            st.subheader("💡 추가 추천 화음 (Top 3 Harmonically Fitting Candidates)")
            
            rec_data = []
            for match in recommended_matches:
                rec_root_midi = match['root_midi']
                rec_chord_type = match['chord_type']
                rec_root_note = note_names[rec_root_midi % 12]
                rec_chord = f"{rec_root_note} {rec_chord_type}"
                
                # Calculate Chord Notes for Recommendation
                # Use the global chord_templates dictionary
                rec_template = chord_templates.get(rec_chord_type, [0, 4, 7]) # Fallback to Major Triad
                rec_chord_note_indices = [((rec_root_midi % 12) + interval) % 12 for interval in rec_template]
                rec_unique_notes_names = [note_names[i] for i in sorted(list(set(rec_chord_note_indices)))]
                rec_notes_output = " - ".join(rec_unique_notes_names)
                
                # Append recommended chord and its constituent notes
                rec_data.append([rec_chord, rec_notes_output])
            
            # Create a simple table for recommendations (Changed st.table to st.dataframe and added hide_index=True)
            st.dataframe(
                pd.DataFrame(
                    rec_data, 
                    columns=['추천 화음', '구성 음정'] # Updated column name to '구성 음정'
                ),
                hide_index=True # 인덱스 숨기기
            )

    else:
        chord = "No chord identified (화음 미식별)"
        st.markdown(f"### **❌ {chord}**")
        st.metric(label="구성 음정 (Constituent Notes)", value="N/A")
        st.info("최소 일치 점수(2점)를 충족하는 화음 후보가 없습니다.")

except Exception as e:
    st.error(f"오디오 분석 중 오류 발생: {e}")
