import math
import struct
import wave
import pathlib
import shutil

def generate_beep_wav(output_path: pathlib.Path):
    sample_rate = 44100
    # Two-tone positive chime: 880 Hz (A5) -> 1318.5 Hz (E6)
    duration_1 = 0.08
    duration_2 = 0.12
    freq_1 = 880.0
    freq_2 = 1318.5

    samples = []
    
    total_samples_1 = int(sample_rate * duration_1)
    for i in range(total_samples_1):
        t = i / sample_rate
        # Amplitude envelope (fade in/out)
        env = min(i / (sample_rate * 0.01), 1.0) * min((total_samples_1 - i) / (sample_rate * 0.01), 1.0)
        val = 0.45 * env * math.sin(2 * math.pi * freq_1 * t)
        samples.append(int(val * 32767))
        
    total_samples_2 = int(sample_rate * duration_2)
    for i in range(total_samples_2):
        t = i / sample_rate
        env = min(i / (sample_rate * 0.01), 1.0) * (1.0 - (i / total_samples_2) ** 0.8)
        val = 0.5 * env * math.sin(2 * math.pi * freq_2 * t)
        samples.append(int(val * 32767))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        raw_bytes = struct.pack(f"<{len(samples)}h", *samples)
        wf.writeframes(raw_bytes)
        
    print(f"Generated clean positive beep: {output_path} ({len(samples)} samples)")

if __name__ == "__main__":
    audio_dir = pathlib.Path("assets/audio")
    beep_wav = audio_dir / "beep.wav"
    generate_beep_wav(beep_wav)
    # Also save as success_beep.wav
    shutil.copy(beep_wav, audio_dir / "success_beep.wav")
    
    # Also ensure wrongsequence.mp3 and stepcomplete.mp3 aliases exist for wrogsequence and stepcompleted
    if (audio_dir / "wrogsequence.mp3").exists() and not (audio_dir / "wrongsequence.mp3").exists():
        shutil.copy(audio_dir / "wrogsequence.mp3", audio_dir / "wrongsequence.mp3")
        print("Created alias wrongsequence.mp3 from wrogsequence.mp3")
        
    if (audio_dir / "stepcompleted.mp3").exists() and not (audio_dir / "stepcomplete.mp3").exists():
        shutil.copy(audio_dir / "stepcompleted.mp3", audio_dir / "stepcomplete.mp3")
        print("Created alias stepcomplete.mp3 from stepcompleted.mp3")
