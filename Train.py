import numpy as np
import soundfile as sf
import os

# Функція для генерації синусоїдальної хвилі
def generate_sine_wave(freq, duration=2.0, sr=22050):
    t = np.linspace(0, duration, int(sr * duration), False)
    tone = 0.5 * np.sin(2 * np.pi * freq * t)
    return tone

# Генерація кількох класів
def create_synthetic_dataset(base_dir="dataset"):
    os.makedirs(f"{base_dir}/speech", exist_ok=True)
    os.makedirs(f"{base_dir}/music", exist_ok=True)
    os.makedirs(f"{base_dir}/transport", exist_ok=True)

    sr = 22050

    # Мова (мінлива частота — імітація інтонації)
    for i in range(5):
        freq = 200 + i * 20
        tone = generate_sine_wave(freq, sr=sr)
        sf.write(f"{base_dir}/speech/speech{i}.wav", tone, sr)

    # Музика (гармоніки)
    for i in range(5):
        tone1 = generate_sine_wave(440, sr=sr)
        tone2 = generate_sine_wave(880, sr=sr)
        tone = tone1 + tone2
        tone /= np.max(np.abs(tone))  # нормалізація
        sf.write(f"{base_dir}/music/music{i}.wav", tone, sr)

    # Транспорт (низькі частоти)
    for i in range(5):
        freq = 80 + i * 10
        tone = generate_sine_wave(freq, sr=sr)
        sf.write(f"{base_dir}/transport/transport{i}.wav", tone, sr)

    print("✅ Синтетичний датасет створено успішно.")

# Виклик функції
create_synthetic_dataset()
