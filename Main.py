import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import os
import streamlit as st
import sounddevice as sd
import tempfile
import joblib

# === 1. Функція для витягування ознак ===
def extract_features(file_path, target_sr=22050):
    y, sr = librosa.load(file_path, sr=target_sr)
    y = librosa.util.normalize(y)
    mfcc = librosa.feature.mfcc(y=y, sr=target_sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc_mean

# === 2. Функція для додавання розпізнавання шуму ===
def add_noise(y, noise_level=0.005):
    noise = np.random.randn(len(y))
    return y + noise_level * noise



# === 3. Завантаження датасету ===
def load_dataset(folder):
    features = []
    labels = []
    for label in os.listdir(folder):
        class_folder = os.path.join(folder, label)
        if not os.path.isdir(class_folder):
            continue
        for file in os.listdir(class_folder):
            if file.endswith(".wav"):
                file_path = os.path.join(class_folder, file)
                y, sr = librosa.load(file_path, sr=22050)
                y = librosa.util.normalize(y)

                # звичайні ознаки
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                mfcc_mean = np.mean(mfcc.T, axis=0)
                features.append(mfcc_mean)
                labels.append(label)

                # аугментовані ознаки (зі шумом)
                y_noisy = add_noise(y)
                mfcc_noisy = librosa.feature.mfcc(y=y_noisy, sr=sr, n_mfcc=13)
                mfcc_noisy_mean = np.mean(mfcc_noisy.T, axis=0)
                features.append(mfcc_noisy_mean)
                labels.append(label)
    return np.array(features), np.array(labels)


# === 4. Навчання та збереження моделі ===
def train_model(dataset_path="dataset"):
    X, y = load_dataset(dataset_path)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    joblib.dump(model, "sound_model.pkl")
    joblib.dump(le, "label_encoder.pkl")
    return classification_report(y_test, y_pred, target_names=le.classes_), confusion_matrix(y_test, y_pred)

# === 5. Запис звуку з мікрофона ===
def record_audio(duration=3, sr=22050):
    st.info(f"Запис звуку ({duration} секунд)...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait()
    audio = audio.flatten()
    audio = librosa.util.normalize(audio)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(temp_file.name, audio, sr, subtype='PCM_16')
    return temp_file.name


# === 6. Класифікація звуку ===
def classify_audio(file_path):
    model = joblib.load("sound_model.pkl")
    le = joblib.load("label_encoder.pkl")
    features = extract_features(file_path).reshape(1, -1)
    pred = model.predict(features)
    label = le.inverse_transform(pred)[0]
    return label

# === 7. Візуалізація спектрограми ===
def plot_spectrogram(file_path):
    y, sr = librosa.load(file_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_DB = librosa.power_to_db(S, ref=np.max)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel', ax=ax)
    ax.set(title='Мел-спектрограма')
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    return fig

# === 8. Інтерфейс користувача (Streamlit) ===
st.set_page_config(page_title="Класифікатор звуків", layout="centered")
st.title("🔊 Sound Source Classifier")
st.write("Ця система дозволяє розпізнавати джерела звуку: мова, музика, транспорт, природа тощо.")

option = st.sidebar.radio("Оберіть дію:", ["Класифікувати аудіофайл", "Записати та класифікувати звук", "Навчити модель"])

if option == "Класифікувати аудіофайл":
    uploaded_file = st.file_uploader("Завантажте WAV-файл", type=["wav"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
            temp.write(uploaded_file.read())
            label = classify_audio(temp.name)
            st.success(f"✅ Результат класифікації: {label}")
            st.pyplot(plot_spectrogram(temp.name))

elif option == "Записати та класифікувати звук":
    if st.button("🎙️ Записати звук"):
        file_path = record_audio()
        label = classify_audio(file_path)
        st.success(f"✅ Результат класифікації: {label}")
        st.pyplot(plot_spectrogram(file_path))

elif option == "Навчити модель":
    st.info("Вкажіть директорію з підпапками-класами (наприклад: dataset/transport, dataset/speech)")
    dataset_path = st.text_input("📂 Шлях до датасету:", value="dataset")
    if st.button("🚀 Почати навчання"):
        with st.spinner("Навчання моделі..."):
            try:
                report, cm = train_model(dataset_path)
                st.success("Модель успішно навчена!")
                st.text("📊 Звіт класифікації:")
                st.text(report)
                st.text("Матриця плутанини:")
                st.text(str(cm))
            except Exception as e:
                st.error(f"❌ Помилка під час навчання: {str(e)}")

# === 9. Додаткові вказівки ===
st.sidebar.markdown("---")
st.sidebar.markdown("**Інструкції:**")
st.sidebar.markdown("1. Структура датасету повинна мати підпапки для кожного класу.")
st.sidebar.markdown("2. Файли повинні бути у форматі WAV.")
st.sidebar.markdown("3. Натисніть 'Навчити модель', щоб згенерувати класифікатор.")
