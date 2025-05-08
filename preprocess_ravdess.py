import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
raw_data_path = './Ravdess/audio_speech_actors_01-24/'
output_base = './data/wav'
emotion_map = {
    '01': 0,  # neutral
    '03': 1,  # happy
    '04': 2,  # sad
    '05': 3   # angry
}
all_data = []
for actor_folder in os.listdir(raw_data_path):
    actor_path = os.path.join(raw_data_path, actor_folder)
    if not os.path.isdir(actor_path):
        continue

    for file in os.listdir(actor_path):
        if not file.endswith('.wav'):
            continue
        parts = file.split('-')
        emotion_code = parts[2]
        if emotion_code not in emotion_map:
            continue

        label = emotion_map[emotion_code]
        filepath = os.path.join(actor_path, file)

        try:
            y, sr = librosa.load(filepath, sr=16000)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            mfcc = mfcc.T  # shape [time, freq]
            all_data.append((mfcc, label, file))
        except Exception as e:
            print(f"Failed to process {file}: {e}")
# Split into train/valid/test (80/10/10)
train, temp = train_test_split(all_data, test_size=0.2, random_state=42)
valid, test = train_test_split(temp, test_size=0.5, random_state=42)

def save_split(data, split_name):
    split_path = os.path.join(output_base, split_name)
    os.makedirs(split_path, exist_ok=True)
    for i, (mfcc, label, fname) in enumerate(data):
        filename = f'{label}_{i}.npy'
        save_path = os.path.join(split_path, filename)
        np.save(save_path, mfcc)
save_split(train, 'train')
save_split(valid, 'valid')
save_split(test, 'test')

print("Preprocessing complete.")