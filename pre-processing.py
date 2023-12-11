import os
import librosa
import numpy as np
from sklearn.preprocessing import MinMaxScaler

input_folder = '/home/bantai/PycharmProjects/Audio_Log/Dataset/Ahh'
output_folder = '/home/bantai/PycharmProjects/Audio_Log/Output/test'

minmax_output_folder = os.path.join(output_folder, 'MinMax')

os.makedirs(minmax_output_folder, exist_ok=True)

for file_name in os.listdir(input_folder):
    if file_name.endswith('.wav'):
        audio_file = os.path.join(input_folder, file_name)
        try:
            y, sr = librosa.load(audio_file)
            D = np.abs(librosa.stft(y))
            log_power = librosa.amplitude_to_db(D, ref=np.max)

            scaler = MinMaxScaler()
            scaled_audio_data = scaler.fit_transform(log_power)

            output_file_path = os.path.join(minmax_output_folder, os.path.splitext(file_name)[0])
            np.save(output_file_path, scaled_audio_data)

            print(f"Processed {file_name}. Scaled data shape:", scaled_audio_data.shape)
            print(f"Saved scaled data to: {output_file_path}")

        except Exception as e:
            print(f"Error processing {audio_file}: {e}")