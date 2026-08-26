import csv
from pathlib import Path 
from sklearn.datasets import load_diabetes

#load dataset
dataset = load_diabetes()

#nama fitur
feature_names = dataset.feature_names

#data input (x) dan target (y)
X = dataset.data
y = dataset.target

#simpan file dataset ke folder "data"
data_folder = Path("data")
data_file = data_folder / "diabetes_regression.csv"

#buat folder "data" jika belum tersedia
data_folder.mkdir(exist_ok=True)

#buka file csv dataset
with open(data_file, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)

    header = list(feature_names) + ["target"]
    writer.writerow(header)

    for features, target in zip(X,y):
        row = list(features) + [target]
        writer.writerow(row)

#menampilkan informasi hasil
print("Dataset berhasil disimpan:")
print(f"Lokasi file: {data_file}")
print(f"Jumlah data: {len(y)}")
print(f"Jumlah fitur: {len(feature_names)}")
print(f"Nama Fitur: {', '.join(feature_names)}")
