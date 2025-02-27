import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

# 1. Memuat dataset Iris dari scikit-learn
iris = datasets.load_iris()

X = iris.data    # Inputan untuk machine learning
y = iris.target  # Output yang diinginkan dari machine learning

# 2. Mengonversi data fitur dan target menjadi DataFrame
df_X = pd.DataFrame(X, columns=iris.feature_names)
df_y = pd.Series(y, name='target')

# Gabungkan fitur dan target dalam satu DataFrame
df = pd.concat([df_X, df_y], axis=1)

# Menampilkan informasi dataset
print("Informasi Dataset:")
print(df.info())

print("\nStatistik Deskriptif Dataset:")
print(df.describe())

print("\nNilai Unik pada Target:")
print(df['target'].unique())

# 3. Split Data
X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2, random_state=42)

# 4. Train the Model dengan K-Nearest Neighbors (KNN)
model = KNeighborsClassifier(n_neighbors=3)  # Menggunakan 3 tetangga terdekat
model.fit(X_train, y_train)

# 5. Predict & Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\nLaporan Klasifikasi:")
print(f"Akurasi: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 6. Visualisasi Confusion Matrix dengan Matplotlib
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalisasi matrix
plt.imshow(norm_cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(iris.target_names))
plt.xticks(tick_marks, iris.target_names, rotation=45)
plt.yticks(tick_marks, iris.target_names)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, f'{cm[i, j]}\n({norm_cm[i, j]:.2f})', 
                 horizontalalignment="center", 
                 color="white" if norm_cm[i, j] > 0.5 else "black")

plt.xlabel("Prediksi")
plt.ylabel("Aktual")
plt.show()

# 7. Visualisasi Distribusi Fitur dengan Histogram
plt.figure(figsize=(12, 8))
for i, feature in enumerate(iris.feature_names):
    plt.subplot(2, 2, i+1)
    for target in range(len(iris.target_names)):  # Pastikan target dalam rentang yang valid
        sns.histplot(df_X.loc[df_y[df_y == target].index, feature], kde=True, label=iris.target_names[target], alpha=0.6)
    plt.xlabel(feature)
    plt.legend()
plt.suptitle('Histogram Distribusi Fitur')
plt.tight_layout()
plt.show()

# 8. Visualisasi Hasil Prediksi
results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
results_df['Correct'] = results_df['Actual'] == results_df['Predicted']

plt.figure(figsize=(10, 6))
plt.scatter(results_df['Actual'], results_df['Predicted'], c=results_df['Correct'], cmap='coolwarm')
plt.xlabel("Aktual")
plt.ylabel("Prediksi")
plt.title("Perbandingan Hasil Prediksi dengan Aktual")
plt.xticks(ticks=[0, 1, 2], labels=iris.target_names)
plt.yticks(ticks=[0, 1, 2], labels=iris.target_names)
plt.colorbar(label='Benar (True) / Salah (False)')
plt.show()

# 9. Visualisasi Data yang Telah Diklasterisasi
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_test.iloc[:, 0], y=X_test.iloc[:, 1], hue=y_pred, palette='viridis', style=y_pred, s=100)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title("Visualisasi Hasil Klasifikasi dengan KNN")
plt.legend(title='Cluster')
plt.show()
