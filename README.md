#  RNN Time Series Forecasting with PyTorch

Proyek peramalan (forecasting) data time series menggunakan **Recurrent Neural Network (RNN)** yang dibangun dari nol dengan **PyTorch**, dibantu library [`jcopdl`](https://pypi.org/project/jcopdl/) untuk custom dataset, konfigurasi, dan callback training.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![Status](https://img.shields.io/badge/status-active-success)
![License](https://img.shields.io/badge/license-MIT-green)

---

##  Deskripsi

Studi kasus proyek ini adalah **memprediksi suhu harian minimum** (*daily minimum temperature*) menggunakan pendekatan deep learning berbasis **RNN**. Notebook ini membahas alur lengkap time series forecasting, mulai dari eksplorasi data, resampling, pembuatan custom dataset sequence-based, hingga evaluasi hasil forecasting jangka pendek maupun jangka panjang (multi-step forecasting).

Proyek ini cocok sebagai referensi belajar:
- Cara membangun **custom `Dataset`** untuk data time series di PyTorch
- Perbedaan pendekatan **`TimeSeriesDataset`** bawaan `jcopdl` vs custom dataset manual
- Arsitektur **RNN** sederhana untuk regresi time series
- Konsep **forecasting** vs **prediction dengan data historis**, serta keterbatasan RNN saat memprediksi berdasarkan hasil prediksinya sendiri (*recursive forecasting*)

---

##  Struktur Proyek

```
.
├── Part_2_-_RNN_With_Pytorch_copy.ipynb   # Notebook utama
├── data/
│   └── daily_min_temp.csv                 # Dataset suhu harian
├── utils.py                                # Helper: data4pred & pred4pred
├── model/
│   └── rnn/                                # Output checkpoint model (otomatis dibuat)
└── README.md
```

> Dataset berupa data deret waktu (time series) univariat dengan satu kolom target: **`Temp`**.

---

##  Instalasi

1. Clone repository ini:
```bash
git clone https://github.com/username/nama-repo.git
cd nama-repo
```

2. Buat virtual environment (opsional tapi disarankan):
```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
```

3. Install dependencies:
```bash
pip install torch torchvision pandas numpy matplotlib scikit-learn tqdm jupyter
pip install jcopdl luwiji
```

---

##  Alur Data (Data Pipeline)

1. **Load & Parse Date** — data suhu harian dibaca dengan `Date` sebagai index, di-*parse* sebagai datetime.
2. **Resampling Mingguan** — data harian yang noisy di-*resample* menjadi rata-rata mingguan (`resample("W").mean()`) agar tren lebih halus (*smooth*).
3. **Train-Test Split** — data dibagi menjadi `ts_train` dan `ts_test` dengan `shuffle=False` (data time series **tidak boleh diacak**).
4. **Windowing / Sequencing** — data diubah menjadi pasangan sequence `(X, y)` menggunakan `seq_len` (jumlah langkah waktu sebelumnya yang digunakan untuk memprediksi langkah berikutnya).

Notebook ini mendemonstrasikan **dua pendekatan** untuk membangun dataset sequence:

| Pendekatan | Deskripsi |
|---|---|
| `TimeSeriesDataset` (jcopdl) | Custom dataset siap pakai dari library `jcopdl.utils.dataloader` |
| `MyTimeSeriesDataset` (manual) | Custom `Dataset` PyTorch buatan sendiri — meng-*override* `__init__`, `__len__`, dan `__getitem__` secara eksplisit untuk kontrol penuh atas proses windowing |

---

##  Arsitektur Model

```python
class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        x, hidden = self.rnn(x, hidden)
        x = self.fc(x)
        return x, hidden
```

**Konfigurasi model:**

| Parameter      | Nilai |
|-----------------|-------|
| `input_size`    | Jumlah fitur (1, hanya kolom `Temp`) |
| `hidden_size`   | 64 |
| `num_layers`    | 2 |
| `dropout`       | 0 |
| `output_size`   | 1 |
| `seq_len`       | 6 (dapat diuji ulang dengan nilai lain, mis. 16) |

---

##  Training

| Komponen | Detail |
|---|---|
| **Loss function** | `MSELoss` (Mean Squared Error) — cocok untuk masalah regresi |
| **Optimizer** | `AdamW`, learning rate `0.001` |
| **Batch size** | 16–32 (disesuaikan dengan ukuran dataset yang kecil) |
| **Early stopping** | Otomatis berdasarkan `test_cost` via `jcopdl.callback` |

Jalankan seluruh sel notebook secara berurutan, atau langsung via terminal:

```bash
jupyter notebook RNN_With_Pytorch.ipynb
```

Selama training, akan tampil progress bar per epoch serta plot cost secara real-time, dengan checkpoint model otomatis tersimpan di `model/rnn/`.

---

##  Forecasting: Dua Skema Evaluasi

Bagian paling menarik dari notebook ini adalah **perbandingan dua skema forecasting**:

### 1️ `data4pred` — Prediksi dengan Data Aktual (One-step Prediction)
Model memprediksi satu langkah ke depan menggunakan **data historis asli** sebagai input di setiap langkah. Hasilnya terlihat sangat baik karena model masih "dibantu" oleh data ground-truth.

### 2`pred4pred` — Prediksi Rekursif (Multi-step Forecasting)
Model memprediksi masa depan menggunakan **hasil prediksinya sendiri** sebagai input untuk langkah berikutnya (tanpa data asli). Ini adalah simulasi forecasting nyata — dan hasilnya menunjukkan akurasi menurun seiring bertambahnya *horizon* waktu prediksi.

>  **Insight kunci:** RNN memprediksi berdasarkan pola pada data historis. Untuk data suhu, pola musiman (*seasonality*) jauh lebih berpengaruh dibanding nilai suhu hari/minggu sebelumnya secara langsung — sehingga forecasting jangka panjang tanpa fitur musiman cenderung meleset semakin jauh horizonnya.

---

##  Tech Stack

- **PyTorch** — Deep learning framework
- **jcopdl** — Custom dataset, config, dan callback management
- **pandas** & **NumPy** — Manipulasi dan resampling data time series
- **Matplotlib** — Visualisasi tren & hasil forecasting
- **scikit-learn** — Train-test split
- **tqdm** — Progress bar training

---

##  Lisensi

Proyek ini menggunakan lisensi **MIT** — bebas digunakan, dimodifikasi, dan didistribusikan ulang untuk keperluan pembelajaran maupun pengembangan lebih lanjut.

---

##  Kontribusi

Kontribusi, saran, dan diskusi sangat terbuka! Silakan buat *issue* atau *pull request* jika ingin membantu mengembangkan proyek ini — terutama eksperimen dengan fitur musiman (*seasonal features*) untuk meningkatkan akurasi forecasting jangka panjang.

---

<p align="center">Made with  using PyTorch</p>
