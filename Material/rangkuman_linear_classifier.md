# Rangkuman Materi: Linear Classifier & Single Layer Perceptron

Dokumen ini berisi rangkuman lengkap dari sesi diskusi mengenai konsep *Linear Classifier*, representasi matematisnya, fungsi aktivasi, optimasi parameter, serta perluasannya untuk klasifikasi multi-class.

---

## 1. Fungsi Garis Linear pada Linear Classifier
Garis linear (atau secara umum disebut **Decision Boundary** / Batas Keputusan) memiliki fungsi utama sebagai **pembatas atau pemisah antar kelas data**.

* **Representasi Dimensi Fitur:**
  * **2 Dimensi (2 Fitur):** Berbentuk **garis lurus**.
  * **3 Dimensi (3 Fitur):** Berbentuk **bidang datar (plane)**.
  * **Lebih dari 3 Dimensi:** Berbentuk **hiperbidang (hyperplane)**.

* **Fungsi Matematika:**
  $$f(x) = \theta^T x$$
  * Jika $\theta^T x > 0 \to$ Prediksi Kelas Positif ($1$).
  * Jika $\theta^T x < 0 \to$ Prediksi Kelas Negatif ($0$ atau $-1$).
  * Jika $\theta^T x = 0 \to$ Titik data berada tepat di atas garis perbatasan.

* **Tingkat Keyakinan (Confidence):**
  Semakin jauh posisi suatu titik data dari garis linear ke arah wilayah kelasnya, semakin tinggi tingkat keyakinan model terhadap prediksi tersebut.

---

## 2. Definisi Matematis Transpose ($\theta^T$)
Simbol $\theta^T$ dibaca sebagai **"theta transpose"**. Transpose adalah operasi aljabar linear untuk menukar baris menjadi kolom, or sebaliknya.

Operasi ini digunakan agar perkalian vektor bobot ($\theta$) dan vektor fitur input ($x$) dapat dilakukan secara matematis menggunakan aturan perkalian matriks (*dot product*).

* **Vektor Bobot/Parameter ($\theta$):**
  $$\theta = \begin{bmatrix} \theta_0 \\ \theta_1 \\ \theta_2 \end{bmatrix} \quad (\text{Ukuran } 3 \times 1)$$
  *(Di mana $\theta_0$ adalah bias $b$, sedangkan $\theta_1, \theta_2$ adalah bobot/weights)*

* **Vektor Fitur Input ($x$):**
  $$x = \begin{bmatrix} 1 \\ x_1 \\ x_2 \end{bmatrix} \quad (\text{Ukuran } 3 \times 1)$$
  *(Angka 1 pada baris pertama ditambahkan sebagai pasangan perkalian untuk bias $\theta_0$)*

* **Vektor Transpose ($\theta^T$):**
  $$\theta^T = \begin{bmatrix} \theta_0 & \theta_1 & \theta_2 \end{bmatrix} \quad (\text{Ukuran } 1 \times 3)$$

* **Perkalian $\theta^T x$:**
  $$\theta^T x = \begin{bmatrix} \theta_0 & \theta_1 & \theta_2 \end{bmatrix} \begin{bmatrix} 1 \\ x_1 \\ x_2 \end{bmatrix} = \theta_0(1) + \theta_1 x_1 + \theta_2 x_2$$

---

## 3. Fungsi Aktivasi $g(z)$ (Sigmoid)
Fungsi $g(z)$ adalah **fungsi aktivasi** yang digunakan untuk memetakan output linear mentah ($z = \theta^T x$) yang bernilai dari $-\infty$ sampai $+\infty$ menjadi nilai probabilitas di dalam rentang **$0$ hingga $1$**.

* **Formula Fungsi Sigmoid:**
  $$g(z) = \frac{1}{1 + e^{-z}}$$

* **Sifat Kurva Sigmoid (Kurva S):**
  * Jika $z \to \infty$, maka $g(z) \to 1$.
  * Jika $z \to -\infty$, maka $g(z) \to 0$.
  * Jika $z = 0$, maka $g(z) = 0.5$.

* **Persamaan Hipotesis Akhir:**
  $$h(x) = g(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}$$
  Output $h(x)$ merepresentasikan probabilitas keanggotaan kelas positif $P(y=1|x)$.

---

## 4. Studi Kasus Rating Film (Single Layer Perceptron)
Diberikan data rating dari Mary dan John untuk memprediksi apakah suatu film disukai (YES/1) atau tidak (NO/0).

* **Fitur:** $x_1$ = Rating Mary, $x_2$ = Rating John.
* **Data Training:**
  1. **LOTR 2**: Mary ($1$), John ($5$) $\to$ Target $y = 0$ (NO)
  2. **Star Wars 1**: Mary ($4.5$), John ($4$) $\to$ Target $y = 1$ (YES)
* **Data Testing:**
  * **Gravity**: Mary ($3$), John ($3$) $\to$ Target $y = ?$

### Skenario 1: Menggunakan Perceptron Klasik (Step Function)
Misalkan didapatkan parameter hasil training: $\theta_0 = -6$, $\theta_1 = 2$, $\theta_2 = -0.5$.
Fungsi aktivasi Step Function:
$$g(z) = \begin{cases} 1 & \text{jika } z \geq 0 \\ 0 & \text{jika } z < 0 \end{cases}$$

* **Prediksi film "Gravity" ($x_1=3, x_2=3$):**
  $$z = -6 + (2 \times 3) + (-0.5 \times 3) = -6 + 6 - 1.5 = -1.5$$
  Karena $z < 0$, maka $g(-1.5) = 0$ (**NO**).

### Skenario 2: Menggunakan Fungsi Aktivasi Sigmoid
Menggunakan data dan parameter yang sama, tetapi output dimasukkan ke fungsi Sigmoid:

* **Prediksi film "Gravity" ($x_1=3, x_2=3$):**
  $$z = -1.5$$
  $$g(-1.5) = \frac{1}{1 + e^{-(-1.5)}} = \frac{1}{1 + e^{1.5}} \approx 0.1825 \quad (18.25\%)$$
  Karena probabilitas $0.1825 < 0.5$, keputusan akhirnya adalah **NO** (tidak disukai) dengan tingkat keyakinan yang tinggi bahwa film tersebut tidak disukai.

---

## 5. Konsep Optimasi: "Learning from Error"
Komputer mencari garis (nilai $\theta$) terbaik menggunakan proses trial-and-error terarah yang terdiri dari:
1. **Prediksi (Forward Pass):** Model menebak output berdasarkan parameter saat ini.
2. **Ukur Kesalahan (Loss/Cost Function):** Menghitung seberapa jauh tebakan dari nilai asli.
3. **Koreksi Diri (Backpropagation & Gradient Descent):** Memperbarui nilai parameter ke arah berlawanan dari gradien kesalahan untuk meminimalkan loss.

---

## 6. Daftar Formula Fungsi Kesalahan (Loss Functions)

### A. Untuk Tugas Klasifikasi
* **Binary Cross-Entropy Loss (Log Loss):**
  $$\text{Loss} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

* **Categorical Cross-Entropy Loss (Multi-class):**
  $$\text{Loss} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{C} y_{ij} \log(\hat{y}_{ij})$$

* **Hinge Loss (SVM):**
  $$\text{Loss} = \frac{1}{N} \sum_{i=1}^{N} \max(0, 1 - y_i \cdot \hat{y}_i) \quad \text{di mana } y_i \in \{-1, 1\}$$

### B. Untuk Tugas Regresi
* **Mean Squared Error (MSE) / L2 Loss:**
  $$\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$

* **Mean Absolute Error (MAE) / L1 Loss:**
  $$\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|$$

*(Keterangan: $y$ = Nilai Asli, $\hat{y}$ = Prediksi Model, $N$ = Jumlah Data)*

---

## 7. Pembaruan Parameter via Gradient Descent

Aturan umum pembaruan parameter dengan *learning rate* $\alpha$:
$$\theta_j \leftarrow \theta_j - \alpha \frac{\partial J}{\partial \theta_j}$$

Berikut adalah perbandingan formula gradien berdasarkan fungsi loss yang digunakan (menggunakan aktivasi Sigmoid):

### Kasus A: Menggunakan Binary Cross-Entropy (BCE) Loss
* **Formula Perubahan Parameter:**
  $$\theta_j \leftarrow \theta_j - \alpha \cdot (Y_{\text{prediksi}} - Y_{\text{asli}}) \cdot x_j$$

### Kasus B: Menggunakan Squared Error Loss
* **Formula Perubahan Parameter (Bentuk Ringkas):**
  $$\theta_j \leftarrow \theta_j - \alpha \cdot (Y_{\text{prediksi}} - Y_{\text{asli}}) \cdot Y_{\text{prediksi}}(1 - Y_{\text{prediksi}}) \cdot x_j$$

* **Formula Gradien Terurai Lengkap (Versi Dosen / Chain Rule):**
  Jika fungsi loss untuk 1 data ke-$i$ didefinisikan sebagai $J = (g(\theta^T x^{(i)} + b) - y^{(i)})^2$:
  $$\Delta \theta_1 = 2 \left[ g(\theta^T x^{(i)} + b) - y^{(i)} \right] \left[ 1 - g(\theta^T x^{(i)} + b) \right] g(\theta^T x^{(i)} + b) x_1^{(i)}$$
  $$\Delta \theta_2 = 2 \left[ g(\theta^T x^{(i)} + b) - y^{(i)} \right] \left[ 1 - g(\theta^T x^{(i)} + b) \right] g(\theta^T x^{(i)} + b) x_2^{(i)}$$
  $$\Delta b = 2 \left[ g(\theta^T x^{(i)} + b) - y^{(i)} \right] \left[ 1 - g(\theta^T x^{(i)} + b) \right] g(\theta^T x^{(i)} + b)$$

* **Formula Pembaruan Parameter Lengkap ($\alpha$ + Gradien Dosen):**
  $$\theta_1^{(\text{baru})} = \theta_1^{(\text{lama})} - \alpha \cdot 2 \left[ g(z) - y \right] \left[ 1 - g(z) \right] g(z) x_1$$
  $$\theta_2^{(\text{baru})} = \theta_2^{(\text{lama})} - \alpha \cdot 2 \left[ g(z) - y \right] \left[ 1 - g(z) \right] g(z) x_2$$
  $$b^{(\text{baru})} = b^{(\text{lama})} - \alpha \cdot 2 \left[ g(z) - y \right] \left[ 1 - g(z) \right] g(z)$$

  *Detail Komponen Chain Rule:*
  1. $2 \left[ g(\theta^T x^{(i)} + b) - y^{(i)} \right] \to$ Turunan Loss terhadap $g(z)$
  2. $\left[ 1 - g(\theta^T x^{(i)} + b) \right] g(\theta^T x^{(i)} + b) \to$ Turunan fungsi Sigmoid $g'(z) = g(z)(1 - g(z))$
  3. $x_1^{(i)}, x_2^{(i)}, 1 \to$ Turunan linear $z = \theta^T x + b$ terhadap masing-masing parameter ($\theta_1, \theta_2, b$)

> [!NOTE]
> **Mengapa ada angka 2 pada rumus dosen, sedangkan rumus umum menggunakan $\alpha$?**
> 1. **Nilai $\Delta \theta$ pada rumus dosen** adalah **gradien murni** $\frac{\partial J}{\partial \theta_j}$. Angka **$2$** muncul secara alami dari turunan bentuk kuadrat $\frac{d}{dx}(x^2) = 2x$.
> 2. **Aturan pembaruan parameter sebenarnya** adalah $\theta_j \leftarrow \theta_j - \alpha \cdot \Delta \theta_j$.
> 3. Banyak literatur mendefinisikan Loss dengan faktor setengah $J = \frac{1}{2}(g(z) - y)^2$ agar saat diturunkan, angka $\frac{1}{2}$ dan $2$ saling mencoret, sehingga angka $2$ tidak perlu ditulis lagi.
> 4. Selain itu, karena $\alpha$ (*learning rate*) adalah konstanta yang kita tentukan sendiri (misal $\alpha = 0.01$), pengali $2$ sering diserap langsung ke dalam $\alpha$.

> [!NOTE]
> Pada **Squared Error**, ada suku pengali tambahan $Y_{\text{prediksi}}(1 - Y_{\text{prediksi}})$ yang berasal dari turunan fungsi sigmoid. Suku ini dapat menyebabkan masalah *vanishing gradient* (pembaruan parameter melambat ketika prediksi mendekati $0$ atau $1$). Oleh karena itu, **Binary Cross-Entropy** lebih disukai untuk klasifikasi.

---

### Kasus C: Perbandingan Formula Update dengan Berbagai Fungsi Aktivasi $g(z)$

Asumsi menggunakan fungsi **Squared Error Loss** $J = \frac{1}{2} (\hat{y} - y)^2$ dengan $z = \theta^T x + b$ dan output prediksi $\hat{y} = g(z)$:

Aturan pembaruan umum berbasis Chain Rule:
$$\theta_j \leftarrow \theta_j - \alpha \cdot (\hat{y} - y) \cdot g'(z) \cdot x_j$$
$$b \leftarrow b - \alpha \cdot (\hat{y} - y) \cdot g'(z)$$

#### 1. Fungsi Sigmoid
* **Fungsi:** $g(z) = \frac{1}{1 + e^{-z}}$
* **Turunan:** $g'(z) = g(z)(1 - g(z)) = \hat{y}(1 - \hat{y})$
* **Update $\theta_j$:** $\theta_j \leftarrow \theta_j - \alpha \cdot (\hat{y} - y) \cdot \hat{y}(1 - \hat{y}) \cdot x_j$
* **Update $b$:** $b \leftarrow b - \alpha \cdot (\hat{y} - y) \cdot \hat{y}(1 - \hat{y})$

#### 2. Fungsi Tanh (Hyperbolic Tangent)
* **Fungsi:** $g(z) = \tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$
* **Turunan:** $g'(z) = 1 - \tanh^2(z) = 1 - \hat{y}^2$
* **Update $\theta_j$:** $\theta_j \leftarrow \theta_j - \alpha \cdot (\hat{y} - y) \cdot (1 - \hat{y}^2) \cdot x_j$
* **Update $b$:** $b \leftarrow b - \alpha \cdot (\hat{y} - y) \cdot (1 - \hat{y}^2)$

#### 3. Fungsi Arctan (Inverse Tangent)
* **Fungsi:** $g(z) = \arctan(z)$
* **Turunan:** $g'(z) = \frac{1}{1 + z^2}$
* **Update $\theta_j$:** $\theta_j \leftarrow \theta_j - \alpha \cdot (\hat{y} - y) \cdot \frac{1}{1 + z^2} \cdot x_j$
* **Update $b$:** $b \leftarrow b - \alpha \cdot (\hat{y} - y) \cdot \frac{1}{1 + z^2}$

#### 4. Fungsi Softsign
* **Fungsi:** $g(z) = \frac{z}{1 + |z|}$
* **Turunan:** $g'(z) = \frac{1}{(1 + |z|)^2}$
* **Update $\theta_j$:** $\theta_j \leftarrow \theta_j - \alpha \cdot (\hat{y} - y) \cdot \frac{1}{(1 + |z|)^2} \cdot x_j$
* **Update $b$:** $b \leftarrow b - \alpha \cdot (\hat{y} - y) \cdot \frac{1}{(1 + |z|)^2}$

---

## 8. Klasifikasi Multi-class (Lebih dari 2 Kelas)
Single Layer Perceptron (SLP) dapat digunakan untuk mengklasifikasikan data ke dalam $C$ kelas ($C > 2$).

### A. Arsitektur Jaringan
* Menggunakan **$C$ buah neuron output** secara paralel pada satu layer.
* Setiap kelas memiliki parameter $\theta$-nya masing-masing.
* Kalkulasi dilakukan secara efisien menggunakan perkalian matriks bobot:
  $$Z = \Theta \cdot x$$

### B. Dua Pendekatan Klasifikasi
1. **Menggunakan Sigmoid (Pendekatan One-vs-All / One-vs-Rest):**
   * Membuat $C$ buah output klasifikasi biner terpisah (contoh: Action vs Lainnya, Comedy vs Lainnya).
   * Nilai probabilitas setiap output bersifat indeks (total jumlah probabilitas tidak harus 1.00).
   * **Pengambilan Keputusan:** Memilih kelas dengan probabilitas terbesar menggunakan fungsi **Argmax**:
     $$\text{Prediksi} = \text{argmax}([P(\text{Kelas}_1), P(\text{Kelas}_2), \dots, P(\text{Kelas}_C)])$$

2. **Menggunakan Softmax (Pendekatan Standar Multi-class):**
   * Menggunakan fungsi aktivasi Softmax pada layer output:
     $$P(\text{Kelas } i) = \frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}}$$
   * Probabilitas seluruh kelas saling mempengaruhi sehingga total penjumlahannya **pasti bernilai 1 (100%)**.
