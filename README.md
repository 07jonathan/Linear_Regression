Penjelasan Program
- Data Training dan Testing:
  - x_train, y_train: Data input dan output untuk pelatihan model.
  - x_test, y_test: Data input dan output untuk menguji model.

- Fungsi compute_cost:
  - Menghitung biaya (Mean Squared Error) berdasarkan model regresi linier dengan parameter ww dan bb.

- Fungsi gradient_descent:
  - Melakukan gradient descent untuk mengoptimalkan parameter ww dan bb.
  - Iteratif memperbarui nilai ww dan bb berdasarkan gradien dari fungsi biaya.

- Parameter Awal dan Proses Gradient Descent:
  - w_init, b_init: Parameter awal untuk ww dan bb.
  - alpha: Learning rate, mengontrol seberapa besar langkah yang diambil dalam setiap iterasi.
  - num_iters: Jumlah iterasi, menentukan seberapa banyak langkah optimisasi dilakukan.

- Prediksi dan Evaluasi Model:
  - Setelah gradient descent selesai, parameter optimal ww dan bb digunakan untuk memprediksi output pada data uji (x_test).
  - Menghitung biaya (cost) pada data uji untuk mengevaluasi seberapa baik model yang dihasilkan bekerja pada data baru.
