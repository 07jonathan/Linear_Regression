import numpy as np

# Data training
x_train = np.array([1.0, 2.0, 4.0])
y_train = np.array([300.0, 500.0, 800.0])

# Data testing
x_test = np.array([1.5, 3.0, 5.0])
y_test = np.array([400.0, 600.0, 900.0])


# Fungsi untuk menghitung biaya (Mean Squared Error)
def compute_cost(x, y, w, b):
    m = len(y)
    predictions = w * x + b
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost


# Fungsi untuk melakukan gradient descent
def gradient_descent(x, y, w_init, b_init, alpha, num_iters):
    w = w_init
    b = b_init
    m = len(y)

    for i in range(num_iters):
        # Hitung gradien
        predictions = w * x + b
        dw = (1 / m) * np.sum((predictions - y) * x)
        db = (1 / m) * np.sum(predictions - y)

        # Update parameter
        w = w - alpha * dw
        b = b - alpha * db

        # Cetak biaya setiap 100 iterasi
        if i % 100 == 0:
            cost = compute_cost(x, y, w, b)
            print(f"Iterasi {i}: Biaya {cost}, w {w}, b {b}")

    return w, b


# Parameter awal
w_init = 0
b_init = 0
alpha = 0.01
num_iters = 1000

# Menjalankan gradient descent pada data training
w_opt, b_opt = gradient_descent(x_train, y_train, w_init, b_init, alpha, num_iters)
print(f"Parameter optimal: w = {w_opt}, b = {b_opt}")


# Menggunakan parameter optimal untuk memprediksi data testing
def predict(x, w, b):
    return w * x + b


y_pred = predict(x_test, w_opt, b_opt)
print("Prediksi:", y_pred)

# Menghitung biaya pada data testing
test_cost = compute_cost(x_test, y_test, w_opt, b_opt)
print(f"Biaya pada data testing: {test_cost}")
