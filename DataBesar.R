# Install dulu jika belum punya: install.packages("data.table")
library(data.table)

# Membaca file .txt (otomatis mendeteksi pemisah kolom)
data_saya <- fread("C:/Users/repandot/Downloads/raw_data_simulasi.txt")

# Load semua library yang dibutuhkan
library(data.table)  # Untuk membaca data besar dengan cepat
library(car)         # Untuk uji Multikolinearitas (VIF)
library(lmtest)      # Untuk uji Heteroskedastisitas dan Autokorelasi
library(nlme)        # Untuk model GLS
library(glmnet)      # Untuk model Ridge Regression
library(Metrics)     # Untuk menghitung RMSE

# ==============================================================================
# 1. LOAD DATA
# ==============================================================================
# Menggunakan fread karena data mencapai 1,5 juta baris. 
# Sesuaikan nama file dengan file yang Anda download dari Google Drive.
df <- fread("C:/Users/repandot/Downloads/raw_data_simulasi.txt")

# Asumsi: Kolom variabel dependen bernama "Y" dan sisanya adalah "X1" sd "X30".
# Jika nama kolom Y berbeda, mohon sesuaikan pada bagian formula.

# ==============================================================================
# 2. PEMBUATAN MODEL OLS & PENGUJIAN ASUMSI KLASIK
# ==============================================================================
# a. Model OLS (Ordinary Least Squares)
model_ols <- lm(Y ~ ., data = df)

# b. Uji Normalitas Sisaan
# Catatan Penting: Karena jumlah n > 5000 (1,5 juta baris), uji shapiro.test akan error.
# Kita gunakan Kolmogorov-Smirnov sebagai alternatif untuk data besar.
uji_normalitas <- ks.test(residuals(model_ols), "pnorm", mean=mean(residuals(model_ols)), sd=sd(residuals(model_ols)))
print("--- Hasil Uji Normalitas ---")
print(uji_normalitas)

# c. Uji Multikolinearitas (VIF)
print("--- Hasil Uji Multikolinearitas (VIF) ---")
print(vif(model_ols)) 
# Interpretasi: Jika VIF > 10, maka ada multikol.

# d. Uji Heteroskedastisitas (Breusch-Pagan Test)
print("--- Hasil Uji Heteroskedastisitas ---")
print(bptest(model_ols)) 

# e. Uji Autokorelasi Sisaan (Durbin-Watson Test)
print("--- Hasil Uji Autokorelasi ---")
print(dwtest(model_ols)) 

# ==============================================================================
# 3. PEMODELAN WLS, GLS, DAN RIDGE REGRESSION
# ==============================================================================

# a. WLS (Weighted Least Squares)
# Menggunakan inverse dari fitted absolute residuals sebagai bobot (weights) sederhana
bobot_wls <- 1 / fitted(lm(abs(residuals(model_ols)) ~ fitted(model_ols)))^2
model_wls <- lm(Y ~ ., data = df, weights = bobot_wls)

# b. GLS (Generalized Least Squares)
# Catatan RAM: Menjalankan GLS pada 1,5 juta baris tanpa spesifikasi korelasi tertentu identik dengan OLS namun memakan memori komputasi yang berat.
model_gls <- gls(Y ~ ., data = df)

# c. Ridge Regression
# glmnet membutuhkan input berupa format matrix, bukan data frame.
x_mat <- as.matrix(df[, .SD, .SDcols = !("Y")]) # Ambil semua kolom kecuali Y
y_vec <- df$Y

# Mencari lambda optimal menggunakan Cross-Validation (alpha = 0 untuk Ridge)
cv_ridge <- cv.glmnet(x_mat, y_vec, alpha = 0)
# Membuat model Ridge terbaik berdasarkan lambda minimum
model_ridge <- glmnet(x_mat, y_vec, alpha = 0, lambda = cv_ridge$lambda.min)

# ==============================================================================
# 4. PERBANDINGAN MODEL BERDASARKAN RMSE DAN AIC[cite: 1]
# ==============================================================================

# Mendapatkan nilai prediksi untuk menghitung RMSE
pred_ols <- predict(model_ols, df)
pred_wls <- predict(model_wls, df)
pred_gls <- predict(model_gls, df)
pred_ridge <- predict(model_ridge, newx = x_mat)

# Hitung RMSE
rmse_ols <- rmse(df$Y, pred_ols)
rmse_wls <- rmse(df$Y, pred_wls)
rmse_gls <- rmse(df$Y, pred_gls)
rmse_ridge <- rmse(df$Y, as.numeric(pred_ridge))

# Hitung AIC
aic_ols <- AIC(model_ols)
aic_wls <- AIC(model_wls)
aic_gls <- AIC(model_gls)

# Menghitung AIC manual untuk Ridge (karena glmnet tidak menyediakan fungsi AIC() bawaan)
dev_ridge <- deviance(model_ridge)
df_ridge <- model_ridge$df
aic_ridge <- dev_ridge + (2 * df_ridge)

# Membuat tabel komparasi akhir
tabel_komparasi <- data.frame(
  Model = c("OLS", "WLS", "GLS", "Ridge"),
  RMSE = c(rmse_ols, rmse_wls, rmse_gls, rmse_ridge),
  AIC = c(aic_ols, aic_wls, aic_gls, aic_ridge)
)

print("=== TABEL PERBANDINGAN MODEL ===")
print(tabel_komparasi)
