---
title: "Tugas Supervised Learning - Analisis Data Simulasi"
author: "Riefan Abdul Hakim"
date: "1 Oktober 2026"
output: github_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE, warning = FALSE, message = FALSE)
```

## Pendahuluan
Dokumen ini berisi pengerjaan tugas Supervised Learning menggunakan data simulasi sebanyak 1.500.000 baris dengan 30 variabel independen (X) dan 1 variabel dependen (Y). Tugas ini meliputi pengujian asumsi klasik, pemodelan menggunakan OLS, WLS, GLS, dan Ridge Regression, serta perbandingan kinerja model.

## 1. Persiapan dan Load Data

Pertama, kita muat semua *libraries* yang dibutuhkan.

```{r libraries}
# Load semua library yang dibutuhkan
library(data.table)  # Untuk membaca data besar dengan cepat
library(car)         # Untuk uji Multikolinearitas (VIF)
library(lmtest)      # Untuk uji Heteroskedastisitas dan Autokorelasi
library(nlme)        # Untuk model GLS
library(glmnet)      # Untuk model Ridge Regression
library(Metrics)     # Untuk menghitung RMSE
library(knitr)       # Untuk menampilkan tabel yang rapi di Github
```

Selanjutnya, membaca data mentah. Kita menggunakan `fread` dari *package* `data.table` karena sangat efisien untuk memuat jutaan baris data.

```{r load_data}
# Sesuaikan direktori dengan lokasi file di komputer lokal
df <- fread("C:/Users/repandot/Downloads/raw_data_simulasi.txt")

# Mengecek dimensi data (baris, kolom)
dim(df)
```

## 2. Pembuatan Model OLS & Pengujian Asumsi Klasik

### a. Model OLS (Ordinary Least Squares)
```{r model_ols}
model_ols <- lm(Y ~ ., data = df)
```

### b. Uji Normalitas Sisaan
Karena jumlah observasi sangat besar ($n > 5000$), uji Shapiro-Wilk tidak dapat digunakan. Sebagai alternatif, kita menggunakan uji Kolmogorov-Smirnov.

```{r uji_normalitas}
uji_normalitas <- ks.test(residuals(model_ols), "pnorm", 
                          mean = mean(residuals(model_ols)), 
                          sd = sd(residuals(model_ols)))
print(uji_normalitas)
```

### c. Uji Multikolinearitas (VIF)
Jika nilai VIF > 10, terdapat indikasi multikolinearitas yang kuat antar variabel independen.

```{r uji_multikol}
print(vif(model_ols))
```

### d. Uji Heteroskedastisitas
Menggunakan Breusch-Pagan Test untuk mendeteksi apakah varians dari sisaan konstan.

```{r uji_hetero}
print(bptest(model_ols))
```

### e. Uji Autokorelasi Sisaan
Menggunakan Durbin-Watson Test untuk mengecek autokorelasi pada sisaan.

```{r uji_autokorelasi}
print(dwtest(model_ols))
```

## 3. Pemodelan Lanjutan (WLS, GLS, dan Ridge Regression)

### a. WLS (Weighted Least Squares)
Kita menggunakan invers dari *fitted absolute residuals* kuadrat sebagai bobot sederhana untuk memitigasi heteroskedastisitas.

```{r model_wls}
bobot_wls <- 1 / fitted(lm(abs(residuals(model_ols)) ~ fitted(model_ols)))^2
model_wls <- lm(Y ~ ., data = df, weights = bobot_wls)
```

### b. GLS (Generalized Least Squares)
*(Catatan: Menjalankan GLS pada 1,5 juta baris memakan memori komputasi yang berat)*

```{r model_gls}
model_gls <- gls(Y ~ ., data = df)
```

### c. Ridge Regression
Karena Ridge Regression dari *package* `glmnet` membutuhkan input berupa *matrix*, kita harus mengubah format data terlebih dahulu.

```{r model_ridge}
x_mat <- as.matrix(df[, .SD, .SDcols = !("Y")]) # Ambil semua kolom kecuali Y
y_vec <- df$Y

# Mencari lambda optimal menggunakan Cross-Validation (alpha = 0 untuk Ridge)
cv_ridge <- cv.glmnet(x_mat, y_vec, alpha = 0)

# Membuat model Ridge terbaik berdasarkan lambda minimum
model_ridge <- glmnet(x_mat, y_vec, alpha = 0, lambda = cv_ridge$lambda.min)

# Menampilkan sebagian koefisien (beta) dan intercept (a0)
# (Tidak di-print semua agar output markdown tidak terlalu panjang)
head(model_ridge$beta)
model_ridge$a0
```

## 4. Perbandingan Model

Tahap terakhir adalah membandingkan keempat model (OLS, WLS, GLS, Ridge) berdasarkan metrik RMSE dan nilai AIC untuk menentukan mana model yang terbaik.

```{r perbandingan_model}
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

# Menghitung AIC manual untuk Ridge
dev_ridge <- deviance(model_ridge)
df_ridge <- model_ridge$df
aic_ridge <- dev_ridge + (2 * df_ridge)

# Membuat dataframe komparasi
tabel_komparasi <- data.frame(
  Model = c("OLS", "WLS", "GLS", "Ridge"),
  RMSE = c(rmse_ols, rmse_wls, rmse_gls, rmse_ridge),
  AIC = c(aic_ols, aic_wls, aic_gls, aic_ridge)
)
```

**Hasil Akhir Perbandingan:**

```{r print_tabel}
# Menggunakan kable agar tabel tampil sangat rapi di Github
kable(tabel_komparasi, caption = "Tabel Perbandingan Evaluasi Model")
```
