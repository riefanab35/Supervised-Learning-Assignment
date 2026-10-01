Komputasi - Supervised Learning - Analisis Data Simulasi Berjumlah Besar
================
Riefan Abdul Hakim
2026-10-01

## Pendahuluan

Dokumen ini berisi pengerjaan tugas Supervised Learning menggunakan data
simulasi sebanyak 1.500.000 baris dengan 30 variabel independen (X) dan
1 variabel dependen (Y). Tugas ini meliputi pengujian asumsi klasik,
pemodelan menggunakan OLS, WLS, GLS, dan Ridge Regression, serta
perbandingan kinerja model.

## 1. Persiapan dan Load Data

Pertama, kita muat semua *libraries* yang dibutuhkan.

``` r
# Load semua library yang dibutuhkan
library(data.table)  # Untuk membaca data besar dengan cepat
library(car)         # Untuk uji Multikolinearitas (VIF)
library(lmtest)      # Untuk uji Heteroskedastisitas dan Autokorelasi
library(nlme)        # Untuk model GLS
library(glmnet)      # Untuk model Ridge Regression
library(Metrics)     # Untuk menghitung RMSE
library(knitr)       # Untuk menampilkan tabel yang rapi di Github
```

Selanjutnya, membaca data mentah. Kita menggunakan `fread` dari
*package* `data.table` karena sangat efisien untuk memuat jutaan baris
data.

``` r
# Sesuaikan direktori dengan lokasi file di komputer lokal
df <- fread("C:/Users/repandot/Downloads/raw_data_simulasi.txt")

# Mengecek dimensi data (baris, kolom)
dim(df)
```

    ## [1] 1500000      31

## 2. Pembuatan Model OLS & Pengujian Asumsi Klasik

### a. Model OLS (Ordinary Least Squares)

``` r
model_ols <- lm(Y ~ ., data = df)
```

### b. Uji Normalitas Sisaan

Karena jumlah observasi sangat besar ($n > 5000$), uji Shapiro-Wilk
tidak dapat digunakan. Sebagai alternatif, kita menggunakan uji
Kolmogorov-Smirnov.

``` r
uji_normalitas <- ks.test(residuals(model_ols), "pnorm", 
                          mean = mean(residuals(model_ols)), 
                          sd = sd(residuals(model_ols)))
print(uji_normalitas)
```

    ## 
    ##  Asymptotic one-sample Kolmogorov-Smirnov test
    ## 
    ## data:  residuals(model_ols)
    ## D = 0.062785, p-value < 2.2e-16
    ## alternative hypothesis: two-sided

### c. Uji Multikolinearitas (VIF)

Jika nilai VIF \> 10, terdapat indikasi multikolinearitas yang kuat
antar variabel independen.

``` r
print(vif(model_ols))
```

    ##         X1         X2         X3         X4         X5         X6         X7 
    ## 139.489451  82.066719  33.211750  17.032378  10.018676  94.600725  54.673488 
    ##         X8         X9        X10        X11        X12        X13        X14 
    ##  21.753808  13.538096   7.600183 202.316283 133.213243  45.773906  17.014603 
    ##        X15        X16        X17        X18        X19        X20        X21 
    ##   9.119131  57.058300  53.002759  23.276802  76.996160  58.976083  73.230650 
    ##        X22        X23        X24        X25        X26        X27        X28 
    ##  55.429828  36.094901  76.461791  59.124578 289.713983 298.471434 333.947299 
    ##        X29        X30 
    ## 328.976320   5.003805

### d. Uji Heteroskedastisitas

Menggunakan Breusch-Pagan Test untuk mendeteksi apakah varians dari
sisaan konstan.

``` r
print(bptest(model_ols))
```

    ## 
    ##  studentized Breusch-Pagan test
    ## 
    ## data:  model_ols
    ## BP = 27.644, df = 30, p-value = 0.5893

### e. Uji Autokorelasi Sisaan

Menggunakan Durbin-Watson Test untuk mengecek autokorelasi pada sisaan.

``` r
print(dwtest(model_ols))
```

    ## 
    ##  Durbin-Watson test
    ## 
    ## data:  model_ols
    ## DW = 0.59981, p-value < 2.2e-16
    ## alternative hypothesis: true autocorrelation is greater than 0

## 3. Pemodelan Lanjutan (WLS, GLS, dan Ridge Regression)

### a. WLS (Weighted Least Squares)

Kita menggunakan invers dari *fitted absolute residuals* kuadrat sebagai
bobot sederhana untuk memitigasi heteroskedastisitas.

``` r
bobot_wls <- 1 / fitted(lm(abs(residuals(model_ols)) ~ fitted(model_ols)))^2
model_wls <- lm(Y ~ ., data = df, weights = bobot_wls)
```

### b. GLS (Generalized Least Squares)

*(Catatan: Menjalankan GLS pada 1,5 juta baris memakan memori komputasi
yang berat)*

``` r
model_gls <- gls(Y ~ ., data = df)
```

### c. Ridge Regression

Karena Ridge Regression dari *package* `glmnet` membutuhkan input berupa
*matrix*, kita harus mengubah format data terlebih dahulu.

``` r
x_mat <- as.matrix(df[, .SD, .SDcols = !("Y")]) # Ambil semua kolom kecuali Y
y_vec <- df$Y

# Mencari lambda optimal menggunakan Cross-Validation (alpha = 0 untuk Ridge)
cv_ridge <- cv.glmnet(x_mat, y_vec, alpha = 0)

# Membuat model Ridge terbaik berdasarkan lambda minimum
model_ridge <- glmnet(x_mat, y_vec, alpha = 0, lambda = cv_ridge$lambda.min)

# Menampilkan sebagian koefisien (beta) dan intercept (a0)
# (Tidak di-print semua agar output markdown tidak terlalu panjang)
head(model_ridge$beta)
```

    ## 6 x 1 sparse Matrix of class "dgCMatrix"
    ##            s0
    ## X1  0.9170827
    ## X2  0.7099223
    ## X3  0.4882533
    ## X4 -1.4295671
    ## X5  0.2024117
    ## X6  0.6739964

``` r
model_ridge$a0
```

    ##       s0 
    ## 4.993045

## 4. Perbandingan Model

Tahap terakhir adalah membandingkan keempat model (OLS, WLS, GLS, Ridge)
berdasarkan metrik RMSE dan nilai AIC untuk menentukan mana model yang
terbaik.

``` r
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

``` r
# Menggunakan kable agar tabel tampil sangat rapi di Github
kable(tabel_komparasi, caption = "Tabel Perbandingan Evaluasi Model")
```

| Model |     RMSE |      AIC |
|:------|---------:|---------:|
| OLS   | 5.715790 |  9486577 |
| WLS   | 5.715790 |  9486575 |
| GLS   | 5.715790 |  9486772 |
| Ridge | 5.725717 | 49175816 |

Tabel Perbandingan Evaluasi Model
