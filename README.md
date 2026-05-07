# 📊 Müşteri Kaybı (Churn) Tahmin Projesi
---

## 🎯 1. Proje Amacı ve İçeriği
Bu çalışma, bir telekomünikasyon şirketinin müşteri verilerini analiz ederek, hangi müşterilerin hizmet almayı bırakacağını (**Churn**) önceden tahmin etmek amacıyla geliştirilmiştir. Yapay sinir ağları kullanılarak müşteri sadakati ölçülmüş ve şirket için kayıp riski taşıyan profillerin tespiti hedeflenmiştir.

**Veri Kaynağı:** Bu projede kullanılan veri seti [Kaggle - Telco Customer Churn (IBM)](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset) üzerinden temin edilmiştir.

### **Proje Kapsamı:**
* **Veri Analizi:** Müşteri demografisi ve kullanım alışkanlıklarının incelenmesi.
* **Veri Ön İşleme:** Eksik verilerin temizlenmesi, aykırı değerlerin yönetimi ve kategorik verilerin sayısallaştırılması (One-Hot Encoding).
* **Model Eğitimi:** **TensorFlow/Keras** kütüphaneleri kullanılarak derin öğrenme tabanlı bir Yapay Sinir Ağları (ANN) mimarisi kurulmuştur.
* **Optimizasyon:** Eğitim grafiklerinden elde edilen verilere dayanarak, modelin ezberlemesini (**Overfitting**) önlemek amacıyla eğitim **25 epoch** ile sınırlandırılmış ve en optimal nokta seçilmiştir.
* **Tahminleme:** Kaydedilen model (`.keras`) ve ölçeklendirme anahtarı (`.pkl`) ile yeni veriler üzerinden canlı tahmin simülasyonu yapılmıştır.

---

## 🛠 2. Kullanılan Teknolojiler
* **Dil:** Python 3.13
* **Veri İşleme:** `Pandas`, `Numpy`
* **Makine Öğrenmesi:** `Scikit-Learn` (Min-Max Scaling, Train-Test Split)
* **Derin Öğrenme:** `TensorFlow & Keras`
* **Görselleştirme:** `Matplotlib`, `Seaborn`

---

## 🏗 3. Model Mimarisi ve Analiz
Model, giriş katmanından sonra gelen ardışık yoğun katmanlardan oluşmaktadır. 
* **Input Layer:** Modern `Input(shape)` yapısı kullanılmıştır.
* **Hidden Layers:** `ReLU` aktivasyon fonksiyonlu gizli katmanlar.
* **Output Layer:** İkili sınıflandırma için `Sigmoid` aktivasyon fonksiyonu.

**Performans Analizi:**
Eğitim sürecinde elde edilen Accuracy ve Loss grafikleri incelenmiş; modelin genelleme yeteneğini en üst seviyede tuttuğu **25. epoch** seviyesinde durdurulmuştur. Hata Matrisi (Confusion Matrix) üzerinden yapılan incelemede, modelin gerçek "Churn" vakalarını yakalama başarısı doğrulanmıştır.

---

## ⚠️ 4. Sistem ve Kurulum Gereksinimleri
Projenin sorunsuz çalışması için **Python 3.13** versiyonu önerilmektedir.

### **Gerekli Paketlerin Kurulması**
```bash
py -3.13 -m pip install notebook tensorflow pandas numpy scikit-learn matplotlib seaborn joblib