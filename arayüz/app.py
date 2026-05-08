import streamlit as st
import pandas as pd
import tensorflow as tf
import pickle
import numpy as np

# --- 1. SAYFA YAPILANDIRMASI ---
st.set_page_config(page_title="Churn Analizi", page_icon="🧬", layout="wide", initial_sidebar_state="expanded")

st.title("🧬 Yapay Sinir Ağları ile Müşteri Kaybı (Churn) Analizi")
st.markdown("*:gray[Bu bilimsel panel, derin öğrenme (Deep Learning) algoritmaları kullanılarak telekomünikasyon müşteri davranışlarını öngörmek ve risk analizleri yapmak için tasarlanmıştır.]*")
st.divider()

# --- 2. MODEL VE SCALER YÜKLEME ---
model_path = 'churn_modeli.keras'
scaler_path = 'scaler.pkl'

try:
    model = tf.keras.models.load_model(model_path)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
except Exception as e:
    st.error(f"❌ Kritik Sistem Hatası: Model veya Scaler yüklenemedi. Detay: {e}")
    st.stop()

# --- SEKME (TAB) YAPISI ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔮 Canlı Tahmin Simülasyonu", 
    "🧪 Metodoloji ve ANN Mimarisi", 
    "📊 Performans ve Optimizasyon",
    "⚡ Canlı Eğitim Laboratuvarı",
    "💻 Teknolojiler ve Araçlar",
    "🚀 Kurulum & İletişim"
])

# --- CANLI TAHMİN SİMÜLASYONU ---
with tab1:
    st.markdown("### 🖥️ Gerçek Zamanlı Müşteri Analizi")
    st.markdown("Kullanıcı verisi alınır ➡️ **:blue[Min-Max Scaler]** ile ölçeklenir ➡️ **:orange[Ağırlıklı ANN Modeli]** olasılık hesaplar.")
    
    st.sidebar.image("https://img.icons8.com/color/96/000000/network.png", width=60)
    st.sidebar.header("Müşteri Girdi Parametreleri")
    tenure = st.sidebar.slider("Abonelik Süresi (Ay)", 0, 72, 12, help="Müşterinin şirkette kaldığı toplam ay süresi.")
    monthly_charges = st.sidebar.number_input("Aylık Ücret ($)", 0.0, 200.0, 50.0, help="Müşterinin aylık ödediği ortalama fatura tutarı.")

    if st.button("Analizi Başlat", type="primary", use_container_width=True):
        try:
            input_data = np.zeros((1, 45))
            temp_df = pd.DataFrame({'tenure': [tenure], 'MonthlyCharges': [monthly_charges]})
            scaled_values = scaler.transform(temp_df)
            
            input_data[0, 0] = scaled_values[0][0]
            input_data[0, 1] = scaled_values[0][1]
            
            prediction = model.predict(input_data)
            probability = float(prediction[0][0])
            
            st.markdown("### 📈 Analiz Çıktıları")
            col_metric, col_info = st.columns([1, 2])
            
            with col_metric:
                delta_val = f"{(probability * 100) - 50:.1f}% (Kritik Eşiğe Göre)"
                st.metric(label="Churn (Ayrılma) Riski", value=f"%{probability * 100:.2f}", delta=delta_val, delta_color="inverse")
                
            with col_info:
                if probability > 0.50:
                    st.error(f"**YÜKSEK RİSK TESPİT EDİLDİ:** Model, bu müşterinin hizmeti bırakma ihtimalini **:red[%{probability*100:.2f}]** olarak hesaplamıştır. Acil sadakat kampanyası (retention) önerilir.")
                else:
                    st.success(f"**DÜŞÜK RİSK:** Müşterinin sadık kalma ihtimali yüksek görünüyor. Modele göre kalma güven skoru: **:green[%{(1-probability)*100:.2f}]**.")
                
        except Exception as prediction_error:
            st.error(f"Tahmin sırasında matematiksel bir hata oluştu: {prediction_error}")

# --- METODOLOJİ VE ANN MİMARİSİ ---
with tab2:
    st.markdown("### 🔬 Bilimsel Altyapı ve Veri İşleme")
    with st.expander("📂 Veri Seti ve Ön İşleme (Data Preprocessing)", expanded=True):
        st.markdown("""
        * **Veri Kaynağı:** Bu projede *Kaggle - Telco Customer Churn (https://www.kaggle.com/datasets/blastchar/telco-customer-churn)* veri seti kullanılmıştır.
        * **Veri Temizliği:** Model için anlam taşımayan `ID` sütunları temizlenmiş, hatalı sayısal formatlar düzeltilmiştir.
        * **Özellik Mühendisliği (Feature Engineering):** Kategorik değişkenler bilgisayarın matematiksel olarak işleyebilmesi için **One-Hot Encoding** yöntemiyle sayısallaştırılmıştır. 
        * **Ölçeklendirme:** Ağırlıkların sapmasını engellemek adına sayısal veriler `MinMaxScaler` ile **0 ile 1** arasına sıkıştırılmıştır.
        * **Veri Ayrımı:** Model %80 Eğitim (Train) ve %20 Test (Validation) olacak şekilde bölünmüştür.
        """)
        
    with st.expander("⚙️ Yapay Sinir Ağı (ANN) Katman Mimarisi", expanded=True):
        colA, colB = st.columns([2, 1])
        with colA:
            st.markdown("""
            **Model Topolojisi:**
            1. **Giriş Katmanı (Input):** Müşteri davranışlarını temsil eden **45 boyutlu** vektör girişi.
            2. **Gizli Katmanlar (Hidden):** Doğrusal olmayan karmaşık ilişkileri çözmek için *ReLU (Rectified Linear Unit)* aktivasyonuna sahip, sırasıyla **20 ve 15 nöronlu** iki derin katman.
            3. **Çıkış Katmanı (Output):** Nihai 0-1 arası olasılık tahmini için *Sigmoid* aktivasyon fonksiyonu.
            """)
        with colB:
            st.code("""
model = Sequential([
    Input(shape=(45,)),
    Dense(20, activation='relu'),
    Dense(15, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            """, language="python")

# --- PERFORMANS VE OPTİMİZASYON ---
with tab3:
    st.markdown("### 📉 Model Başarısı ve Hata Dağılımı")
    st.info("Eğitim süreci, **Adam optimizasyon algoritması** ile 50 epoch boyunca gerçekleştirilmiş olup, genel test doğruluk (accuracy) oranı **~%78** seviyelerinde ölçülmüştür.")
    
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown("#### 1. Öğrenme Eğrisi ve Sweet Spot")
        try:
            st.image("öğrenmeeğrisi.png", caption="Training vs Validation Accuracy/Loss")
        except:
            st.warning("⚠️ Lütfen 'öğrenmeeğrisi.png' dosyasını arayüz klasörüne ekleyin.")
            
        st.markdown("""
        **:blue[Optimizasyon Analizi:]** 
        Grafikler incelendiğinde, modelin en yüksek genelleme yeteneğine **25. Epoch** civarında ulaştığı saptanmıştır. Bu noktadan sonraki eğitimler, modelin veriyi öğrenmekten ziyade ezberlemesine (Overfitting) yol açma eğilimi göstermektedir.
        """)

    with col_g2:
        st.markdown("#### 2. Hata Matrisi (Confusion Matrix)")
        try:
            st.image("hatamatrisi.png", caption="Modelin Gerçek Test Verilerindeki Hata Dağılımı")
        except:
            st.warning("⚠️ Lütfen 'hatamatrisi.png' dosyasını arayüz klasörüne ekleyin.")
            
        st.markdown("""
        **:red[Kritik Hata Analizi (False Negative):]**
        Hata matrisi incelendiğinde şirket için finansal olarak en riskli olan grup **Sol Alt (False Negative)** çeyreğidir. Bu grup, modelin *'Kalacak'* tahmininde bulunduğu ancak gerçekte *aboneliğini iptal eden* müşterileri temsil eder.
        """)

# --- CANLI EĞİTİM LABORATUVARI ---
with tab4:
    st.markdown("### ⚡ Tarayıcı Üzerinde Canlı YSA Eğitimi")
    st.markdown("Aşağıdaki sürgüden epoch (eğitim döngüsü) sayısını ayarlayarak modelin canlı olarak baştan eğitilmesini tetikleyebilirsiniz. Eğitim tamamlandığında öğrenme eğrisi anında interaktif olarak çizilecektir.")
    
    epoch_secimi = st.slider("Eğitim Döngüsü (Epoch) Sayısı", min_value=1, max_value=60, value=25)
    
    if st.button("🚀 Modeli Canlı Eğit", type="primary"):
        # Veri setini hızlıca hazırlayan ve önbelleğe alan fonksiyon
        @st.cache_data
        def hazirla_ve_getir():
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import MinMaxScaler
            
            # Veriyi Jupyter'deki yapıya uygun okuyoruz
            df = pd.read_csv('../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
            df.drop('customerID', axis=1, inplace=True)
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            df.dropna(inplace=True)
            df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
            df_final = pd.get_dummies(df)
            
            X = df_final.drop('Churn', axis=1)
            y = df_final['Churn']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            canli_scaler = MinMaxScaler()
            X_train_scaled = canli_scaler.fit_transform(X_train)
            X_test_scaled = canli_scaler.transform(X_test)
            
            return X_train_scaled, y_train, X_test_scaled, y_test

        try:
            with st.spinner(f"Yapay Sinir Ağı {epoch_secimi} epoch boyunca eğitiliyor. Bu işlem birkaç saniye sürebilir..."):
                X_train, y_train, X_test, y_test = hazirla_ve_getir()
                
                # Yeni ve boş bir model inşa ediyoruz
                from tensorflow.keras.models import Sequential
                from tensorflow.keras.layers import Dense, Input
                
                canli_model = Sequential([
                    Input(shape=(45,)),
                    Dense(20, activation='relu'),
                    Dense(15, activation='relu'),
                    Dense(1, activation='sigmoid')
                ])
                canli_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                
                # Eğitimi başlatıyoruz
                history = canli_model.fit(X_train, y_train, epochs=epoch_secimi, validation_split=0.2, verbose=0)
                
                st.success("✅ Canlı eğitim başarıyla tamamlandı!")
                
                col_chart, col_score = st.columns([3, 1])
                with col_chart:
                    st.markdown("#### 📈 Canlı Öğrenme Eğrisi")
                    # Sonuçları Streamlit'in kendi şık grafiğiyle çizdiriyoruz
                    chart_data = pd.DataFrame({
                        'Eğitim Doğruluğu': history.history['accuracy'],
                        'Doğrulama Doğruluğu': history.history['val_accuracy']
                    })
                    st.line_chart(chart_data)
                
                with col_score:
                    st.markdown("#### 🎯 Test Başarısı")
                    test_loss, test_acc = canli_model.evaluate(X_test, y_test, verbose=0)
                    st.metric(label=f"{epoch_secimi} Epoch Sonucu", value=f"%{test_acc*100:.2f}")

        except FileNotFoundError:
            st.error("❌ Veri seti bulunamadı! Canlı eğitim için lütfen 'WA_Fn-UseC_-Telco-Customer-Churn.csv' dosyasının projenin '../data/' klasöründe olduğundan emin olun.")
        except Exception as e:
            st.error(f"❌ Eğitim sırasında beklenmedik bir hata oluştu: {e}")

# ---  TEKNOLOJİLER VE KÜTÜPHANELER ---
with tab5:
    st.markdown("### 💻 Geliştirme Ortamı ve Araçlar")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.info("**🧠 Yapay Zeka ve Veri Bilimi (Backend)**")
        st.markdown("""
        * **TensorFlow & Keras:** Derin öğrenme mimarisinin inşası.
        * **Scikit-Learn:** Veri ön işleme ve matris hesaplamaları.
        * **Pandas & NumPy:** Veri manipülasyonu.
        """)
    with col_t2:
        st.success("**🖥️ Arayüz ve Araçlar (Frontend & Tools)**")
        st.markdown("""
        * **Streamlit:** Interaktif web arayüzünün geliştirilmesi.
        * **Matplotlib & Seaborn:** Veri görselleştirme.
        * **Pickle / Joblib:** Model dışa aktarımı.
        """)
# --- 6. SEKME: KURULUM VE İLETİŞİM ---
with tab6:
    st.markdown("### 🛠️ Sistemi Kendi Bilgisayarınızda Çalıştırın")
    
    col_dev1, col_dev2 = st.columns(2)
    
    with col_dev1:
        st.markdown("#### 1️⃣ Gereksinimler")
        st.markdown("""
        Bu projeyi çalıştırmak için bilgisayarınızda **Python 3.13.7** yüklü olmalıdır. 
        Gerekli tüm kütüphaneleri tek seferde yüklemek için aşağıdaki komutu terminale yapıştırın:
        """)
        st.code("pip install tensorflow pandas numpy scikit-learn streamlit matplotlib joblib", language="bash")
        
        st.markdown("#### 2️⃣ Kütüphane Detayları")
        with st.expander("Hangi kütüphane ne işe yarıyor?"):
            st.write("- **TensorFlow:** Yapay Sinir Ağı'nın beyni.")
            st.write("- **Streamlit:** Bu gördüğünüz web arayüzünü oluşturur.")
            st.write("- **Scikit-Learn:** Veri ölçeklendirme (Scaler) ve bölme işlemleri.")
            st.write("- **Pandas/NumPy:** Veri tablosu ve matris operasyonları.")

    with col_dev2:
        st.markdown("#### 3️⃣ Çalıştırma")
        st.markdown("Proje klasörüne terminal ile gidip şu komutu yazın:")
        st.code("streamlit run app.py", language="bash")
        
        st.divider()
        
        st.markdown("#### Proje Detayları")
        st.info("Bu proje, Derin Öğrenme ve Müşteri Analitiği üzerine bir vaka çalışması olarak geliştirilmiştir.")
        
        st.markdown("""
        <div style="display: flex; gap: 10px;">
            <a href="https://github.com/gizembsky/Churn_Analizi_Projesi" target="_blank">
                <img src="https://img.shields.io/badge/GitHub-Proje%20Sayfası-black?style=for-the-badge&logo=github" alt="GitHub">
            </a>
           
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.write("📩 Sorularınız için GitHub üzerinden **Issue** açabilir veya doğrudan sorabilirsiniz.")