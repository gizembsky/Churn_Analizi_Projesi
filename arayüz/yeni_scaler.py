import pickle
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Uygulamandaki min-max değerlerine göre sahte bir veri seti oluşturuyoruz
dummy_data = pd.DataFrame({
    'tenure': [0, 72], 
    'MonthlyCharges': [0.0, 200.0]
})

# Scaler'ı eğit
scaler = MinMaxScaler()
scaler.fit(dummy_data)

# Yeni ortama uygun olarak kaydet
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("✅ Yepyeni ve uyumlu scaler.pkl başarıyla oluşturuldu!")