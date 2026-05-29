import streamlit as st
import joblib
import numpy as np

# Kaydedilmiş optimize modeli ve scaleri yükle
model = joblib.load('optimized_rf_model.pkl')
scaler = joblib.load('scaler.pkl')

# Sayfa Genişlik Ayarı ve Başlık
st.set_page_config(layout="wide")
st.title("📊 Amerikan Şirketleri Finansal Risk & İflas Tahmini")
st.write("Şirketin bilanço ve gelir tablosu verilerini girerek iflas riskini yapay zeka modeliyle analiz edin.")
st.markdown("---")

st.subheader("📝 Finansal Göstergeler")

# Girdileri daha düzenli göstermek için 3 geniş sütun oluşturuyoruz
col1, col2, col3 = st.columns(3)

with col1:
    x1 = st.number_input("Dönen Varlıklar / Current Assets (X1)", value=0.0, format="%.3f")
    x2 = st.number_input("Satılan Malın Maliyeti / COGS (X2)", value=0.0, format="%.3f")
    x3 = st.number_input("Amortisman Giderleri / Depreciation (X3)", value=0.0, format="%.3f")
    x4 = st.number_input("FAVÖK / EBITDA (X4)", value=0.0, format="%.3f")
    x5 = st.number_input("Stoklar / Inventory (X5)", value=0.0, format="%.3f")
    x6 = st.number_input("Net Dönem Kârı / Net Income (X6)", value=0.0, format="%.3f")

with col2:
    x7 = st.number_input("Ticari Alacaklar / Total Receivables (X7)", value=0.0, format="%.3f")
    x8 = st.number_input("Piyasa Değeri / Market Value (X8)", value=0.0, format="%.3f")
    x9 = st.number_input("Net Satışlar / Net Sales (X9)", value=0.0, format="%.3f")
    x10 = st.number_input("Toplam Varlıklar / Total Assets (X10)", value=0.0, format="%.3f")
    x11 = st.number_input("Uzun Vadeli Borçlar / Long-term Debt (X11)", value=0.0, format="%.3f")
    x12 = st.number_input("FVÖK / EBIT (X12)", value=0.0, format="%.3f")

with col3:
    x13 = st.number_input("Brüt Kâr / Gross Profit (X13)", value=0.0, format="%.3f")
    x14 = st.number_input("Kısa Vadeli Borçlar / Current Liabilities (X14)", value=0.0, format="%.3f")
    x15 = st.number_input("Geçmiş Yıllar Kârları / Retained Earnings (X15)", value=0.0, format="%.3f")
    x16 = st.number_input("Toplam Hasılat / Total Revenue (X16)", value=0.0, format="%.3f")
    x17 = st.number_input("Toplam Borç / Total Liabilities (X17)", value=0.0, format="%.3f")
    x18 = st.number_input("Faaliyet Giderleri / Operating Expenses (X18)", value=0.0, format="%.3f")

st.markdown("---")

# Tahmin Butonu Merkezleme
_, center_col, _ = st.columns([1, 2, 1])

with center_col:
    if st.button("📊 Şirket İflas Riskini Analiz Et", use_container_width=True):
        # 18 girdiyi tam olarak modelin beklediği sıra ile diziye ekliyoruz
        input_data = np.array([[x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18]])

        # Eğitimde kullanılan ölçekleyici ile veriyi dönüştür
        scaled_data = scaler.transform(input_data)

        # Tahmin üret
        prediction = model.predict(scaled_data)

        st.markdown("<br>", unsafe_allow_html=True)
        if prediction[0] == 1:
            st.error("🚨 **Kritik Risk:** Yapay zeka analizine göre şirketin **İFLAS ETME** olasılığı yüksek (Failed)!")
        else:
            st.success("✅ **Finansal Sağlık İyi:** Yapay zeka analizine göre şirketin **HAYATTA KALMA** olasılığı yüksek (Alive).")
