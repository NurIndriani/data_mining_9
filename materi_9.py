import streamlit as st
import Orange
import numpy as np

st.title("🌸 Prediksi Kategori Bunga Iris (Model Orange)")
st.write("Masukkan nilai fitur untuk memprediksi jenis bunga iris menggunakan model dari Orange Data Mining.")

# ✅ Load model dari file .pkcls dengan fungsi yang benar
model = Orange.misc.pickle_load(open("data_mining_9_materi.pkcls", "rb"))

# Input fitur
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2)

if st.button("🔍 Prediksi"):
    # Ambil domain dari model Orange
    domain = model.domain
    input_data = Orange.data.Instance(domain, [sepal_length, sepal_width, petal_length, petal_width])
    
    # Prediksi
    prediction = model(input_data)
    label = domain.class_var.values[int(prediction)]

    st.subheader("Hasil Prediksi:")
    st.success(f"🌼 Model memprediksi bunga termasuk ke dalam kategori: **{label}**")

st.markdown("---")
st.caption("Dibuat oleh Nur Indriani | UIN Alauddin Makassar")
