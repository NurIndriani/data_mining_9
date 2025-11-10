import streamlit as st
import pickle
import numpy as np

# Judul aplikasi
st.title("🌸 Prediksi Kategori Bunga Iris")
st.write("Masukkan nilai fitur untuk memprediksi jenis bunga iris menggunakan model yang telah dilatih.")

# Load model (ganti nama file sesuai dengan model milikmu)
model_path = "model_iris.pkl"  # contoh: pkcls
with open(model_path, "rb") as file:
    model = pickle.load(file)

# Input fitur dari pengguna
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2)

# Tombol prediksi
if st.button("🔍 Prediksi"):
    # Membuat array input
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Prediksi menggunakan model
    prediction = model.predict(input_data)

    # Hasil prediksi
    st.subheader("Hasil Prediksi:")
    st.success(f"Model memprediksi bunga termasuk ke dalam kategori: **{prediction[0]}**")

# Footer
st.markdown("---")
st.caption("Dibuat oleh Nur Indriani | UIN Alauddin Makassar")
