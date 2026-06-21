import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from groq import Groq

# ==========================================
# 1. KONFIGURASI
# ==========================================
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
GROQ_MODEL = "llama-3.3-70b-versatile"

# ==========================================
# 2. PAGE CONFIG & CUSTOM CSS
# ==========================================
st.set_page_config(
    page_title="FruitSense AI",
    page_icon="🥭",
    layout="wide", # Pakai wide biar layout 2 kolomnya lega
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background-color: #0E1117; /* Background utama gelap */
        color: #FAFAFA; /* Teks utama terang */
    }
    
    /* Header Segar Organik (Tetap dipertahankan karena cocok di dark mode) */
    .main-header {
        text-align: center;
        padding: 2.5rem 1rem;
        background: linear-gradient(135deg, #43A047 0%, #2E7D32 100%);
        border-radius: 24px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    }
    .main-header h1 { font-size: 2.8rem; font-weight: 700; margin-bottom: 0.5rem; }
    .main-header p { font-size: 1.1rem; opacity: 0.9; }
    
    /* Card Styles (Diubah jadi abu-abu gelap) */
    .custom-card {
        background: #1E1E1E; /* Warna card gelap */
        padding: 2rem;
        border-radius: 24px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        margin-bottom: 1.5rem;
        border: 1px solid #333; /* Border disesuaikan */
        color: #FAFAFA;
    }
    
    /* Label Status */
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1.2rem;
        color: white;
        margin-bottom: 1rem;
    }
    .bg-matang { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
    .bg-mentah { background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%); }
    .bg-busuk { background: linear-gradient(135deg, #cb2d3e 0%, #ef473a 100%); }
    
    /* AI Box (Diubah jadi hijau super gelap agar menyatu dengan dark mode) */
    .ai-advice-box {
        background: #1A281A; 
        border-left: 5px solid #43A047;
        padding: 1.5rem;
        border-radius: 0 16px 16px 0;
        margin: 1.5rem 0;
        color: #E0E0E0;
    }
    
    /* Tombol Utama */
    .stButton > button {
        background: linear-gradient(135deg, #43A047 0%, #2E7D32 100%);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 16px;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(67, 160, 71, 0.4);
        color: white;
    }
    
    /* Memastikan input text dan label tetap terlihat jelas di dark mode */
    h3 { color: #FAFAFA !important; }
    .stTextInput > div > div > input { color: #FAFAFA; }
    
    /* Hide elemen bawaan */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. SESSION STATE
# ==========================================
for key in ['analysis_done', 'final_label', 'confidence', 'fruit_name', 'initial_advice', 'detailed_recipe']:
    if key not in st.session_state:
        st.session_state[key] = None

# ==========================================
# 4. FUNGSI API
# ==========================================
@st.cache_resource
def load_cnn_model():
    return tf.keras.models.load_model('model_buah_terbaik.h5')

def get_initial_analysis(fruit, condition):
    client = Groq(api_key=GROQ_API_KEY)
    prompt = f"""Buah: {fruit}
Kondisi: {condition}

Berikan analisis singkat dengan format persis seperti ini:
### 🛡️ Status Kelayakan
[Jelaskan apakah masih aman dimakan atau harus dibuang]

### 💡 Tips Penyimpanan
[Jelaskan cara menyimpan buah ini dengan kondisinya yang sekarang]

### 🍳 Ide Olahan Terbaik
Sebutkan 3 nama olahan spesifik yang sangat cocok untuk kondisi buah ini. (Hanya sebutkan namanya saja dipisah koma, contoh: Smoothie Pisang, Pisang Goreng Keju, Bolu Pisang)"""

    completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=GROQ_MODEL,
    )
    return completion.choices[0].message.content

def get_full_recipe(fruit, condition, dish_name):
    client = Groq(api_key=GROQ_API_KEY)
    prompt = f"Buatkan resep praktis untuk '{dish_name}' menggunakan bahan utama buah {fruit} yang kondisinya {condition}. Sertakan bahan dan langkah-langkah yang jelas."
    completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=GROQ_MODEL,
    )
    return completion.choices[0].message.content

# ==========================================
# 5. HEADER
# ==========================================
st.markdown("""
<div class="main-header">
    <h1>🥭 FruitSense AI</h1>
    <p>Asisten Cerdas untuk Klasifikasi Kelayakan Buah & Rekomendasi Resep</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 6. LAYOUT UTAMA (2 KOLOM)
# ==========================================
col1, col2 = st.columns([1, 1.2], gap="large")

# --- KOLOM KIRI: INPUT ---
with col1:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("### 1. Masukkan Data Buah")
    
    fruit_name = st.text_input("Nama Buah", placeholder="Misal: Pisang, Jambu, Pepaya...")
    input_method = st.radio("Metode Input Gambar:", ["📁 Upload File", "📷 Kamera"], horizontal=True)
    
    image_data = None
    if input_method == "📁 Upload File":
        uploaded_file = st.file_uploader("Upload foto buah", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        if uploaded_file: image_data = Image.open(uploaded_file)
    else:
        camera_input = st.camera_input("Ambil foto buah", label_visibility="collapsed")
        if camera_input: image_data = Image.open(camera_input)
        
    if image_data:
        st.image(image_data, use_container_width=True, caption="Preview Gambar")
        
        if fruit_name:
            if st.button("🔍 Analisis Sekarang", use_container_width=True):
                with st.spinner("Mengolah citra CNN..."):
                    st.session_state.fruit_name = fruit_name
                    model = load_cnn_model()
                    
                    # Preprocess Image
                    img = image_data.resize((224, 224))
                    img_array = np.array(img) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)
                    
                    # Predict
                    preds = model.predict(img_array)
                    labels = ['Busuk', 'Matang', 'Mentah']
                    st.session_state.final_label = labels[np.argmax(preds)]
                    st.session_state.confidence = np.max(preds) * 100
                
                with st.spinner("Meminta saran pakar nutrisi AI..."):
                    st.session_state.initial_advice = get_initial_analysis(fruit_name, st.session_state.final_label)
                    st.session_state.analysis_done = True
                    st.session_state.detailed_recipe = None # Reset resep
                st.rerun()
        else:
            st.warning("⚠️ Ketik nama buah terlebih dahulu.")
    st.markdown('</div>', unsafe_allow_html=True)

# --- KOLOM KANAN: HASIL ---
with col2:
    if st.session_state.analysis_done:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("### 2. Hasil Analisis AI")
        
        # Tentukan warna badge
        badge_class = "bg-matang"
        emoji = "✅"
        if st.session_state.final_label == "Busuk": badge_class, emoji = "bg-busuk", "⚠️"
        elif st.session_state.final_label == "Mentah": badge_class, emoji = "bg-mentah", "🟡"
        
        st.markdown(f"""
        <div class="status-badge {badge_class}">
            {emoji} {st.session_state.final_label} ({st.session_state.confidence:.1f}%)
        </div>
        """, unsafe_allow_html=True)
        
        # Tampilkan Saran Awal
        st.markdown(f'<div class="ai-advice-box">{st.session_state.initial_advice}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Fitur Resep
        if st.session_state.final_label != "Busuk":
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.markdown("### 3. Eksekusi Ide Menu")
            st.write("Tertarik bikin salah satu menu di atas? Ketik di bawah ini untuk mendapatkan resep lengkapnya!")
            
            pilihan_resep = st.text_input("Ketik nama olahan pilihanmu:", placeholder="Misal: Smoothie Pisang")
            
            if st.button("📖 Beri Tahu Resepnya!"):
                if pilihan_resep:
                    with st.spinner(f"Menyusun resep {pilihan_resep}..."):
                        st.session_state.detailed_recipe = get_full_recipe(
                            st.session_state.fruit_name, 
                            st.session_state.final_label, 
                            pilihan_resep
                        )
                else:
                    st.warning("Ketik nama menu dulu ya!")
            
            if st.session_state.detailed_recipe:
                st.markdown("---")
                st.markdown(st.session_state.detailed_recipe)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.error("Karena terdeteksi Busuk, AI tidak merekomendasikan resep apapun demi kesehatan Anda.")
    else:
        # Tampilan kosong di awal
        st.markdown('<div class="custom-card" style="text-align:center; padding: 4rem 2rem; color: #888;">', unsafe_allow_html=True)
        st.markdown("<h2>🤖</h2>", unsafe_allow_html=True)
        st.markdown("Tunggu apa lagi? Upload foto buahmu di sebelah kiri, dan biarkan AI menganalisisnya.")
        st.markdown('</div>', unsafe_allow_html=True)
