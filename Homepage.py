import streamlit as st

st.set_page_config(
    page_title="Homepage",
    page_icon="😄",
    layout="wide"
)

st.sidebar.markdown("# Halo-Halo 😄")
st.markdown(
    """
    <h1> Halo-Halo 😄</h1>
    <p>Perkenalkan namaku Apping. Kesibukan sekarang menjadi mahasiswa tingkat akhir yang benar-benar diakhir.<br>
    Topik tugas akhirnya dibidang NLP (Natural Languange Processing) Part of Speech Tagging untuk Bahasa Indonesia.
    Kalian bisa lihat profil lengkapku di <a href='https://haloapping.github.io/' target='blank'>haloapping</a>.</p>
    <h4>Kenapa website ini dibuat?</h4>
    <p>Website ini aku buat sebagai catatan apa saja yang sudah aku pelajari sejauh ini, sekaligus mau #sharingtipistipis ke teman-teman yang mungkin tertarik dengan machine learning, tapi bingung mau mulai darimana.<br>
    Dari hasil belajar dan eksplorasi aku banyak menemui hal-hal yang simpel, tapi sebenarnya fundamental yang mungkin gak didapatin di perkuliahan atau ditempat lain.
    Dimana hal fundamental ini sangat berguna untuk dapat mengerti topik yang lebih kompleks.</p>
    <h4>Mau bahas apa??</h4>
    <p>Aku akan berbagi hal-hal simpel dan fundamental terkait machine learning dengan gaya yang aku harap mudah dimengerti buat teman-teman yang baru mau mengenal machine learning.
    Topik yang aku akan bahas, seperti tools dan library machine learning, Python programming, dan mungkin juga deep learning.</p>
    <h4>Harapannya apa nih?!</h4>
    <p>Semoga dengan website ini menjadi langkah awal  teman-teman untuk tertarik dengan topik machine learning. Machine learning memang sulit, tapi bukan berarti karena sulit gak bisa dipelajari.
    <h4>Semoga bermanfaat yah!</h4>
    <p>💚</p>""",
    unsafe_allow_html=True
)
