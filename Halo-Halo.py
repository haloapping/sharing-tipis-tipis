import streamlit as st

st.set_page_config(
    page_title="Halo-Halo",
    page_icon="😄",
    layout="wide"
)

st.sidebar.markdown("# Halo-Halo 😄")
st.markdown(
    """
    <h1>Halo-Halo 😄</h1>
    <p>Perkenalkan nama saya Apping. Sekarang sedang berkuliah di salah satu kampus di Bandung.
    Untuk lebih lengkapnya kalian bisa lihat profil saya di <a href='https://haloapping.github.io/' target='blank'>haloapping</a>.</p>
    <p>Pada webiste ini saya akan <i>#sharingtipistipis</i> catatan selama belajar mengenai <i>machine learning</i>, <i>deep learning</i> dan masih banyak lagi.
    Bahasannya akan seputar hal-hal yang sifatnya simpel dan fundamental.</p>
    <p>Saya berharap semoga catatan simpel saya ini bisa bermanfaat buat teman-teman sekalian. Semangat belajar!</p>""",
    unsafe_allow_html=True
)
