import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression


st.set_page_config(
    page_title="Regresi Linier",
    page_icon="🚀",
    layout="wide"
)

st.sidebar.markdown("## Regresi Linier")
st.markdown(
    body="<h2 style='text-align: center; margin-bottom: 50px'>Regresi Linier</h2>",
    unsafe_allow_html=True
)

st.markdown(
    body="""<h4>Intuisi dan Konsep</h4>
    <p style='margin-bottom: 30px;'>Buat sebuah garis lurus yang dapat menghampiri semua sampel atau data.
    Jarak antara garis dengan setiap sampel sekecil mungkin. Pola dari data harus cukup linier.</p>""",
    unsafe_allow_html=True
)

st.markdown(
    body="<h4>Persamaan Matematika</h4>",
    unsafe_allow_html=True
)

st.latex(
    body=r'''
        \begin{align*}
            y &= wx + b\\\\
            \text{dimana, }y &= \text{target (prediksi),}\\
            w &= \text{bobot,}\\
            x &= \text{fitur,}\\
            b &= \text{bias}
        \end{align*}
    '''
)

col_1, col_2 = st.columns(2)

with col_1:
    st.markdown("<h4>Konfigurasi Dataset</h4>", unsafe_allow_html=True)
    n_samples = st.text_input(
        label="Jumlah sampel",
        value="20",
        help="Ukuran sampel yang diinginkan."
    )
    noise = st.text_input(
        label="Noise",
        value="10",
        help="Seberapa besar simpangan baku dari Gaussian Noise."
    )
    random_state = st.text_input(
        label="Random state",
        value="42",
        help="Menentukan pembuatan nomor acak untuk pembuatan kumpulan data."
    )
    show_model = st.checkbox("Tampilkan model regresi")

with col_2:
    X, y = make_regression(n_samples=int(n_samples),
                           n_features=1,
                           n_targets=1,
                           noise=int(noise),
                           random_state=int(random_state))
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    r2_score = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    st.markdown(
        body="<h4 style='text-align: center'>Visualisasi Regresi Linier</h4>",
        unsafe_allow_html=True)
    fig, ax = plt.subplots()
    ax.set_title(
        f"$R^2$: {r2_score:.3f} | Mean Absolute Error: {mae:.3f}", size=13)
    ax.set_xlabel("X (fitur)")
    ax.set_ylabel("y (target)")
    ax.grid()
    ax.scatter(X, y, c="green", label="Sampel")
    
    if show_model:
        ax.plot(X, y_pred, c="gold", label="Prediksi Model")
    ax.legend()
    st.pyplot(fig)

st.markdown(
    body="""<h4>Kode</h4>
    <p>Implementasi regresi linier menggunakan pustaka <a href='https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression' target='_blank'>Scikit-Learn - Linear Regression</a></p>""",
    unsafe_allow_html=True
)

st.code(
    body="""## Linear Regression with Scikit-Learn ##

# step 1: import modules
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# step 2: create dummy data, X (feature) and y (target)
X, y = make_regression(n_samples=20,
                       n_features=1,
                       n_targets=1,
                       noise=10,
                       random_state=42)

# step 3: training
model = LinearRegression().fit(X, y)

# step 4: predict
y_pred = model.predict(X)

# step 5: evaluate
r_square_score = r2_score(y, y_pred)
mae = mean_absolute_error(y, y_pred)

# step 6: create plot
plt.figure(figsize=(5, 5))
plt.title(f"$R^2$: {r_square_score:.3f} | Mean Absolute Error: {mae:.3f}", size=13)
plt.xlabel("X (fitur)")
plt.ylabel("y (target)")
plt.grid()
plt.scatter(X, y, c="green", label="Samples")
plt.plot(X, y_pred, c="gold", label="Prediksi Model")
plt.legend()

# step 7: save plot
plt.savefig("linear_regression.jpg", dpi=300)

# step 8: show plot
plt.show();

## Made with 💚 by haloapping (https://haloapping.github.io/) ##""",
    language="python"
)

st.markdown(
    body="""<h4>Referensi</h4>
    <a href='https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py'>Scikit-Learn Tutorial</a>""",
    unsafe_allow_html=True
)
