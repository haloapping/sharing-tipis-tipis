import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy import special
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


st.set_page_config(
    page_title="Regresi Logistik",
    page_icon="🐉",
    layout="wide"
)

st.sidebar.markdown("## Regresi Logistik")
st.markdown(
    body="<h2 style='text-align: center; margin-bottom: 50px'>Regresi Logistik</h2>",
    unsafe_allow_html=True
)

st.markdown(
    body="""<h4>Intuisi dan Konsep</h4>
    <p style='margin-bottom: 30px;'>Regresi logistik menggunakan fungsi sigmoid untuk memprediksi suatu nilai. Fungsi linier diubah menjadi non-linier menggunakan fungsi sigmoid.
    Walaupun namanya regresi logistik, tetapi model ini digunakan pada kasus klasifikasi. Nilai keluaran dari model berupa probabilitas dengan rentang nilai adalah 0 sampai 1.</p>""",
    unsafe_allow_html=True
)

st.markdown(
    body="<h4>Persamaan Matematika</h4>",
    unsafe_allow_html=True
)

st.latex(
    body=r'''
        \begin{align*}
            y &= \frac{1}{1 + {e^{-x}}}\\\\
            \text{dimana, }y &= \text{target (prediksi),}\\
            x &= \text{fitur,}\\
            e &= \text{fungsi eksponensial}
        \end{align*}
    '''
)

col_1, col_2 = st.columns(2)

with col_1:
    st.markdown("<h4>Konfigurasi Dataset</h4>", unsafe_allow_html=True)
    n_samples = st.text_input(
        label="Jumlah sampel",
        value="100",
        help="Ukuran sampel yang diinginkan."
    )
    n_range = st.text_input(
        label="Rentang nilai",
        value="10",
        help="Rentang nilai sumbu x (-rentang nilai, +rentang nilai)"
    )
    random_state = st.text_input(
        label="Random state",
        value="42",
        help="Menentukan pembuatan nomor acak untuk pembuatan kumpulan data."
    )
    show_model = st.checkbox("Tampilkan decision function")

with col_2:
    np.random.seed(int(random_state))
    # Step 1: create dummy data
    X = np.linspace(-int(n_range), int(n_range), int(n_samples)).reshape(-1, 1)
    y = special.expit(X + np.random.randn(int(n_samples), 1))

    # Step 2: training
    model = LogisticRegression().fit(X, np.where(y <= 0.5, 0, 1))
    y_pred = special.expit(model.decision_function(X))

    # Step 3: evaluate
    acc = accuracy_score(np.where(y <= 0.5, 0, 1), model.predict(X))

    # Step 4: plot
    fig, ax = plt.subplots(figsize=(5, 5))
    if show_model:
        ax.set_title(f"Accuracy: {acc}", size=13)
    ax.set_xlabel("X (fitur)")
    ax.set_ylabel("y (target/probabilitas)")
    ax.scatter(X, y, c="green", label="Sampel")
    
    if show_model:
        ax.plot(X, y_pred, c="gold", label="Decision Function")
        
    ax.grid()
    ax.legend()
    st.pyplot(fig)

st.markdown(
    body="""<h4>Kode</h4>
    <p>Implementasi regresi logistik menggunakan pustaka <a href='https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression' target='_blank'>Scikit-Learn - Logistic Regression</a></p>""",
    unsafe_allow_html=True
)

st.code(
    body="""## Logistic Regression with Scikit-Learn ##

# Step 1: Import modules
import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 2: create dummy data
n_sample = 100
n_range = 10
X = np.linspace(-n_range, n_range, n_sample).reshape(-1, 1)
y = special.expit(X + np.random.randn(n_sample, 1))

# Step 3: training
model = LogisticRegression().fit(X, np.where(y <= 0.5, 0, 1))
y_pred = special.expit(model.decision_function(X))

# Step 4: evaluate
acc = accuracy_score(np.where(y <= 0.5, 0, 1), model.predict(X))

# Step 5: plot
plt.title(f"Accuracy: {acc}")
plt.scatter(X, y, label="Sampel", c="g")
plt.scatter(X, y_pred, label="Decision Function", c="gold")
plt.xlabel("x (fitur)")
plt.ylabel("y (target/probabilitas)")
plt.legend()
plt.grid()

# step 6: save plot
plt.savefig("logistic_regression.jpg", dpi=300)

# step 7: show plot
plt.show()

## Made with 💚 by haloapping (https://haloapping.github.io/) ##""",
    language="python"
)

st.markdown(
    body="""<h4>Referensi</h4>
    <a href='https://scikit-learn.org/stable/auto_examples/linear_model/plot_logistic.html#sphx-glr-auto-examples-linear-model-plot-logistic-py'>Scikit-Learn Tutorial</a>""",
    unsafe_allow_html=True
)
