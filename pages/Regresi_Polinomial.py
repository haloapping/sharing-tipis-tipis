import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import PolynomialFeatures


st.set_page_config(
    page_title="Regresi Polinomial",
    page_icon="🍩",
    layout="wide"
)

st.sidebar.markdown("## Regresi Polinomial")
st.markdown(
    body="<h2 style='text-align: center; margin-bottom: 50px'>Regresi Polinomial</h2>",
    unsafe_allow_html=True
)

st.markdown(
    body="""<h4>Intuisi dan Konsep</h4>
    <p style='margin-bottom: 30px;'>Regresi polinomial adalah regresi yang khusus digunakan pada kasus non-linear.
    Data masukan diubah menjadi non-linear menggunakan fungsi polinomial dengan derajat tertentu.</p>""",
    unsafe_allow_html=True
)

st.markdown(
    body="<h4>Persamaan Matematika</h4>",
    unsafe_allow_html=True
)

st.latex(
    body=r'''
        \begin{align*}
            y &= w~.~polynomial(x) + b\\\\
            \text{dimana, }polynomial &= \text{fungsi polinomial}\\
            y &= \text{target (prediksi),}\\
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
        help="Ukuran noise."
    )
    random_state = st.text_input(
        label="Random state",
        value="42",
        help="Menentukan pseudorandom."
    )

    st.markdown("<h4>Konfigurasi Model</h4>", unsafe_allow_html=True)
    degree = st.text_input(
        label="Derajat Polinomial",
        value="2",
        help="Ukuran derajat polinomial."
    )
    
    show_model = st.checkbox(label="Tampilkan decision function")

with col_2:
    st.markdown("<h4>Visualisasi Regresi Polinomial</h4>",
                unsafe_allow_html=True)

    np.random.seed(int(random_state))
    X = np.linspace(1, 100, int(n_samples)).reshape(-1, 1)
    y = np.power(X, 2).flatten() + int(noise)**3 * \
        np.random.randn(int(n_samples), )
    polynomial_features = PolynomialFeatures(degree=int(degree))
    X_poly = polynomial_features.fit_transform(X)
    model = LinearRegression().fit(X_poly, y)
    y_pred = model.predict(X_poly)
    r_square_score = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    
    if show_model:
        ax.set_title(f"$R^2$: {r_square_score:.3f} | Mean Absolute Error: {mae:.3f}", size=13)
    ax.set_xlabel("X (fitur)")
    ax.set_ylabel("y (target)")
    ax.scatter(X, y, c="green", label="Sampel")
    
    if show_model:
        ax.plot(X, y_pred, c="gold", label="Prediksi Model")
        
    ax.grid()
    ax.legend()
    st.pyplot(fig)

st.markdown(
    body="""<h4>Kode</h4>
    <p>Implementasi regresi polinomial menggunakan pustaka <a href='https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html#sklearn.preprocessing.PolynomialFeatures' target='_blank'>Scikit-Learn - Polynomial Features</a></p>""",
    unsafe_allow_html=True
)

st.code(
    body="""## Polynomial Regression with Scikit-Learn ##

# step 1: import modules
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

# step 2: create dummy data, X (feature) and y (target)
np.random.seed(42)
size = 20
noise = 400
X = np.linspace(1, 100, size).reshape(-1, 1)
y = np.power(X, 2).flatten() + noise * np.random.randn(size, )

# step 3: create polynomial feature
degree = 2
polynomial_features = PolynomialFeatures(degree=degree)
X_poly = polynomial_features.fit_transform(X)

# step 4: training
model = LinearRegression().fit(X_poly, y)

# step 5: predict
y_pred = model.predict(X_poly)

# step 6: evaluate
r_square_score = r2_score(y, y_pred)
mae = mean_absolute_error(y, y_pred)

# step 7: create plot
plt.figure(figsize=(5, 5))
plt.title(f"$R^2$: {r_square_score:.3f} | Mean Absolute Error: {mae:.3f}", size=13)
plt.xlabel("X (fitur)")
plt.ylabel("y (target)")
plt.scatter(X, y, c="green", label="Sampel")
plt.plot(X, y_pred, c="gold", label="Prediksi Model")
plt.grid()
plt.legend()

# step 8: save plot
plt.savefig("polynomial_regression.jpg", dpi=300)

# step 9: show plot
plt.show()

## Made with 💚 by haloapping (https://haloapping.github.io/) ##""",
    language="python"
)

st.markdown(
    body="""<h4>Referensi</h4>
    <ol>
        <li><a href='https://scikit-learn.org/stable/auto_examples/linear_model/plot_polynomial_interpolation.html#sphx-glr-auto-examples-linear-model-plot-polynomial-interpolation-py'>Scikit-Learn: Plot Polynomial Interpolation</a></li>
    </ol>""",
    unsafe_allow_html=True
)
