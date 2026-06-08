from pathlib import Path
import sys
import tempfile

import streamlit as st
from PIL import Image


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

sys.path.append(str(SRC_DIR))

from predict import predict_image


st.set_page_config(
    page_title="Clasificador de calidad de frutas",
    page_icon="🍎",
    layout="wide"
)


st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0px;
    }
    .subtitle {
        font-size: 18px;
        color: #555;
        margin-top: 0px;
        margin-bottom: 25px;
    }
    .info-card {
        padding: 18px;
        border-radius: 14px;
        background-color: #f7f7f7;
        border: 1px solid #e5e5e5;
        margin-bottom: 15px;
    }
    .small-text {
        color: #666;
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown('<div class="main-title">🍎 Clasificador de calidad de frutas frescas</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Sistema de visión por computador para estimar la calidad y el tamaño relativo de frutas.</div>',
    unsafe_allow_html=True
)


with st.sidebar:
    st.header("Configuración")

    model_option = st.selectbox(
        "Modelo de predicción",
        ["random_forest", "svm", "cnn"],
        index=1,
        help="SVM queda seleccionado por defecto porque obtuvo el mejor desempeño general."
    )

    input_option = st.radio(
        "Modo de entrada",
        ["Subir imagen", "Usar cámara"],
        index=0
    )

    st.divider()

    st.subheader("Frutas del dataset")
    st.write("Manzana, banano, guayaba, limón, naranja y granada.")

    st.caption(
        "El sistema clasifica la calidad de la fruta en bueno, regular o malo. "
        "No identifica la fruta como salida principal."
    )


left_col, right_col = st.columns([1.05, 0.95], gap="large")


with left_col:
    st.subheader("Entrada de imagen")

    uploaded_file = None
    camera_file = None

    if input_option == "Subir imagen":
        uploaded_file = st.file_uploader(
            "Sube una imagen de una fruta",
            type=["jpg", "jpeg", "png", "webp"]
        )
        image_source = uploaded_file
    else:
        camera_file = st.camera_input("Toma una foto de la fruta")
        image_source = camera_file

    st.markdown(
        """
        <div class="info-card">
        <b>Recomendación:</b> usa imágenes de una sola fruta, con fondo simple y buena iluminación.
        El modelo fue entrenado con frutas en tres estados: bueno, regular y malo.
        </div>
        """,
        unsafe_allow_html=True
    )


with right_col:
    st.subheader("Información del sistema")

    st.markdown(
        """
        <div class="info-card">
        <b>Salida del sistema:</b><br>
        • Calidad estimada: bueno, regular o malo.<br>
        • Tamaño relativo: pequeño, mediano o grande.<br>
        • Confianza aproximada de la predicción.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-card">
        <b>Modelos disponibles:</b><br>
        • Random Forest<br>
        • SVM<br>
        • CNN
        </div>
        """,
        unsafe_allow_html=True
    )


if image_source is not None:
    st.divider()

    image = Image.open(image_source).convert("RGB")

    preview_col, result_col = st.columns([1, 1], gap="large")

    with preview_col:
        st.subheader("Imagen cargada")
        st.image(image, use_container_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
        image.save(temp_file.name)
        temp_path = temp_file.name

    with result_col:
        st.subheader("Resultado")

        if st.button("Analizar fruta", type="primary", use_container_width=True):
            try:
                with st.spinner("Analizando imagen..."):
                    result = predict_image(temp_path, model_type=model_option)

                quality = result["quality"]
                confidence = result["confidence"]
                size = result["size"]
                area_ratio = result["area_ratio"]

                confidence_value = confidence if confidence is not None else 0

                metric_col_1, metric_col_2, metric_col_3 = st.columns(3)

                with metric_col_1:
                    st.metric("Calidad", quality.upper())

                with metric_col_2:
                    st.metric("Tamaño", size.upper())

                with metric_col_3:
                    if confidence is not None:
                        st.metric("Confianza", f"{confidence_value * 100:.1f}%")
                    else:
                        st.metric("Confianza", "N/A")

                st.write(f"**Modelo usado:** `{model_option}`")
                st.write(f"**Área relativa aproximada:** `{area_ratio:.2f}`")

                if quality == "bueno":
                    st.success("La fruta parece estar en buen estado.")
                elif quality == "regular":
                    st.warning("La fruta presenta señales intermedias de desgaste o madurez.")
                else:
                    st.error("La fruta parece estar en mal estado o presenta deterioro visible.")

                if confidence is not None and confidence < 0.60:
                    st.warning(
                        "La confianza del modelo es baja. Esto puede pasar cuando la fruta tiene características ambiguas "
                        "entre dos categorías, especialmente entre bueno y regular o entre regular y malo."
                    )

                st.markdown(
                    """
                    <div class="small-text">
                    Nota: la estimación de tamaño es relativa a la proporción que ocupa la fruta en la imagen,
                    no corresponde a una medición física real en centímetros.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            except Exception as error:
                st.error("Ocurrió un error al analizar la imagen.")
                st.exception(error)

else:
    st.divider()
    st.info("Sube una imagen o toma una foto para comenzar el análisis.")
