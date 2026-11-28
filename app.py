"""
Aplicación Streamlit para predecir la intención de compra en e-commerce
"""

import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, FunctionTransformer
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from xgboost import XGBClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# Configuración de la página
st.set_page_config(
    page_title="Intención de Compra - Grupo 2",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
    <style>
    .tarjeta {
        background-color: #f0f2f6;
        padding: 18px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .tarjeta h2 {
        margin: 0;
    }
    .supuesto {
        border-left: 6px solid #1f77b4;
        background-color: #eef4fb;
        padding: 12px 16px;
        border-radius: 6px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Título principal
st.title("🛒 Predicción de Intención de Compra en E-Commerce")
st.markdown("### Aplicación Interactiva con Sesiones de Navegación Web")
st.markdown("---")

# Sidebar para navegación
st.sidebar.title("🎯 Navegación")
pagina = st.sidebar.radio(
    "Selecciona una sección:",
    ["📊 Exploración de Datos",
     "⚙️ Entrenamiento del Modelo",
     "🔮 Hacer Predicciones",
     "📈 Análisis de Resultados"]
)
st.sidebar.markdown("---")
st.sidebar.caption("Business Predictive Analytics - UPC\n\nGrupo 2 - Sección 9341")

# Variables del modelo (las mismas del notebook)
cols_num = [
    "administrative", "administrative_duration",
    "informational", "informational_duration",
    "product_related", "product_related_duration",
    "page_values", "special_day"
]
cols_nominales = ["month", "visitor_type"]
cols_bool = ["weekend"]
cols_modelo = cols_num + cols_nominales + cols_bool
orden_meses = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Inicializar variables en session_state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'modelo' not in st.session_state:
    st.session_state.modelo = None
if 'modelo_smote' not in st.session_state:
    st.session_state.modelo_smote = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'metricas_base' not in st.session_state:
    st.session_state.metricas_base = None
if 'metricas_smote' not in st.session_state:
    st.session_state.metricas_smote = None


def limpiar(df):
    # misma limpieza que el notebook
    df.columns = df.columns.str.strip().str.lower()
    df = df.rename(columns={
        "productrelated": "product_related",
        "productrelated_duration": "product_related_duration",
        "bouncerates": "bounce_rates",
        "exitrates": "exit_rates",
        "pagevalues": "page_values",
        "specialday": "special_day",
        "operatingsystems": "operating_systems",
        "traffictype": "traffic_type",
        "visitortype": "visitor_type",
    })
    df = df.astype({"month": "string", "visitor_type": "string"})
    df = df.drop_duplicates()
    df = df.drop(columns=["operating_systems", "browser", "region",
                          "bounce_rates", "exit_rates", "traffic_type"])
    df["venta"] = df["revenue"].astype(str).str.lower().eq("true").astype("int8")
    return df


def a_entero(x):
    return x.astype(int)


def crear_preprocessor():
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler())
        ]), cols_num),
        ("cat_nom", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False))
        ]), cols_nominales),
        ("bool", Pipeline([
            ("to_int", FunctionTransformer(a_entero, feature_names_out="one-to-one")),
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(drop="if_binary", sparse_output=False))
        ]), cols_bool)
    ])


def crear_xgb():
    # parámetros que eligió el GridSearch del notebook
    return XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, subsample=0.8,
                         random_state=42, eval_metric="logloss", n_jobs=-1)


def calcular_metricas(y_real, y_pred, y_proba):
    tn, fp, fn, tp = confusion_matrix(y_real, y_pred).ravel()
    return {
        'accuracy': accuracy_score(y_real, y_pred),
        'precision': precision_score(y_real, y_pred),
        'recall': recall_score(y_real, y_pred),
        'f1': f1_score(y_real, y_pred),
        'auc': roc_auc_score(y_real, y_proba),
        'fp': fp,
        'fn': fn
    }


def matriz(y_real, y_pred, titulo, color):
    cm = confusion_matrix(y_real, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap=color, ax=ax,
                xticklabels=['No compra', 'Compra'],
                yticklabels=['No compra', 'Compra'])
    ax.set_ylabel('Real')
    ax.set_xlabel('Predicho')
    ax.set_title(titulo)
    return fig


# ========================================
# PÁGINA 1: EXPLORACIÓN DE DATOS
# ========================================
if pagina == "📊 Exploración de Datos":
    st.header("📊 Exploración y Análisis del Dataset")

    uploaded_file = st.file_uploader(
        "Sube el archivo online_shoppers_intention.csv:",
        type=['csv'],
        help="Dataset de UCI (Sakar et al., 2019) con la columna Revenue"
    )
    usar_proyecto = st.button("📂 Usar el dataset del proyecto")

    if uploaded_file is not None or usar_proyecto:
        try:
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_csv("online_shoppers_intention.csv")

            if 'Revenue' not in df.columns and 'revenue' not in df.columns:
                st.error(f"""
                ❌ **Error: no se encontró la columna 'Revenue'**

                **Columnas del archivo:** {', '.join(df.columns.tolist())}
                """)
                st.stop()

            st.session_state.df = limpiar(df)
            st.success(f"✅ Archivo cargado: {df.shape[0]:,} filas originales, "
                       f"{len(st.session_state.df):,} sin duplicados")
        except Exception as e:
            st.error(f"❌ Error al cargar el archivo: {e}")
            st.stop()

    if st.session_state.df is not None:
        df = st.session_state.df

        st.markdown("""
        <div class="supuesto">
        <b>Supuestos del proyecto:</b> cada sesión es independiente (el dataset no tiene ID de usuario)
        y cada mes es un ciclo de compra independiente. Por eso los datos se separan al azar y el mes
        es una variable más del modelo.
        </div>
        """, unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["📋 Vista General", "📊 Análisis de Desbalanceo", "📈 Conversión"])

        with tab1:
            st.subheader("Vista General del Dataset")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total de Sesiones", f"{df.shape[0]:,}")
            with col2:
                st.metric("Variables del Modelo", len(cols_modelo))
            with col3:
                st.metric("Valores Nulos", int(df.isnull().sum().sum()))

            st.write("**Primeras 10 filas del dataset:**")
            st.dataframe(df[cols_modelo + ['revenue']].head(10), use_container_width=True)

        with tab2:
            st.subheader("Análisis del Desbalanceo de Clases")

            class_counts = df['venta'].value_counts()
            class_percentages = df['venta'].value_counts(normalize=True) * 100

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("No Compra (revenue = False)", f"{class_counts[0]:,}",
                          f"{class_percentages[0]:.1f}%")
            with col2:
                st.metric("Compra (revenue = True)", f"{class_counts[1]:,}",
                          f"{class_percentages[1]:.1f}%")
            with col3:
                st.metric("Ratio de Desbalanceo", f"1:{class_counts[0] / class_counts[1]:.1f}")

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(['No compra', 'Compra'], class_counts.sort_index().values, color=['#3498db', '#2ecc71'])
            ax.set_ylabel('Sesiones')
            ax.set_title('Distribución de revenue')
            st.pyplot(fig)

            st.info("Con 15.6% de compras el balanceo no es obligatorio (está entre 10% y 20%), "
                    "así que lo probamos en la siguiente sección para decidir.")

        with tab3:
            st.subheader("Tasa de conversión")
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Por mes (%)**")
                meses = (df.groupby('month')['venta'].mean() * 100).reindex(orden_meses).round(2)
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(meses.index, meses.values, color='#1f77b4')
                ax.set_ylabel('Conversión (%)')
                ax.tick_params(axis='x', rotation=45)
                st.pyplot(fig)

            with col2:
                st.write("**Por tiempo en páginas de producto**")
                rangos = pd.cut(df['product_related_duration'] / 60, bins=[-1, 2, 10, 30, 1000],
                                labels=["0 - 2 min", "2 - 10 min", "10 - 30 min", "+30 min"])
                tabla = df.groupby(rangos, observed=True)['venta'].agg(sesiones='size', compras='sum')
                tabla['conversión (%)'] = (tabla['compras'] / tabla['sesiones'] * 100).round(2)
                st.dataframe(tabla, use_container_width=True)
    else:
        st.info("👆 Sube el archivo CSV o usa el dataset del proyecto para comenzar el análisis.")
        st.markdown("""
        **El archivo debe contener las columnas originales del dataset, entre ellas:**
        - `Administrative`, `Informational`, `ProductRelated` y sus duraciones
        - `PageValues`, `SpecialDay`, `Month`, `VisitorType`, `Weekend`
        - `Revenue`: variable objetivo (True = la sesión terminó en compra)
        """)

# ========================================
# PÁGINA 2: ENTRENAMIENTO DEL MODELO
# ========================================
elif pagina == "⚙️ Entrenamiento del Modelo":
    st.header("⚙️ Entrenamiento del Modelo")

    if st.session_state.df is None:
        st.warning("⚠️ Por favor, carga primero el dataset en la sección de Exploración de Datos.")
    else:
        df = st.session_state.df

        st.markdown("""
        ### Modelo: XGBoost optimizado

        Usamos el XGBoost con los parámetros que eligió el GridSearch del notebook
        (`n_estimators=100`, `max_depth=5`, `learning_rate=0.05`, `subsample=0.8`) y el mismo
        preprocesamiento: mediana + RobustScaler para las numéricas y One-Hot para mes, tipo de
        visitante y weekend.

        **Paso 1:** se entrena sin balanceo. **Paso 2:** se entrena con SMOTE para comparar.
        """)

        st.subheader("🎛️ Configuración de Parámetros")

        col1, col2, col3 = st.columns(3)

        with col1:
            sampling_strategy = st.slider(
                "Sampling Strategy",
                min_value=0.2,
                max_value=1.0,
                value=0.5,
                step=0.1,
                help="0.5 significa que la clase compra queda en el 50% de la clase no compra"
            )

        with col2:
            k_neighbors = st.number_input(
                "K Neighbors",
                min_value=1,
                max_value=10,
                value=5,
                help="Número de vecinos cercanos que usa SMOTE"
            )

        with col3:
            test_size = st.slider(
                "Tamaño del Test Set",
                min_value=0.1,
                max_value=0.4,
                value=0.2,
                step=0.05,
                help="En el notebook usamos 20% para test"
            )

        if st.button("🚀 Entrenar Modelo", type="primary", use_container_width=True):
            with st.spinner("Entrenando modelo... Esto puede tomar unos segundos."):

                X = df.drop(columns=['venta', 'revenue'])
                y = df['venta']

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )

                # winsorización con el P98 de train
                tope_alto = round(np.percentile(X_train['product_related_duration'], 98), 2)
                X_train['product_related_duration'] = X_train['product_related_duration'].clip(upper=tope_alto)
                X_test['product_related_duration'] = X_test['product_related_duration'].clip(upper=tope_alto)

                X_train = X_train[cols_modelo]
                X_test = X_test[cols_modelo]
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test

                st.write("### 📊 Distribución de Datos")
                col1, col2 = st.columns(2)
                train_dist = Counter(y_train)
                test_dist = Counter(y_test)
                with col1:
                    st.write("**Conjunto de Entrenamiento:**")
                    st.write(f"- No compra: {train_dist[0]:,}")
                    st.write(f"- Compra: {train_dist[1]:,}")
                with col2:
                    st.write("**Conjunto de Prueba:**")
                    st.write(f"- No compra: {test_dist[0]:,}")
                    st.write(f"- Compra: {test_dist[1]:,}")

                # Paso 1: sin balanceo
                st.write("### 🔵 Paso 1: XGBoost sin balanceo")
                progress_bar = st.progress(0)

                modelo = Pipeline([("prep", crear_preprocessor()), ("model", crear_xgb())])
                modelo.fit(X_train, y_train)
                progress_bar.progress(25)

                y_proba_base = modelo.predict_proba(X_test)[:, 1]
                y_pred_base = (y_proba_base >= 0.5).astype(int)
                metricas_base = calcular_metricas(y_test, y_pred_base, y_proba_base)
                st.session_state.metricas_base = metricas_base
                st.session_state.modelo = modelo
                progress_bar.progress(50)

                # Paso 2: con SMOTE (solo en train, dentro del pipeline de imblearn)
                st.write("### 🟢 Paso 2: XGBoost con SMOTE")

                modelo_smote = ImbPipeline([
                    ("prep", crear_preprocessor()),
                    ("smote", SMOTE(sampling_strategy=sampling_strategy, k_neighbors=k_neighbors,
                                    random_state=42)),
                    ("model", crear_xgb())
                ])
                modelo_smote.fit(X_train, y_train)
                progress_bar.progress(75)

                smote_dist = Counter(modelo_smote.named_steps['smote'].fit_resample(
                    modelo_smote.named_steps['prep'].transform(X_train), y_train)[1])
                st.write("**Distribución de train después de SMOTE:**")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("No compra", f"{smote_dist[0]:,}")
                with col2:
                    st.metric("Compra", f"{smote_dist[1]:,}", f"+{smote_dist[1] - train_dist[1]:,}")

                y_proba_smote = modelo_smote.predict_proba(X_test)[:, 1]
                y_pred_smote = (y_proba_smote >= 0.5).astype(int)
                metricas_smote = calcular_metricas(y_test, y_pred_smote, y_proba_smote)
                st.session_state.metricas_smote = metricas_smote
                st.session_state.modelo_smote = modelo_smote
                progress_bar.progress(100)

                st.success("✅ ¡Modelos entrenados!")

                st.write("### 📊 Comparación de Resultados en Test")
                nombres = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Falsos positivos', 'Falsos negativos']
                keys = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'fp', 'fn']
                comparacion = pd.DataFrame({
                    'Métrica': nombres,
                    'Sin balanceo': [round(float(metricas_base[k]), 4) for k in keys],
                    'Con SMOTE': [round(float(metricas_smote[k]), 4) for k in keys],
                    'Diferencia': [round(float(metricas_smote[k] - metricas_base[k]), 4) for k in keys]
                })
                st.dataframe(comparacion, use_container_width=True, hide_index=True)

                st.write("### 🔍 Matrices de Confusión")
                col1, col2 = st.columns(2)
                with col1:
                    st.pyplot(matriz(y_test, y_pred_base, 'Sin balanceo', 'Blues'))
                with col2:
                    st.pyplot(matriz(y_test, y_pred_smote, 'Con SMOTE', 'Greens'))

# ========================================
# PÁGINA 3: HACER PREDICCIONES
# ========================================
elif pagina == "🔮 Hacer Predicciones":
    st.header("🔮 Predecir si una Sesión Termina en Compra")

    if st.session_state.modelo is None:
        st.warning("⚠️ Por favor, entrena primero el modelo en la sección de Entrenamiento.")
    else:
        st.markdown("""
        ### Ingresa los datos acumulados de una sesión

        `revenue` no se ingresa: es lo que el modelo predice. Puedes usar una de las sesiones de
        ejemplo del informe o una sesión real del conjunto de prueba.
        """)

        version = st.radio("Modelo a usar:", ["Sin balanceo (el elegido en el informe)", "Con SMOTE"],
                           horizontal=True)
        modelo = st.session_state.modelo if version.startswith("Sin") else st.session_state.modelo_smote

        ejemplos = {
            "Sesión 1: recurrente en noviembre, mucho catálogo": dict(
                administrative=3, administrative_duration=90.0, informational=0, informational_duration=0.0,
                product_related=45, product_related_duration=1800.0, page_values=40.0, special_day=0.0,
                month="Nov", visitor_type="Returning_Visitor", weekend=False),
            "Sesión 2: recurrente en diciembre, navegación media": dict(
                administrative=2, administrative_duration=60.0, informational=1, informational_duration=30.0,
                product_related=25, product_related_duration=900.0, page_values=8.0, special_day=0.0,
                month="Dec", visitor_type="Returning_Visitor", weekend=False),
            "Sesión 3: nueva en mayo, pocas páginas": dict(
                administrative=0, administrative_duration=0.0, informational=0, informational_duration=0.0,
                product_related=6, product_related_duration=150.0, page_values=0.0, special_day=0.0,
                month="May", visitor_type="New_Visitor", weekend=True),
        }

        col1, col2 = st.columns([3, 1])
        with col1:
            usar_random = st.checkbox("Usar sesión aleatoria del conjunto de prueba")
        with col2:
            if usar_random and st.button("🎲 Generar Aleatorio"):
                st.session_state.idx_random = np.random.randint(0, len(st.session_state.X_test))

        if usar_random and 'idx_random' in st.session_state:
            idx = st.session_state.idx_random
            muestra = st.session_state.X_test.iloc[idx]
            real_class = int(st.session_state.y_test.iloc[idx])
            st.info(f"**Sesión #{idx} del test** - Resultado real: "
                    f"{'🟢 COMPRA' if real_class == 1 else '🔴 NO COMPRA'}")
            base = {c: muestra[c] for c in cols_modelo}
        else:
            eleccion = st.selectbox("Sesión de ejemplo:", list(ejemplos.keys()))
            base = ejemplos[eleccion]

        st.subheader("📝 Datos de la Sesión")
        meses_opc = orden_meses if base['month'] in orden_meses else orden_meses + [base['month']]
        tipos = ["Returning_Visitor", "New_Visitor"]
        # Other solo aparece si la sesión aleatoria del test lo trae
        if base['visitor_type'] not in tipos:
            tipos.append(base['visitor_type'])

        tab1, tab2 = st.tabs(["🗓️ Mes y visitante", "📄 Páginas y tiempos"])
        input_data = {}

        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                input_data['month'] = st.selectbox("Mes", meses_opc, index=meses_opc.index(base['month']))
                input_data['visitor_type'] = st.selectbox("Tipo de visitante", tipos,
                                                          index=tipos.index(base['visitor_type']))
            with col2:
                input_data['weekend'] = st.checkbox("Visita en fin de semana", value=bool(base['weekend']))
                input_data['special_day'] = st.slider("Cercanía a fecha especial", 0.0, 1.0,
                                                      float(base['special_day']), 0.2)

        with tab2:
            cols = st.columns(3)
            campos = [
                ('product_related', "Páginas de producto"),
                ('product_related_duration', "Tiempo en producto (s)"),
                ('page_values', "page_values"),
                ('administrative', "Páginas administrativas"),
                ('administrative_duration', "Tiempo en administrativas (s)"),
                ('informational', "Páginas informativas"),
                ('informational_duration', "Tiempo en informativas (s)"),
            ]
            for i, (col_name, etiqueta) in enumerate(campos):
                with cols[i % 3]:
                    input_data[col_name] = st.number_input(etiqueta, min_value=0.0,
                                                           value=float(base[col_name]), format="%.2f")

        st.markdown("---")
        if st.button("🔮 Predecir Compra", type="primary", use_container_width=True):
            input_df = pd.DataFrame([input_data])[cols_modelo]
            probabilidad = modelo.predict_proba(input_df)[0]
            prediccion = int(probabilidad[1] >= 0.5)

            st.markdown("---")
            st.subheader("📊 Resultado de la Predicción")

            # tarjeta con HTML: verde si compra, naranja si no
            color = "#2e7d32" if prediccion == 1 else "#ef6c00"
            texto = "COMPRA PROBABLE" if prediccion == 1 else "NO COMPRA PROBABLE"
            accion = "Sin cupón" if prediccion == 1 else "Candidata a cupón"
            st.markdown(f"""
            <div class="tarjeta" style="border-left: 8px solid {color};">
                <p style="margin: 0; color: #555;">Probabilidad de compra (revenue = True)</p>
                <h2 style="color: {color};">{probabilidad[1]:.1%} - {texto}</h2>
                <p style="margin: 0;"><b>Acción:</b> {accion}</p>
            </div>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Probabilidad de Compra", f"{probabilidad[1]:.2%}")
            with col2:
                st.metric("Probabilidad de No Compra", f"{probabilidad[0]:.2%}")

            fig, ax = plt.subplots(figsize=(8, 3))
            ax.barh(['No compra', 'Compra'], probabilidad, color=['#ef6c00', '#2e7d32'])
            ax.set_xlabel('Probabilidad')
            ax.set_xlim([0, 1])
            ax.set_title('Distribución de Probabilidades')
            for i, v in enumerate(probabilidad):
                ax.text(v, i, f' {v:.2%}', va='center')
            st.pyplot(fig)

            if usar_random and 'idx_random' in st.session_state:
                st.markdown("---")
                st.subheader("🎯 Validación")
                if prediccion == real_class:
                    st.success(f"✅ **Predicción correcta**: la sesión {'SÍ' if real_class == 1 else 'NO'} terminó en compra.")
                else:
                    st.error(f"❌ **Predicción incorrecta**: el modelo predijo "
                             f"{'COMPRA' if prediccion == 1 else 'NO COMPRA'}, pero el resultado real fue "
                             f"{'COMPRA' if real_class == 1 else 'NO COMPRA'}.")

# ========================================
# PÁGINA 4: ANÁLISIS DE RESULTADOS
# ========================================
elif pagina == "📈 Análisis de Resultados":
    st.header("📈 Análisis Detallado de Resultados")

    if st.session_state.metricas_base is None or st.session_state.metricas_smote is None:
        st.warning("⚠️ Por favor, entrena primero el modelo en la sección de Entrenamiento.")
    else:
        st.subheader("📊 Comparación de Métricas (SMOTE frente a sin balanceo)")

        metricas_base = st.session_state.metricas_base
        metricas_smote = st.session_state.metricas_smote

        metricas_nombres = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        metricas_keys = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        cols = st.columns(5)

        for col, nombre, key in zip(cols, metricas_nombres, metricas_keys):
            with col:
                st.metric(nombre, f"{metricas_smote[key]:.3f}",
                          f"{metricas_smote[key] - metricas_base[key]:+.3f}")

        st.subheader("📊 Visualización Comparativa")

        fig, ax = plt.subplots(figsize=(12, 5))
        x = np.arange(len(metricas_nombres))
        width = 0.35
        base_vals = [metricas_base[k] for k in metricas_keys]
        smote_vals = [metricas_smote[k] for k in metricas_keys]

        ax.bar(x - width / 2, base_vals, width, label='Sin balanceo', color='#1f77b4', alpha=0.8)
        ax.bar(x + width / 2, smote_vals, width, label='Con SMOTE', color='#2ecc71', alpha=0.8)
        ax.set_ylabel('Valor')
        ax.set_title('Comparación de Métricas en Test: Sin balanceo vs SMOTE')
        ax.set_xticks(x)
        ax.set_xticklabels(metricas_nombres)
        ax.legend()
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)
        for i, (b_val, s_val) in enumerate(zip(base_vals, smote_vals)):
            ax.text(i - width / 2, b_val, f'{b_val:.3f}', ha='center', va='bottom', fontsize=9)
            ax.text(i + width / 2, s_val, f'{s_val:.3f}', ha='center', va='bottom', fontsize=9)
        st.pyplot(fig)

        st.markdown("---")
        st.subheader("💡 Interpretación de Resultados")

        st.markdown(f"""
        **Recall (compras detectadas):** sin balanceo {metricas_base['recall']:.1%}, con SMOTE {metricas_smote['recall']:.1%}.

        **Precision (de las que marca como compra, cuántas compran):** sin balanceo {metricas_base['precision']:.1%},
        con SMOTE {metricas_smote['precision']:.1%}.

        **Errores en test:**
        - Falsos negativos (cupón regalado a quien ya iba a comprar): {metricas_base['fn']} sin balanceo y {metricas_smote['fn']} con SMOTE.
        - Falsos positivos (venta perdida: no se ofreció cupón y la sesión se fue): {metricas_base['fp']} sin balanceo y {metricas_smote['fp']} con SMOTE.

        > En nuestro caso la venta perdida cuesta más que el 15% de un cupón, por eso el error que más cuidamos es el falso positivo.
        """)

        st.markdown("---")
        st.subheader("🎯 Recomendaciones")

        if metricas_smote['fp'] > metricas_base['fp']:
            st.warning(f"""
            ⚠️ **Nos quedamos sin balanceo.**

            SMOTE detecta más compras, pero agrega {metricas_smote['fp'] - metricas_base['fp']} ventas perdidas
            (falsos positivos) a cambio de {metricas_base['fn'] - metricas_smote['fn']} cupones regalados menos.
            Como la venta perdida es el error más caro, el modelo sin balanceo es el recomendado.
            """)
        else:
            st.success("✅ Con esta configuración SMOTE no aumenta las ventas perdidas; se podría considerar.")

        st.info("Limitación: el modelo depende sobre todo de `page_values`. En el notebook, sin esa variable "
                "el ROC-AUC bajó de 0.93 a 0.75.")

        st.markdown("---")
        st.subheader("💾 Exportar Reporte")

        reporte = pd.DataFrame({
            'Métrica': metricas_nombres,
            'Sin balanceo': base_vals,
            'Con SMOTE': smote_vals,
            'Diferencia': [s - b for s, b in zip(smote_vals, base_vals)]
        })
        st.download_button(
            label="📥 Descargar reporte.csv",
            data=reporte.to_csv(index=False),
            file_name="reporte_metricas.csv",
            mime="text/csv",
            use_container_width=True
        )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>Predicción de Intención de Compra en E-Commerce</strong></p>
    <p>Business Predictive Analytics - UPC - Grupo 2</p>
</div>
""", unsafe_allow_html=True)
