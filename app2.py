
import streamlit as st
import requests
from streamlit_lottie import st_lottie
from PIL import Image

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

#Creamos datos sintéticos realistas
np.random.seed(42)
fechas = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
n_productos = ['Laptop', 'Mouse', 'Teclado', 'Monitor', 'Auriculares']
regiones = ['Norte', 'Sur', 'Este', 'Oeste', 'Centro']

#Generamos el DataSet
data = []
for fecha in fechas:
    for _ in range(np.random.poisson(10)):   #10 Ventas promedio por día
        data.append({
            'fecha': fecha,
            'producto': np.random.choice(n_productos),
            'region': np.random.choice(regiones),
            'cantidad': np.random.randint(1, 6),
            'precio_unitario': np.random.uniform(50, 1500),
            'vendedor': f'Vendedor_{np.random.randint(1, 21)}'
        })

df = pd.DataFrame(data)
df['venta_total'] = df['cantidad'] * df['precio_unitario']

# Funciones para generar gráficos
def graficar_ventas_mensuales(df_filtered):
    df_monthly = df_filtered.groupby(df_filtered['fecha'].dt.to_period('M'))['venta_total'].sum().reset_index()
    df_monthly['fecha'] = df_monthly['fecha'].astype(str)

    fig_monthly = px.line(df_monthly, x='fecha', y='venta_total',
                          title='Tendencia de Ventas Mensuales',
                          labels={'venta_total': 'Ventas ($)', 'fecha': 'Mes'})
    fig_monthly.update_traces(line=dict(width=3))
    return fig_monthly

def graficar_top_productos(df_filtered):
    df_productos = df_filtered.groupby('producto')['venta_total'].sum().sort_values(ascending=True)
    fig_productos = px.bar(x=df_productos.values, y=df_productos.index,
                           orientation='h', title='Ventas por Producto',
                           labels={'x': 'Ventas Totales ($)', 'y': 'Producto'})
    return fig_productos

def graficar_analisis_geografico(df_filtered):
    df_regiones = df_filtered.groupby('region')['venta_total'].sum().reset_index()
    fig_regiones = px.pie(df_regiones, values='venta_total', names='region',
                          title='Distribución de Ventas por Región',
                          labels={'venta_total': 'Ventas Totales ($)'})
    return fig_regiones

def graficar_correlacion_variables(df_filtered):
    df_corr = df_filtered[['cantidad', 'precio_unitario', 'venta_total']].corr()
    fig_heatmap = px.imshow(df_corr, text_auto=True, aspect="auto",
                            title='Correlación entre Variables',
                            labels=dict(x="Variables", y="Variables", color="Correlación"))
    return fig_heatmap

def graficar_distribucion_ventas(df_filtered):
    fig_dist = px.histogram(df_filtered, x='venta_total', nbins=50,
                            title='Distribución de Ventas Individuales')
    fig_dist.update_layout(bargap=0.2)
    return fig_dist


# configuración de la página
st.set_page_config(page_title="Dashboard de Ventas",
                   page_icon=":bar_chart:", layout="wide")

st.title("Dashboard de Análisis de Ventas")
st.markdown("---")

#SiderBar para filtros
st.sidebar.header("Filtros")
productos_seleccionados = st.sidebar.multiselect(
    "Selecciona Productos:",
    options=df['producto'].unique(),
    default=df['producto'].unique(),
)


regiones_seleccionadas = st.sidebar.multiselect(
    "Selecciona Regiones:",
    options=df['region'].unique(),
    default=df['region'].unique(),
)

#Filtrar los datos basado en la selección
df_filtered = df[
    (df['producto'].isin(productos_seleccionados)) &
    (df['region'].isin(regiones_seleccionadas))
]

#Métricas principales
col1, col2, col3, col4 = st.columns(4)
with col1:
  st.metric("Ventas Totales", f"${df_filtered['venta_total'].sum():,.0f}")
with col2:
  st.metric("Promedio por Venta", f"${df_filtered['venta_total'].mean():.0f}")
with col3:
    st.metric("Número de Ventas", f"{len(df_filtered):,}")
with col4:
    ventas_2024 = df_filtered[df_filtered['fecha'] >= '2024-01-01']['venta_total'].sum()
    ventas_2023 = df_filtered[df_filtered['fecha'] < '2024-01-01']['venta_total'].sum()
    if ventas_2023 > 0:
        crecimiento = ((ventas_2024 / ventas_2023) - 1) * 100
        st.metric("Crecimiento de Ventas 2024", f"{crecimiento:.1f}%")
    else:
        st.metric("Crecimiento de Ventas 2024", "N/A")


#Generar gráficos con datos filtrados
fig_monthly = graficar_ventas_mensuales(df_filtered)
fig_productos = graficar_top_productos(df_filtered)
fig_regiones = graficar_analisis_geografico(df_filtered)
fig_heatmap = graficar_correlacion_variables(df_filtered)
fig_dist = graficar_distribucion_ventas(df_filtered)

#Layout con dos columnas
col1, col2 = st.columns(2)
with col1:
  st.plotly_chart(fig_monthly, use_container_width=True)
  st.plotly_chart(fig_productos, use_container_width=True)
with col2:
  st.plotly_chart(fig_regiones, use_container_width=True)
  st.plotly_chart(fig_heatmap, use_container_width=True)

#Gráfico completo en la parte inferior
st.plotly_chart(fig_dist, use_container_width=True)
