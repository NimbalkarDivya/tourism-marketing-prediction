import streamlit as st
import subprocess

st.title("📊 Model Comparison Dashboard")

st.write("Run multi-model comparison in terminal:")
st.code("python -m src.multi_model_training")

st.info("Check terminal output for accuracy comparison and best model.")