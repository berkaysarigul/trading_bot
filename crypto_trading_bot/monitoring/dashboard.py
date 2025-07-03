import streamlit as st
import pandas as pd

def show_dashboard(trade_log, performance_metrics):
    st.title("Crypto Trading Bot Dashboard")
    st.subheader("Portföy Değeri")
    st.line_chart(trade_log['portfolio_value'])
    st.subheader("İşlem Geçmişi")
    st.dataframe(trade_log)
    st.subheader("Performans Metrikleri")
    for k, v in performance_metrics.items():
        st.write(f"{k}: {v}") 