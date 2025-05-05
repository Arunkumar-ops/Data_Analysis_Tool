import streamlit as st
import pandas as pd
import plotly.express as px
import json
import io
from agent import create_agent, query_agent  # Import Functions from agents.py

st.set_page_config(page_title="CSV Chatbot Analyst", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
    <style>
        body { background-color: #0E1117; color: white; }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 Chat with Your CSV Data")

# File upload
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.session_state.agent_df = df
    st.dataframe(df)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.chat_input("Ask something about the data...")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        try:
            response = query_agent(st.session_state.agent_df, user_input)
        except Exception as e:
            response = {"error": str(e)}

        st.session_state.chat_history.append({"role": "assistant", "content": response})

    # Show chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            content = msg["content"]
            if isinstance(content, dict):
                if "answer" in content:
                    st.write(content["answer"])
                elif "bar" in content:
                    bar = content["bar"]
                    df_bar = pd.DataFrame(bar["data"], columns=bar["columns"])
                    if len(bar["columns"]) > 1:
                        df_bar.set_index(bar["columns"][0], inplace=True)
                    st.bar_chart(df_bar)
                elif "line" in content:
                    line = content["line"]
                    df_line = pd.DataFrame(line["data"], columns=line["columns"])
                    if len(line["columns"]) > 1:
                        df_line.set_index(line["columns"][0],inplace=True)
                    st.line_chart(df_line)
                elif "table" in content:
                    table = content["table"]
                    df_table = pd.DataFrame(table["data"], columns=table["columns"])
                    st.dataframe(df_table)
                elif "text" in content:
                    st.write(content)
                elif "error" in content:
                    st.error(content["error"])
                else:
                    st.json(content)
            else:
                st.write(content)
else:
    st.info("⬆️ Upload a CSV file to get started.")
