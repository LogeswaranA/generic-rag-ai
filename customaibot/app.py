# app.py
import streamlit as st
from customaibot.knowledgebase import app

st.set_page_config(page_title="Tathmeer Research Assistant", page_icon="🤖")

st.title("🤖 Tathmeer AI Assistant")
st.write("Ask questions about your documents or general knowledge")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What would you like to know?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        response = app.invoke({"question": prompt})
        answer = response['answer']
        st.markdown(answer)
    
    st.session_state.messages.append({"role": "assistant", "content": answer})