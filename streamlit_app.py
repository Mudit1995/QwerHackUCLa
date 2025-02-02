import streamlit as st
import requests

# API endpoint
API_URL = "http://localhost:5000/ask"

st.title("ChAI - A friend indeed")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_input = st.text_input("Ask a question:", "", key="input")
if st.button("Send") and user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Send request to API
    response = requests.post(API_URL, json={"question": user_input})
    bot_reply = response.json().get("answer", "Error: No response")
    
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    st.rerun()
