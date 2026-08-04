# app.py - Main Streamlit Application

import os
import streamlit as st
from dotenv import load_dotenv
from agents import TutorAgentSystem
import time

load_dotenv(dotenv_path=".env")
load_dotenv(dotenv_path="new.env")

st.set_page_config(
    page_title="AI Tutor - ඉගෙනුම් සහායක",
    page_icon="📚",
    layout="wide"
)

st.title("🎓 AI Tutor - ඔබේ පෞද්ගලික ඉගෙනුම් සහායක")
st.markdown("""
    මෙම AI Tutor ඔබට ඕනෑම විෂයක් පිළිබඳ ප්‍රශ්න ඇසීමට,
    සරල පැහැදිලි කිරීම් ලබා ගැනීමට සහ ඔබේ දැනුම පරීක්ෂා කර ගැනීමට උදව් කරයි.
""")

missing_keys = [key for key in ("GROQ_API_KEY", "OPENROUTER_API_KEY") if not os.getenv(key)]
if missing_keys:
    st.warning(
        "API key configuration is incomplete. Create a `.env` file in the project root with:\n"
        "GROQ_API_KEY=your_groq_key\n"
        "OPENROUTER_API_KEY=your_openrouter_key"
    )
    st.stop()

@st.cache_resource
def load_agent_system():
    return TutorAgentSystem()

agent_system = load_agent_system()

with st.sidebar:
    st.header("ℹ️ තොරතුරු")
    st.markdown("""
    **AI Agents කණ්ඩායම:**
    1. 🏷️ **Subject Classifier** - විෂය හඳුනා ගැනීම
    2. 📝 **Explanation Agent** - පැහැදිලි කිරීම්
    3. ❓ **Quiz Agent** - ප්‍රශ්න සහ පරාවර්තන
    """)

    st.divider()

    st.subheader("🤖 භාවිතා කරන Models")
    st.info("""
    - **Classifier**: Groq (Llama 3.1 8B) - වේගවත්
    - **Explanation**: OpenRouter (Claude/GPT-4) - ගැඹුරු
    """)

    if st.button("🗑️ සංවාදය මකන්න"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("ඔබේ ප්‍රශ්නය මෙතන ටයිප් කරන්න..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🤔 Agents එකට වැඩ කරනවා..."):
            try:
                response = agent_system.process_question(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"දෝෂයක් සිදුවිය: {str(e)}")
                st.info("කරුණාකර නැවත උත්සාහ කරන්න.")
