"""Small owner-password gate for the hosted Streamlit UI; not tenant authentication."""
import hmac
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


def require_access():
    expected = os.getenv("APP_PASSWORD", "")
    if not expected:
        if os.getenv("RENDER"):
            st.error("Access is not configured yet. Please contact the app owner.")
            st.stop()
        return
    if st.session_state.get("access_granted"):
        return
    st.title("Sign in")
    with st.form("app_access"):
        password = st.text_input("App password", type="password")
        submitted = st.form_submit_button("Continue")
    if submitted:
        if hmac.compare_digest(password.encode(), expected.encode()):
            st.session_state.access_granted = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()
