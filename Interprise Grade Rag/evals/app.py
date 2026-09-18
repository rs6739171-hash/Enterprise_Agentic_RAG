"""Optional standalone dashboard; the main deployed UI includes this page."""
import streamlit as st
from deployment_access import require_access
from evals.dashboard import render_dashboard

st.set_page_config(page_title="RAG evaluation", layout="wide")
require_access()
render_dashboard()
