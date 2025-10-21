
# state.py
# -*- coding: utf-8 -*-
import streamlit as st



def ensure_defaults() -> None:
    """Sørg for at alle nøkler finnes i session_state med defaults."""
    from constants import DEFAULTS
    for k, v in DEFAULTS.items():
        st.session_state.setdefault(k, v)

def reset_results() -> None:
    """Tøm tidligere simuleringsresultater (trygt å kalle når inputs endres)."""
    for k in ("freqs", "genotypes", "df_cache"):
        st.session_state.pop(k, None)

def reset_to_defaults() -> None:
    """Tilbakestill alle innstillinger til standardverdier (DEFAULTS)."""
    from constants import DEFAULTS
    st.session_state.clear()
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
