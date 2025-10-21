# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 19:52:34 2025

@author: johvik
"""

# app.py
# -*- coding: utf-8 -*-
import streamlit as st

pg = st.navigation([
    st.Page("pages/setup.py", title="Oppsett & mekanismer"),
    st.Page("pages/results.py", title="Resultater"),
])

pg.run()
