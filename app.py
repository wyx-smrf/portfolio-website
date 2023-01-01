import streamlit as st
from streamlit_option_menu import option_menu

selected = option_menu(
    menu_title = None,
    options = ['Home', 'Projects', 'Contact'],
    icons = ['house', 'book', 'envelope'],
    menu_icon = 'cast',
    default_index=0,
    orientation='horizontal',
    styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "green"},
    }
)

if selected == 'Home':
    st.title(f'You have selected {selected}')
if selected == 'Projects':
    st.title(f'You have selected {selected}')
if selected == 'Contact':
    st.title(f'You have selected {selected}')