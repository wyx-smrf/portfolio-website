import streamlit as st
from streamlit_option_menu import option_menu


def st_page_configurations():
    page_config = st.set_page_config(
        page_title = 'Streamlit / OpenAI', # Title of the webpage
        page_icon = "👨‍💻",
        layout = 'wide',
        menu_items = {
            'Get Help': None,
            'Report a bug': None,
            'About': "Author: [Roi Jacob C. Olfindo](https://github.com/wyx-smrf)"
        })

    return page_config

def title_page():
    st.title('Roi Jacob C. Olfindo')
    st.write('Welcome to my simple website')
    st.markdown('---')
    pass


pc = st_page_configurations()
t_page = title_page()


selected = option_menu(
    menu_title = None,
    options = ['Home', 'Projects', 'Contact'],
    icons = ['house', 'book', 'envelope'], # Change via bootstrap icons
    menu_icon = 'cast',
    default_index=0,
    orientation='horizontal')

def projects():
    st.write('Steel Industry Data (Soon) :(')
    pass



if selected == 'Home':
    st.title(f'You have selected {selected}')

if selected == 'Projects':
    st.title('Personal Projects')
    my_projects = projects()

if selected == 'Contact':
    st.title(f'You have selected {selected}')



