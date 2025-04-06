import streamlit as st

# Add a custom background image
def add_bg_from_url():
    st.markdown(
        f
        <style>
        stApp {{
            background-image: url("/Users/tejashwinigampa/Downloads/download.png");
            background-size: cover;
        }}
        </style>
        ,
        
        unsafe_allow_html=True
    )

add_bg_from_url()

st.title("🌙 Smart Sleep Tracking & Optimization System")
st.write("### Get personalized insights for better sleep quality!")
