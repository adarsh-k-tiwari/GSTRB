import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import time
import torch
import torchvision.transforms as transforms
from model import GTSRBModel

# Set page title and icon
st.set_page_config(
    page_title="GTSRB",
    page_icon=":traffic_light:",
    layout="wide"
)

@st.cache_resource  # Cache the model to avoid reloading on every interaction
def load_model():
    model = torch.load("model.pth", map_location=torch.device("cpu"))
    model.eval()  # Set the model to evaluation mode
    return model

model = load_model()

# Define the transformation to preprocess the image
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize to match model input size
    transforms.ToTensor(),          # Convert to PyTorch tensor    
])

@st.dialog("Model Result")
def output_model(image):
    with st.spinner("Loading Model..."):
        predicted_class = ''
        if image is not None:
            image = Image.open(image).convert('RGB')
            input_tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                output = model(input_tensor)
                _, predicted_class = torch.max(output.data, 1)

            col1, col2 = st.columns(2)
            with col1:
                st.image(image, width=200)
            with col2:
                st.write(f"The predicted image is:{predicted_class}")
                # with st.empty():
                    # sentiment_mapping = [":material/thumb_down:", ":material/thumb_up:"]
                feedback = st.feedback("thumbs")
                if feedback is not None:
                    st.markdown(f"Thanks for your feedback")


# Add CSS to set background image
def set_background(image_url):
    st.markdown(f"""
    <style>
        [data-testid="stAppViewContainer"]{{
        background-image: url("{image_url}");
        background-position: center;
        background-repeat: no-repeat;
        background-size: cover;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        }}

        [data-testid="stHeader"]{{
        background-color: transparent;}}

       
    </style>
""", unsafe_allow_html=True)


landing_bg_url = "https://images.unsplash.com/photo-1730704430871-971cc92f45f3?q=80&w=1932&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
main_bg_url = "https://images.unsplash.com/photo-1730712659277-d8a9b2663579?q=80&w=1932&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
# Load HTML content from index.html
# with open("index.html", "r") as file:
    # html_content = file.read()

# Render the HTML file content in Streamlit
# components.html(html_content, height=600, scrolling=True)


# Initialize session state for page navigation
if 'page' not in st.session_state:
    st.session_state.page = 'landing'

# Function to switch pages
def go_to_page(page_name):
    st.session_state.click = True
    st.session_state.page = page_name

# Landing Page
if st.session_state.page == 'landing':
    st.markdown(f"""
    <style>
        div.stButton > button:first-child {{
            background-color: #ffffff;
            color: #2C3844;
            font-size: 32px;
            font-weight: bold;
            height: 1em;
            width: 100%;
            border-radius: 35px;
            margin-top: 20em;
            justify-content: center;
        }}

        div.stButton > button:first-child:hover {{
            border-color: #FFC500;
            background-color: #FFC500;
            color: #FFFFFF;
            transition: background-color 0.3s ease;
        }}

    </style> """, unsafe_allow_html=True)
    
    set_background(landing_bg_url)

    col1, col2, col3, col4= st.columns([4,2,2,4], vertical_alignment="center")
    # st.html("<style> .main {overflow: hidden} </style>")
    st.markdown('<div class="button-container">', unsafe_allow_html=True)
    with col2:
        if st.button(":material/lab_profile: Blog Post", key="research", use_container_width=True):
            st.session_state.page = 'research'
            st.rerun()
        
    
    st.markdown('<div class="button-container">', unsafe_allow_html=True)
    with col3:
        if st.button(":material/batch_prediction: Model Demo", key="demo", use_container_width=True):
            st.session_state.page = 'demo'
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    

# Research Paper Page
elif st.session_state.page == 'research':
    set_background(main_bg_url)
    st.markdown(f"""
    <style>
        div.stButton > button:first-child {{
            background-color: #ffffff;
            color: #2C3844;
            font-size: 36px;
            font-weight: bold;
            height: 1em;
            width: 100%;
            border-radius: 35px;
            margin-top: 2em;
            justify-content: center;
        }}

        div.stButton > button:first-child:hover {{
            border-color: #FFC500;
            background-color: #FFC500;
            color: #FFFFFF;
            transition: background-color 0.3s ease;
        }}

        div.stButton > baseButton-secondary{{
            margin-top: 0px;
        }}
    </style> """, unsafe_allow_html=True)
    st.title("Research Paper")
    st.write("This page displays the research paper details, including text and images.")
    
    # Example content for the research paper
    st.subheader("Abstract")
    st.write("This research paper explores the impact of X on Y using various machine learning models...")
    
    st.subheader("Introduction")
    st.write("The goal of this research is to understand...")
    col1, col2, col3, col4 = st.columns([4,2,2,4], vertical_alignment="center")
    
    st.markdown('<div class="button-container">', unsafe_allow_html=True)
    with col2:
        if st.button(":material/home: Main Menu", key="landingR", use_container_width=True):
            st.session_state.page = 'landing'
            st.rerun()
        
    
    st.markdown('<div class="button-container">', unsafe_allow_html=True)
    with col3:
        if st.button(":material/batch_prediction: Model Demo", key="demoR", use_container_width=True):
            st.session_state.page = 'demo'
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    
# Model Demo Page
elif st.session_state.page == 'demo':
    set_background(main_bg_url)
    st.markdown(f"""
    <style>
        div.stButton > button:first-child {{
            background-color: #ffffff;
            color: #2C3844;
            font-size: 36px;
            font-weight: bold;
            height: 1em;
            width: 100%;
            border-radius: 35px;
            margin-top: 2em;
            justify-content: center;
        }}

        div.stButton > button:first-child:not([disabled]):hover {{
            border-color: #FFC500;
            background-color: #FFC500;
            color: #FFFFFF;
            transition: background-color 0.3s ease;
        }}

        div.stButton > button:first-child[disabled] {{
            background-color: #d3d3d3;
            color: #888888;
            cursor: not-allowed;
            border: none;
        }}

        div.stButton > baseButton-primary{{
            margin-top: 20px;
                width: 50%;
        }}

    </style> """, unsafe_allow_html=True)

    st.title("Machine Learning Model Demo")
    st.write("Please upload the image in ppm format only")
    
    # Example inputs for the model
    input_1 = st.file_uploader(label='', accept_multiple_files=False)
    is_button_disabled = input_1 is None
    
    col1, col2, col3, col4 = st.columns([2, 6, 2, 2])

    # Dummy model prediction
    with col1:
        if st.button(":material/neurology: Run Model", key="runModel", use_container_width=False, disabled=is_button_disabled):
                output_model(input_1)
        # Add your model prediction code here
    
    
    
    st.markdown('<div class="button-container">', unsafe_allow_html=True)
    with col3:
        if st.button(":material/home: Main Menu", key="landingD", use_container_width=True, args=("button2",)):
            st.session_state.page = 'landing'
            st.rerun()
        
    
    st.markdown('<div class="button-container">', unsafe_allow_html=True)
    with col4:
        if st.button(":material/lab_profile: Blog Post", key="researchD", use_container_width=True, args=("button1",)):
            st.session_state.page = 'research'
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)