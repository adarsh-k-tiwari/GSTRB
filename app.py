# Importing libraries
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import torch
import torchvision.transforms as transforms
from model import GTSRBModel
import random
import os

# Set page title and icon
st.set_page_config(
    page_title="GTSRB",
    page_icon=":traffic_light:",
    layout="wide"
)

# Load the GTSRB model
@st.cache_resource 
def load_model():
    model = torch.load("model.pth", map_location=torch.device("cpu"))
    model.eval()
    return model

model = load_model()

# Transform the model
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.CenterCrop(28),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
])

# Prediction function
def predict_class(image):
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_class = torch.max(output.data, 1)
    return predicted_class.item()

# Dialog box for classification results
@st.dialog("GTSRB Classifier")
def output_model(image):
    with st.spinner("Loading Model..."):
        predicted_class = ''
        if image is not None:
            image = Image.open(image).convert('RGB')
            input_tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                output = model(input_tensor)
                _, predicted_class = torch.max(output.data, 1)
                predicted_class = str(predicted_class).strip('tensor([])')

            col1, col2 = st.columns(2)
            with col1:
                st.image(image, width=200)
            with col2:
                st.write(f"The predicted image is: {label[predicted_class]}")
                feedback = st.feedback("thumbs")
                if feedback is not None:
                    st.markdown(f"Thanks for your feedback")

# Encoding class labels to text labels
label = {
'0' : 'Speed limit (20km/h)',
'1' : 'Speed limit (30km/h)',
'2' : 'Speed limit (50km/h)',
'3' : 'Speed limit (60km/h)',
'4' : 'Speed limit (70km/h)',
'5' : 'Speed limit (80km/h)',
'6' : 'End of speed limit (80km/h)',
'7' : 'Speed limit (100km/h)',
'8' : 'Speed limit (120km/h)',
'9' : 'No passing',
'10' : 'No passing for vehicles over 3.5 tons',
'11' : 'Priority',
'12' : 'Priority road',
'13' : 'Yield',
'14' : 'Stop',
'15' : 'Road Closed',
'16' : 'Vehicles over 3.5 tons prohibited',
'17' : 'Do not enter',
'18' : 'General Danger',
'19' : 'Left Curve',
'20' : 'Right Curve',
'21' : 'Double Curve',
'22' : 'Bumpy Road',
'23' : 'Slippery Road',
'24' : 'Road narrows',
'25' : 'Roadworks',
'26' : 'Traffic signals ahead',
'27' : 'Pedestrians Crossing',
'28' : 'Watch for children',
'29' : 'Bicycle crossing',
'30' : 'Ice/Snow',
'31' : 'Wild animals crossing',
'32' : 'End of all restrictions',
'33' : 'Turn right ahead',
'34' : 'Turn left ahead',
'35' : 'Ahead only',
'36' : 'Ahead or turn right',
'37' : 'Ahead or turn left',
'38' : 'Pass by on right',
'39' : 'Pass by on left',
'40' : 'Roundabout mandatory',
'41' : 'End of no passing zone',
'42' : 'End of no passing for vehicles over 3.5 tons'}

# Dialog box for multiple classifications
@st.dialog("GTSRB Classifier: Predicting Classes for 5 Random Images", width='large')
def predict_multiple(image):
    columns = st.columns(5)
    for idx, img in enumerate(image):
        col = columns[idx % 5]
        with st.spinner("Predicting..."):
            predicted_class = predict_class(img)
            with col:
                st.image(img, width=100)
                st.write(f"Predicted image is:{label[str(predicted_class)]}")

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
        
        /* Mobile screens */
        @media screen and (max-width: 640px) {{
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
        }}

        /* Tablet screens */
        @media screen and (min-width: 641px) and (max-width: 1024px) {{
            [data-testid="stAppViewContainer"]{{
            background-image: url("{image_url}");
            background-position: center;
            background-repeat: no-repeat;
            background-size: cover;
            top: 0;
            left: 0;
            width: 70%;
            height: 70%;
            }}
        }}

        @media screen and (min-width: 1025px) {{
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
        }}

    </style>
""", unsafe_allow_html=True)

# Load images from a specified directory
def load_images_from_folder():
    images = []
    folder = "image/"
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

# Main Page background image
landing_bg_url = "https://images.unsplash.com/photo-1730704430871-971cc92f45f3?q=80&w=1932&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
main_bg_url = "https://images.unsplash.com/photo-1730712659277-d8a9b2663579?q=80&w=1932&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"

# Initialize session state for page navigation
if 'page' not in st.session_state:
    st.session_state.page = 'landing'

# Function to switch pages
def go_to_page(page_name):
    st.session_state.click = True
    st.session_state.page = page_name

# Landing Page
if st.session_state.page == 'landing':

    # Styling the landing page buttons
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
            margin-top: 20;
            justify-content: center;
        }}

        div.stButton > button:first-child:hover {{
            border-color: #FFC500;
            background-color: #FFC500;
            color: #FFFFFF;
            transition: background-color 0.3s ease;
        }}
                
         /* Mobile screens */
        @media screen and (max-width: 640px) {{
            div.stButton > button {{
                font-size: 0.6rem;
                padding: 0.5rem;
                width: 100%;
                margin-top: 2em;
            }}
        }}
    
        /* Tablet screens */
        @media screen and (min-width: 641px) and (max-width: 1024px) {{
            div.stButton > button {{
                font-size: 1rem;
                padding: 0.75rem;
                width: 100%;
                margin-top: 10em;
            }}
        }}
    
        /* Desktop screens */
        @media screen and (min-width: 1025px) {{
            div.stButton > button {{
                font-size: 1.2rem;
                padding: 1rem;
                width: 100%;
                margin-top: 20rem;
            }}
        }}
                
        @media screen and (min-width: 1920px) {{
            div.stButton > button {{
                font-size: 1.2rem;
                padding: 1rem;
                width: 100%;
                margin-top: 20em;
            }}
        }}

    </style> """, unsafe_allow_html=True)
    set_background(landing_bg_url)

    col1, col2, col3, col4= st.columns([4,2,2,4], vertical_alignment="center")
    # st.html("<style> .main {overflow: hidden} </style>")
    # st.markdown('<div class="button-container">', unsafe_allow_html=True)
    with col2:
        if st.button(":material/lab_profile: Blog Post", key="research", use_container_width=True):
            st.session_state.page = 'research'
            st.rerun()
        
    
    # st.markdown('<div class="button-container">', unsafe_allow_html=True)
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
    st.markdown('<h1 style="text-align: center;">Harnessing the GTSRB Dataset for Smarter Navigation and Safer Roads</h1>', unsafe_allow_html=True)
    st.latex(
        r'''{Adarsh\>Kumar}\qquad{Hwiyoon\>Kim}\qquad {Jonathan\>Gover}\qquad{Neha\>Joshi}\qquad{Neil\>Roy}''')
    
    # Content for the blog post
    st.subheader("Abstract")
    st.write("This report explores the German Traffic Sign Recognition Benchmark (GTSRB) dataset, as a foundation for developing intelligent navigation systems. Using Convolutional Neural Networks (CNNs), we explore the potential of this dataset to enable more accurate and responsive traffic sign detection, contributing to road safety and efficient autonomous navigation.")
    st.divider()

    st.subheader("Introduction")
    st.write("The German Traffic Sign Recognition Benchmark (GTSRB) dataset is a well-established resource for research in traffic sign classification and recognition. This dataset is highly regarded in the computer vision community for developing and benchmarking traffic sign recognition models that contribute to applications in intelligent transportation systems (ITS), advanced driver assistance systems (ADAS), and autonomous driving. GTSRB has become essential for experiments that target real-world road conditions and ensure road safety through accurate traffic sign classification. The proposed approach aims to leverage these insights by designing and training a CNN that can recognize traffic signs accurately, potentially achieving a similar high performance. Through this experiment, we anticipate contributing further to the development of safer navigation systems.")
    st.divider()

    st.subheader("Method")
    st.write("The GTSRB dataset contains 43 distinct classes of traffic signs and a substantial number of labeled images—26,640 for training and 12,630 for testing. To process this dataset, we will use a Convolutional Neural Network (CNN), a model well-suited for image classification tasks due to its ability to capture spatial hierarchies and patterns within images. CNNs are particularly effective for traffic sign recognition, where robust detection and classification of visual cues are critical. Numerous studies have shown CNN-based models to be effective in handling traffic signs across varying light conditions, perspectives, and partial occlusions, which are common in real-world driving scenarios. For instance, CNN models have achieved high accuracy on GTSRB by employing regularization techniques like dropout and batch normalization, and modifications such as attention mechanisms, which enhance the model's focus on crucial image regions(e.g., using SE and VGG-16 based architectures)​<- OUR BASE MODEL->")

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
    
    # Example inputs for the model
    input_1 = st.file_uploader(label='', accept_multiple_files=False, help='Please upload the image in ppm format only')
    is_button_disabled = input_1 is None
    
    col1, col2, col3, col4, col5 = st.columns([2, 2, 6, 2, 2])

    # Dummy model prediction
    with col1:
        if st.button(":material/neurology: Run Model", key="runModel", use_container_width=False, disabled=is_button_disabled):
                output_model(input_1)
                input_1 = None
        # Add your model prediction code here
    
    st.markdown('<div class="button-container">', unsafe_allow_html=True)
    with col2:
        if st.button(":ghost: Surprise Me", key="surpriseM", use_container_width=True):
            all_images = load_images_from_folder()
            random_images = random.sample(all_images, 5)
            predict_multiple(random_images)
    
    st.markdown('<div class="button-container">', unsafe_allow_html=True)
    with col4:
        if st.button(":material/home: Main Menu", key="landingD", use_container_width=True, args=("button2",)):
            st.session_state.page = 'landing'
            st.rerun()
        
    
    st.markdown('<div class="button-container">', unsafe_allow_html=True)
    with col5:
        if st.button(":material/lab_profile: Blog Post", key="researchD", use_container_width=True, args=("button1",)):
            st.session_state.page = 'research'
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
