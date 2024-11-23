# Importing libraries
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import torch
import torchvision.transforms as transforms
from model import GTSRBModel
import random
import os
import pandas as pd

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

    st.markdown('<h1 style="text-align: center"><br><br><span style="background-color:#FFC500; color:#FFFFFF; font-family: Poppins, sans-serif; font-weight: 600; font-style: normal;">  A I  -  P O W E R E D </span></h1>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; font-size: 8em; line-height:95%;font-family: Roboto, sans-serif; font-weight: 900; font-style: normal;"><strong>Traffic Sign</strong><br><div style="color: #FFC500"><strong>Recognition</strong></div></div>', unsafe_allow_html=True)
    st.markdown('<h4 style="text-align: center; color: #A9A9A9"><br>Harnessing the GTSRB Dataset for Smarter Navigation and Safer Roads</h4>', unsafe_allow_html=True)
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
            # margin-top: 20;
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
                # margin-top: 2em;
            }}
        }}
    
        /* Tablet screens */
        @media screen and (min-width: 641px) and (max-width: 1024px) {{
            div.stButton > button {{
                font-size: 1rem;
                padding: 0.75rem;
                width: 100%;
                # margin-top: 10em;
            }}
        }}
    
        /* Desktop screens */
        @media screen and (min-width: 1025px) {{
            div.stButton > button {{
                font-size: 1.2rem;
                padding: 1rem;
                width: 100%;
                # margin-top: 20rem;
            }}
        }}
                
        @media screen and (min-width: 1920px) {{
            div.stButton > button {{
                font-size: 1.2rem;
                padding: 1rem;
                width: 100%;
                # margin-top: 20em;
            }}
        }}

    </style> """, unsafe_allow_html=True)
    set_background(main_bg_url)

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
    st.write("As self-driving vehicle technology continues to evolve, recognizing traffic signs is becoming increasingly essential. Utilizing the German Traffic Sign Recognition Benchmark (GTSRB) dataset, up to 43 types of traffic signs can be classified. The topic of this study is to combine various pre-processing techniques to refine the model\'s accuracy, focusing on applying to various types convolutional neural networks (CNNs). While 2 state-of-the-art CNN architectures (ResNet and MobileNet) and additional 3 models were tested (Baseline CNN, Self Attention-CNN and N-Adam CNN), the results highlight that CNNs with Spatial SelfAttention mechanisms reached the highest validation accuracy. The proposed technique can be applied to other downstream tasks for traffic sign detection.")
    st.divider()

    st.subheader("Introduction")
    st.write("Accurately recognizing traffic signs is a critical component of intelligent transportation systems can be effective in various systems, ranging from advanced driver assistance systems (ADAS) to fully autonomous vehicles. Traffic sign recognition ensures road safety as well as enhancing driving efficiency by providing real-time information to drivers and vehicle systems. However, recognizing accurate traffic signs can face difficulties due to different circumstances such as lighting, weather conditions, and the diverse designs of traffic signs across regions, as shown in figure 1:")
    
    st.image('signs.png', caption='Fig. 1: Sample Images from Training Set (top) and Testing Set (bottom).', width=600)
    st.divider()

    st.subheader("Methodology")
    st.write("There are three stages in the proposed pipeline. First, the images are subjected to one of three different sets of image transformations, each of which produces their splits for  testing, validation, and training. Each transformed image set is then used to train and evaluate the models in the model set. Finally, the performance of each model is compared across all transformed feature sets")
    st.image('Model_Pipeline.png', caption='Fig. 2: Model Pipeline', width=600)
    st.markdown("""**1. Pre-Processing Techniques:**
    - Grayscale conversion, CLAHE (Contrast Limited Adaptive Histogram Equalization), cropping, sharpening, rotation, affine transformation, and color jitter were employed to enhance image features.
- Three distinct pre-processing pipelines were designed:
    - Set 1: Focused on contrast enhancement (Grayscale → Cropping → CLAHE → Sharpening).
    - Set 2: Included data augmentation (Cropping → Rotation → Affine → Color Jitter).
    - Set 3: Baseline with only cropping and normalization.""")
    st.image('Feauter_processing.png', width=500, caption='Fig. 3: Images after different transformations')
    st.markdown("""**2. Deep Learning Architectures Evaluated:**
- Baseline CNN: A standard CNN with max-pooling and dropout layers.
- CNN + NAdam: Integrated Nesterov Accelerated Gradient (NAdam) optimizer for faster convergence.
- CNN with Attention: Incorporated spatial self-attention modules for enhanced feature extraction.
- ResNet-18 and MobileNetv2: Pre-trained state-of-the-art models fine-tuned on the GTSRB dataset.""")
    st.image('model.png', width=500, caption='Fig. 4: Model Architecture for CNN with Self-Attention')
    st.markdown("""**3. Experimental Findings:**
- CNN with self-attention mechanisms achieved the highest accuracy (97.28%) across all pre-processing sets.
- Feature Set 1 (focused on contrast enhancements) consistently outperformed the others, confirming the effectiveness of pre-processing in improving model accuracy.
- Attention CNN excelled in precision, recall, and F1 score metrics, demonstrating robust and reliable performance.
- Visualization of attention maps showed the model effectively focused on critical regions of traffic signs while ignoring irrelevant background details.""")
    st.image('model_loss.jpg', width=800,  caption='Fig. 5: Loss vs Epochs and Accuracy vs Epochs across all models built')
    st.markdown("""**4. Training Setup:**
- The models were trained on 26,640 images (split into 80% training and 20% validation) and tested on 12,630 images.
- Hyperparameters such as learning rate, batch size, and epochs were optimized using GridSearch.
- All five models were trained and validated on three pre-processing sets, which comprised 26,640 images (80% training and 20% validation).
- Results showed that CNN with Self-Attention trained on Set 1 consistently achieved the highest performance across metrics, leveraging the benefits of contrast-enhancing transformations like CLAHE and sharpening.""")
    st.write(pd.DataFrame({
        'Pre-processing/Models' : ['Set 1', 'Set 2', 'Set 3'],
'Baseline CNN' : ['92.00%', '72.92%', '83.89%'],
'CNN + Attention Module' : ['97.28%', '96.56%', '94.19%'],
'ResNet-18' : ['94.38%', '91.88%', '94.16%'],
'MobileNetv2' : ['96.04%', '94.49%', '95.08%'],
'CNN + N Adam' : ['93.30%', '93.40%', '94.00%'],
    }, index=None))

    st.markdown("""**Conclusion and Future Work:**

- This study demonstrates combining pre-processing techniques with deep learning models for traffic sign recognition. Using the GTSRB dataset from PyTorch, multiple preprocessing pipelines and model architectures were tested, with the conclusion that CNNs with Spatial Self-Attention mechanisms reached the highest test accuracy of 97.28%. The attention map visualization highlights that the model focuses on key features while also giving less importance to the dark background areas.
- This approach can be applied internationally to recognize traffic signs from any country, providing accurate and reliable detection for autonomous vehicles, which can alert drivers to traffic signs and reduce the risk of accidents. Additionally, the model can assess traffic sign conditions, identifying those that are damaged, poorly maintained, or missing.
                
This projects contributes to advancing intelligent transportation systems, showcasing high accuracy and practical applicability in traffic sign recognition tasks.""")
                
    st.markdown('<div class="button-container"><strong><center>You can test the model by clicking on the \'Model Demo\' button.</center></strong> </div>', unsafe_allow_html=True)
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
    st.markdown("""To test the model, you have two options:

- **Upload an Image**: Select an image of your choice and click the 'Predict' button to see the model's prediction.
- **Surprise Me**: Let the model randomly select 5 images from the test dataset and display their predictions.""")
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
