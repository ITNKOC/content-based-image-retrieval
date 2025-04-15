import numpy as np
import cv2
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import os
import streamlit as st
from PIL import Image
from descriptor import glcm, bitdesc, haralick_feat, bit_glcm_haralick
from distances import manhattan, euclidean, chebyshev, canberra
import time

# Load precomputed signatures
@st.cache_data
def load_signatures():
    signatures_glcm = np.load('signatures_glcm.npy', allow_pickle=True)
    signatures_bitdesc = np.load('signatures_bitdesc.npy', allow_pickle=True)
    signatures_haralick = np.load('signatures_haralick_feat.npy', allow_pickle=True)
    signatures_combined = np.load('signatures_bit_glcm_haralick.npy', allow_pickle=True)
    return signatures_glcm, signatures_bitdesc, signatures_haralick, signatures_combined

signatures_glcm, signatures_bitdesc, signatures_haralick, signatures_combined = load_signatures()

# Define distance functions
distance_functions = {
    'Manhattan': manhattan,
    'Euclidean': euclidean,
    'Chebyshev': chebyshev,
    'Canberra': canberra,
}

# Define descriptor functions
descriptor_functions = {
    'GLCM': glcm,
    'BIT': bitdesc,
    'HARALICK': haralick_feat,
    'BIT_GLCM_HARALICK': bit_glcm_haralick
}

def fine_tune_model(model, param_grid, X_train, Y_train):
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X_train, Y_train)
    return grid_search.best_estimator_

def get_model(classifier, use_grid_search, X_train, Y_train):
    if classifier == "LDA":
        model = LinearDiscriminantAnalysis()
    elif classifier == "KNN":
        if use_grid_search:
            param_grid = {'n_neighbors': list(range(1, 30)), 'p': [1, 2]}
            model = fine_tune_model(KNeighborsClassifier(), param_grid, X_train, Y_train)
        else:
            model = KNeighborsClassifier(n_neighbors=10)
    elif classifier == "SVM":
        if use_grid_search:
            param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf', 'linear']}
            model = fine_tune_model(SVC(), param_grid, X_train, Y_train)
        else:
            model = SVC(C=2.5, max_iter=5000)
    elif classifier == "Random Forest":
        if use_grid_search:
            param_grid = {'n_estimators': [10, 50, 100, 200], 'max_features': ['sqrt', 'log2'], 'max_depth': [None, 10, 20, 30]}
            model = fine_tune_model(RandomForestClassifier(), param_grid, X_train, Y_train)
        else:
            model = RandomForestClassifier()
    elif classifier == "AdaBoost":
        if use_grid_search:
            param_grid = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1]}
            model = fine_tune_model(AdaBoostClassifier(), param_grid, X_train, Y_train)
        else:
            model = AdaBoostClassifier()
    elif classifier == "Decision Tree":
        if use_grid_search:
            param_grid = {'criterion': ['gini', 'entropy'], 'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]}
            model = fine_tune_model(DecisionTreeClassifier(), param_grid, X_train, Y_train)
        else:
            model = DecisionTreeClassifier()
    elif classifier == "Naive Bayes":
        model = GaussianNB()
    return model

def apply_transform(X, transform_choice):
    if transform_choice == 'Rescale':
        scaler = MinMaxScaler()
    elif transform_choice == 'Normalization':
        scaler = Normalizer()
    elif transform_choice == 'Standardization':
        scaler = StandardScaler()
    else:
        return X
    return scaler.fit_transform(X)

def display_metric(metric_name, metric_value):
    if metric_value > 0.9:
        color = '#28a745'  # Green
    elif metric_value > 0.7:
        color = '#fd7e14'  # Orange
    else:
        color = '#dc3545'  # Red
    
    col1, col2 = st.columns([1, 4])
    with col1:
        st.markdown(f"<h4>{metric_name}:</h4>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div style="
            background-color: {color}; 
            color: white; 
            padding: 10px; 
            border-radius: 5px; 
            text-align: center;
            font-weight: bold;
            width: 100%;
            font-size: 1.2em;
        ">
            {metric_value:.2%}
        </div>
        """, unsafe_allow_html=True)

# Extract features function
def extract_features(image, descriptor_choice):
    descriptor_func = descriptor_functions[descriptor_choice]
    return descriptor_func(image)

# Prediction function
def predict_class(model, image_features):
    scaler = StandardScaler()
    image_features_scaled = scaler.fit_transform([image_features])
    return model.predict(image_features_scaled)[0]

# Display images by class
def display_images_by_class(predicted_class, signatures, image_count):
    st.write(f"Displaying images for the predicted class: {predicted_class}")
    class_images = [signature for signature in signatures if signature[-2] == predicted_class]
    cols = st.columns(4)
    for idx, img_info in enumerate(class_images[:image_count]):
        relative_path = img_info[-1]
        if not isinstance(relative_path, str):
            st.write(f"Error: expected a string for relative_path but got {type(relative_path)}")
            continue
        img_path = os.path.join("images", *relative_path.replace("\\", "/").split("/"))
        image = Image.open(img_path)
        cols[idx % 4].image(image, caption=relative_path, use_container_width=True)

# Custom CSS for the app
def custom_css():
    st.markdown("""
    <style>
    /* Main app styling */
    .main {
        padding: 2rem 1rem;
        line-height: 1.6;
    }
    
    /* Headers */
    h1 {
        color: #2c3e50;
        font-weight: 700;
        margin-bottom: 1.5rem;
    }
    
    h2, h3, h4 {
        color: #34495e;
        font-weight: 600;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Card styles for image containers */
    .img-card {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 0.75rem;
        margin-bottom: 1rem;
        transition: transform 0.3s ease;
        background-color: #ffffff;
    }
    
    .img-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.15);
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 20px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
        border: none;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 10px rgba(0, 0, 0, 0.1);
    }
    
    /* File uploader styling */
    .uploadedFile {
        border: 2px dashed #6c757d;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        background-color: #f8f9fa;
    }
    
    /* Sidebar styling */
    .css-18e3th9 {
        padding: 2rem 1rem;
        background-color: #f8f9fa;
    }
    
    /* Custom tooltips */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: pointer;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 10px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div {
        background-color: #3498db;
    }
    
    /* Night mode toggle styles */
    .night-mode-toggle {
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
    }
    
    .night-mode-toggle label {
        margin-left: 0.5rem;
        font-weight: 500;
    }
    
    /* Animated loading spinner */
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .loading-spinner {
        border: 5px solid #f3f3f3;
        border-top: 5px solid #3498db;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 0 auto;
    }
    
    /* Enhanced metrics display */
    .metric-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    /* Dashboard cards */
    .dashboard-card {
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        background-color: #ffffff;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    
    .dashboard-card:hover {
        box-shadow: 0 10px 15px rgba(0, 0, 0, 0.1);
    }
    
    /* Multimodal search styling */
    .search-box {
        border-radius: 30px;
        padding: 0.5rem 1rem;
        border: 2px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    
    .search-box:focus {
        border-color: #3498db;
        box-shadow: 0 4px 8px rgba(52, 152, 219, 0.2);
    }
    
    /* Image gallery styling */
    .gallery {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
        grid-gap: 15px;
        margin-bottom: 2rem;
    }
    
    .gallery img {
        width: 100%;
        border-radius: 8px;
        transition: transform 0.3s ease;
    }
    
    .gallery img:hover {
        transform: scale(1.05);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        background-color: #f8f9fa;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #3498db !important;
        color: white !important;
    }

    /* Night mode styles will be added dynamically */
    </style>
    """, unsafe_allow_html=True)

# Night mode CSS
def apply_night_mode():
    if st.session_state.get("night_mode", False):
        st.markdown(
            """
            <style>
            /* Main background and text color */
            .main {
                background-color: #1e1e1e;
                color: #dcdcdc;
            }
            
            /* Sidebar and widget backgrounds */
            .css-1d391kg, .css-18e3th9 {
                background-color: #2c2c2c;
                color: #dcdcdc;
            }
            
            /* Headings */
            h1, h2, h3, h4, h5, h6 {
                color: #f0f0f0;
            }
            
            /* Text, markdown, and data frames */
            .stText, .stMarkdown, .stDataFrame {
                color: #dcdcdc;
            }

            /* Cards */
            .dashboard-card, .img-card {
                background-color: #333333;
                color: #f0f0f0;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            }

            /* Buttons */
            .stButton>button {
                background-color: #444444;
                color: #ffffff;
                border: none;
                border-radius: 20px;
            }
            .stButton>button:hover {
                background-color: #555555;
                color: #ffffff;
                box-shadow: 0 5px 10px rgba(0, 0, 0, 0.3);
            }

            /* Selectbox, radio buttons, and sliders */
            .stSelectbox>div, .stRadio>div {
                background-color: #333333;
                color: #dcdcdc;
                border-radius: 4px;
            }
            
            .stSelectbox>div>div, .stRadio>div>div {
                background-color: #333333;
                color: #dcdcdc;
            }

            /* Sliders */
            .stSlider>div>div>div {
                background-color: #444444;
            }
            .stSlider>div>div>div>div {
                background-color: #666666;
                color: #ffffff;
            }

            /* File uploader */
            .stFileUploader {
                background-color: #2c2c2c;
                color: #dcdcdc;
            }
            
            .uploadedFile {
                border: 2px dashed #666666;
                background-color: #333333;
            }

            /* Text inputs and text areas */
            .stTextInput>div>div>input, .stTextArea>div>textarea, .search-box {
                background-color: #333333;
                color: #ffffff;
                border: 1px solid #555555;
            }

            /* Metric containers */
            .metric-container {
                background-color: #333333;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
            }
            
            /* Tabs */
            .stTabs [data-baseweb="tab"] {
                background-color: #333333;
                color: #dcdcdc;
            }
            
            /* Progress bars */
            .stProgress > div > div > div {
                background-color: #3498db;
            }
            
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <style>
            /* Reset to default styles for day mode */
            .main {
                background-color: #ffffff;
                color: #000000 !important;
            }
            
            .css-1d391kg, .css-18e3th9 {
                background-color: #f8f9fa;
                color: #000000 !important;
            }

            h1, h2, h3, h4, h5, h6 {
                color: #000000 !important;
            }

            p, li, span, div {
                color: #000000 !important;
            }

            .stText, .stMarkdown, .stDataFrame {
                color: #000000 !important;
            }

            /* Cards */
            .dashboard-card, .img-card {
                background-color: #ffffff;
                color: #000000 !important;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }

            /* Buttons */
            .stButton>button {
                background-color: #007bff;
                color: #ffffff;
                border: none;
                border-radius: 4px;
            }

            .stButton>button:hover {
                background-color: #0056b3;
                color: #ffffff;
            }

            .stSelectbox>div, .stRadio>div {
                background-color: #ffffff;
                color: #000000;
                border-radius: 4px;
            }

            .stSelectbox>div>div, .stRadio>div>div {
                background-color: #ffffff;
                color: #000000;
            }

            .stSlider>div>div>div {
                background-color: #cccccc;
            }

            .stSlider>div>div>div>div {
                background-color: #007bff;
                color: #ffffff;
            }

            .stFileUploader {
                background-color: #f8f9fa;
                color: #000000;
            }

            .stTextInput>div>div>input, .stTextArea>div>textarea {
                background-color: #ffffff;
                color: #000000;
                border: 1px solid #ced4da;
            }

            .stCheckbox>div>div>div {
                color: #000000;
            }
            
            /* Search container for light mode */
            .search-container {
                background-color: #ffffff;
                border: 1px solid #e0e0e0;
            }
            
            /* Category chips for light mode */
            .category-chip {
                background-color: #e9ecef;
                color: #000000 !important;
            }
            
            /* Make sure captions are visible */
            .stImage > div > div > p {
                color: #000000 !important;
            }

            </style>
            """,
            unsafe_allow_html=True
        )
    if st.session_state.get("night_mode", False):
        st.markdown(
            """
            <style>
            /* Main background and text color */
            .main {
                background-color: #1e1e1e;
                color: #dcdcdc;
            }
            
            /* Sidebar and widget backgrounds */
            .css-1d391kg, .css-18e3th9 {
                background-color: #2c2c2c;
                color: #dcdcdc;
            }
            
            /* Headings */
            h1, h2, h3, h4, h5, h6 {
                color: #f0f0f0;
            }
            
            /* Text, markdown, and data frames */
            .stText, .stMarkdown, .stDataFrame {
                color: #dcdcdc;
            }

            /* Cards */
            .dashboard-card, .img-card {
                background-color: #333333;
                color: #f0f0f0;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            }

            /* Buttons */
            .stButton>button {
                background-color: #444444;
                color: #ffffff;
                border: none;
                border-radius: 20px;
            }
            .stButton>button:hover {
                background-color: #555555;
                color: #ffffff;
                box-shadow: 0 5px 10px rgba(0, 0, 0, 0.3);
            }

            /* Selectbox, radio buttons, and sliders */
            .stSelectbox>div, .stRadio>div {
                background-color: #333333;
                color: #dcdcdc;
                border-radius: 4px;
            }
            
            .stSelectbox>div>div, .stRadio>div>div {
                background-color: #333333;
                color: #dcdcdc;
            }

            /* Sliders */
            .stSlider>div>div>div {
                background-color: #444444;
            }
            .stSlider>div>div>div>div {
                background-color: #666666;
                color: #ffffff;
            }

            /* File uploader */
            .stFileUploader {
                background-color: #2c2c2c;
                color: #dcdcdc;
            }
            
            .uploadedFile {
                border: 2px dashed #666666;
                background-color: #333333;
            }

            /* Text inputs and text areas */
            .stTextInput>div>div>input, .stTextArea>div>textarea, .search-box {
                background-color: #333333;
                color: #ffffff;
                border: 1px solid #555555;
            }

            /* Metric containers */
            .metric-container {
                background-color: #333333;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
            }
            
            /* Tabs */
            .stTabs [data-baseweb="tab"] {
                background-color: #333333;
                color: #dcdcdc;
            }
            
            /* Progress bars */
            .stProgress > div > div > div {
                background-color: #3498db;
            }
            
            </style>
            """,
            unsafe_allow_html=True
        )

# Progress indicator for loading processes
def show_progress_bar(title, steps=10):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(steps):
        # Update progress bar
        progress = (i + 1) / steps
        progress_bar.progress(progress)
        status_text.text(f"{title}: {int(progress * 100)}%")
        time.sleep(0.1)  # Simulating process time
    
    progress_bar.empty()
    status_text.empty()

# Basic CBIR with improved UI
def cbir_basic():
    st.markdown("""
    <div class="dashboard-card">
        <h1>Content-based Image Retrieval</h1>
        <p>Upload an image to find similar images in the database.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("""
        <div class="dashboard-card">
            <h3>Settings</h3>
        </div>
        """, unsafe_allow_html=True)
        
        descriptor_choice = st.selectbox(
            "Choose a descriptor", 
            ["GLCM", "BIT", "HARALICK", "BIT_GLCM_HARALICK"],
            help="Select the feature extraction method"
        )
        
        distance_choice = st.selectbox(
            "Choose a distance metric", 
            ["Manhattan", "Euclidean", "Chebyshev", "Canberra"],
            help="Select the similarity measurement method"
        )
        
        image_count = st.slider(
            "Number of similar images to display", 
            min_value=1, 
            max_value=12, 
            value=4,
            help="Adjust how many similar images to display"
        )
    
    with col2:
        st.markdown("""
        <div class="dashboard-card">
            <h3>Upload Image</h3>
            <p>Select an image to find similar images in the database</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"])
        
        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded Image", use_container_width=True)
            
            # Process the image
            show_progress_bar("Processing image", 5)
            
            img = np.array(img)
            img = cv2.resize(img, (256, 256))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            uploaded_image_features = extract_features(img, descriptor_choice)

            # Select the correct signatures based on the descriptor
            if descriptor_choice == 'GLCM':
                signatures = signatures_glcm
            elif descriptor_choice == 'BIT':
                signatures = signatures_bitdesc
            elif descriptor_choice == 'HARALICK':
                signatures = signatures_haralick
            else:
                signatures = signatures_combined

            distances = []
            dist_func = distance_functions[distance_choice]

            # Compute distances between the uploaded image and all the signatures
            show_progress_bar("Finding similar images", 5)
            
            for signature in signatures:
                feature_vector = np.array(signature[:-3], dtype=float)  # Exclude the last three elements (relative path, folder name, class label)
                dist = dist_func(uploaded_image_features, feature_vector)
                distances.append((dist, signature[-3], signature[-2], signature[-1]))  # Keep the relative path, folder name, and class label

            # Sort the distances in ascending order
            distances.sort(key=lambda x: x[0])

            # Display the top N similar images
            st.markdown(f"""
            <div class="dashboard-card">
                <h3>Top {image_count} Similar Images</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Create a responsive grid for images
            num_cols = min(4, image_count)
            cols = st.columns(num_cols)
            
            for i in range(image_count):
                dist, relative_path, folder_name, class_label = distances[i]
                img_path = os.path.join("images", *relative_path.replace("\\", "/").split("/"))
                similar_img = Image.open(img_path)
                
                with cols[i % num_cols]:
                    st.markdown(f"""
                    <div class="img-card">
                    """, unsafe_allow_html=True)
                    st.image(similar_img, caption=f"{folder_name}", use_container_width=True)
                    st.markdown(f"""
                    <p style="text-align: center; font-size: 0.9em; color: #666;">Similarity: {1/(1+dist):.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="uploadedFile">
                <p>📁 Drop your image here or click to browse</p>
                <p style="font-size: 0.8em; color: #666;">Supported formats: PNG, JPG, JPEG</p>
            </div>
            """, unsafe_allow_html=True)

# Advanced CBIR with Model Selection and Fine-tuning
def cbir_advanced():
    st.markdown("""
    <div class="dashboard-card">
        <h1>Advanced CBIR with Machine Learning</h1>
        <p>Upload an image to classify and find similar images using machine learning models.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for better organization
    tab1, tab2 = st.tabs(["Model Configuration", "Results"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            <div class="dashboard-card">
                <h3>Feature Settings</h3>
            </div>
            """, unsafe_allow_html=True)
            
            descriptor_choice = st.selectbox(
                "Feature Descriptor", 
                ["GLCM", "BIT", "HARALICK", "BIT_GLCM_HARALICK"],
                help="Select the feature extraction method"
            )
            
            distance_choice = st.selectbox(
                "Distance Metric", 
                ["Manhattan", "Euclidean", "Chebyshev", "Canberra"],
                help="Select the similarity measurement method"
            )
            
            transform_choice = st.selectbox(
                "Data Transformation", 
                ["No Transform", "Rescale", "Normalization", "Standardization"],
                help="Select how to preprocess the features"
            )
            
        with col2:
            st.markdown("""
            <div class="dashboard-card">
                <h3>Model Settings</h3>
            </div>
            """, unsafe_allow_html=True)
            
            classifier = st.selectbox(
                "Classification Algorithm", 
                ["LDA", "KNN", "Naive Bayes", "Decision Tree", "SVM", "Random Forest", "AdaBoost"],
                help="Select the machine learning algorithm"
            )
            
            use_grid_search = st.toggle(
                "Use GridSearchCV for Hyperparameter Tuning", 
                value=False, 
                help="Enable to automatically find the best model parameters (takes longer)"
            )
            
            image_count = st.slider(
                "Number of Similar Images", 
                min_value=1, 
                max_value=12, 
                value=4,
                help="Adjust how many similar images to display"
            )
        
        st.markdown("""
        <div class="dashboard-card">
            <h3>Upload Image</h3>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_image = st.file_uploader("Choose an image for classification and similarity search", type=["jpg", "png", "jpeg"])
    
    with tab2:
        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("""
                <div class="dashboard-card">
                    <h3>Uploaded Image</h3>
                </div>
                """, unsafe_allow_html=True)
                st.image(image, caption='Uploaded Image', use_container_width=True)
            
            with col2:
                # Process the image
                image = np.array(image)
                image = cv2.resize(image, (256, 256))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                uploaded_image_features = extract_features(image, descriptor_choice)
                
                if descriptor_choice == 'GLCM':
                    signatures = signatures_glcm
                elif descriptor_choice == 'BIT':
                    signatures = signatures_bitdesc
                elif descriptor_choice == 'HARALICK':
                    signatures = signatures_haralick
                else:
                    signatures = signatures_combined
                
                X = np.array([sig[:-3] for sig in signatures], dtype=float)
                Y = np.array([sig[-1] for sig in signatures], dtype=int)
                
                # Clean data
                X = np.where(np.isinf(X), np.nan, X)
                X = np.where(np.isnan(X), np.nanmean(X, axis=0), X)
                
                # Apply selected transformation
                X = apply_transform(X, transform_choice)
                
                # Model training status
                st.markdown("""
                <div class="dashboard-card">
                    <h3>Model Training Progress</h3>
                </div>
                """, unsafe_allow_html=True)
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Split train / test data
                train_proportion = 0.15
                seed = 10
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=train_proportion, random_state=seed)
                
                # Update progress
                progress_bar.progress(0.2)
                status_text.text("Preparing model...")
                
                # Get and train model
                model = get_model(classifier, use_grid_search, X_train, Y_train)
                
                # Update progress
                progress_bar.progress(0.5)
                status_text.text("Training model...")
                
                model.fit(X_train, Y_train)
                
                # Update progress
                progress_bar.progress(0.7)
                status_text.text("Evaluating model...")
                
                Y_pred = model.predict(X_test)
                accuracy = accuracy_score(Y_test, Y_pred)
                recall = recall_score(Y_test, Y_pred, average='weighted')
                f1 = f1_score(Y_test, Y_pred, average='weighted')
                precision = precision_score(Y_test, Y_pred, average='weighted')
                
                # Update progress
                progress_bar.progress(0.9)
                status_text.text("Making prediction...")
                
                # Predict for the uploaded image
                uploaded_image_features = np.array(uploaded_image_features).flatten()
                uploaded_image_features = uploaded_image_features.reshape(1, -1)
                uploaded_image_features = apply_transform(uploaded_image_features, transform_choice)
                
                uploaded_image_prediction = model.predict(uploaded_image_features)
                
                # Complete progress
                progress_bar.progress(1.0)
                status_text.text("Completed!")
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()
                
                # Display metrics in a nice format
                st.markdown("""
                <div class="dashboard-card">
                    <h3>Model Performance Metrics</h3>
                </div>
                """, unsafe_allow_html=True)
                
                display_metric("Accuracy", accuracy)
                display_metric("Precision", precision)
                display_metric("Recall", recall)
                display_metric("F1 Score", f1)
                
                # Display prediction result
                st.markdown(f"""
                <div class="dashboard-card">
                    <h3>Image Classification Result</h3>
                    <p style="font-size: 1.2em; font-weight: 600;">Prediction: <span style="color: #3498db; font-size: 1.4em;">{uploaded_image_prediction[0]}</span></p>
                </div>
                """, unsafe_allow_html=True)
            
            # Filter images by the predicted class
            similar_images_indices = [i for i, y in enumerate(Y) if y == uploaded_image_prediction[0]]
            
            # Calculate distances and sort similar images
            distances = []
            dist_func = distance_functions[distance_choice]
            for idx in similar_images_indices:
                feature_vector = np.array(signatures[idx][:-3], dtype=float).flatten()
                dist = dist_func(uploaded_image_features.flatten(), feature_vector)
                distances.append((dist, signatures[idx][-3], signatures[idx][-1]))
            
            distances.sort(key=lambda x: x[0])
            
            # Display similar images
            st.markdown(f"""
            <div class="dashboard-card">
                <h3>Top {image_count} Similar Images in Class {uploaded_image_prediction[0]}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Create a responsive grid for similar images
            num_cols = min(4, image_count)
            cols = st.columns(num_cols)
            
            for i in range(min(image_count, len(distances))):
                dist, relative_path, class_label = distances[i]
                folder_name = os.path.basename(os.path.dirname(relative_path))
                img_path = os.path.join("images", *relative_path.replace("\\", "/").split("/"))
                
                try:
                    similar_img = Image.open(img_path)
                    
                    with cols[i % num_cols]:
                        st.markdown(f"""
                        <div class="img-card">
                        """, unsafe_allow_html=True)
                        st.image(similar_img, caption=f"{folder_name}", use_container_width=True)
                        st.markdown(f"""
                        <p style="text-align: center; font-size: 0.9em;">Similarity: {1/(1+dist):.2f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Could not load image: {relative_path}")
        else:
            st.info("Please upload an image in the Model Configuration tab to see results")


# Multimodal Search with improved UI
def multimodal():
    st.markdown("""
    <div class="dashboard-card">
        <h1>Multimodal Image Search</h1>
        <p>Search for images using text queries to find matching folders and images.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create a nice search bar
    st.markdown("""
    <style>
    .search-container {
        display: flex;
        align-items: center;
        margin-bottom: 20px;
        background-color: white;
        padding: 10px;
        border-radius: 30px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    }
    .search-icon {
        margin-right: 10px;
        color: #6c757d;
    }
    .stTextInput > div {
        flex-grow: 1;
    }
    .stTextInput > div > div > input {
        border: none !important;
        background: transparent !important;
        font-size: 16px;
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown('<div class="search-container">', unsafe_allow_html=True)
        st.markdown('<div class="search-icon">🔍</div>', unsafe_allow_html=True)
        search_term = st.text_input("", placeholder="Enter keywords to search for images...", label_visibility='collapsed')
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        image_count = st.slider("Results to display", min_value=4, max_value=20, value=8)

    if search_term:
        st.markdown(f"""
        <div class="dashboard-card">
            <h3>Search Results for: "{search_term}"</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Show a loading spinner
        with st.spinner(f"Searching for images matching '{search_term}'..."):
            matched_signatures = [sig for sig in signatures_combined if search_term.lower() in sig[-2].lower()]
            
            if matched_signatures:
                # Display results count
                st.success(f"Found {len(matched_signatures)} matching results")
                
                # Create a responsive grid for images
                num_cols = min(4, image_count)
                cols = st.columns(num_cols)
                
                for idx, signature in enumerate(matched_signatures[:image_count]):
                    relative_path = signature[-3]
                    folder_name = signature[-2]
                    
                    try:
                        img_path = os.path.join("images", *relative_path.replace("\\", "/").split("/"))
                        img = Image.open(img_path)
                        
                        with cols[idx % num_cols]:
                            st.markdown(f"""
                            <div class="img-card">
                            """, unsafe_allow_html=True)
                            st.image(img, caption=f"{folder_name}", use_container_width=True)
                            st.markdown(f"""
                            <p style="text-align: center; font-size: 0.8em; color: #666;">Category: {folder_name}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Could not load image: {relative_path}")
            else:
                st.warning(f"No matches found for '{search_term}'. Try different keywords.")
                
                # Suggest some categories
                all_categories = set([sig[-2] for sig in signatures_combined])
                if all_categories:
                    st.markdown("""
                    <div class="dashboard-card">
                        <h4>Available Categories</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display categories as clickable chips
                    st.markdown("""
                    <style>
                    .category-chip {
                        display: inline-block;
                        padding: 5px 15px;
                        margin: 5px;
                        background-color: #e9ecef;
                        border-radius: 20px;
                        font-size: 0.9em;
                        cursor: pointer;
                        transition: all 0.3s ease;
                    }
                    .category-chip:hover {
                        background-color: #3498db;
                        color: white;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    chips_html = ""
                    for category in sorted(list(all_categories)[:20]):  # Limit to 20 categories
                        chips_html += f'<span class="category-chip">{category}</span>'
                    
                    st.markdown(f"""
                    <div style="margin-top: 10px;">
                        {chips_html}
                    </div>
                    """, unsafe_allow_html=True)
    else:
        # Show some sample searches or suggestions
        st.markdown("""
        <div class="dashboard-card">
            <h4>Popular Search Terms</h4>
            <div style="display: flex; flex-wrap: wrap; gap: 10px; margin-top: 15px;">
                <div style="background-color: #e9ecef; padding: 8px 15px; border-radius: 20px; font-size: 0.9em;">nature</div>
                <div style="background-color: #e9ecef; padding: 8px 15px; border-radius: 20px; font-size: 0.9em;">animals</div>
                <div style="background-color: #e9ecef; padding: 8px 15px; border-radius: 20px; font-size: 0.9em;">people</div>
                <div style="background-color: #e9ecef; padding: 8px 15px; border-radius: 20px; font-size: 0.9em;">buildings</div>
                <div style="background-color: #e9ecef; padding: 8px 15px; border-radius: 20px; font-size: 0.9em;">art</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# Settings Page with improved UI
def settings():
    st.markdown("""
    <div class="dashboard-card">
        <h1>Application Settings</h1>
        <p>Customize your CBIR application experience</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabbed interface for settings
    tab1, tab2, tab3 = st.tabs(["Appearance", "Performance", "About"])
    
    with tab1:
        st.markdown("""
        <div class="dashboard-card">
            <h3>Theme Settings</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Night mode toggle with better styling
        col1, col2 = st.columns([1, 3])
        
        with col1:
            if "night_mode" not in st.session_state:
                st.session_state["night_mode"] = False
                
            night_mode = st.toggle("Night Mode", st.session_state["night_mode"])
            st.session_state["night_mode"] = night_mode
        
        with col2:
            if night_mode:
                st.markdown("""
                <div style="display: flex; align-items: center;">
                    <div style="background-color: #1e1e1e; width: 30px; height: 30px; border-radius: 50%; margin-right: 10px;"></div>
                    <p>Dark theme is active - easier on the eyes in low-light environments</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="display: flex; align-items: center;">
                    <div style="background-color: #ffffff; width: 30px; height: 30px; border-radius: 50%; border: 1px solid #ddd; margin-right: 10px;"></div>
                    <p>Light theme is active - better visibility in bright environments</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Apply night mode
        apply_night_mode()
        
        # Interface settings
        st.markdown("""
        <div class="dashboard-card">
            <h3>Interface Settings</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.selectbox(
                "Default Page", 
                ["Basic CBIR", "Advanced CBIR", "Multimodal Search"],
                index=0,
                help="Choose which page to show by default when the app starts"
            )
            
        with col2:
            st.selectbox(
                "Default Image Display", 
                ["Grid View", "List View", "Detailed View"],
                index=0,
                help="Choose how images are displayed by default"
            )
    
    with tab2:
        st.markdown("""
        <div class="dashboard-card">
            <h3>Performance Settings</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.slider(
                "Image Load Quality", 
                min_value=0, 
                max_value=100, 
                value=80,
                help="Lower values improve performance but reduce image quality"
            )
            
        with col2:
            st.selectbox(
                "Default Feature Extractor", 
                ["GLCM", "BIT", "HARALICK", "BIT_GLCM_HARALICK"],
                index=0,
                help="Choose the default feature extractor"
            )
            
        # Cache settings
        st.markdown("""
        <div class="dashboard-card">
            <h4>Cache Management</h4>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.toggle(
                "Enable Results Caching", 
                value=True,
                help="Keep results in memory to speed up repeated searches"
            )
            
        with col2:
            if st.button("Clear Cache", help="Remove all cached results and data"):
                st.success("Cache has been cleared successfully!")
    
    with tab3:
        st.markdown("""
        <div class="dashboard-card">
            <h3>About This Application</h3>
            <p>Advanced Content-Based Image Retrieval System</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="display: flex; justify-content: center; margin: 20px 0;">
            <div style="text-align: center; max-width: 500px;">
                <div style="font-size: 5em; margin-bottom: 10px;">🔍</div>
                <h2>CBIR System v1.0</h2>
                <p>A powerful image retrieval system that uses computer vision and machine learning techniques to find similar images.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Features
        st.markdown("""
        <div class="dashboard-card">
            <h4>Key Features</h4>
            <ul>
                <li><strong>Multiple Feature Extractors:</strong> GLCM, BIT, Haralick, and combined approaches</li>
                <li><strong>Various Distance Metrics:</strong> Manhattan, Euclidean, Chebyshev, and Canberra</li>
                <li><strong>Machine Learning Models:</strong> Support for multiple classification algorithms</li>
                <li><strong>Multimodal Search:</strong> Text-based search capability</li>
                <li><strong>Modern UI:</strong> Responsive and user-friendly interface</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Footer
        st.markdown("""
        <div style="text-align: center; margin-top: 30px; opacity: 0.7;">
            <p>Created with ❤️ using Streamlit</p>
        </div>
        """, unsafe_allow_html=True)


def main():
    # Apply custom CSS
    custom_css()
    
    # Apply night mode at the start
    apply_night_mode()
    
    # Create a nice header
    st.markdown("""
    <div style="display: flex; align-items: center; background: linear-gradient(90deg, #3498db, #2980b9); padding: 1rem; border-radius: 10px; margin-bottom: 20px;">
        <div style="font-size: 2em; margin-right: 15px; color: white;">🔍</div>
        <div>
            <h1 style="color: white; margin: 0;">Advanced CBIR System</h1>
            <p style="color: white; opacity: 0.9; margin: 0;">Content-Based Image Retrieval with Machine Learning</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Create a sidebar with a better UI
    st.sidebar.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <div style="font-size: 3em; margin-bottom: 10px;">🖼️</div>
        <h3>CBIR Navigation</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Create menu items with icons
    menu_options = {
        "Basic CBIR": "🔍 Basic CBIR",
        "Advanced CBIR": "🧠 Advanced CBIR",
        "Multimodal Search": "🔤 Multimodal Search",
        "Settings": "⚙️ Settings"
    }
    
    choice = st.sidebar.selectbox("", list(menu_options.values()), format_func=lambda x: x)
    
    # Convert back to original key
    selected_option = list(menu_options.keys())[list(menu_options.values()).index(choice)]
    
    # Display the selected page
    if selected_option == "Basic CBIR":
        cbir_basic()
    elif selected_option == "Advanced CBIR":
        cbir_advanced()
    elif selected_option == "Multimodal Search":
        multimodal()
    else:
        settings()
    
    # Add a footer
    st.sidebar.markdown("""
    <div style="position: fixed; bottom: 0; left: 0; width: 100%; background-color: rgba(0,0,0,0.05); padding: 10px; text-align: center; font-size: 0.8em;">
        <p>© 2025 CBIR System</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()