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

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="CBIR System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "night_mode" not in st.session_state:
    st.session_state.night_mode = False
if "default_descriptor" not in st.session_state:
    st.session_state.default_descriptor = "GLCM"
if "default_distance" not in st.session_state:
    st.session_state.default_distance = "Euclidean"
if "default_image_count" not in st.session_state:
    st.session_state.default_image_count = 6

# Load precomputed signatures with error handling
@st.cache_data
def load_signatures():
    try:
        signatures_glcm = np.load('signatures_glcm.npy', allow_pickle=True)
        signatures_bitdesc = np.load('signatures_bitdesc.npy', allow_pickle=True)
        signatures_haralick = np.load('signatures_haralick_feat.npy', allow_pickle=True)
        signatures_combined = np.load('signatures_bit_glcm_haralick.npy', allow_pickle=True)
        return signatures_glcm, signatures_bitdesc, signatures_haralick, signatures_combined
    except FileNotFoundError as e:
        st.error(f"Signature files not found. Please run data_processing.py first.")
        return None, None, None, None

signatures_glcm, signatures_bitdesc, signatures_haralick, signatures_combined = load_signatures()

# Get all available categories from dataset
@st.cache_data
def get_categories():
    if signatures_combined is not None:
        return sorted(list(set([sig[-2] for sig in signatures_combined])))
    return []

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

# Custom CSS
def apply_custom_css():
    is_dark = st.session_state.night_mode

    if is_dark:
        bg_primary = "#0e1117"
        bg_secondary = "#1a1f2e"
        bg_card = "#262c3d"
        text_primary = "#fafafa"
        text_secondary = "#b0b8c4"
        accent_color = "#667eea"
        accent_gradient = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
        border_color = "#3d4663"
        hover_bg = "#323a52"
    else:
        bg_primary = "#f8fafc"
        bg_secondary = "#ffffff"
        bg_card = "#ffffff"
        text_primary = "#1e293b"
        text_secondary = "#64748b"
        accent_color = "#6366f1"
        accent_gradient = "linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)"
        border_color = "#e2e8f0"
        hover_bg = "#f1f5f9"

    st.markdown(f"""
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        /* Global styles */
        .stApp {{
            background-color: {bg_primary};
            font-family: 'Inter', sans-serif;
        }}

        /* Hide Streamlit branding */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}

        /* Main content area */
        .main .block-container {{
            padding: 2rem 3rem;
            max-width: 1400px;
        }}

        /* Sidebar styling */
        section[data-testid="stSidebar"] {{
            background: {bg_secondary};
            border-right: 1px solid {border_color};
        }}

        section[data-testid="stSidebar"] .block-container {{
            padding: 2rem 1.5rem;
        }}

        /* Headers */
        h1, h2, h3, h4, h5, h6 {{
            color: {text_primary} !important;
            font-weight: 600 !important;
        }}

        p, span, label, .stMarkdown {{
            color: {text_secondary} !important;
        }}

        /* Custom card component */
        .custom-card {{
            background: {bg_card};
            border: 1px solid {border_color};
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            transition: all 0.3s ease;
        }}

        .custom-card:hover {{
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
            transform: translateY(-2px);
        }}

        .custom-card h3 {{
            margin-bottom: 0.5rem;
            font-size: 1.1rem;
        }}

        .custom-card p {{
            margin: 0;
            font-size: 0.9rem;
        }}

        /* Hero header */
        .hero-header {{
            background: {accent_gradient};
            border-radius: 20px;
            padding: 2rem 2.5rem;
            margin-bottom: 2rem;
            color: white !important;
        }}

        .hero-header h1 {{
            color: white !important;
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }}

        .hero-header p {{
            color: rgba(255, 255, 255, 0.9) !important;
            font-size: 1rem;
            margin: 0;
        }}

        /* Image cards */
        .image-card {{
            background: {bg_card};
            border: 1px solid {border_color};
            border-radius: 12px;
            padding: 0.75rem;
            margin-bottom: 1rem;
            transition: all 0.3s ease;
            overflow: hidden;
        }}

        .image-card:hover {{
            border-color: {accent_color};
            box-shadow: 0 8px 25px rgba(99, 102, 241, 0.15);
            transform: translateY(-4px);
        }}

        .image-card img {{
            border-radius: 8px;
        }}

        .image-caption {{
            text-align: center;
            padding: 0.5rem 0;
            font-size: 0.85rem;
            color: {text_secondary};
            font-weight: 500;
        }}

        .similarity-badge {{
            display: inline-block;
            background: {accent_gradient};
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-top: 0.5rem;
        }}

        /* Metric cards */
        .metric-card {{
            background: {bg_card};
            border: 1px solid {border_color};
            border-radius: 12px;
            padding: 1.25rem;
            text-align: center;
            transition: all 0.3s ease;
        }}

        .metric-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }}

        .metric-value {{
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }}

        .metric-label {{
            font-size: 0.85rem;
            color: {text_secondary};
            font-weight: 500;
        }}

        .metric-excellent {{
            color: #10b981;
        }}

        .metric-good {{
            color: #f59e0b;
        }}

        .metric-poor {{
            color: #ef4444;
        }}

        /* Buttons */
        .stButton > button {{
            background: {accent_gradient};
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.6rem 1.5rem;
            font-weight: 600;
            font-size: 0.9rem;
            transition: all 0.3s ease;
            width: 100%;
        }}

        .stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(99, 102, 241, 0.3);
        }}

        /* File uploader */
        .stFileUploader > div {{
            background: {bg_card};
            border: 2px dashed {border_color};
            border-radius: 12px;
            padding: 2rem;
            transition: all 0.3s ease;
        }}

        .stFileUploader > div:hover {{
            border-color: {accent_color};
            background: {hover_bg};
        }}

        /* Select boxes */
        .stSelectbox > div > div {{
            background: {bg_card};
            border: 1px solid {border_color};
            border-radius: 10px;
        }}

        /* Sliders */
        .stSlider > div > div > div {{
            background: {border_color};
        }}

        .stSlider > div > div > div > div {{
            background: {accent_color};
        }}

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 0.5rem;
            background: transparent;
        }}

        .stTabs [data-baseweb="tab"] {{
            background: {bg_card};
            border: 1px solid {border_color};
            border-radius: 10px;
            padding: 0.75rem 1.5rem;
            font-weight: 500;
            color: {text_secondary};
        }}

        .stTabs [aria-selected="true"] {{
            background: {accent_gradient} !important;
            border: none !important;
            color: white !important;
        }}

        /* Progress bar */
        .stProgress > div > div > div {{
            background: {accent_gradient};
            border-radius: 10px;
        }}

        /* Text input */
        .stTextInput > div > div > input {{
            background: {bg_card};
            border: 1px solid {border_color};
            border-radius: 10px;
            padding: 0.75rem 1rem;
            color: {text_primary};
        }}

        .stTextInput > div > div > input:focus {{
            border-color: {accent_color};
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
        }}

        /* Category chips */
        .category-chip {{
            display: inline-block;
            background: {bg_card};
            border: 1px solid {border_color};
            color: {text_secondary};
            padding: 0.5rem 1rem;
            margin: 0.25rem;
            border-radius: 25px;
            font-size: 0.85rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
        }}

        .category-chip:hover {{
            background: {accent_gradient};
            color: white;
            border-color: transparent;
        }}

        /* Toggle switch */
        .stToggle > label > div {{
            background: {border_color};
        }}

        /* Info boxes */
        .stAlert {{
            border-radius: 12px;
            border: none;
        }}

        /* Sidebar navigation */
        .nav-item {{
            display: flex;
            align-items: center;
            padding: 0.875rem 1rem;
            margin: 0.25rem 0;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.2s ease;
            color: {text_secondary};
        }}

        .nav-item:hover {{
            background: {hover_bg};
            color: {text_primary};
        }}

        .nav-item.active {{
            background: {accent_gradient};
            color: white;
        }}

        .nav-icon {{
            font-size: 1.25rem;
            margin-right: 0.75rem;
        }}

        /* Spinner */
        .stSpinner > div {{
            border-color: {accent_color};
        }}

        /* Expander */
        .streamlit-expanderHeader {{
            background: {bg_card};
            border: 1px solid {border_color};
            border-radius: 10px;
        }}

        /* Results grid */
        .results-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 1rem;
        }}

        /* Stats bar */
        .stats-bar {{
            display: flex;
            justify-content: space-around;
            background: {bg_card};
            border: 1px solid {border_color};
            border-radius: 12px;
            padding: 1rem;
            margin-bottom: 1.5rem;
        }}

        .stat-item {{
            text-align: center;
        }}

        .stat-value {{
            font-size: 1.5rem;
            font-weight: 700;
            color: {accent_color};
        }}

        .stat-label {{
            font-size: 0.8rem;
            color: {text_secondary};
        }}
    </style>
    """, unsafe_allow_html=True)

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

def display_metric_card(label, value, col):
    if value > 0.9:
        color_class = "metric-excellent"
    elif value > 0.7:
        color_class = "metric-good"
    else:
        color_class = "metric-poor"

    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-value {color_class}">{value:.1%}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

def extract_features(image, descriptor_choice):
    descriptor_func = descriptor_functions[descriptor_choice]
    return descriptor_func(image)

def get_signatures(descriptor_choice):
    if descriptor_choice == 'GLCM':
        return signatures_glcm
    elif descriptor_choice == 'BIT':
        return signatures_bitdesc
    elif descriptor_choice == 'HARALICK':
        return signatures_haralick
    else:
        return signatures_combined

def display_image_card(img, caption, similarity=None, col=None):
    target = col if col else st
    with target:
        st.markdown('<div class="image-card">', unsafe_allow_html=True)
        st.image(img, use_container_width=True)
        st.markdown(f'<div class="image-caption">{caption}</div>', unsafe_allow_html=True)
        if similarity is not None:
            st.markdown(f'<div style="text-align:center;"><span class="similarity-badge">{similarity:.1%} match</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Basic CBIR Page
def cbir_basic():
    st.markdown("""
    <div class="hero-header">
        <h1>Basic Image Retrieval</h1>
        <p>Upload an image to find visually similar images in the database using feature descriptors</p>
    </div>
    """, unsafe_allow_html=True)

    if signatures_glcm is None:
        st.error("Database not loaded. Please run `python data_processing.py` first.")
        return

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("### Configuration")

        descriptor_choice = st.selectbox(
            "Feature Descriptor",
            ["GLCM", "BIT", "HARALICK", "BIT_GLCM_HARALICK"],
            index=["GLCM", "BIT", "HARALICK", "BIT_GLCM_HARALICK"].index(st.session_state.default_descriptor),
            help="Algorithm used to extract image features"
        )

        distance_choice = st.selectbox(
            "Distance Metric",
            ["Manhattan", "Euclidean", "Chebyshev", "Canberra"],
            index=["Manhattan", "Euclidean", "Chebyshev", "Canberra"].index(st.session_state.default_distance),
            help="Method to measure similarity between images"
        )

        image_count = st.slider(
            "Results to Display",
            min_value=2,
            max_value=12,
            value=st.session_state.default_image_count,
            help="Number of similar images to show"
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # Stats
        categories = get_categories()
        signatures = get_signatures(descriptor_choice)
        st.markdown(f"""
        <div class="stats-bar">
            <div class="stat-item">
                <div class="stat-value">{len(signatures) if signatures is not None else 0:,}</div>
                <div class="stat-label">Images</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{len(categories)}</div>
                <div class="stat-label">Categories</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("### Upload Image")
        uploaded_file = st.file_uploader("Choose an image to search", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            st.image(img, caption="Query Image", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        with st.spinner("Analyzing image and finding similar matches..."):
            img_array = np.array(img)
            img_array = cv2.resize(img_array, (256, 256))
            if len(img_array.shape) == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

            uploaded_features = extract_features(img_array, descriptor_choice)
            signatures = get_signatures(descriptor_choice)

            distances = []
            dist_func = distance_functions[distance_choice]

            for signature in signatures:
                feature_vector = np.array(signature[:-3], dtype=float)
                dist = dist_func(uploaded_features, feature_vector)
                distances.append((dist, signature[-3], signature[-2], signature[-1]))

            distances.sort(key=lambda x: x[0])

        st.markdown("### Similar Images Found")

        num_cols = min(4, image_count)
        cols = st.columns(num_cols)

        for i in range(image_count):
            dist, relative_path, folder_name, class_label = distances[i]
            img_path = os.path.join("images", *relative_path.replace("\\", "/").split("/"))

            try:
                similar_img = Image.open(img_path)
                similarity = 1 / (1 + dist)
                display_image_card(similar_img, folder_name, similarity, cols[i % num_cols])
            except Exception as e:
                cols[i % num_cols].error(f"Could not load: {relative_path}")

# Advanced CBIR Page
def cbir_advanced():
    st.markdown("""
    <div class="hero-header">
        <h1>ML-Powered Image Retrieval</h1>
        <p>Combine machine learning classification with visual similarity search</p>
    </div>
    """, unsafe_allow_html=True)

    if signatures_glcm is None:
        st.error("Database not loaded. Please run `python data_processing.py` first.")
        return

    tab1, tab2 = st.tabs(["Configuration", "Results"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.markdown("### Feature Settings")

            descriptor_choice = st.selectbox(
                "Feature Descriptor",
                ["GLCM", "BIT", "HARALICK", "BIT_GLCM_HARALICK"],
                key="adv_descriptor"
            )

            distance_choice = st.selectbox(
                "Distance Metric",
                ["Manhattan", "Euclidean", "Chebyshev", "Canberra"],
                key="adv_distance"
            )

            transform_choice = st.selectbox(
                "Data Transformation",
                ["No Transform", "Rescale", "Normalization", "Standardization"]
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.markdown("### Model Settings")

            classifier = st.selectbox(
                "Classification Algorithm",
                ["LDA", "KNN", "Naive Bayes", "Decision Tree", "SVM", "Random Forest", "AdaBoost"]
            )

            use_grid_search = st.toggle(
                "Hyperparameter Tuning (GridSearchCV)",
                value=False,
                help="Automatically find optimal model parameters"
            )

            image_count = st.slider(
                "Results to Display",
                min_value=2,
                max_value=12,
                value=6,
                key="adv_count"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("### Upload Image")
        uploaded_image = st.file_uploader("Choose an image for classification", type=["jpg", "png", "jpeg"], key="adv_upload")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        if uploaded_image is not None:
            image = Image.open(uploaded_image)

            col1, col2 = st.columns([1, 3])

            with col1:
                st.markdown('<div class="custom-card">', unsafe_allow_html=True)
                st.markdown("### Query Image")
                st.image(image, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                # Process image
                img_array = np.array(image)
                img_array = cv2.resize(img_array, (256, 256))
                if len(img_array.shape) == 3:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

                uploaded_features = extract_features(img_array, descriptor_choice)
                signatures = get_signatures(descriptor_choice)

                X = np.array([sig[:-3] for sig in signatures], dtype=float)
                Y = np.array([sig[-1] for sig in signatures], dtype=int)

                # Clean data
                X = np.where(np.isinf(X), np.nan, X)
                col_means = np.nanmean(X, axis=0)
                inds = np.where(np.isnan(X))
                X[inds] = np.take(col_means, inds[1])

                X = apply_transform(X, transform_choice)

                progress = st.progress(0, text="Training model...")

                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=10)
                progress.progress(20, text="Training model...")

                model = get_model(classifier, use_grid_search, X_train, Y_train)
                progress.progress(50, text="Training model...")

                model.fit(X_train, Y_train)
                progress.progress(70, text="Evaluating...")

                Y_pred = model.predict(X_test)
                accuracy = accuracy_score(Y_test, Y_pred)
                recall = recall_score(Y_test, Y_pred, average='weighted')
                f1 = f1_score(Y_test, Y_pred, average='weighted')
                precision = precision_score(Y_test, Y_pred, average='weighted')

                progress.progress(90, text="Making prediction...")

                uploaded_features_flat = np.array(uploaded_features).flatten().reshape(1, -1)
                uploaded_features_flat = apply_transform(uploaded_features_flat, transform_choice)
                prediction = model.predict(uploaded_features_flat)[0]

                progress.progress(100, text="Complete!")
                progress.empty()

                # Display metrics
                st.markdown("### Model Performance")
                metric_cols = st.columns(4)
                display_metric_card("Accuracy", accuracy, metric_cols[0])
                display_metric_card("Precision", precision, metric_cols[1])
                display_metric_card("Recall", recall, metric_cols[2])
                display_metric_card("F1 Score", f1, metric_cols[3])

                # Display prediction
                folder_name = [sig[-2] for sig in signatures if sig[-1] == prediction][0] if prediction else "Unknown"
                st.markdown(f"""
                <div class="custom-card" style="text-align: center;">
                    <h3>Predicted Category</h3>
                    <div style="font-size: 2rem; font-weight: 700; color: #6366f1;">{folder_name}</div>
                    <p style="margin-top: 0.5rem;">Class ID: {prediction}</p>
                </div>
                """, unsafe_allow_html=True)

            # Find similar images in predicted class
            similar_indices = [i for i, y in enumerate(Y) if y == prediction]

            distances = []
            dist_func = distance_functions[distance_choice]
            for idx in similar_indices:
                feature_vector = np.array(signatures[idx][:-3], dtype=float).flatten()
                dist = dist_func(uploaded_features_flat.flatten(), feature_vector)
                distances.append((dist, signatures[idx][-3], signatures[idx][-2]))

            distances.sort(key=lambda x: x[0])

            st.markdown(f"### Similar Images in Category: {folder_name}")

            num_cols = min(4, image_count)
            cols = st.columns(num_cols)

            for i in range(min(image_count, len(distances))):
                dist, relative_path, cat_name = distances[i]
                img_path = os.path.join("images", *relative_path.replace("\\", "/").split("/"))

                try:
                    similar_img = Image.open(img_path)
                    similarity = 1 / (1 + dist)
                    display_image_card(similar_img, cat_name, similarity, cols[i % num_cols])
                except Exception:
                    cols[i % num_cols].error(f"Could not load image")
        else:
            st.info("Please upload an image in the Configuration tab to see results.")

# Multimodal Search Page
def multimodal():
    st.markdown("""
    <div class="hero-header">
        <h1>Multimodal Search</h1>
        <p>Search images by category name or keywords</p>
    </div>
    """, unsafe_allow_html=True)

    if signatures_combined is None:
        st.error("Database not loaded. Please run `python data_processing.py` first.")
        return

    categories = get_categories()

    col1, col2 = st.columns([3, 1])

    with col1:
        search_term = st.text_input(
            "Search",
            placeholder="Enter category name or keyword...",
            label_visibility="collapsed"
        )

    with col2:
        image_count = st.slider("Results", min_value=4, max_value=24, value=12, label_visibility="collapsed")

    if search_term:
        with st.spinner(f"Searching for '{search_term}'..."):
            matched = [sig for sig in signatures_combined if search_term.lower() in sig[-2].lower()]

        if matched:
            st.success(f"Found {len(matched)} images matching '{search_term}'")

            num_cols = min(4, image_count)
            cols = st.columns(num_cols)

            for idx, sig in enumerate(matched[:image_count]):
                relative_path = sig[-3]
                folder_name = sig[-2]
                img_path = os.path.join("images", *relative_path.replace("\\", "/").split("/"))

                try:
                    img = Image.open(img_path)
                    display_image_card(img, folder_name, col=cols[idx % num_cols])
                except Exception:
                    cols[idx % num_cols].error("Could not load image")
        else:
            st.warning(f"No matches found for '{search_term}'")

            st.markdown("### Available Categories")
            st.markdown('<div style="margin-top: 1rem;">', unsafe_allow_html=True)

            chips_html = ""
            for cat in categories[:30]:
                chips_html += f'<span class="category-chip">{cat}</span>'

            st.markdown(chips_html, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("### Browse Categories")
        st.markdown("Click a category name to search, or type in the search box above.")
        st.markdown('<div style="margin-top: 1rem;">', unsafe_allow_html=True)

        chips_html = ""
        for cat in categories:
            chips_html += f'<span class="category-chip">{cat}</span>'

        st.markdown(chips_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Settings Page
def settings():
    st.markdown("""
    <div class="hero-header">
        <h1>Settings</h1>
        <p>Customize your CBIR experience</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Appearance", "Defaults", "About"])

    with tab1:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("### Theme")

        col1, col2 = st.columns([1, 3])

        with col1:
            night_mode = st.toggle("Dark Mode", st.session_state.night_mode)
            if night_mode != st.session_state.night_mode:
                st.session_state.night_mode = night_mode
                st.rerun()

        with col2:
            if st.session_state.night_mode:
                st.markdown("Dark theme - easier on the eyes in low-light environments")
            else:
                st.markdown("Light theme - better visibility in bright environments")

        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("### Default Settings")

        col1, col2 = st.columns(2)

        with col1:
            default_descriptor = st.selectbox(
                "Default Feature Descriptor",
                ["GLCM", "BIT", "HARALICK", "BIT_GLCM_HARALICK"],
                index=["GLCM", "BIT", "HARALICK", "BIT_GLCM_HARALICK"].index(st.session_state.default_descriptor)
            )
            if default_descriptor != st.session_state.default_descriptor:
                st.session_state.default_descriptor = default_descriptor

            default_distance = st.selectbox(
                "Default Distance Metric",
                ["Manhattan", "Euclidean", "Chebyshev", "Canberra"],
                index=["Manhattan", "Euclidean", "Chebyshev", "Canberra"].index(st.session_state.default_distance)
            )
            if default_distance != st.session_state.default_distance:
                st.session_state.default_distance = default_distance

        with col2:
            default_count = st.slider(
                "Default Results Count",
                min_value=2,
                max_value=12,
                value=st.session_state.default_image_count
            )
            if default_count != st.session_state.default_image_count:
                st.session_state.default_image_count = default_count

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("### Cache Management")

        if st.button("Clear Application Cache"):
            st.cache_data.clear()
            st.success("Cache cleared successfully!")

        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown("""
        <div class="custom-card" style="text-align: center;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">🔍</div>
            <h2>CBIR System v2.0</h2>
            <p>Content-Based Image Retrieval with Machine Learning</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="custom-card">
            <h3>Features</h3>
            <ul>
                <li><strong>Multiple Feature Extractors:</strong> GLCM, BIT, Haralick, and combined</li>
                <li><strong>Distance Metrics:</strong> Manhattan, Euclidean, Chebyshev, Canberra</li>
                <li><strong>ML Models:</strong> LDA, KNN, SVM, Random Forest, AdaBoost, Decision Tree, Naive Bayes</li>
                <li><strong>Multimodal Search:</strong> Text-based category search</li>
                <li><strong>Modern UI:</strong> Dark/Light mode, responsive design</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        categories = get_categories()
        signatures = get_signatures("GLCM")
        st.markdown(f"""
        <div class="stats-bar">
            <div class="stat-item">
                <div class="stat-value">{len(signatures) if signatures is not None else 0:,}</div>
                <div class="stat-label">Total Images</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{len(categories)}</div>
                <div class="stat-label">Categories</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">4</div>
                <div class="stat-label">Descriptors</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">7</div>
                <div class="stat-label">ML Models</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def main():
    apply_custom_css()

    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0 2rem 0;">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🔍</div>
            <h2 style="margin: 0; font-size: 1.25rem;">CBIR System</h2>
        </div>
        """, unsafe_allow_html=True)

        page = st.radio(
            "Navigation",
            ["Basic Search", "ML Search", "Multimodal", "Settings"],
            label_visibility="collapsed"
        )

        st.markdown("---")

        # Quick stats in sidebar
        categories = get_categories()
        st.markdown(f"""
        <div style="text-align: center; opacity: 0.8; font-size: 0.85rem;">
            <p><strong>{len(signatures_glcm) if signatures_glcm is not None else 0:,}</strong> images indexed</p>
            <p><strong>{len(categories)}</strong> categories</p>
        </div>
        """, unsafe_allow_html=True)

    # Main content
    if page == "Basic Search":
        cbir_basic()
    elif page == "ML Search":
        cbir_advanced()
    elif page == "Multimodal":
        multimodal()
    else:
        settings()

if __name__ == '__main__':
    main()
