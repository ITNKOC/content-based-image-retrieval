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

# Load precomputed signatures
signatures_glcm = np.load('signatures_glcm.npy', allow_pickle=True)
signatures_bitdesc = np.load('signatures_bitdesc.npy', allow_pickle=True)
signatures_haralick = np.load('signatures_haralick_feat.npy', allow_pickle=True)
signatures_combined = np.load('signatures_bit_glcm_haralick.npy', allow_pickle=True)

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

def display_metric_with_color(metric_name, metric_value):
    if metric_value > 0.9:
        color = 'green'
    elif metric_value > 0.7:
        color = 'orange'
    else:
        color = 'red'
    st.markdown(f"<h2 style='color:{color};'>{metric_name}: {metric_value:.2%}</h2>", unsafe_allow_html=True)


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
        img_path = os.path.join('images', relative_path)
        image = Image.open(img_path)
        cols[idx % 4].image(image, caption=relative_path, use_column_width=True)


def cbir_basic():
    st.sidebar.header("Descriptor")
    descriptor_choice = st.sidebar.radio("", ("GLCM", "BIT", "HARALICK", "BIT_GLCM_HARALICK"))
    st.sidebar.header("Distances")
    distance_choice = st.sidebar.radio("", ("Manhattan", "Euclidean", "Chebyshev", "Canberra"))
    st.sidebar.header("Nombre d'Images")
    image_count = st.sidebar.number_input("", min_value=1, value=4, step=1)
    st.title("Content-based Image Retrieval")
    uploaded_file = st.file_uploader("Téléverser une image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)
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
        for signature in signatures:
            feature_vector = np.array(signature[:-3], dtype=float)  # Exclude the last three elements (relative path, folder name, class label)
            dist = dist_func(uploaded_image_features, feature_vector)
            distances.append((dist, signature[-3], signature[-2], signature[-1]))  # Keep the relative path, folder name, and class label

        # Sort the distances in ascending order
        distances.sort(key=lambda x: x[0])

        # Display the top N similar images
        st.header(f"Top {image_count} images similaires")
        cols = st.columns(4)
        for i in range(image_count):
            dist, relative_path, folder_name, class_label = distances[i]
            img_path = os.path.join('images', relative_path)
            similar_img = Image.open(img_path)
            # Show the image with the folder name as the caption
            cols[i % 4].image(similar_img, caption=f"{folder_name}", use_column_width=True)
    else:
        st.write("Veuillez téléverser une image pour commencer.")



# Advanced CBIR with Model Selection and Fine-tuning
def cbir_advanced():
    st.sidebar.header("Descriptor")
    descriptor_choice = st.sidebar.radio("", ("GLCM", "BIT", "HARALICK", "BIT_GLCM_HARALICK"))
    st.sidebar.header("Distances")
    distance_choice = st.sidebar.radio("", ("Manhattan", "Euclidean", "Chebyshev", "Canberra"))
    st.sidebar.header("Nombre d'Images")
    image_count = st.sidebar.number_input("", min_value=1, value=4, step=1)
    st.sidebar.header("GridSearch")
    use_grid_search = st.sidebar.checkbox("Use GridSearchCV")
    st.sidebar.header("Transformation")
    transform_choice = st.sidebar.selectbox("Select Transformation", ["No Transform", "Rescale", "Normalization", "Standardization"])

    st.write("Select Classification Algorithm")
    classifier = st.selectbox("Classifier", ["LDA", "KNN", "Naive Bayes", "Decision Tree", "SVM", "Random Forest", "AdaBoost"])
    st.write("Upload an image for prediction")
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)
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
        
        # Replace inf values with finite values
        X = np.where(np.isinf(X), np.nan, X)
        X = np.where(np.isnan(X), np.nanmean(X, axis=0), X)  # Impute NaN values with column mean
        
        # Apply selected transformation
        X = apply_transform(X, transform_choice)
        
        train_proportion = 0.15
        seed = 10
        # Split train / test data
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=train_proportion, random_state=seed)
        
        model = get_model(classifier, use_grid_search, X_train, Y_train)
        model.fit(X_train, Y_train)
        
        Y_pred = model.predict(X_test)
        accuracy = accuracy_score(Y_test, Y_pred)
        recall = recall_score(Y_test, Y_pred, average='weighted')
        f1 = f1_score(Y_test, Y_pred, average='weighted')
        precision = precision_score(Y_test, Y_pred, average='weighted')
        
        display_metric_with_color('Accuracy', accuracy)
        display_metric_with_color('Recall', recall)
        display_metric_with_color('F1 Score', f1)
        display_metric_with_color('Precision', precision)
        
        # Predict for the uploaded image
        uploaded_image_features = np.array(uploaded_image_features).flatten()  # Convert to NumPy array and flatten
        uploaded_image_features = uploaded_image_features.reshape(1, -1)  # Reshape to 2D array
        uploaded_image_features = apply_transform(uploaded_image_features, transform_choice)
        
        uploaded_image_prediction = model.predict(uploaded_image_features)
        
        st.write(f'Prediction for uploaded image: {uploaded_image_prediction[0]}')
        
        # Filter images by the predicted class
        similar_images_indices = [i for i, y in enumerate(Y) if y == uploaded_image_prediction[0]]
        
        # Calculate distances and sort similar images
        distances = []
        dist_func = distance_functions[distance_choice]
        for idx in similar_images_indices:
            feature_vector = np.array(signatures[idx][:-3], dtype=float).flatten()  # Ensure feature_vector is 1-D
            dist = dist_func(uploaded_image_features.flatten(), feature_vector)
            distances.append((dist, signatures[idx][-3], signatures[idx][-1]))  # Ensure correct indices
        
        distances.sort(key=lambda x: x[0])
        
        # Display similar images
        st.header(f"Top {image_count} images similaires dans la classe {uploaded_image_prediction[0]}")
        cols = st.columns(4)
        for i in range(min(image_count, len(distances))):
            dist, relative_path, class_label = distances[i]
            folder_name = os.path.basename(os.path.dirname(relative_path))  # Get folder name
            img_path = os.path.join('images', relative_path)
            similar_img = Image.open(img_path)
            cols[i % 4].image(similar_img, caption=f"{folder_name}", use_column_width=True)


# Multimodal Search
def multimodal():
    st.sidebar.header("Nombre d'Images")
    image_count = st.sidebar.number_input("", min_value=1, value=4, step=1)
    search_term = st.text_input("Entrez un mot ou une phrase pour la recherche")

    if search_term:
        st.write(f"Recherche de dossiers contenant: {search_term}")

        matched_signatures = [sig for sig in signatures_combined if search_term.lower() in sig[-2].lower()]

        if matched_signatures:
            st.header(f"Top {image_count} images similaires pour '{search_term}'")
            cols = st.columns(4)
            for idx, signature in enumerate(matched_signatures[:image_count]):
                relative_path = signature[-3]
                img_path = os.path.join('images', relative_path)
                img = Image.open(img_path)
                folder_name = signature[-2]
                cols[idx % 4].image(img, caption=f"{folder_name}", use_column_width=True)
        else:
            st.write("Aucun dossier correspondant trouvé.")
    else:
        st.write("Veuillez entrer un mot ou une phrase pour effectuer la recherche.")


# Settings Page
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

            /* Buttons */
            .stButton>button {
                background-color: #444444;
                color: #ffffff;
                border: none;
                border-radius: 4px;
            }
            .stButton>button:hover {
                background-color: #555555;
                color: #ffffff;
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

            /* Text inputs and text areas */
            .stTextInput>div>div>input, .stTextArea>div>textarea {
                background-color: #333333;
                color: #ffffff;
                border: 1px solid #555555;
            }

            /* Checkbox and labels */
            .stCheckbox>div>div>div {
                color: #dcdcdc;
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
                color: #000000;
            }
            
            .css-1d391kg, .css-18e3th9 {
                background-color: #f8f9fa;
                color: #000000;
            }

            h1, h2, h3, h4, h5, h6 {
                color: #000000;
            }

            .stText, .stMarkdown, .stDataFrame {
                color: #000000;
            }

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

            </style>
            """,
            unsafe_allow_html=True
        )

def settings():
    if "night_mode" not in st.session_state:
        st.session_state["night_mode"] = False  # Default is day mode

    st.sidebar.header("Mode Nuit")
    night_mode = st.sidebar.checkbox("Activer le mode nuit", st.session_state["night_mode"])

    st.session_state["night_mode"] = night_mode
    apply_night_mode()

    st.write("Le mode nuit est activé." if night_mode else "Le mode jour est activé.")

def main():
    st.title("Advanced CBIR System")
    apply_night_mode()  # Apply night mode at the start of the app
    menu = ["Basic CBIR", "Advanced CBIR", "Multimodal Search", "Settings"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Basic CBIR":
        cbir_basic()
    elif choice == "Advanced CBIR":
        cbir_advanced()
    elif choice == "Multimodal Search":
        multimodal()
    else:
        settings()

if __name__ == '__main__':
    main()