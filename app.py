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

# Page configuration
st.set_page_config(
    page_title="CBIR System",
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='%236366f1'><circle cx='11' cy='11' r='8'/><path d='m21 21-4.35-4.35'/></svg>",
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

# Load signatures
@st.cache_data
def load_signatures():
    try:
        return (
            np.load('signatures_glcm.npy', allow_pickle=True),
            np.load('signatures_bitdesc.npy', allow_pickle=True),
            np.load('signatures_haralick_feat.npy', allow_pickle=True),
            np.load('signatures_bit_glcm_haralick.npy', allow_pickle=True)
        )
    except FileNotFoundError:
        st.error("Database not found. Please run data_processing.py first.")
        return None, None, None, None

signatures_glcm, signatures_bitdesc, signatures_haralick, signatures_combined = load_signatures()

@st.cache_data
def get_categories():
    if signatures_combined is not None:
        return sorted(list(set([sig[-2] for sig in signatures_combined])))
    return []

distance_functions = {
    'Manhattan': manhattan,
    'Euclidean': euclidean,
    'Chebyshev': chebyshev,
    'Canberra': canberra,
}

descriptor_functions = {
    'GLCM': glcm,
    'BIT': bitdesc,
    'HARALICK': haralick_feat,
    'BIT_GLCM_HARALICK': bit_glcm_haralick
}

# SVG Icons
ICONS = {
    'search': '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"></circle><path d="m21 21-4.35-4.35"></path></svg>''',
    'brain': '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z"></path><path d="M12 5a3 3 0 1 1 5.997.125 4 4 0 0 1 2.526 5.77 4 4 0 0 1-.556 6.588A4 4 0 1 1 12 18Z"></path><path d="M15 13a4.5 4.5 0 0 1-3-4 4.5 4.5 0 0 1-3 4"></path><path d="M17.599 6.5a3 3 0 0 0 .399-1.375"></path><path d="M6.003 5.125A3 3 0 0 0 6.401 6.5"></path><path d="M3.477 10.896a4 4 0 0 1 .585-.396"></path><path d="M19.938 10.5a4 4 0 0 1 .585.396"></path><path d="M6 18a4 4 0 0 1-1.967-.516"></path><path d="M19.967 17.484A4 4 0 0 1 18 18"></path></svg>''',
    'layers': '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z"></path><path d="m22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65"></path><path d="m22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65"></path></svg>''',
    'settings': '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"></path><circle cx="12" cy="12" r="3"></circle></svg>''',
    'upload': '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" x2="12" y1="3" y2="15"></line></svg>''',
    'image': '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="18" x="3" y="3" rx="2" ry="2"></rect><circle cx="9" cy="9" r="2"></circle><path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"></path></svg>''',
    'grid': '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="7" height="7" x="3" y="3" rx="1"></rect><rect width="7" height="7" x="14" y="3" rx="1"></rect><rect width="7" height="7" x="14" y="14" rx="1"></rect><rect width="7" height="7" x="3" y="14" rx="1"></rect></svg>''',
    'sliders': '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="4" x2="4" y1="21" y2="14"></line><line x1="4" x2="4" y1="10" y2="3"></line><line x1="12" x2="12" y1="21" y2="12"></line><line x1="12" x2="12" y1="8" y2="3"></line><line x1="20" x2="20" y1="21" y2="16"></line><line x1="20" x2="20" y1="12" y2="3"></line><line x1="2" x2="6" y1="14" y2="14"></line><line x1="10" x2="14" y1="8" y2="8"></line><line x1="18" x2="22" y1="16" y2="16"></line></svg>''',
    'cpu': '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2"></rect><rect x="9" y="9" width="6" height="6"></rect><path d="M15 2v2"></path><path d="M15 20v2"></path><path d="M2 15h2"></path><path d="M2 9h2"></path><path d="M20 15h2"></path><path d="M20 9h2"></path><path d="M9 2v2"></path><path d="M9 20v2"></path></svg>''',
    'target': '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><circle cx="12" cy="12" r="6"></circle><circle cx="12" cy="12" r="2"></circle></svg>''',
    'chart': '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"></path><path d="m19 9-5 5-4-4-3 3"></path></svg>''',
    'tag': '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12.586 2.586A2 2 0 0 0 11.172 2H4a2 2 0 0 0-2 2v7.172a2 2 0 0 0 .586 1.414l8.704 8.704a2.426 2.426 0 0 0 3.42 0l6.58-6.58a2.426 2.426 0 0 0 0-3.42z"></path><circle cx="7.5" cy="7.5" r=".5" fill="currentColor"></circle></svg>''',
    'moon': '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"></path></svg>''',
    'sun': '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4"></circle><path d="M12 2v2"></path><path d="M12 20v2"></path><path d="m4.93 4.93 1.41 1.41"></path><path d="m17.66 17.66 1.41 1.41"></path><path d="M2 12h2"></path><path d="M20 12h2"></path><path d="m6.34 17.66-1.41 1.41"></path><path d="m19.07 4.93-1.41 1.41"></path></svg>''',
    'palette': '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="13.5" cy="6.5" r=".5" fill="currentColor"></circle><circle cx="17.5" cy="10.5" r=".5" fill="currentColor"></circle><circle cx="8.5" cy="7.5" r=".5" fill="currentColor"></circle><circle cx="6.5" cy="12.5" r=".5" fill="currentColor"></circle><path d="M12 2C6.5 2 2 6.5 2 12s4.5 10 10 10c.926 0 1.648-.746 1.648-1.688 0-.437-.18-.835-.437-1.125-.29-.289-.438-.652-.438-1.125a1.64 1.64 0 0 1 1.668-1.668h1.996c3.051 0 5.555-2.503 5.555-5.555C21.965 6.012 17.461 2 12 2z"></path></svg>''',
    'database': '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"></ellipse><path d="M3 5V19A9 3 0 0 0 21 19V5"></path><path d="M3 12A9 3 0 0 0 21 12"></path></svg>''',
    'folder': '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z"></path></svg>''',
    'trash': '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18"></path><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"></path><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"></path></svg>''',
    'info': '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><path d="M12 16v-4"></path><path d="M12 8h.01"></path></svg>''',
    'check': '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"></path></svg>''',
    'sparkles': '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"></path><path d="M5 3v4"></path><path d="M19 17v4"></path><path d="M3 5h4"></path><path d="M17 19h4"></path></svg>''',
}

def icon(name, size=24, color="currentColor"):
    svg = ICONS.get(name, ICONS['search'])
    svg = svg.replace('width="24"', f'width="{size}"')
    svg = svg.replace('height="24"', f'height="{size}"')
    if color != "currentColor":
        svg = svg.replace('stroke="currentColor"', f'stroke="{color}"')
    return svg

# ==================== MODERN CSS ====================
def apply_modern_css():
    is_dark = st.session_state.night_mode

    if is_dark:
        css = """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

        :root {
            --bg-primary: #0a0a0f;
            --bg-secondary: #12121a;
            --bg-card: #1a1a24;
            --bg-card-hover: #22222e;
            --text-primary: #f0f0f5;
            --text-secondary: #9898a6;
            --text-muted: #5a5a6e;
            --accent-primary: #6366f1;
            --accent-secondary: #818cf8;
            --accent-gradient: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
            --border-color: #2a2a3a;
            --shadow-color: rgba(99, 102, 241, 0.1);
            --success: #22c55e;
            --warning: #eab308;
            --error: #ef4444;
        }

        * { font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important; }

        .stApp {
            background: var(--bg-primary);
            background-image:
                radial-gradient(ellipse at 0% 0%, rgba(99, 102, 241, 0.08) 0%, transparent 50%),
                radial-gradient(ellipse at 100% 100%, rgba(139, 92, 246, 0.06) 0%, transparent 50%);
        }

        #MainMenu, footer, header { visibility: hidden; }

        section[data-testid="stSidebar"] {
            background: var(--bg-secondary);
            border-right: 1px solid var(--border-color);
        }

        h1, h2, h3, h4, h5, h6 { color: var(--text-primary) !important; font-weight: 600 !important; }
        p, span, label, div, .stMarkdown { color: var(--text-secondary) !important; }

        .hero-section {
            background: var(--accent-gradient);
            border-radius: 20px;
            padding: 2.5rem 3rem;
            margin-bottom: 2rem;
            position: relative;
            overflow: hidden;
        }

        .hero-section::after {
            content: '';
            position: absolute;
            top: 0; right: 0; bottom: 0; left: 0;
            background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, transparent 50%);
            pointer-events: none;
        }

        .hero-section h1 { color: #fff !important; font-size: 2rem !important; font-weight: 700 !important; margin: 0 0 0.5rem 0; }
        .hero-section p { color: rgba(255,255,255,0.85) !important; margin: 0; font-size: 1rem; }
        .hero-icon { display: inline-flex; align-items: center; justify-content: center; width: 48px; height: 48px; background: rgba(255,255,255,0.15); border-radius: 12px; margin-bottom: 1rem; }
        .hero-icon svg { stroke: white; }

        .glass-card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1.25rem;
            transition: all 0.3s ease;
        }

        .glass-card:hover {
            border-color: var(--accent-primary);
            box-shadow: 0 8px 32px var(--shadow-color);
            transform: translateY(-2px);
        }

        .card-header { display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1rem; }
        .card-header svg { stroke: var(--accent-primary); }
        .card-header h3 { margin: 0 !important; font-size: 1rem !important; color: var(--text-primary) !important; }

        .image-card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 14px;
            overflow: hidden;
            transition: all 0.3s ease;
        }

        .image-card:hover {
            transform: translateY(-6px);
            box-shadow: 0 16px 40px var(--shadow-color);
            border-color: var(--accent-primary);
        }

        .image-info { padding: 1rem; text-align: center; }
        .image-title { font-weight: 600; color: var(--text-primary) !important; font-size: 0.875rem; margin-bottom: 0.5rem; }

        .similarity-badge {
            display: inline-flex; align-items: center; gap: 0.375rem;
            background: var(--accent-gradient);
            color: #fff !important;
            padding: 0.375rem 0.875rem;
            border-radius: 50px;
            font-size: 0.75rem;
            font-weight: 600;
        }

        .metric-card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 14px;
            padding: 1.25rem;
            text-align: center;
            transition: all 0.3s ease;
        }

        .metric-card:hover { transform: translateY(-3px); box-shadow: 0 12px 28px var(--shadow-color); }
        .metric-value { font-size: 2rem; font-weight: 700; }
        .metric-value.excellent { color: var(--success) !important; }
        .metric-value.good { color: var(--warning) !important; }
        .metric-value.poor { color: var(--error) !important; }
        .metric-label { color: var(--text-muted) !important; font-size: 0.8rem; font-weight: 500; margin-top: 0.25rem; }

        .stats-container { display: flex; gap: 1rem; margin-bottom: 1.25rem; }
        .stat-box {
            flex: 1;
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 14px;
            padding: 1.25rem;
            text-align: center;
            transition: all 0.3s ease;
        }
        .stat-box:hover { border-color: var(--accent-primary); }
        .stat-value { font-size: 1.5rem; font-weight: 700; color: var(--accent-primary) !important; }
        .stat-label { font-size: 0.75rem; color: var(--text-muted) !important; margin-top: 0.25rem; text-transform: uppercase; letter-spacing: 0.5px; }

        .chip-container { display: flex; flex-wrap: wrap; gap: 0.5rem; }
        .category-chip {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            color: var(--text-secondary) !important;
            padding: 0.5rem 1rem;
            border-radius: 50px;
            font-size: 0.8rem;
            font-weight: 500;
            transition: all 0.25s ease;
            cursor: pointer;
        }
        .category-chip:hover {
            background: var(--accent-gradient);
            border-color: transparent;
            color: #fff !important;
            transform: translateY(-2px);
        }

        .stButton > button {
            background: var(--accent-gradient) !important;
            color: #fff !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 0.625rem 1.5rem !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
        }
        .stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 24px var(--shadow-color) !important; }

        .stSelectbox > div > div, .stTextInput > div > div > input {
            background: var(--bg-card) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 10px !important;
            color: var(--text-primary) !important;
        }
        .stSelectbox > div > div:hover, .stTextInput > div > div > input:focus {
            border-color: var(--accent-primary) !important;
        }

        .stFileUploader > div {
            background: var(--bg-card) !important;
            border: 2px dashed var(--border-color) !important;
            border-radius: 14px !important;
        }
        .stFileUploader > div:hover { border-color: var(--accent-primary) !important; }

        .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; background: transparent; }
        .stTabs [data-baseweb="tab"] {
            background: var(--bg-card) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 10px !important;
            color: var(--text-secondary) !important;
            padding: 0.625rem 1.25rem !important;
        }
        .stTabs [aria-selected="true"] {
            background: var(--accent-gradient) !important;
            border: none !important;
            color: #fff !important;
        }

        .stProgress > div > div > div { background: var(--accent-gradient) !important; border-radius: 8px !important; }
        .stSlider > div > div > div > div { background: var(--accent-primary) !important; }

        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: var(--bg-secondary); }
        ::-webkit-scrollbar-thumb { background: var(--border-color); border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: var(--accent-primary); }

        .sidebar-logo { text-align: center; padding: 1.5rem 0; }
        .sidebar-logo-icon { width: 56px; height: 56px; background: var(--accent-gradient); border-radius: 14px; display: inline-flex; align-items: center; justify-content: center; margin-bottom: 0.75rem; }
        .sidebar-logo-icon svg { stroke: white; width: 28px; height: 28px; }
        .sidebar-title { font-size: 1.125rem !important; font-weight: 700 !important; color: var(--text-primary) !important; margin: 0; }
        .sidebar-subtitle { font-size: 0.75rem; color: var(--text-muted) !important; }
        .sidebar-stats { text-align: center; font-size: 0.8rem; color: var(--text-muted) !important; }
        .sidebar-stats strong { color: var(--text-secondary) !important; }

        /* Style Streamlit's native sidebar collapse button as hamburger */
        [data-testid="stSidebarCollapseButton"] {
            position: fixed !important;
            top: 12px !important;
            left: 12px !important;
            z-index: 999999 !important;
            width: 40px !important;
            height: 40px !important;
            background: var(--bg-card) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 10px !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.12) !important;
            transition: all 0.3s ease !important;
        }
        [data-testid="stSidebarCollapseButton"]:hover {
            background: var(--accent-primary) !important;
            border-color: var(--accent-primary) !important;
            transform: scale(1.05) !important;
        }
        [data-testid="stSidebarCollapseButton"] svg {
            stroke: var(--text-secondary) !important;
            transition: all 0.3s ease !important;
        }
        [data-testid="stSidebarCollapseButton"]:hover svg {
            stroke: white !important;
        }

        /* Style the expand button when sidebar is collapsed */
        [data-testid="stSidebarCollapsedControl"] {
            position: fixed !important;
            top: 12px !important;
            left: 12px !important;
            z-index: 999999 !important;
        }
        [data-testid="stSidebarCollapsedControl"] > div {
            width: 40px !important;
            height: 40px !important;
            background: var(--bg-card) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 10px !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.12) !important;
            transition: all 0.3s ease !important;
        }
        [data-testid="stSidebarCollapsedControl"] > div:hover {
            background: var(--accent-primary) !important;
            border-color: var(--accent-primary) !important;
            transform: scale(1.05) !important;
        }
        [data-testid="stSidebarCollapsedControl"] svg {
            stroke: var(--text-secondary) !important;
        }
        [data-testid="stSidebarCollapsedControl"] > div:hover svg {
            stroke: white !important;
        }

        /* Sidebar smooth transition */
        section[data-testid="stSidebar"] {
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }
        </style>
        """
    else:
        css = """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

        :root {
            --bg-primary: #fafbfc;
            --bg-secondary: #ffffff;
            --bg-card: #ffffff;
            --bg-card-hover: #f5f7fa;
            --text-primary: #111827;
            --text-secondary: #4b5563;
            --text-muted: #9ca3af;
            --accent-primary: #6366f1;
            --accent-secondary: #818cf8;
            --accent-gradient: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
            --border-color: #e5e7eb;
            --shadow-color: rgba(99, 102, 241, 0.08);
            --success: #22c55e;
            --warning: #eab308;
            --error: #ef4444;
        }

        * { font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important; }

        .stApp {
            background: var(--bg-primary);
            background-image:
                radial-gradient(ellipse at 0% 0%, rgba(99, 102, 241, 0.04) 0%, transparent 50%),
                radial-gradient(ellipse at 100% 100%, rgba(139, 92, 246, 0.03) 0%, transparent 50%);
        }

        #MainMenu, footer, header { visibility: hidden; }

        section[data-testid="stSidebar"] {
            background: var(--bg-secondary);
            border-right: 1px solid var(--border-color);
        }

        h1, h2, h3, h4, h5, h6 { color: var(--text-primary) !important; font-weight: 600 !important; }
        p, span, label, div, .stMarkdown { color: var(--text-secondary) !important; }

        .hero-section {
            background: var(--accent-gradient);
            border-radius: 20px;
            padding: 2.5rem 3rem;
            margin-bottom: 2rem;
            position: relative;
            overflow: hidden;
            box-shadow: 0 10px 40px var(--shadow-color);
        }

        .hero-section::after {
            content: '';
            position: absolute;
            top: 0; right: 0; bottom: 0; left: 0;
            background: linear-gradient(135deg, rgba(255,255,255,0.15) 0%, transparent 50%);
            pointer-events: none;
        }

        .hero-section h1 { color: #fff !important; font-size: 2rem !important; font-weight: 700 !important; margin: 0 0 0.5rem 0; }
        .hero-section p { color: rgba(255,255,255,0.9) !important; margin: 0; font-size: 1rem; }
        .hero-icon { display: inline-flex; align-items: center; justify-content: center; width: 48px; height: 48px; background: rgba(255,255,255,0.2); border-radius: 12px; margin-bottom: 1rem; }
        .hero-icon svg { stroke: white; }

        .glass-card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1.25rem;
            transition: all 0.3s ease;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        }

        .glass-card:hover {
            border-color: var(--accent-primary);
            box-shadow: 0 8px 32px var(--shadow-color);
            transform: translateY(-2px);
        }

        .card-header { display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1rem; }
        .card-header svg { stroke: var(--accent-primary); }
        .card-header h3 { margin: 0 !important; font-size: 1rem !important; color: var(--text-primary) !important; }

        .image-card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 14px;
            overflow: hidden;
            transition: all 0.3s ease;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        }

        .image-card:hover {
            transform: translateY(-6px);
            box-shadow: 0 16px 40px var(--shadow-color);
            border-color: var(--accent-primary);
        }

        .image-info { padding: 1rem; text-align: center; }
        .image-title { font-weight: 600; color: var(--text-primary) !important; font-size: 0.875rem; margin-bottom: 0.5rem; }

        .similarity-badge {
            display: inline-flex; align-items: center; gap: 0.375rem;
            background: var(--accent-gradient);
            color: #fff !important;
            padding: 0.375rem 0.875rem;
            border-radius: 50px;
            font-size: 0.75rem;
            font-weight: 600;
        }

        .metric-card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 14px;
            padding: 1.25rem;
            text-align: center;
            transition: all 0.3s ease;
        }

        .metric-card:hover { transform: translateY(-3px); box-shadow: 0 12px 28px var(--shadow-color); }
        .metric-value { font-size: 2rem; font-weight: 700; }
        .metric-value.excellent { color: var(--success) !important; }
        .metric-value.good { color: var(--warning) !important; }
        .metric-value.poor { color: var(--error) !important; }
        .metric-label { color: var(--text-muted) !important; font-size: 0.8rem; font-weight: 500; margin-top: 0.25rem; }

        .stats-container { display: flex; gap: 1rem; margin-bottom: 1.25rem; }
        .stat-box {
            flex: 1;
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 14px;
            padding: 1.25rem;
            text-align: center;
            transition: all 0.3s ease;
        }
        .stat-box:hover { border-color: var(--accent-primary); box-shadow: 0 8px 24px var(--shadow-color); }
        .stat-value { font-size: 1.5rem; font-weight: 700; color: var(--accent-primary) !important; }
        .stat-label { font-size: 0.75rem; color: var(--text-muted) !important; margin-top: 0.25rem; text-transform: uppercase; letter-spacing: 0.5px; }

        .chip-container { display: flex; flex-wrap: wrap; gap: 0.5rem; }
        .category-chip {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            color: var(--text-secondary) !important;
            padding: 0.5rem 1rem;
            border-radius: 50px;
            font-size: 0.8rem;
            font-weight: 500;
            transition: all 0.25s ease;
            cursor: pointer;
        }
        .category-chip:hover {
            background: var(--accent-gradient);
            border-color: transparent;
            color: #fff !important;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px var(--shadow-color);
        }

        .stButton > button {
            background: var(--accent-gradient) !important;
            color: #fff !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 0.625rem 1.5rem !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 2px 8px var(--shadow-color) !important;
        }
        .stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 24px var(--shadow-color) !important; }

        .stSelectbox > div > div, .stTextInput > div > div > input {
            background: var(--bg-card) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 10px !important;
            color: var(--text-primary) !important;
        }
        .stSelectbox > div > div:hover, .stTextInput > div > div > input:focus {
            border-color: var(--accent-primary) !important;
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.08) !important;
        }

        .stFileUploader > div {
            background: var(--bg-card) !important;
            border: 2px dashed var(--border-color) !important;
            border-radius: 14px !important;
        }
        .stFileUploader > div:hover { border-color: var(--accent-primary) !important; background: var(--bg-card-hover) !important; }

        .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; background: transparent; }
        .stTabs [data-baseweb="tab"] {
            background: var(--bg-card) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 10px !important;
            color: var(--text-secondary) !important;
            padding: 0.625rem 1.25rem !important;
        }
        .stTabs [aria-selected="true"] {
            background: var(--accent-gradient) !important;
            border: none !important;
            color: #fff !important;
        }

        .stProgress > div > div > div { background: var(--accent-gradient) !important; border-radius: 8px !important; }
        .stSlider > div > div > div > div { background: var(--accent-primary) !important; }

        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: var(--bg-primary); }
        ::-webkit-scrollbar-thumb { background: var(--border-color); border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: var(--accent-primary); }

        .sidebar-logo { text-align: center; padding: 1.5rem 0; }
        .sidebar-logo-icon { width: 56px; height: 56px; background: var(--accent-gradient); border-radius: 14px; display: inline-flex; align-items: center; justify-content: center; margin-bottom: 0.75rem; box-shadow: 0 4px 12px var(--shadow-color); }
        .sidebar-logo-icon svg { stroke: white; width: 28px; height: 28px; }
        .sidebar-title { font-size: 1.125rem !important; font-weight: 700 !important; color: var(--text-primary) !important; margin: 0; }
        .sidebar-subtitle { font-size: 0.75rem; color: var(--text-muted) !important; }
        .sidebar-stats { text-align: center; font-size: 0.8rem; color: var(--text-muted) !important; }
        .sidebar-stats strong { color: var(--text-secondary) !important; }

        /* Style Streamlit's native sidebar collapse button as hamburger */
        [data-testid="stSidebarCollapseButton"] {
            position: fixed !important;
            top: 12px !important;
            left: 12px !important;
            z-index: 999999 !important;
            width: 40px !important;
            height: 40px !important;
            background: var(--bg-card) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 10px !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
            transition: all 0.3s ease !important;
        }
        [data-testid="stSidebarCollapseButton"]:hover {
            background: var(--accent-primary) !important;
            border-color: var(--accent-primary) !important;
            transform: scale(1.05) !important;
        }
        [data-testid="stSidebarCollapseButton"] svg {
            stroke: var(--text-secondary) !important;
            transition: all 0.3s ease !important;
        }
        [data-testid="stSidebarCollapseButton"]:hover svg {
            stroke: white !important;
        }

        /* Style the expand button when sidebar is collapsed */
        [data-testid="stSidebarCollapsedControl"] {
            position: fixed !important;
            top: 12px !important;
            left: 12px !important;
            z-index: 999999 !important;
        }
        [data-testid="stSidebarCollapsedControl"] > div {
            width: 40px !important;
            height: 40px !important;
            background: var(--bg-card) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 10px !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
            transition: all 0.3s ease !important;
        }
        [data-testid="stSidebarCollapsedControl"] > div:hover {
            background: var(--accent-primary) !important;
            border-color: var(--accent-primary) !important;
            transform: scale(1.05) !important;
        }
        [data-testid="stSidebarCollapsedControl"] svg {
            stroke: var(--text-secondary) !important;
        }
        [data-testid="stSidebarCollapsedControl"] > div:hover svg {
            stroke: white !important;
        }

        /* Sidebar smooth transition */
        section[data-testid="stSidebar"] {
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }
        </style>
        """

    st.markdown(css, unsafe_allow_html=True)

# ==================== HELPER FUNCTIONS ====================

def fine_tune_model(model, param_grid, X_train, Y_train):
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X_train, Y_train)
    return grid_search.best_estimator_

def get_model(classifier, use_grid_search, X_train, Y_train):
    models = {"LDA": lambda: LinearDiscriminantAnalysis(), "Naive Bayes": lambda: GaussianNB()}
    grid_models = {
        "KNN": (KNeighborsClassifier(), {'n_neighbors': list(range(1, 30)), 'p': [1, 2]}, KNeighborsClassifier(n_neighbors=10)),
        "SVM": (SVC(), {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf', 'linear']}, SVC(C=2.5, max_iter=5000)),
        "Random Forest": (RandomForestClassifier(), {'n_estimators': [10, 50, 100, 200], 'max_features': ['sqrt', 'log2'], 'max_depth': [None, 10, 20, 30]}, RandomForestClassifier()),
        "AdaBoost": (AdaBoostClassifier(), {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1]}, AdaBoostClassifier()),
        "Decision Tree": (DecisionTreeClassifier(), {'criterion': ['gini', 'entropy'], 'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]}, DecisionTreeClassifier()),
    }
    if classifier in models: return models[classifier]()
    if classifier in grid_models:
        base, param_grid, default = grid_models[classifier]
        return fine_tune_model(base, param_grid, X_train, Y_train) if use_grid_search else default
    return LinearDiscriminantAnalysis()

def apply_transform(X, transform_choice):
    scalers = {'Rescale': MinMaxScaler(), 'Normalization': Normalizer(), 'Standardization': StandardScaler()}
    return scalers[transform_choice].fit_transform(X) if transform_choice in scalers else X

def get_signatures(descriptor_choice):
    return {'GLCM': signatures_glcm, 'BIT': signatures_bitdesc, 'HARALICK': signatures_haralick, 'BIT_GLCM_HARALICK': signatures_combined}.get(descriptor_choice, signatures_combined)

def extract_features(image, descriptor_choice):
    return descriptor_functions[descriptor_choice](image)

# ==================== UI COMPONENTS ====================

def render_hero(title, subtitle, icon_name="search"):
    st.markdown(f'''
    <div class="hero-section">
        <div class="hero-icon">{icon(icon_name, 28)}</div>
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    ''', unsafe_allow_html=True)

def render_card_header(title, icon_name):
    return f'<div class="card-header">{icon(icon_name, 20)}<h3>{title}</h3></div>'

def render_stats(images_count, categories_count):
    st.markdown(f'''
    <div class="stats-container">
        <div class="stat-box"><div class="stat-value">{images_count:,}</div><div class="stat-label">Images</div></div>
        <div class="stat-box"><div class="stat-value">{categories_count}</div><div class="stat-label">Categories</div></div>
    </div>
    ''', unsafe_allow_html=True)

def render_metric(label, value, col):
    cls = "excellent" if value > 0.9 else "good" if value > 0.7 else "poor"
    col.markdown(f'<div class="metric-card"><div class="metric-value {cls}">{value:.1%}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

def render_image_card(img, title, similarity=None, col=None):
    target = col if col else st
    with target:
        st.markdown('<div class="image-card">', unsafe_allow_html=True)
        st.image(img, use_container_width=True)
        badge = f'<span class="similarity-badge">{icon("target", 12)} {similarity:.1%}</span>' if similarity else ''
        st.markdown(f'<div class="image-info"><div class="image-title">{title}</div>{badge}</div></div>', unsafe_allow_html=True)

# ==================== PAGES ====================

def page_basic_search():
    render_hero("Visual Search", "Find similar images using advanced feature descriptors", "search")
    if signatures_glcm is None: return

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown(f'<div class="glass-card">{render_card_header("Configuration", "sliders")}', unsafe_allow_html=True)
        descriptor = st.selectbox("Feature Descriptor", ["GLCM", "BIT", "HARALICK", "BIT_GLCM_HARALICK"], index=["GLCM", "BIT", "HARALICK", "BIT_GLCM_HARALICK"].index(st.session_state.default_descriptor))
        distance = st.selectbox("Distance Metric", ["Manhattan", "Euclidean", "Chebyshev", "Canberra"], index=["Manhattan", "Euclidean", "Chebyshev", "Canberra"].index(st.session_state.default_distance))
        count = st.slider("Results", 2, 12, st.session_state.default_image_count)
        st.markdown('</div>', unsafe_allow_html=True)
        render_stats(len(get_signatures(descriptor)) if get_signatures(descriptor) is not None else 0, len(get_categories()))

    with col2:
        st.markdown(f'<div class="glass-card">{render_card_header("Upload Image", "upload")}', unsafe_allow_html=True)
        uploaded = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
        if uploaded:
            img = Image.open(uploaded)
            st.image(img, caption="Query Image", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if uploaded:
        with st.spinner("Finding similar images..."):
            img_array = np.array(img)
            img_array = cv2.resize(img_array, (256, 256))
            if len(img_array.shape) == 3: img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            features = extract_features(img_array, descriptor)
            signatures = get_signatures(descriptor)
            distances = [(distance_functions[distance](features, np.array(sig[:-3], dtype=float)), sig[-3], sig[-2], sig[-1]) for sig in signatures]
            distances.sort(key=lambda x: x[0])

        st.markdown(f'<div style="margin: 1.5rem 0 1rem;">{render_card_header("Similar Images", "target")}</div>', unsafe_allow_html=True)
        cols = st.columns(min(4, count))
        for i in range(count):
            d, path, folder, _ = distances[i]
            try: render_image_card(Image.open(os.path.join("images", *path.replace("\\", "/").split("/"))), folder, 1/(1+d), cols[i % len(cols)])
            except: cols[i % len(cols)].error("Image not found")

def page_ml_search():
    render_hero("ML-Powered Search", "Classify and search using machine learning", "brain")
    if signatures_glcm is None: return

    # Configuration section
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f'<div class="glass-card">{render_card_header("Features", "layers")}', unsafe_allow_html=True)
        descriptor = st.selectbox("Descriptor", ["GLCM", "BIT", "HARALICK", "BIT_GLCM_HARALICK"], key="ml_desc")
        distance = st.selectbox("Distance", ["Manhattan", "Euclidean", "Chebyshev", "Canberra"], key="ml_dist")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown(f'<div class="glass-card">{render_card_header("Model", "cpu")}', unsafe_allow_html=True)
        classifier = st.selectbox("Algorithm", ["LDA", "KNN", "Naive Bayes", "Decision Tree", "SVM", "Random Forest", "AdaBoost"])
        transform = st.selectbox("Transform", ["No Transform", "Rescale", "Normalization", "Standardization"])
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown(f'<div class="glass-card">{render_card_header("Options", "sliders")}', unsafe_allow_html=True)
        grid_search = st.toggle("Hyperparameter Tuning", False)
        count = st.slider("Results", 2, 12, 6, key="ml_count")
        st.markdown('</div>', unsafe_allow_html=True)

    # Upload section
    st.markdown(f'<div class="glass-card">{render_card_header("Upload Image", "upload")}', unsafe_allow_html=True)
    uploaded = st.file_uploader("Choose image", type=["jpg", "png", "jpeg"], key="ml_upload")
    st.markdown('</div>', unsafe_allow_html=True)

    # Results section
    if uploaded:
        image = Image.open(uploaded)

        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown(f'<div class="glass-card">{render_card_header("Query", "image")}', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            with st.spinner("Processing..."):
                # Prepare image
                img_array = np.array(image)
                img_array = cv2.resize(img_array, (256, 256))
                if len(img_array.shape) == 3:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

                # Extract features
                features = extract_features(img_array, descriptor)
                signatures = get_signatures(descriptor)

                # Prepare data
                X = np.array([s[:-3] for s in signatures], dtype=float)
                Y = np.array([s[-1] for s in signatures], dtype=int)

                # Clean data
                X = np.where(np.isinf(X), np.nan, X)
                col_means = np.nanmean(X, axis=0)
                inds = np.where(np.isnan(X))
                X[inds] = np.take(col_means, inds[1])
                X = apply_transform(X, transform)

                # Train model
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=10)
                model = get_model(classifier, grid_search, X_train, Y_train)
                model.fit(X_train, Y_train)

                # Evaluate
                Y_pred = model.predict(X_test)
                metrics = {
                    'Accuracy': accuracy_score(Y_test, Y_pred),
                    'Precision': precision_score(Y_test, Y_pred, average='weighted', zero_division=0),
                    'Recall': recall_score(Y_test, Y_pred, average='weighted', zero_division=0),
                    'F1 Score': f1_score(Y_test, Y_pred, average='weighted', zero_division=0)
                }

                # Predict
                feat_flat = np.array(features).flatten().reshape(1, -1)
                feat_flat = apply_transform(feat_flat, transform)
                prediction = model.predict(feat_flat)[0]

            # Display metrics
            st.markdown(f'<div style="margin-bottom: 1rem;">{render_card_header("Performance", "chart")}</div>', unsafe_allow_html=True)
            metric_cols = st.columns(4)
            for (label, value), mcol in zip(metrics.items(), metric_cols):
                render_metric(label, value, mcol)

            # Display prediction
            folder_name = [s[-2] for s in signatures if s[-1] == prediction]
            folder_name = folder_name[0] if folder_name else "Unknown"
            st.markdown(f'''<div class="glass-card" style="text-align: center;">
                {render_card_header("Predicted Category", "tag")}
                <div style="font-size: 1.75rem; font-weight: 700; color: var(--accent-primary);">{folder_name}</div>
            </div>''', unsafe_allow_html=True)

        # Similar images
        indices = [i for i, y in enumerate(Y) if y == prediction]
        dists = []
        dist_func = distance_functions[distance]
        for idx in indices:
            feat = np.array(signatures[idx][:-3], dtype=float).flatten()
            d = dist_func(feat_flat.flatten(), feat)
            dists.append((d, signatures[idx][-3], signatures[idx][-2]))
        dists.sort(key=lambda x: x[0])

        st.markdown(f'<div style="margin: 1.5rem 0 1rem;">{render_card_header(f"Similar in {folder_name}", "target")}</div>', unsafe_allow_html=True)
        cols = st.columns(min(4, count))
        for i in range(min(count, len(dists))):
            d, path, name = dists[i]
            img_path = os.path.join("images", *path.replace("\\", "/").split("/"))
            try:
                sim_img = Image.open(img_path)
                render_image_card(sim_img, name, 1/(1+d), cols[i % len(cols)])
            except:
                pass

def page_multimodal():
    render_hero("Multimodal Search", "Search images by category or keywords", "layers")
    if signatures_combined is None: return

    categories = get_categories()
    col1, col2 = st.columns([3, 1])
    with col1: search = st.text_input("Search", placeholder="Enter category name...", label_visibility="collapsed")
    with col2: count = st.slider("Results", 4, 24, 12, label_visibility="collapsed")

    if search:
        matched = [s for s in signatures_combined if search.lower() in s[-2].lower()]
        if matched:
            st.success(f"Found {len(matched)} images matching '{search}'")
            cols = st.columns(min(4, count))
            for i, sig in enumerate(matched[:count]):
                try: render_image_card(Image.open(os.path.join("images", *sig[-3].replace("\\", "/").split("/"))), sig[-2], col=cols[i % len(cols)])
                except: pass
        else:
            st.warning(f"No matches for '{search}'")
            st.markdown(f'<div style="margin: 1rem 0;">{render_card_header("Available Categories", "folder")}</div>', unsafe_allow_html=True)
            chips = "".join([f'<span class="category-chip">{c}</span>' for c in categories[:30]])
            st.markdown(f'<div class="chip-container">{chips}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="margin: 1rem 0;">{render_card_header("Browse Categories", "folder")}</div>', unsafe_allow_html=True)
        chips = "".join([f'<span class="category-chip">{c}</span>' for c in categories])
        st.markdown(f'<div class="chip-container">{chips}</div>', unsafe_allow_html=True)

def page_settings():
    render_hero("Settings", "Customize your experience", "settings")

    tab1, tab2, tab3 = st.tabs(["Theme", "Defaults", "About"])

    with tab1:
        st.markdown(f'<div class="glass-card">{render_card_header("Appearance", "palette")}', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 3])
        with col1:
            dark = st.toggle("Dark Mode", st.session_state.night_mode)
            if dark != st.session_state.night_mode:
                st.session_state.night_mode = dark
                st.rerun()
        with col2:
            theme_icon = icon("moon", 18) if st.session_state.night_mode else icon("sun", 18)
            theme_text = "Dark mode - easier on the eyes" if st.session_state.night_mode else "Light mode - bright and clean"
            st.markdown(f'<div style="display: flex; align-items: center; gap: 0.5rem;">{theme_icon}<span>{theme_text}</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown(f'<div class="glass-card">{render_card_header("Default Settings", "sliders")}', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            desc = st.selectbox("Default Descriptor", ["GLCM", "BIT", "HARALICK", "BIT_GLCM_HARALICK"], index=["GLCM", "BIT", "HARALICK", "BIT_GLCM_HARALICK"].index(st.session_state.default_descriptor))
            if desc != st.session_state.default_descriptor: st.session_state.default_descriptor = desc
            dist = st.selectbox("Default Distance", ["Manhattan", "Euclidean", "Chebyshev", "Canberra"], index=["Manhattan", "Euclidean", "Chebyshev", "Canberra"].index(st.session_state.default_distance))
            if dist != st.session_state.default_distance: st.session_state.default_distance = dist
        with col2:
            cnt = st.slider("Default Results", 2, 12, st.session_state.default_image_count)
            if cnt != st.session_state.default_image_count: st.session_state.default_image_count = cnt
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="glass-card">{render_card_header("Cache", "trash")}', unsafe_allow_html=True)
        if st.button("Clear Cache"):
            st.cache_data.clear()
            st.success("Cache cleared!")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown(f'''<div class="glass-card" style="text-align: center;">
            <div style="width: 64px; height: 64px; background: var(--accent-gradient); border-radius: 16px; display: inline-flex; align-items: center; justify-content: center; margin-bottom: 1rem;">{icon("search", 32, "white")}</div>
            <h2 style="margin: 0 0 0.5rem 0;">CBIR System v2.0</h2>
            <p>Content-Based Image Retrieval with Machine Learning</p>
            <p style="margin-top: 1rem; opacity: 0.6;">Built with Streamlit by Koceila Djaballah</p>
        </div>''', unsafe_allow_html=True)
        categories = get_categories()
        st.markdown(f'''<div class="stats-container">
            <div class="stat-box"><div class="stat-value">{len(signatures_glcm) if signatures_glcm is not None else 0:,}</div><div class="stat-label">Images</div></div>
            <div class="stat-box"><div class="stat-value">{len(categories)}</div><div class="stat-label">Categories</div></div>
            <div class="stat-box"><div class="stat-value">4</div><div class="stat-label">Descriptors</div></div>
            <div class="stat-box"><div class="stat-value">7</div><div class="stat-label">ML Models</div></div>
        </div>''', unsafe_allow_html=True)

# ==================== MAIN ====================

def main():
    apply_modern_css()

    with st.sidebar:
        st.markdown(f'''<div class="sidebar-logo">
            <div class="sidebar-logo-icon">{icon("search", 28)}</div>
            <div class="sidebar-title">CBIR System</div>
            <div class="sidebar-subtitle">Image Retrieval</div>
        </div>''', unsafe_allow_html=True)

        st.markdown("---")
        page = st.radio("", ["Visual Search", "ML Search", "Multimodal", "Settings"], label_visibility="collapsed")
        st.markdown("---")

        categories = get_categories()
        st.markdown(f'''<div class="sidebar-stats">
            <p><strong>{len(signatures_glcm) if signatures_glcm is not None else 0:,}</strong> images</p>
            <p><strong>{len(categories)}</strong> categories</p>
        </div>''', unsafe_allow_html=True)

    {"Visual Search": page_basic_search, "ML Search": page_ml_search, "Multimodal": page_multimodal, "Settings": page_settings}[page]()

if __name__ == '__main__':
    main()
