from skimage.feature import graycomatrix, graycoprops
from mahotas.features import haralick
import cv2
import numpy as np

# Try to import Bitdesc, fallback to custom implementation
try:
    from Bit import bio_taxo as _bio_taxo
    BITDESC_AVAILABLE = True
except ImportError:
    BITDESC_AVAILABLE = False

def _custom_bio_taxo(data):
    """
    Custom implementation of bio-inspired texture descriptor.
    Computes biodiversity and taxonomic indices from image data.
    Returns 14 features (7 biodiversity + 7 taxonomic indices).
    """
    if data is None or data.size == 0:
        return [0.0] * 14

    # Flatten and get histogram
    pixels = data.flatten().astype(float)
    hist, _ = np.histogram(pixels, bins=256, range=(0, 256))
    hist = hist.astype(float)
    hist = hist[hist > 0]  # Remove zero counts

    if len(hist) == 0:
        return [0.0] * 14

    # Total number of pixels
    N = np.sum(hist)
    # Number of species (unique intensity levels)
    S = len(hist)
    # Proportions
    p = hist / N

    # === Biodiversity Indices (7) ===

    # 1. Shannon Index (H')
    shannon = -np.sum(p * np.log(p + 1e-10))

    # 2. Simpson Index (D)
    simpson = np.sum(p ** 2)

    # 3. Margalef Index
    margalef = (S - 1) / (np.log(N + 1) + 1e-10)

    # 4. Menhinick Index
    menhinick = S / (np.sqrt(N) + 1e-10)

    # 5. Equitability (Pielou's J)
    H_max = np.log(S + 1e-10)
    equitability = shannon / (H_max + 1e-10) if H_max > 0 else 0

    # 6. Berger-Parker Index
    berger_parker = np.max(p)

    # 7. Fisher Alpha (approximation)
    fisher_alpha = S / (np.log(N / S + 1) + 1e-10) if S > 0 else 0

    # === Taxonomic Indices (7) ===

    # Use co-occurrence for taxonomic relationships
    if len(data.shape) == 2:
        h, w = data.shape
        if h > 1 and w > 1:
            # Compute local variance as taxonomic distinctness proxy
            kernel = np.ones((3, 3)) / 9
            local_mean = cv2.filter2D(data.astype(float), -1, kernel)
            local_var = cv2.filter2D((data.astype(float) - local_mean) ** 2, -1, kernel)

            # 8. Average Taxonomic Distinctness (Delta+)
            avg_tax_dist = np.mean(local_var)

            # 9. Variation in Taxonomic Distinctness (Lambda+)
            var_tax_dist = np.var(local_var)

            # 10. Total Taxonomic Distinctness (sDelta+)
            total_tax_dist = np.sum(local_var) / (N + 1e-10)

            # 11. Taxonomic Diversity (Delta)
            tax_diversity = np.std(pixels)

            # 12. Phylogenetic Diversity
            gradient_x = cv2.Sobel(data, cv2.CV_64F, 1, 0, ksize=3)
            gradient_y = cv2.Sobel(data, cv2.CV_64F, 0, 1, ksize=3)
            phylo_div = np.mean(np.sqrt(gradient_x**2 + gradient_y**2))

            # 13. Taxonomic Evenness
            tax_evenness = 1 - simpson

            # 14. Quadratic Entropy
            quad_entropy = np.sum(p * (1 - p))
        else:
            avg_tax_dist = var_tax_dist = total_tax_dist = 0
            tax_diversity = phylo_div = tax_evenness = quad_entropy = 0
    else:
        avg_tax_dist = var_tax_dist = total_tax_dist = 0
        tax_diversity = phylo_div = tax_evenness = quad_entropy = 0

    features = [
        shannon, simpson, margalef, menhinick, equitability, berger_parker, fisher_alpha,
        avg_tax_dist, var_tax_dist, total_tax_dist, tax_diversity, phylo_div, tax_evenness, quad_entropy
    ]

    # Normalize features to avoid extreme values
    features = [float(f) if np.isfinite(f) else 0.0 for f in features]

    return features

def bio_taxo(data):
    """Bio-inspired texture descriptor - uses Bitdesc if available, else custom implementation."""
    if BITDESC_AVAILABLE:
        try:
            return _bio_taxo(data)
        except Exception:
            return _custom_bio_taxo(data)
    return _custom_bio_taxo(data)

def haralick_feat(data):
    """Extract Haralick texture features."""
    try:
        return haralick(data).mean(0).tolist()
    except Exception:
        return [0.0] * 13

def haralick_feat_beta(image_path):
    data = cv2.imread(image_path, 0)
    return haralick_feat(data)

def glcm(data):
    """Extract GLCM texture features."""
    try:
        co_matrix = graycomatrix(data, [1], [np.pi/4], None, symmetric=False, normed=False)
        dissimilarity = graycoprops(co_matrix, 'dissimilarity')[0, 0]
        cont = graycoprops(co_matrix, 'contrast')[0, 0]
        corr = graycoprops(co_matrix, 'correlation')[0, 0]
        ener = graycoprops(co_matrix, 'energy')[0, 0]
        asm = graycoprops(co_matrix, 'ASM')[0, 0]
        homo = graycoprops(co_matrix, 'homogeneity')[0, 0]
        return [dissimilarity, cont, corr, ener, asm, homo]
    except Exception:
        return [0.0] * 6

def glcm_beta(image_path):
    data = cv2.imread(image_path, 0)
    return glcm(data)

def bitdesc(data):
    """Bio-inspired texture descriptor."""
    if data is None or data.size == 0:
        return [0.0] * 14
    return bio_taxo(data)

def bitdesc_(image_path):
    data = cv2.imread(image_path, 0)
    return bio_taxo(data)

def bit_glcm_haralick(data):
    """Combined descriptor: BIT + GLCM + Haralick."""
    return bitdesc(data) + glcm(data) + haralick_feat(data)

def imagePyramid(image_path: str, levels: int):
    """Create image pyramid."""
    pyramids = []
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Image not loaded properly.")
        pyramids.append(img)
        for i in range(levels):
            pyr_level = cv2.pyrDown(pyramids[i])
            pyramids.append(pyr_level)
        return pyramids
    except Exception:
        return []

def features_concat(pyramids: list):
    """Concatenate features from image pyramid."""
    if len(pyramids) == 4:
        try:
            l0_feat = haralick_feat(cv2.cvtColor(pyramids[0], cv2.COLOR_BGR2GRAY))
            l1_feat = glcm(cv2.cvtColor(pyramids[1], cv2.COLOR_BGR2GRAY))
            l2_feat = bitdesc(cv2.cvtColor(pyramids[2], cv2.COLOR_BGR2GRAY))
            l3_feat = bit_glcm_haralick(cv2.cvtColor(pyramids[3], cv2.COLOR_BGR2GRAY))
            return l0_feat + l1_feat + l2_feat + l3_feat
        except Exception:
            return []
    return []

def features_concat_beta(pyramids: list):
    """Alternative feature concatenation."""
    descr_list = [haralick_feat, glcm, bitdesc, bit_glcm_haralick]
    all_features = []
    if len(pyramids) == 4:
        for i, desc in enumerate(descr_list):
            all_features = all_features + desc(cv2.cvtColor(pyramids[i-1], cv2.COLOR_BGR2GRAY))
        return all_features
    return []
