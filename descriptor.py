from skimage.feature import graycomatrix, graycoprops
from BiT import bio_taxo
from mahotas.features import haralick
import cv2
import numpy as np


def haralick_feat(data):
    return haralick(data).mean(0).tolist()    

def haralick_feat_beta(image_path):
    data = cv2.imread(image_path, 0)
    return haralick(data).mean(0).tolist() 
   
def glcm(data):
    
    co_matrix = graycomatrix(data, [1], [np.pi/4], None,symmetric=False, normed=False )
    dissimilarity = graycoprops(co_matrix, 'dissimilarity')[0, 0]
    cont = graycoprops(co_matrix, 'contrast')[0, 0]
    corr = graycoprops(co_matrix, 'correlation')[0, 0]
    ener = graycoprops(co_matrix, 'energy')[0, 0]
    asm = graycoprops(co_matrix, 'ASM')[0, 0]
    homo = graycoprops(co_matrix, 'homogeneity')[0, 0]
    return [dissimilarity, cont, corr, ener, asm, homo]

def glcm_beta(image_path):
    data = cv2.imread(image_path, 0)
    co_matrix = graycomatrix(data, [1], [np.pi/4], None,symmetric=False, normed=False )
    dissimilarity = graycoprops(co_matrix, 'dissimilarity')[0, 0]
    cont = graycoprops(co_matrix, 'contrast')[0, 0]
    corr = graycoprops(co_matrix, 'correlation')[0, 0]
    ener = graycoprops(co_matrix, 'energy')[0, 0]
    asm = graycoprops(co_matrix, 'ASM')[0, 0]
    homo = graycoprops(co_matrix, 'homogeneity')[0, 0]
    return [dissimilarity, cont, corr, ener, asm, homo]

def bitdesc(data):
    if data is None or data.size == 0:
        print("Error: Empty image data provided to bitdesc.")
        return []
    return bio_taxo(data)



def bitdesc_(image_path):
    data = cv2.imread(image_path, 0)
    return bio_taxo(data)

def bit_glcm_haralick(data):
    return bitdesc(data) + glcm(data) + haralick_feat(data)

def imagePyramid(image_path: str, levels: int):
    pyramids = []
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Image not loaded properly.")
        pyramids.append(img)
        print(f'Initial image: {img.shape}')
        for i in range(0, levels):
            pyr_level = cv2.pyrDown(pyramids[i])
            print(f'Level {i} shape: {pyr_level.shape}')
            pyramids.append(pyr_level)
        return pyramids
    except Exception as e:
        print(f'Error: {e}')
        return []

        
def features_concat(pyramids: list):
    if len(pyramids) == 4:
        try:
            l0_feat = haralick_feat(cv2.cvtColor(pyramids[0], cv2.COLOR_BGR2GRAY))
            l1_feat = glcm(cv2.cvtColor(pyramids[1], cv2.COLOR_BGR2GRAY))
            l2_feat = bitdesc(cv2.cvtColor(pyramids[2], cv2.COLOR_BGR2GRAY))
            l3_feat = bit_glcm_haralick(cv2.cvtColor(pyramids[3], cv2.COLOR_BGR2GRAY))
            return l0_feat + l1_feat + l2_feat + l3_feat
        except Exception as e:
            print(f"Error in feature extraction: {e}")
            return []
    else:
        return []


def features_concat_beta(pyramids: list):
    descr_list = [haralick_feat, glcm, bitdesc, bit_glcm_haralick]
    all_features = []
    if len(pyramids) == 4:
        for i, desc in enumerate(descr_list):
            all_features = all_features + desc(cv2.cvtColor(pyramids[i-1], cv2.COLOR_BGR2GRAY))
        return all_features
    else:
        return False