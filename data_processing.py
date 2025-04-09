import cv2
import os
import numpy as np
from descriptor import glcm, bitdesc, haralick_feat, bit_glcm_haralick

# List of descriptors
descriptors = [glcm, bitdesc, haralick_feat, bit_glcm_haralick]

def process_datasets(root_folder, descriptors):
    all_features = {descriptor.__name__: [] for descriptor in descriptors}  # Dictionary to store features for each descriptor
    class_list = {}  # Dictionary to store class labels
    class_counter = 1  # Counter for assigning class labels

    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                relative_path = os.path.relpath(os.path.join(root, file), root_folder)
                relative_path = str(relative_path)  # Ensure relative_path is a string
                file_name = f'{relative_path.split(os.sep)[0]}_{file}'  
                image_rel_path = os.path.join(root, file)
                folder_name = os.path.basename(os.path.dirname(image_rel_path))

                print(f"Processing file: {image_rel_path}")

                img = cv2.imread(image_rel_path, 0)
                if img is not None:
                    for descriptor in descriptors:
                        features = descriptor(img)
                        if features is not None:
                            if folder_name not in class_list:
                                class_list[folder_name] = class_counter
                                class_counter += 1
                            features = features + [relative_path,folder_name,class_list[folder_name]]
                            all_features[descriptor.__name__].append(features)
                else:
                    print(f"Failed to read image: {image_rel_path}")

    # Convert lists to numpy arrays and save
    for descriptor in descriptors:
        descriptor_name = descriptor.__name__
        signatures = np.array(all_features[descriptor_name], dtype=object)
        np.save(f'signatures_{descriptor_name}.npy', signatures)

    print('Class List:', class_list)
    print('Successfully stored!')

# Process datasets
process_datasets('./images', descriptors)
