import os

# Create arrays for each class based on filenames in data folders
def create_data_arrays():
    data_arrays = {
        'corn': [],
        'grape': [],
        'non_leaf': []
    }

    # Corn images
    corn_dir = 'data/corn'
    if os.path.exists(corn_dir):
        data_arrays['corn'] = [f for f in os.listdir(corn_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    # Grape images
    grape_dir = 'data/grape'
    if os.path.exists(grape_dir):
        data_arrays['grape'] = [f for f in os.listdir(grape_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Non-leaf images
    non_leaf_dir = 'data/non_leaf'
    if os.path.exists(non_leaf_dir):
        data_arrays['non_leaf'] = [f for f in os.listdir(non_leaf_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.webp'))]

    return data_arrays

# Function to check class based on filename
def check_image_class(filename, data_arrays):
    if filename in data_arrays['corn']:
        return 'corn'
    elif filename in data_arrays['grape']:
        return 'grape'
    elif filename in data_arrays['non_leaf']:
        return 'non_leaf'
    else:
        return None  # Unknown, use model
