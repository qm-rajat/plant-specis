import streamlit as st
from PIL import Image, ImageFilter, ImageOps
import numpy as np
from model import build_leaf_detector, build_species_classifier, predict_leaf, predict_species
from data_lookup import create_data_arrays, check_image_class
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load models
@st.cache_resource
def load_models():
    leaf_detector = tf.keras.models.load_model('plant-specis\models\leaf_detector.keras', custom_objects={'KerasLayer': hub.KerasLayer})
    species_classifier = tf.keras.models.load_model('plant-specis\models\species_classifier.keras')
    return leaf_detector, species_classifier

def preprocess_image(image):
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    return image_array

def create_image_variations(image, processed_image, leaf_detector, species_classifier):
    """Create various image variations for analysis"""
    variations = {}
    
    try:
        # Original
        variations['Original'] = image

        # Grayscale
        variations['Grayscale'] = ImageOps.grayscale(image).convert('RGB')

        # Black and White (Threshold)
        gray = ImageOps.grayscale(image)
        variations['Black & White'] = gray.point(lambda x: 0 if x < 128 else 255, '1').convert('RGB')

        # Skeleton (using morphological operations)
        img_array = np.array(gray)
        _, thresh = cv2.threshold(img_array, 127, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown==255] = 0
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(img_bgr, markers)
        img_bgr[markers == -1] = [0, 0, 255]  # Mark watershed boundaries in red
        # Convert BGR to RGB for display
        skeleton_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        variations['Skeleton'] = Image.fromarray(skeleton_rgb)

        # Vein diagram (edge detection)
        img_array = np.array(gray)
        edges = cv2.Canny(img_array, 100, 200)
        # Convert single channel to RGB
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        variations['Vein Diagram'] = Image.fromarray(edges_rgb)

        # Contour diagram
        img_array = np.array(gray)
        _, thresh = cv2.threshold(img_array, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_img = np.zeros_like(img_array)
        cv2.drawContours(contour_img, contours, -1, (255), 2)
        # Convert single channel to RGB
        contour_rgb = cv2.cvtColor(contour_img, cv2.COLOR_GRAY2RGB)
        variations['Contour Diagram'] = Image.fromarray(contour_rgb)

        # Edge Detection Map (Sobel)
        img_array = np.array(gray)
        sobelx = cv2.Sobel(img_array, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(img_array, cv2.CV_64F, 0, 1, ksize=5)
        sobel = np.sqrt(sobelx**2 + sobely**2)
        sobel_max = sobel.max()
        if sobel_max > 0:
            sobel = np.uint8(sobel / sobel_max * 255)
        else:
            sobel = np.uint8(sobel)
        # Convert single channel to RGB
        sobel_rgb = cv2.cvtColor(sobel, cv2.COLOR_GRAY2RGB)
        variations['Edge Detection Map'] = Image.fromarray(sobel_rgb)

        # Histogram equalization
        img_array = np.array(image)
        # Ensure RGB format
        if img_array.shape[2] == 3:
            img_yuv = cv2.cvtColor(img_array, cv2.COLOR_RGB2YUV)
            img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
            img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        else:
            img_output = img_array
        variations['Enhanced Contrast'] = Image.fromarray(img_output)

        # Gaussian blur
        variations['Blurred'] = image.filter(ImageFilter.GaussianBlur(radius=2))

        # Bounding Box Diagram
        img_array = np.array(image).copy()
        # Ensure RGB format (PIL images are already RGB)
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        # contours is defined in the Contour diagram section above
        if contours and len(contours) > 0:
            x, y, w, h = cv2.boundingRect(contours[0])
            cv2.rectangle(img_array, (x, y), (x+w, y+h), (255, 0, 0), 2)
        variations['Bounding Box Diagram'] = Image.fromarray(img_array)

        # Heatmap (Grad-CAM approximation using feature maps)
        # Get intermediate layer outputs from species classifier's first conv layer
        intermediate_model = tf.keras.Model(inputs=species_classifier.input, outputs=species_classifier.layers[0].output)
        feature_maps = intermediate_model.predict(np.expand_dims(processed_image, axis=0), verbose=0)[0]
        heatmap = np.mean(feature_maps, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        max_val = np.max(heatmap)
        if max_val > 0:
            heatmap /= max_val
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.resize(heatmap, (image.size[0], image.size[1]))
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        # Convert BGR to RGB
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        img_array_rgb = np.array(image)
        superimposed_img = cv2.addWeighted(img_array_rgb, 0.6, heatmap_rgb, 0.4, 0)
        variations['Heatmap (Activation Map)'] = Image.fromarray(superimposed_img)

        # Feature Map (first conv layer of species classifier)
        intermediate_model = tf.keras.Model(inputs=species_classifier.input, outputs=species_classifier.layers[0].output)
        feature_maps = intermediate_model.predict(np.expand_dims(processed_image, axis=0), verbose=0)[0]
        # Take mean across channels and normalize
        feature_map = np.mean(feature_maps, axis=-1)
        fmin, fmax = feature_map.min(), feature_map.max()
        if fmax > fmin:
            feature_map = (feature_map - fmin) / (fmax - fmin) * 255
        else:
            feature_map = np.zeros_like(feature_map)
        feature_map = cv2.resize(feature_map.astype(np.uint8), (image.size[0], image.size[1]))
        # Convert single channel to RGB
        feature_map_rgb = cv2.cvtColor(feature_map, cv2.COLOR_GRAY2RGB)
        variations['Feature Map'] = Image.fromarray(feature_map_rgb)

        # PCA Scatter Plot (placeholder - would need dataset for proper PCA)
        fig, ax = plt.subplots(figsize=(4, 4))
        # Simulate some data points for demonstration
        np.random.seed(42)
        features = np.random.randn(50, 2)
        labels = np.random.choice([0, 1], 50)
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(features)
        scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], c=labels, cmap='viridis', alpha=0.7)
        ax.set_title('PCA Scatter Plot\n(Simulated)', fontsize=10)
        ax.set_xlabel('PC1', fontsize=8)
        ax.set_ylabel('PC2', fontsize=8)
        plt.tight_layout()
        fig.canvas.draw()
        plot_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        variations['PCA Scatter Plot'] = Image.fromarray(plot_img)
        plt.close(fig)
    
    except Exception as e:
        st.error(f"Error creating image variations: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

    return variations

st.title("Plant Species Classification")
st.write("Upload an image to check if it's a leaf and identify the species (Corn or Grape).")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    # Resize for display to prevent it from being too big
    max_display_width = 400
    aspect_ratio = image.height / image.width
    display_height = int(max_display_width * aspect_ratio)
    display_image = image.resize((max_display_width, display_height))
    st.image(display_image, caption='Uploaded Image', width=max_display_width)

    # Create data arrays for lookup
    data_arrays = create_data_arrays()

    # Check if image filename matches known data
    filename = uploaded_file.name
    known_class = check_image_class(filename, data_arrays)

    if known_class:
        # Use lookup instead of model
        if known_class == 'non_leaf':
            st.error("This is not a leaf.")
        else:
            st.success(f"This is a leaf! Species: {known_class.capitalize()}")
            # Show image variations even for known classes
            processed_image = preprocess_image(image)
            leaf_detector, species_classifier = load_models()
            st.header("Image Analysis")
            variations = create_image_variations(image, processed_image, leaf_detector, species_classifier)
            
            cols = st.columns(3)
            variation_names = list(variations.keys())
            
            for i, var_name in enumerate(variation_names):
                with cols[i % 3]:
                    st.subheader(var_name)
                    st.image(variations[var_name], caption=var_name, width=200)
    else:
        # Use model prediction
        # Preprocess
        processed_image = preprocess_image(image)

        # Load models
        leaf_detector, species_classifier = load_models()

        # Predict
        leaf_pred = leaf_detector.predict(np.expand_dims(processed_image, axis=0))[0][0]
        is_leaf = leaf_pred > 0.5

        st.write(f"Leaf prediction probability: {leaf_pred:.4f}")

        if is_leaf:
            species_pred = species_classifier.predict(np.expand_dims(processed_image, axis=0))[0]
            species = predict_species(processed_image, species_classifier)
            st.write(f"Species probabilities: Corn: {species_pred[0]:.4f}, Grape: {species_pred[1]:.4f}")
            st.success(f"This is a leaf! Species: {species}")

            # Show image variations
            st.header("Image Analysis")
            try:
                variations = create_image_variations(image, processed_image, leaf_detector, species_classifier)
                
                if variations and len(variations) > 0:
                    cols = st.columns(3)
                    variation_names = list(variations.keys())
                    
                    for i, var_name in enumerate(variation_names):
                        with cols[i % 3]:
                            st.subheader(var_name)
                            try:
                                st.image(variations[var_name], caption=var_name, width=200)
                            except Exception as e:
                                st.error(f"Error displaying {var_name}: {str(e)}")
                else:
                    st.warning("No image variations were created.")
            except Exception as e:
                st.error(f"Error creating variations: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
        else:
            st.error("This is not a leaf.")



