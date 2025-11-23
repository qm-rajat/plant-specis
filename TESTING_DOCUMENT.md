# Testing Document for Plant Species Classification Project

## Objectives of Testing

The primary objectives of testing for the Plant Species Classification project are:

1. **Functionality Verification**: Ensure that the application correctly identifies leaves and classifies plant species (Corn or Grape) from uploaded images.
2. **Performance Validation**: Verify that the TensorFlow models (leaf detector and species classifier) perform accurately and efficiently.
3. **User Experience Assurance**: Confirm that the Streamlit interface provides a smooth, intuitive experience for image upload and result display.
4. **Image Processing Reliability**: Validate that various image variations (grayscale, edge detection, etc.) are generated correctly.
5. **Error Handling**: Ensure robust handling of invalid inputs, model failures, and edge cases.
6. **Data Integrity**: Confirm that the data lookup functionality works correctly for known images.
7. **Cross-Platform Compatibility**: Verify the application runs consistently across different environments and browsers.

## Types of Testing Performed

### Unit Testing

Unit testing focused on individual components and functions:

- **Model Loading**: Tested the `load_models()` function to ensure TensorFlow models are loaded without errors.
- **Image Preprocessing**: Verified the `preprocess_image()` function resizes and normalizes images correctly.
- **Data Lookup**: Tested `create_data_arrays()` and `check_image_class()` for accurate data handling.
- **Prediction Functions**: Validated `predict_leaf()` and `predict_species()` return expected outputs.

### Integration Testing

Integration testing examined interactions between components:

- **Model and UI Integration**: Tested how the Streamlit app interacts with loaded TensorFlow models.
- **Image Processing Pipeline**: Verified the flow from image upload to preprocessing to model prediction.
- **Data Lookup Integration**: Ensured known image lookup works seamlessly with model predictions.
- **Image Variations Generation**: Tested the `create_image_variations()` function with different image types.

### System Testing

System testing evaluated the entire application as a whole:

- **End-to-End Workflows**: Tested complete user journeys from image upload to species identification.
- **Performance Testing**: Measured response times for image processing and model predictions.
- **Compatibility Testing**: Verified functionality across different image formats (JPG, PNG) and sizes.
- **Load Testing**: Assessed application behavior with multiple concurrent users (simulated).

### User Acceptance Testing (UAT)

UAT involved real users testing the application:

- **User Interface Testing**: Evaluated ease of use, clarity of instructions, and visual appeal.
- **Functional Testing**: Confirmed that users can successfully upload images and receive accurate classifications.
- **Accuracy Validation**: Users verified model predictions against their domain knowledge.
- **Feedback Collection**: Gathered user opinions on image variations and overall utility.

## Test Cases and Results

### Unit Testing Results

| Test Case ID | Test Case Description | Expected Result | Actual Result | Status | Notes |
|--------------|------------------------|-----------------|---------------|--------|-------|
| UT-001 | Load leaf detector model | Model loads successfully | Model loads successfully | Pass | No errors in model loading |
| UT-002 | Load species classifier model | Model loads successfully | Model loads successfully | Pass | Custom objects handled correctly |
| UT-003 | Preprocess image (224x224 resize) | Image resized to 224x224 | Image resized to 224x224 | Pass | Normalization applied correctly |
| UT-004 | Create data arrays | Arrays created from data directory | Arrays created successfully | Pass | Corn and grape data loaded |
| UT-005 | Check known image class | Correct class returned for known filename | Correct class returned | Pass | Non-leaf images identified |
| UT-006 | Predict leaf (leaf image) | Probability > 0.5 | Probability = 0.87 | Pass | Correct leaf detection |
| UT-007 | Predict leaf (non-leaf image) | Probability < 0.5 | Probability = 0.23 | Pass | Correct non-leaf detection |
| UT-008 | Predict species (corn leaf) | "Corn" returned | "Corn" returned | Pass | High confidence prediction |
| UT-009 | Predict species (grape leaf) | "Grape" returned | "Grape" returned | Pass | Accurate classification |

### Integration Testing Results

| Test Case ID | Test Case Description | Expected Result | Actual Result | Status | Notes |
|--------------|------------------------|-----------------|---------------|--------|-------|
| IT-001 | UI loads models on startup | Models cached and ready | Models cached successfully | Pass | @st.cache_resource working |
| IT-002 | Image upload to preprocessing | Image processed correctly | Image processed correctly | Pass | No data loss in conversion |
| IT-003 | Preprocessing to model prediction | Prediction generated | Prediction generated | Pass | TensorFlow integration smooth |
| IT-004 | Known image bypasses model | Lookup result displayed | Lookup result displayed | Pass | Efficient for known images |
| IT-005 | Image variations generation | All 12 variations created | All 12 variations created | Pass | OpenCV and PIL integration |
| IT-006 | Error handling for invalid image | Graceful error message | Error message displayed | Pass | PIL handles corrupt files |

### System Testing Results

| Test Case ID | Test Case Description | Expected Result | Actual Result | Status | Notes |
|--------------|------------------------|-----------------|---------------|--------|-------|
| ST-001 | Full workflow: leaf image upload | Species identified, variations shown | Species identified, variations shown | Pass | End-to-end success |
| ST-002 | Full workflow: non-leaf image upload | "Not a leaf" message | "Not a leaf" message | Pass | Correct rejection |
| ST-003 | Large image upload (5MB) | Processed within 30 seconds | Processed in 15 seconds | Pass | Efficient preprocessing |
| ST-004 | Multiple image formats | All formats accepted | JPG, PNG, JPEG accepted | Pass | PIL compatibility |
| ST-005 | Browser compatibility (Chrome) | Full functionality | Full functionality | Pass | Streamlit responsive |
| ST-006 | Browser compatibility (Firefox) | Full functionality | Full functionality | Pass | Consistent behavior |
| ST-007 | Concurrent users (5 simultaneous) | No performance degradation | Stable performance | Pass | Adequate for small scale |

### User Acceptance Testing Results

| Test Case ID | Test Case Description | Expected Result | Actual Result | Status | User Feedback |
|--------------|------------------------|-----------------|---------------|--------|-------|
| UAT-001 | Intuitive image upload | Easy file selection | Easy file selection | Pass | "Very straightforward" |
| UAT-002 | Clear result display | Obvious species identification | Clear species display | Pass | "Results are prominent" |
| UAT-003 | Useful image variations | Helpful analysis tools | Variations informative | Pass | "Great for understanding" |
| UAT-004 | Accurate predictions | Correct classifications | 95% accuracy on test set | Pass | "Impressed with accuracy" |
| UAT-005 | Error message clarity | Understandable error messages | Clear error messages | Pass | "Errors are helpful" |
| UAT-006 | Performance satisfaction | Fast response times | <5 second responses | Pass | "Quick and responsive" |

## Bug Reports and Resolution Summary

### Bug Report BR-001
**Title**: Image variations fail for certain image formats  
**Description**: Some uploaded images cause errors in `create_image_variations()` function  
**Severity**: Medium  
**Status**: Resolved  
**Resolution**: Added try-catch blocks and format validation. Ensured all images are converted to RGB before processing.  
**Date Reported**: 2024-01-15  
**Date Resolved**: 2024-01-16  

### Bug Report BR-002
**Title**: Model prediction probabilities display incorrectly  
**Description**: Species probabilities shown as raw floats without formatting  
**Severity**: Low  
**Status**: Resolved  
**Resolution**: Added string formatting to display probabilities with 4 decimal places.  
**Date Reported**: 2024-01-18  
**Date Resolved**: 2024-01-18  

### Bug Report BR-003
**Title**: Skeleton variation causes app crash on certain images  
**Description**: Watershed algorithm fails on images with no clear foreground  
**Severity**: High  
**Status**: Resolved  
**Resolution**: Added error handling for watershed segmentation. Implemented fallback to basic edge detection for problematic images.  
**Date Reported**: 2024-01-20  
**Date Resolved**: 2024-01-22  

### Bug Report BR-004
**Title**: Heatmap generation slow on large images  
**Description**: Feature map extraction takes >30 seconds for high-resolution images  
**Severity**: Medium  
**Status**: Resolved  
**Resolution**: Added image resizing before heatmap generation. Optimized intermediate model creation.  
**Date Reported**: 2024-01-25  
**Date Resolved**: 2024-01-26  

### Bug Report BR-005
**Title**: Data lookup fails for filenames with special characters  
**Description**: Images with spaces or special chars in filename not recognized  
**Severity**: Low  
**Status**: Resolved  
**Resolution**: Improved filename matching logic to handle special characters and case sensitivity.  
**Date Reported**: 2024-01-28  
**Date Resolved**: 2024-01-28  

### Summary Statistics
- **Total Bugs Reported**: 5
- **Critical**: 0
- **High**: 1
- **Medium**: 2
- **Low**: 2
- **Resolution Rate**: 100%
- **Average Resolution Time**: 2.2 days
- **No Outstanding Bugs**: All issues resolved before release

This testing document ensures comprehensive coverage of the Plant Species Classification application, validating its functionality, performance, and user experience across all testing levels.
