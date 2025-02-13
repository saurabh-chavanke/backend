from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from skimage.morphology import white_tophat, disk, remove_small_objects
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import label, regionprops
from skimage import img_as_ubyte
from scipy import ndimage as ndi
import base64
import io
from PIL import Image

app = Flask(__name__)
CORS(app)

def encode_image(image):
    """Convert an image to Base64 format for frontend display."""
    _, buffer = cv2.imencode(".jpg", image)
    return base64.b64encode(buffer).decode("utf-8")

def process_fundus_image(image, original_image):
    """Process fundus image and draw detection boxes on the original image."""
    image = cv2.resize(image, (512, 512))
    original_image = cv2.resize(original_image, (512, 512))  # Keep original size

    steps = []  # Store processing steps

    # Step 1: Extract Green Channel
    green_channel = image[:, :, 1]
    steps.append({"step": "Extract Green Channel", "image": encode_image(cv2.merge([green_channel, green_channel, green_channel]))})

    # Step 2: Apply Gaussian Blur
    blurred_img = gaussian(green_channel, sigma=1)
    blurred_img = img_as_ubyte(blurred_img)
    steps.append({"step": "Apply Gaussian Blur", "image": encode_image(cv2.merge([blurred_img, blurred_img, blurred_img]))})

    # Step 3: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(blurred_img)
    steps.append({"step": "Apply CLAHE", "image": encode_image(cv2.merge([enhanced_img, enhanced_img, enhanced_img]))})

    # Step 4: Invert Image
    inverted_img = cv2.bitwise_not(enhanced_img)
    steps.append({"step": "Invert Image", "image": encode_image(cv2.merge([inverted_img, inverted_img, inverted_img]))})

    # Step 5: Morphological Closing
    selem = disk(3)
    closed_img = cv2.morphologyEx(inverted_img, cv2.MORPH_CLOSE, selem)
    steps.append({"step": "Morphological Closing", "image": encode_image(cv2.merge([closed_img, closed_img, closed_img]))})

    # Step 6: Subtraction
    subtracted = cv2.subtract(closed_img, inverted_img)
    steps.append({"step": "Subtraction", "image": encode_image(cv2.merge([subtracted, subtracted, subtracted]))})

    # Step 7: Vessel Removal via Thresholding
    thresh_val = threshold_otsu(subtracted)
    vessel_removed = subtracted > thresh_val
    vessel_removed_img = img_as_ubyte(vessel_removed)
    steps.append({"step": "Vessel Removal", "image": encode_image(cv2.merge([vessel_removed_img, vessel_removed_img, vessel_removed_img]))})

    # Step 8: White Tophat Filtering
    structuring_element = disk(8)
    tophat_img = white_tophat(vessel_removed_img, footprint=structuring_element)
    steps.append({"step": "White Tophat Filtering", "image": encode_image(cv2.merge([tophat_img, tophat_img, tophat_img]))})

    # Step 9: Binary Thresholding
    thresh_val = threshold_otsu(tophat_img)
    binary_img = tophat_img > thresh_val
    cleaned_img = remove_small_objects(binary_img, min_size=50)
    filled_img = ndi.binary_fill_holes(cleaned_img)
    filtered_img = img_as_ubyte(filled_img)
    steps.append({"step": "Binary Thresholding", "image": encode_image(cv2.merge([filtered_img, filtered_img, filtered_img]))})

    # Step 10: Detect Hemorrhages and Draw Bounding Boxes
    labeled_img = label(filtered_img)
    detection_img = original_image.copy()  # Use the original image for bounding boxes
    hemorrhage_count = 0

    for prop in regionprops(labeled_img):
        area = prop.area
        eccentricity = prop.eccentricity
        compactness = (prop.perimeter ** 2) / area if prop.perimeter > 0 else 1
        if compactness > 1 and eccentricity < 0.9:
            hemorrhage_count += 1
            minr, minc, maxr, maxc = prop.bbox
            cv2.rectangle(detection_img, (minc, minr), (maxc, maxr), (255, 255, 0), 2) 

    steps.append({"step": "Final Detection", "image": encode_image(detection_img)})

    return steps, hemorrhage_count

@app.route("/process", methods=["POST"])
def process_image():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    image = Image.open(io.BytesIO(file.read())).convert("RGB")
    image = np.array(image)
    processed_steps, hemorrhage_count = process_fundus_image(image, image.copy())

    return jsonify({"steps": processed_steps, "hemorrhage_count": hemorrhage_count})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=10000)
