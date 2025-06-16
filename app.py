import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

class ImageProcessor:
    def __init__(self, image):
        self.image = image

    # Basic Operations
    def flip(self, flip_code):
        return cv2.flip(self.image, flip_code)

    def grayscale(self):
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def resize(self, height, width):
        return cv2.resize(self.image, (width, height))

    def rotate(self, angle):
        (h, w) = self.image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(self.image, M, (w, h))

    def extract_channel(self, channel_name):
        if channel_name == "Blue":
            return self.image[:, :, 0]
        elif channel_name == "Green":
            return self.image[:, :, 1]
        elif channel_name == "Red":
            return self.image[:, :, 2]
        else:
            raise ValueError("Invalid channel name")

    # Filtering
    def apply_blur(self, blur_type):
        if blur_type == 'boxFilter':
            kernel_size = st.sidebar.slider("Kernel Size:", min_value=3, max_value=89, step=2, value=23)
            depth = st.sidebar.select_slider("Select the desired depth", options=[-1, 0, 1, 2, 3, 4, 5, 6, 7, 8])
            return cv2.boxFilter(self.image, depth, (kernel_size, kernel_size))
        elif blur_type == 'bilateralFilter':
            diameter = st.sidebar.slider("Select diameter of each pixel's neighborhood:", min_value=5, max_value=51, step=2, value=9)
            sigma_color = st.sidebar.slider("Larger value means that farther colors within the pixel neighborhood will be mixed together:", min_value=1.0, max_value=50.0, value=10.0)
            sigma_space = st.sidebar.slider("Filter sigma in the coordinate space:", min_value=1.0, max_value=50.0, value=10.0)
            return cv2.bilateralFilter(self.image, diameter, sigma_color, sigma_space)
        elif blur_type == 'GaussianBlur':
            kernel_size = st.sidebar.slider("Select the kernel size:", min_value=3, max_value=89, step=2, value=5)
            sigma_x = st.sidebar.slider("Select Gaussian kernel standard deviation in X direction:", min_value=0.0, max_value=50.0, value=1.0)
            sigma_y = st.sidebar.slider("Select Gaussian kernel standard deviation in Y direction:", min_value=0.0, max_value=50.0, value=1.0)
            return cv2.GaussianBlur(self.image, (kernel_size, kernel_size), sigma_x, sigmaY=sigma_y)
        elif blur_type == 'medianBlur':
            kernel_size = st.sidebar.slider("Select the kernel size:", min_value=3, max_value=89, step=2, value=5)
            return cv2.medianBlur(self.image, kernel_size)
        else:
            raise ValueError("Invalid blur type")

    def detect_edges_canny(self):
        low_threshold = st.sidebar.slider("Low Threshold:", min_value=0, max_value=255, value=100)
        high_threshold = st.sidebar.slider("High Threshold:", min_value=0, max_value=255, value=200)
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        return cv2.Canny(gray, low_threshold, high_threshold)

    def detect_edges_sobel(self):
        dx = st.sidebar.checkbox("Sobel X", value=True)
        dy = st.sidebar.checkbox("Sobel Y", value=False)
        kernel_size = st.sidebar.slider("Kernel Size:", min_value=1, max_value=7, step=2, value=3)
        if dx and dy:
            sobelx = cv2.Sobel(self.image, cv2.CV_64F, 1, 0, ksize=kernel_size)
            sobely = cv2.Sobel(self.image, cv2.CV_64F, 0, 1, ksize=kernel_size)
            sobel = cv2.magnitude(sobelx, sobely)
            return np.uint8(sobel)
        elif dx:
            return cv2.Sobel(self.image, cv2.CV_64F, 1, 0, ksize=kernel_size)
        elif dy:
            return cv2.Sobel(self.image, cv2.CV_64F, 0, 1, ksize=kernel_size)
        else:
            st.sidebar.write("Please select at least one direction (X or Y).")
            return self.image

    def detect_edges_laplacian(self):
        kernel_size = st.sidebar.slider("Kernel Size:", min_value=1, max_value=31, step=2, value=3)
        return cv2.Laplacian(self.image, cv2.CV_64F, ksize=kernel_size)

    # Morphology
    def apply_erosion(self):
        kernel_size = st.sidebar.slider("Kernel Size:", min_value=3, max_value=89, step=2, value=5)
        iterations = st.sidebar.slider("Select the number of iterations:", min_value=1, max_value=10, step=1, value=1)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.erode(self.image, kernel, iterations=iterations)

    def apply_dilation(self):
        kernel_size = st.sidebar.slider("Kernel Size:", min_value=3, max_value=89, step=2, value=5)
        iterations = st.sidebar.slider("Select the number of iterations:", min_value=1, max_value=10, step=1, value=1)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.dilate(self.image, kernel, iterations=iterations)

    def apply_morphological_gradient(self):
        kernel_size = st.sidebar.slider("Kernel Size:", min_value=3, max_value=89, step=2, value=5)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.morphologyEx(self.image, cv2.MORPH_GRADIENT, kernel)

    def apply_morphological_opening(self):
        kernel_size = st.sidebar.slider("Kernel Size:", min_value=3, max_value=89, step=2, value=5)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.morphologyEx(self.image, cv2.MORPH_OPEN, kernel)

    def apply_morphological_closing(self):
        kernel_size = st.sidebar.slider("Kernel Size:", min_value=3, max_value=89, step=2, value=5)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.morphologyEx(self.image, cv2.MORPH_CLOSE, kernel)

    # Contours
    def find_contours(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_image = self.image.copy()
        cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
        return contour_image

# Helper function to convert NumPy array to PIL Image
def convert_to_pil_image(image):
    if len(image.shape) == 2:  # Grayscale image
        return Image.fromarray(image, mode='L')
    elif len(image.shape) == 3:  # Color image
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        raise ValueError("Unsupported image shape")

# Helper function to convert NumPy array to bytes
def convert_to_bytes(image):
    if len(image.shape) == 2:  # Grayscale image
        _, buffer = cv2.imencode('.png', image)
    else:  # Color image
        _, buffer = cv2.imencode('.png', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return buffer.tobytes()

st.title("Image Editing App")
edited_image = None

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)

    # Convert RGB to BGR for OpenCV operations
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    st.image(convert_to_pil_image(image_bgr), caption='Original Image', use_column_width=True)

    st.sidebar.header("Select Operation Category")
    category = st.sidebar.selectbox("Choose a category", ["Basic Operations", "Filtering", "Morphology", "Contours"])

    processor = ImageProcessor(image_bgr)

    if category == "Basic Operations":
        st.sidebar.header("Basic Operations")
        operation = st.sidebar.selectbox("Choose an operation", ["Flip", "Resize", "Grayscaling", "Show Channels", "Rotate"])
        if operation == "Flip":
            flip_type = st.sidebar.selectbox("Select the flip type:", ["Horizontal Flip", "Vertical Flip", "Horizontal + Vertical Flip"])
            flip_code = {
                "Horizontal Flip": 1,
                "Vertical Flip": 0,
                "Horizontal + Vertical Flip": -1
            }
            edited_image = processor.flip(flip_code[flip_type])
        elif operation == "Resize":
            height = st.sidebar.slider("Select the height:", min_value=round(image_bgr.shape[0] / 5), max_value=round(image_bgr.shape[0] * 5), value=round(image_bgr.shape[0]))
            width = st.sidebar.slider("Select the width:", min_value=round(image_bgr.shape[1] / 5), max_value=round(image_bgr.shape[1] * 5), value=round(image_bgr.shape[1]))
            edited_image = processor.resize(height, width)
        elif operation == "Grayscaling":
            edited_image = processor.grayscale()
        elif operation == "Show Channels":
            channel_name = st.sidebar.selectbox("Select the channel", ["Blue", "Green", "Red"])
            edited_image = processor.extract_channel(channel_name)
        elif operation == "Rotate":
            angle = st.sidebar.slider("Select angle:", min_value=0, max_value=360, value=90)
            edited_image = processor.rotate(angle)

    elif category == "Filtering":
        st.sidebar.header("Filtering")
        operation = st.sidebar.selectbox("Choose an operation", ["Blur", "Edge Detection"])
        if operation == "Blur":
            blur_type = st.sidebar.selectbox("Select the type of blur:", ['boxFilter', 'bilateralFilter', 'GaussianBlur', 'medianBlur'])
            edited_image = processor.apply_blur(blur_type)
        elif operation == "Edge Detection":
            edge_type = st.sidebar.selectbox("Select the type of edge detection:", ['Canny', 'Sobel', 'Laplacian'])
            if edge_type == 'Canny':
                edited_image = processor.detect_edges_canny()
            elif edge_type == 'Sobel':
                edited_image = processor.detect_edges_sobel()
            elif edge_type == 'Laplacian':
                edited_image = processor.detect_edges_laplacian()

    elif category == "Morphology":
        st.sidebar.header("Morphology")
        operation = st.sidebar.selectbox("Choose an operation", ["Erosion", "Dilation", "Morphological Gradient", "Morphological Opening", "Morphological Closing"])
        if operation == "Erosion":
            edited_image = processor.apply_erosion()
        elif operation == "Dilation":
            edited_image = processor.apply_dilation()
        elif operation == "Morphological Gradient":
            edited_image = processor.apply_morphological_gradient()
        elif operation == "Morphological Opening":
            edited_image = processor.apply_morphological_opening()
        elif operation == "Morphological Closing":
            edited_image = processor.apply_morphological_closing()

    elif category == "Contours":
        st.sidebar.header("Contours")
        operation = st.sidebar.selectbox("Choose an operation", ["Find Contours"])
        if operation == "Find Contours":
            edited_image = processor.find_contours()

    if edited_image is not None:
        st.image(convert_to_pil_image(edited_image), caption='Edited Image', use_container_width=True)
      
        
        edited_image_bytes = convert_to_bytes(edited_image[:,:,[2,1,0]])   # conversion to RGB from BGR other wise bluish image would be downloaded
        # Add download button
        st.download_button(
            label="Download Edited Image",
            data=edited_image_bytes,
            file_name="edited_image.png",
            mime="image/png"
        )

else:
    st.write("Please upload an image to edit.")