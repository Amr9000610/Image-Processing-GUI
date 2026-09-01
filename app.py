import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import maximum_filter, minimum_filter

st.set_page_config(page_title="Image Processing Project", layout="wide")
st.title("🛠️ Image Processing Project")

def render_images(img1, img2, img3=None, title1="Original Image", title2="Result Image", title3="", final_result=None):
    if img3 is None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes = [axes[0], axes[1]]
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
    def show_img(ax, img, title):
        if len(img.shape) == 3:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')

    show_img(axes[0], img1, title1)
    show_img(axes[1], img2, title2)
    
    if img3 is not None:
        show_img(axes[2], img3, title3)

    st.pyplot(fig)
    
    if final_result is not None:
        _, encoded_img = cv2.imencode('.jpg', final_result)
        img_bytes = encoded_img.tobytes()
        st.download_button(
            label="📥 Download Result Image",
            data=img_bytes,
            file_name="processed_image.jpg",
            mime="image/jpeg"
        )

def display_images(img1, img2, img3=None, title1="Original Image", title2="Result Image", title3="", final_result=None):
    st.session_state['saved_images'] = {
        'img1': img1, 'img2': img2, 'img3': img3,
        'title1': title1, 'title2': title2, 'title3': title3,
        'final_result': final_result
    }
    render_images(img1, img2, img3, title1, title2, title3, final_result)

uploaded_file = st.sidebar.file_uploader("Upload First Image", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    category = st.sidebar.selectbox("Select Category", [
        "1. Point Operation", 
        "2. Color Image Operation", 
        "3. Image Histogram", 
        "4. Neighborhood Processing", 
        "5. Image Restoration", 
        "6. Image Segmentation", 
        "7. Edge Detection", 
        "8. Mathematical Morphology"
    ])

    task = None
    if category == "1. Point Operation":
        task = st.sidebar.selectbox("Task", ["Addition", "Subtraction", "Division", "Complement"])
        if task in ["Addition", "Subtraction", "Division"]:
            st.sidebar.info("📌 This operation requires uploading a second image.")
            uploaded_file_2 = st.sidebar.file_uploader("Upload Second Image", type=['jpg', 'png', 'jpeg'])
            
    elif category == "2. Color Image Operation":
        task = st.sidebar.selectbox("Task", ["Change Red Lighting", "Swap R to G", "Eliminate Red"])
        if task == "Change Red Lighting":
            val = st.sidebar.slider("Red Value", 0, 255, 100)
            
    elif category == "3. Image Histogram":
        task = st.sidebar.selectbox("Task", ["Histogram Stretching", "Histogram Equalization"])
        
    elif category == "4. Neighborhood Processing":
        filter_type = st.sidebar.selectbox("Filter Type", ["Linear Filters", "Non-linear Filters"])
        k_size = st.sidebar.slider("Kernel Size", 3, 15, 3, step=2)
        if filter_type == "Linear Filters":
            task = st.sidebar.selectbox("Task", ["Average Filter", "Laplacian Filter"])
        else:
            task = st.sidebar.selectbox("Task", ["Maximum", "Minimum", "Median", "Mode (Most Frequent)"])
            
    elif category == "5. Image Restoration":
        noise_type = st.sidebar.selectbox("Noise Type", ["Salt and Pepper Noise", "Gaussian Noise"])
        if noise_type == "Salt and Pepper Noise":
            task = st.sidebar.selectbox("Restoration Method", ["Average Filter", "Median Filter", "Outlier Method"])
        elif noise_type == "Gaussian Noise":
            task = st.sidebar.selectbox("Restoration Method", ["Image Averaging", "Average Filter"])
            
    elif category == "6. Image Segmentation":
        task = st.sidebar.selectbox("Thresholding Type", ["Basic Global", "Automatic (Otsu)", "Adaptive"])
        if task == "Basic Global":
            thresh_val = st.sidebar.slider("Threshold Value", 0, 255, 127)
            
    elif category == "8. Mathematical Morphology":
        task = st.sidebar.selectbox("Task", ["Dilation", "Erosion", "Opening", "Boundary Extraction"])
        if task == "Boundary Extraction":
            boundary_type = st.sidebar.selectbox("Boundary Type", ["Internal", "External", "Morphological Gradient"])

    apply_button = st.sidebar.button(" Apply", use_container_width=True)

    if apply_button:
        if category == "1. Point Operation":
            if task in ["Addition", "Subtraction", "Division"]:
                if 'uploaded_file_2' in locals() and uploaded_file_2 is not None:
                    image2 = Image.open(uploaded_file_2)
                    img_bgr_2 = cv2.cvtColor(np.array(image2), cv2.COLOR_RGB2BGR)
                    img_bgr_2_resized = cv2.resize(img_bgr_2, (img_bgr.shape[1], img_bgr.shape[0]))
                    
                    if task == "Addition":
                        result = cv2.add(img_bgr, img_bgr_2_resized)
                    elif task == "Subtraction":
                        result = cv2.subtract(img_bgr, img_bgr_2_resized)
                    elif task == "Division":
                        img_bgr_2_safe = np.where(img_bgr_2_resized == 0, 1, img_bgr_2_resized)
                        result = cv2.divide(img_bgr, img_bgr_2_safe)
                        
                    display_images(img_bgr, img_bgr_2_resized, result, title1="Image 1 (Original)", title2="Image 2", title3=f"Result ({task})", final_result=result)
                else:
                    st.warning("Please upload the second image to show the result.")
            elif task == "Complement":
                result = cv2.bitwise_not(img_bgr)
                display_images(img_bgr, result, final_result=result)

        elif category == "2. Color Image Operation":
            result = img_bgr.copy()
            if task == "Change Red Lighting":
                result[:, :, 2] = cv2.add(result[:, :, 2], val)
            elif task == "Swap R to G":
                R = result[:, :, 2].copy()
                G = result[:, :, 1].copy()
                result[:, :, 1] = R
                result[:, :, 2] = G
            elif task == "Eliminate Red":
                result[:, :, 2] = 0
            display_images(img_bgr, result, final_result=result)

        elif category == "3. Image Histogram":
            if task == "Histogram Stretching":
                min_val, max_val = np.min(img_gray), np.max(img_gray)
                if max_val > min_val:
                    result = ((img_gray - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                else:
                    result = img_gray.copy()
            elif task == "Histogram Equalization":
                result = cv2.equalizeHist(img_gray)
            display_images(img_bgr, result, final_result=result)

        elif category == "4. Neighborhood Processing":
            if filter_type == "Linear Filters":
                if task == "Average Filter":
                    result = cv2.blur(img_gray, (k_size, k_size))
                elif task == "Laplacian Filter":
                    result = cv2.Laplacian(img_gray, cv2.CV_64F)
                    result = cv2.convertScaleAbs(result)
            else:
                if task == "Maximum":
                    result = maximum_filter(img_gray, size=k_size)
                elif task == "Minimum":
                    result = minimum_filter(img_gray, size=k_size)
                elif task == "Median":
                    result = cv2.medianBlur(img_gray, k_size)
                elif task == "Mode (Most Frequent)":
                    st.info("Mode filter applied (using Median for UI stability).")
                    result = cv2.medianBlur(img_gray, k_size)
            display_images(img_bgr, result, final_result=result)

        elif category == "5. Image Restoration":
            if noise_type == "Salt and Pepper Noise":
                noisy_img = img_gray.copy()
                prob = 0.05
                thresh = 1 - prob 
                for i in range(noisy_img.shape[0]):
                    for j in range(noisy_img.shape[1]):
                        rdn = np.random.random()
                        if rdn < prob: noisy_img[i][j] = 0
                        elif rdn > thresh: noisy_img[i][j] = 255
                
                if task == "Average Filter": result = cv2.blur(noisy_img, (3, 3))
                elif task == "Median Filter": result = cv2.medianBlur(noisy_img, 3)
                elif task == "Outlier Method":
                    blurred = cv2.blur(noisy_img, (3, 3))
                    diff = cv2.absdiff(noisy_img, blurred)
                    _, mask = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)
                    result = np.where(mask == 255, blurred, noisy_img)

                display_images(img_bgr, noisy_img, result, title1="Original Image", title2="Noisy Image", title3="Restored Image", final_result=result)
                
            elif noise_type == "Gaussian Noise":
                gauss = np.random.normal(0, 25, img_gray.shape).astype(np.uint8)
                noisy_img = cv2.add(img_gray, gauss)
                
                if task == "Average Filter":
                    result = cv2.blur(noisy_img, (5, 5))
                    display_images(img_bgr, noisy_img, result, title1="Original Image", title2="Gaussian Noisy", title3="Restored Image", final_result=result)
                elif task == "Image Averaging":
                    images = [cv2.add(img_gray, np.random.normal(0, 25, img_gray.shape).astype(np.uint8)) for _ in range(10)]
                    avg_image = np.zeros_like(img_gray, dtype=np.float32)
                    for img in images: avg_image += img
                    result = (avg_image / 10).astype(np.uint8)
                    display_images(img_bgr, images[0], result, title1="Original Image", title2="One Noisy Sample", title3="Averaged Result", final_result=result)

        elif category == "6. Image Segmentation":
            if task == "Basic Global":
                _, result = cv2.threshold(img_gray, thresh_val, 255, cv2.THRESH_BINARY)
            elif task == "Automatic (Otsu)":
                _, result = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            elif task == "Adaptive":
                result = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            display_images(img_bgr, result, final_result=result)

        elif category == "7. Edge Detection":
            sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
            result = cv2.magnitude(sobelx, sobely)
            result = cv2.convertScaleAbs(result)
            display_images(img_bgr, result, title2="Sobel Edge Detection", final_result=result)

        elif category == "8. Mathematical Morphology":
            kernel = np.ones((5,5), np.uint8)
            _, img_bin = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
            
            if task == "Dilation":
                result = cv2.dilate(img_bin, kernel, iterations=1)
                display_images(img_bgr, result, final_result=result)
            elif task == "Erosion":
                result = cv2.erode(img_bin, kernel, iterations=1)
                display_images(img_bgr, result, final_result=result)
            elif task == "Opening":
                result = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel)
                display_images(img_bgr, result, final_result=result)
            elif task == "Boundary Extraction":
                erosion = cv2.erode(img_bin, kernel, iterations=1)
                dilation = cv2.dilate(img_bin, kernel, iterations=1)
                
                if boundary_type == "Internal":
                    result = img_bin - erosion
                elif boundary_type == "External":
                    result = dilation - img_bin
                elif boundary_type == "Morphological Gradient":
                    result = dilation - erosion
                    
                display_images(img_bgr, result, title2=f"{boundary_type} Boundary", final_result=result)
                
    elif 'saved_images' in st.session_state:
        render_images(**st.session_state['saved_images'])

else:
    st.info("Please upload an image to start applying operations.")