from flask import Flask, render_template, jsonify, redirect, url_for
import cv2
import numpy as np
import base64
import os
import traceback

# --- WEB APP SETUP ---
# Flask is our minimal web framework. We create the app instance here.
app = Flask(__name__)

# --- PATH CONFIGURATION ---
# We define where the application lives and where to find the source images.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_FOLDER = os.path.join(BASE_DIR, 'static', 'images')


# --- IMAGE UTILITY FUNCTIONS ---

def mat_to_base64(mat):
    """Converts the OpenCV numerical image data (matrix) into a Base64 string for display in a web browser."""
    if mat is None: return ""
    # We use JPEG compression (.jpg) here to reduce file size for faster web transfer.
    _, buffer = cv2.imencode('.jpg', mat)
    return base64.b64encode(buffer).decode('utf-8')

def resize_image_fixed_width(img, target_width=600):
    """Resizes the image to a standardized width (600px). This dramatically speeds up SIFT/stitching performance."""
    if img is None: return None
    h, w = img.shape[:2]
    if w <= target_width: return img 
    
    scale = target_width / w
    # cv2.INTER_AREA is best for shrinking images to avoid artifacts.
    return cv2.resize(img, (target_width, int(h * scale)), interpolation=cv2.INTER_AREA)

def crop_black_borders(img):
    """
    Crops the black, empty space around the final panoramic image after warping.
    This keeps the image canvas and file size manageable.
    """
    if img is None: return None
    
    # Convert to grayscale and threshold to find non-black pixels (the image content)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
    # Find the largest boundary (contour) around the image content
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        return img[y:y+h, x:x+w] # Crop the image array
        
    return img


# --- CORE STITCHING LOGIC: MANUAL FALLBACK ---

def stitch_two_manually(img1, img2):
    """
    Stitches img2 onto img1 using the robust RANSAC/Homography method. 
    This is the fallback if the automatic stitcher fails.
    """
    print(f"Stitching pair: {img1.shape} and {img2.shape}")
    
    # 1. Detect & Match Features (Using SIFT: Scale-Invariant Feature Transform)
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        print("SIFT failed to find descriptors for a robust match.")
        return img1

    # FLANN Matcher setup 
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2) # Find the 2 best matches for each keypoint

    # Filter for good matches using Lowe's ratio test 
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 4:
        print(f"Not enough good matches ({len(good)} < 4) to calculate Homography.")
        return img1

    # 2. Calculate Homography (The Perspective Transformation Matrix, H)
    src_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)

    H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 15.0) 
    
    if H is None:
        print("Homography calculation failed (H is None).")
        return img1

    # 3. Calculate New Canvas Size to fit BOTH images after warping
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Find where the corners of img2 will land after transformation H
    pts_corners_img2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    pts_corners_img2_transformed = cv2.perspectiveTransform(pts_corners_img2, H)
    
    pts_corners_img1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    
    all_corners = np.concatenate((pts_corners_img1, pts_corners_img2_transformed), axis=0)
    
    [xmin, ymin] = np.int32(all_corners.min(axis=0).flatten() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).flatten() + 0.5)

    if (xmax - xmin) > 6000 or (ymax - ymin) > 6000:
        print("Canvas too big (bad match detected). Skipping this pair.")
        return img1

    # Create a Translation Matrix to shift the image so all coordinates are positive
    t = [-xmin, -ymin]
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]], dtype=np.float32)

    # 4. Warp img2 onto the new canvas
    # The total transformation is Ht * H
    result = cv2.warpPerspective(img2, Ht.dot(H), (xmax - xmin, ymax - ymin))

    # 5. Place img1 onto the canvas (only shifted by Ht)
    img1_shifted = cv2.warpPerspective(img1, Ht, (xmax - xmin, ymax - ymin))
    
    # Create a mask to ensure img1 is placed without black boxes or simple addition
    gray = cv2.cvtColor(img1_shifted, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
    # Overlay img1 onto the warped result (no blending, just overlap)
    result[mask > 0] = img1_shifted[mask > 0] 

    # 6. CRITICAL STEP: Crop the final image to remove external black borders
    return crop_black_borders(result)

# --- LOGIC: STITCHING CONTROLLER ---
def process_stitching():

    filenames = ['IMG1.jpg', 'IMG2.jpg', 'IMG3.jpg', 'IMG4.jpg']
    images = []

    for fname in filenames:
        path = os.path.join(IMAGE_FOLDER, fname)
        img = cv2.imread(path)
        if img is None: 
            if not os.path.isdir(IMAGE_FOLDER):
                return None, f"Image folder not found: {IMAGE_FOLDER}. Please create it and place images."
            return None, f"Missing {fname}. Ensure {fname} is in the 'static/images' folder."
        images.append(resize_image_fixed_width(img, 600))

    print("Trying Auto Stitch...")
    try:
        # Use the default panorama mode
        stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
        status, pano = stitcher.stitch(images)
        
        if status == cv2.Stitcher_OK:
            print("Auto Stitch Successful.")
            return crop_black_borders(pano), None
        
        print(f"Auto Stitch failed with status code {status}. Falling back to manual.")
    except Exception as e:
        print(f"Auto Stitch crashed: {e}")
        pass # Fall through to manual

    print("Running Robust Manual Stitch...")
    try:
        res = images[0]
        for i in range(1, len(images)):
            res = stitch_two_manually(res, images[i])
        
        return crop_black_borders(res), None
    except Exception as e:
        print("Manual Stitching failed!")
        traceback.print_exc()
        return None, f"Manual stitching failed: {str(e)}"


# --- LOGIC: SIFT COMPARISON ---
def compute_sift_logic():
    """Compares a basic Difference of Gaussians (DoG) feature detector against official OpenCV SIFT."""
    path = os.path.join(IMAGE_FOLDER, 'IMG2.jpg')
    img_orig = cv2.imread(path)
    if img_orig is None: 
        return None, None, None, None, "Missing IMG2.jpg. Cannot run SIFT comparison."

    img = resize_image_fixed_width(img_orig, 600)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. My "From Scratch" Logic
    try:
        # Blur the image at two different scales
        g1 = cv2.GaussianBlur(gray, (3, 3), 1.3)
        g2 = cv2.GaussianBlur(gray, (5, 5), 2.6)
        dog = cv2.absdiff(g1, g2) # Difference of Gaussians (DoG) highlights edges/corners

        dog_norm = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX)
        
        # Threshold the DoG result to find the most extreme keypoint candidates
        thresh_val = 20 
        _, dog_thresh = cv2.threshold(dog_norm, thresh_val, 255, cv2.THRESH_BINARY)
        
        # Find contours around these candidates to determine keypoint locations
        contours, _ = cv2.findContours(dog_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        my_keypoints = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 2: 
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    # Calculate the center (centroid) of the blob
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    my_keypoints.append(cv2.KeyPoint(float(cX), float(cY), 5))

        # Draw the custom keypoints in Green
        img_scratch = cv2.drawKeypoints(img, my_keypoints, None, color=(0, 255, 0))
        count_scratch = len(my_keypoints)
    except Exception as e:
        img_scratch = img
        count_scratch = 0
        print(f"Custom SIFT logic failed: {e}")


    # 2. OpenCV Reference (Full SIFT)
    try:
        sift = cv2.SIFT_create()
        kp_cv, _ = sift.detectAndCompute(gray, None)
        # Draw the official, robust keypoints in Blue
        img_cv = cv2.drawKeypoints(img, kp_cv, None, color=(255, 0, 0)) 
        count_cv = len(kp_cv)
    except Exception as e:
        img_cv = img
        count_cv = 0
        print(f"OpenCV SIFT failed: {e}")
        
    return img_scratch, img_cv, count_scratch, count_cv, None


# --- WEB APPLICATION ROUTES ---

@app.route('/')
def home():
    """Default route, redirects users to the main module page."""
    return redirect(url_for('module4_index'))

@app.route('/module4')
def module4_index():
    """Renders the main HTML page for the demo."""
    return render_template('index.html') 

@app.route('/module4/run_stitch', methods=['POST'])
def run_stitch():
    """API endpoint triggered by the user to start the panorama stitching."""
    pano, error = process_stitching()
    if error: 
        return jsonify({'success': False, 'error': error})
    
    # Load a separate 'phone' image to show alongside the panorama
    phone_img = cv2.imread(os.path.join(IMAGE_FOLDER, 'phone.jpg'))
    phone_b64 = mat_to_base64(resize_image_fixed_width(phone_img, 600)) if phone_img is not None else ""

    return jsonify({
        'success': True, 
        'pano_image': mat_to_base64(pano), 
        'phone_image': phone_b64
    })

@app.route('/module4/run_sift', methods=['POST'])
def run_sift():
    """API endpoint to run the feature detection comparison (custom vs. official SIFT)."""
    scratch, cv_ver, count_scratch, count_cv, error = compute_sift_logic()
    if error: 
        return jsonify({'success': False, 'error': error})
        
    return jsonify({
        'success': True, 
        'scratch_img': mat_to_base64(scratch), 
        'cv_img': mat_to_base64(cv_ver),
        'scratch_count': count_scratch, 
        'cv_count': count_cv
    })

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 10000))  
    app.run(host="0.0.0.0", port=port, debug=False)
