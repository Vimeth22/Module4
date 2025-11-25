# Module 4 – Panorama Synthesis & SIFT Feature Analysis

This module implements an interactive Flask-based web application for two major computer vision tasks:

1. **Panoramic Image Synthesis (Part 1)**
2. **SIFT Feature Descriptor Analysis (Part 2)**

The user interface is built using Bootstrap 5 with a dark academic-themed design.

## 1. Project Structure

```
Module4/
└── static/
│    └── images/
         ├── IMG1.jpg
         ├── IMG2.jpg
         ├── IMG3.jpg
         ├── IMG4.jpg
         └── phone.jpg
│── templates/
│    └── index.html         
│── app.py
```

API Endpoints required:
- POST `/module4/run_stitch`
- POST `/module4/run_sift`

---

## 2. Features

### Part 1 — Panoramic Image Synthesis
- Multi-image registration and blending
- Automatically stitches images IMG1–IMG4
- Displays:
  - Synthesized panorama
  - Mobile panorama ground truth

Expected backend JSON response:
```
{
  "success": true,
  "pano_image": "<BASE64_IMAGE>",
  "phone_image": "<BASE64_IMAGE>"
}
```

---

### Part 2 — SIFT Feature Descriptor Analysis
Compares:
- Custom DoG-based SIFT implementation
- OpenCV SIFT

Displays:
- Two SIFT visualizations
- Feature counts for each method

Expected backend JSON response:
```
{
  "success": true,
  "scratch_img": "<BASE64_IMAGE>",
  "cv_img": "<BASE64_IMAGE>",
  "scratch_count": <int>,
  "cv_count": <int>
}
```

---

## 3. Running the Application

Install dependencies:
```
pip install flask opencv-python numpy
```

Run the Flask server:
```
python app.py
```

Open in your browser:
```
Click the link that comes when you run the app.py

 * Running on http://127.0.0.1:5000
```

---

## 4. Output
- Once you go to the web page there are two tabs which is part 1 and part 2.
- Click part 1 and click initiate Panorama Synthesis blue box for the result. 
- Click part 2 and click Execute Keypoint Detection Analysi green box for the results.

---

## 5. Notes
- Backend must return base64‑encoded JPEG images.
- JavaScript fetch() handles real-time updates.
- Requires Bootstrap 5 (already included via CDN).

---

## 6. Summary
This module provides a full interactive environment for:
- Multi-image panorama stitching
- SIFT feature comparison (custom vs OpenCV)
- Real-time visualization
- Research-oriented evaluation
