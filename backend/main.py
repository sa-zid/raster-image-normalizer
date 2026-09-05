from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import io
import base64
from PIL import Image

app = FastAPI(title="Raster Image Normalizer API")

# Netlify ফ্রন্টএন্ড থেকে রিকোয়েস্ট আসার জন্য CORS অনুমতি
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # পরে Netlify ডোমেইন ইউআরএল দিতে পারেন
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def compute_stats(img_array):
    """ইমেজের পিক্সেল পরিসংখ্যান বের করে"""
    return {
        "min": float(np.min(img_array)),
        "max": float(np.max(img_array)),
        "mean": float(np.mean(img_array)),
        "std": float(np.std(img_array))
    }

def min_max_normalize(img, target_min=0, target_max=255):
    img_float = img.astype(np.float32)
    min_val, max_val = np.min(img_float), np.max(img_float)
    if max_val == min_val:
        return np.full_like(img, target_min, dtype=np.uint8)
    normalized = (img_float - min_val) / (max_val - min_val)
    scaled = normalized * (target_max - target_min) + target_min
    return np.clip(scaled, 0, 255).astype(np.uint8)

def percentile_clip_normalize(img, low_p=1, high_p=99):
    img_float = img.astype(np.float32)
    low_val, high_val = np.percentile(img_float, (low_p, high_p))
    clipped = np.clip(img_float, low_val, high_val)
    if high_val == low_val:
        return np.full_like(img, 0, dtype=np.uint8)
    normalized = (clipped - low_val) / (high_val - low_val) * 255.0
    return np.clip(normalized, 0, 255).astype(np.uint8)

def z_score_normalize(img):
    img_float = img.astype(np.float32)
    mean, std = np.mean(img_float), np.std(img_float)
    if std == 0:
        return np.zeros_like(img, dtype=np.uint8)
    standardized = (img_float - mean) / std
    # z-score (সাধারণত -3 থেকে +3) কে 0-255 এ রূপান্তর
    scaled = (standardized + 3) / 6.0 * 255.0
    return np.clip(scaled, 0, 255).astype(np.uint8)

def histogram_equalization(img):
    if len(img.shape) == 2:
        return cv2.equalizeHist(img)
    elif len(img.shape) == 3:
        # YUV কালার স্পেসে নিয়ে Luminance চ্যানেল ইককুয়ালাইজ করা
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    return img

@app.post("/api/normalize")
async def normalize_image(
    file: UploadFile = File(...),
    method: str = Form("minmax"),
    target_min: float = Form(0),
    target_max: float = Form(255),
    grayscale: bool = Form(False)
):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        if grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        orig_stats = compute_stats(img)

        # নির্বাচিত মেথড অনুযায়ী প্রসেসিং
        if method == "minmax":
            processed = min_max_normalize(img, target_min, target_max)
        elif method == "percentile":
            processed = percentile_clip_normalize(img, 1, 99)
        elif method == "zscore":
            processed = z_score_normalize(img)
        elif method == "histeq":
            processed = histogram_equalization(img)
        else:
            processed = min_max_normalize(img, target_min, target_max)

        norm_stats = compute_stats(processed)

        # ইমেজ কোড করে Base64 স্ট্রিং এ রূপান্তর
        _, encoded_img = cv2.imencode('.png', processed)
        base64_str = base64.b64encode(encoded_img).decode('utf-8')

        return {
            "status": "success",
            "original_stats": orig_stats,
            "normalized_stats": norm_stats,
            "image": f"data:image/png;base64,{base64_str}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
