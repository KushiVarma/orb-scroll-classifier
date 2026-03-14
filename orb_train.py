import os
import cv2
import numpy as np
import joblib

DATA_DIR = "data"

# ================= ORB CONFIG =================
orb = cv2.ORB_create(
    nfeatures=1500,
    scaleFactor=1.2,
    nlevels=8,
    edgeThreshold=5,
    patchSize=31
)
# ================= CAMERA AUGMENT =================
def camera_augment(img):

    h, w = img.shape[:2]

    # Perspective warp
    shift = 0.08 * w
    pts1 = np.float32([[0,0],[w,0],[0,h],[w,h]])
    pts2 = np.float32([
        [np.random.uniform(0,shift), np.random.uniform(0,shift)],
        [w-np.random.uniform(0,shift), np.random.uniform(0,shift)],
        [np.random.uniform(0,shift), h-np.random.uniform(0,shift)],
        [w-np.random.uniform(0,shift), h-np.random.uniform(0,shift)]
    ])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    img = cv2.warpPerspective(img, M, (w,h))

    # Brightness
    alpha = np.random.uniform(0.7, 1.3)
    img = np.clip(img * alpha, 0, 255).astype(np.uint8)

    # Blur
    if np.random.rand() > 0.6:
        img = cv2.GaussianBlur(img, (3,3), 0)

    # Noise
    if np.random.rand() > 0.6:
        noise = np.random.normal(0, 8, img.shape)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)

    return img

# ================= DATABASE =================
database = {
    "real": [],
    "fake": []
}

# ================= TRAIN =================
def process_folder(label):

    folder = os.path.join(DATA_DIR, label)

    for file in os.listdir(folder):

        path = os.path.join(folder, file)

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        img = cv2.resize(img, (400,400))

        # Original
        kp, des = orb.detectAndCompute(img, None)
        if des is not None:
            database[label].append(des)

        # Augmented views
        for _ in range(12):
            aug = camera_augment(img)
            kp, des = orb.detectAndCompute(aug, None)
            if des is not None:
                database[label].append(des)

print("Processing REAL...")
process_folder("real")

print("Processing FAKE...")
process_folder("fake")

joblib.dump(database, "orb_database.joblib")

print("Database saved.")
print("Real descriptors:", len(database["real"]))
print("Fake descriptors:", len(database["fake"]))
