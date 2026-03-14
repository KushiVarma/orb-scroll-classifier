import cv2
import numpy as np
import joblib

# ================= LOAD DATABASE =================
database = joblib.load("orb_database.joblib")

# ================= PI OPTIMIZED ORB =================
orb = cv2.ORB_create(
    nfeatures=1500,
    scaleFactor=1.2,
    nlevels=8,
    edgeThreshold=5,
    patchSize=31
)

bf = cv2.BFMatcher(cv2.NORM_HAMMING)

# ================= MATCH FUNCTION =================
def match_score(des_live, des_list):

    if des_live is None:
        return 0

    total_good = 0

    for des_db in des_list:

        matches = bf.knnMatch(des_live, des_db, k=2)

        good_count = 0
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_count += 1

            # Early exit if already strong match
            if good_count > 25:
                break

        total_good += good_count

        # Early exit global
        if total_good > 60:
            break

    return total_good

# ================= CLASSIFIER =================
def classify_orb(roi):

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Smaller ORB input = big speed gain
    gray = cv2.resize(gray, (160,160))

    # SYMBOL MASK
    _, mask = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Remove noise
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Early exit → skip useless ORB compute
    if cv2.countNonZero(mask) < 400:
        return "UNKNOWN", 0

    symbol = cv2.bitwise_and(gray, gray, mask=mask)

    kp, des_live = orb.detectAndCompute(symbol, None)

    if des_live is None or len(kp) < 6:
        return "UNKNOWN", 0

    real_score = match_score(des_live, database["real"])
    fake_score = match_score(des_live, database["fake"])

    total = real_score + fake_score

    if total < 5:
        return "UNKNOWN", total

    # Normalized scoring
    real_norm = real_score / len(des_live)
    fake_norm = fake_score / len(des_live)

    # Soft decision margin
    if real_norm > fake_norm * 1.15:
        return "REAL KFS", real_norm
    elif fake_norm > real_norm * 1.15:
        return "FAKE KFS", fake_norm
    else:
        return "UNCERTAIN", max(real_norm, fake_norm)

# ================= CAMERA =================
cap = cv2.VideoCapture(0)

# Optional: Lower camera resolution → BIG Pi boost
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_count = 0
FRAME_SKIP = 2   # Process every 2nd frame

while True:

    ret, frame = cap.read()
    if not ret:
        continue

    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        cv2.imshow("ORB Pi Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    frame = cv2.resize(frame, (640,480))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # STRICT WHITE FILTER → reduces shirt detection
    lower_white = np.array([0, 0, 170])
    upper_white = np.array([180, 70, 255])

    mask = cv2.inRange(hsv, lower_white, upper_white)

    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in contours:

        area = cv2.contourArea(cnt)
        if area < 4000:
            continue

        x,y,w,h = cv2.boundingRect(cnt)

        roi = frame[y:y+h, x:x+w]
        if roi.size == 0:
            continue

        label, score = classify_orb(roi)

        if "REAL" in label:
            color = (0,255,0)
        elif "FAKE" in label:
            color = (0,0,255)
        else:
            color = (0,255,255)

        cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)

        cv2.putText(
            frame,
            f"{label} {score:.2f}",
            (x,y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    cv2.imshow("ORB Pi Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()