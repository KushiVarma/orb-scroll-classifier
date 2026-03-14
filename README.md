# ORB-based Image Authenticity Detection

This project detects whether an image is real or fake using ORB feature extraction.

Dataset
- 30 real images
- 30 fake images

Method
- ORB feature extraction
- Descriptor matching

Files
- orb_train.py : trains the ORB model
- orb_infer.py : predicts real vs fake

Run Training
python src/orb_train.py

Run Inference
python src/orb_infer.py