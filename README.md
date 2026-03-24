# Geospatial-Image-Segmentation

A geospatial deep-learning image segmentation project implementing modern segmentation architectures (U-Net, LinkNet, DeepLab v3+) to process overhead/satellite imagery and extract land-cover/feature masks.  
Built and maintained by Om Roy.

---

## 🚀 Overview  
This repository contains Jupyter notebooks and trained models for segmenting geospatial imagery. You’ll find:  
- Custom architecture explorations: U-Net, LinkNet, DeepLab v3+  
- Datasets of satellite/overhead imagery with corresponding masks  
- Trained model checkpoints for inference or further fine-tuning  
- Example notebooks demonstrating model training, evaluation and visualization  

---

## 🛠 Features  
- Ability to train segmentation models on geospatial/deep-learning image data  
- Support for multiple architectures for comparison  
- Clear pipeline: data loading → preprocessing → model training → evaluation → prediction visualization  
- Helps land-cover mapping, feature extraction, remote‐sensing workflows  

---

## 🚀 Getting Started

### 🧰 Prerequisites
Make sure you have the following installed:
- Python 3.8+
- Jupyter Notebook or Google Colab
- Required libraries:
  ```bash
  pip install numpy pandas matplotlib opencv-python tensorflow torch torchvision
  ```

### Usage  
1. Clone the repository:  
   ```bash
   git clone https://github.com/omroy07/Geospatial-Image-Segmentation.git
   cd Geospatial-Image-Segmentation
   ```
2. Open a notebook such as:

- `Custom_UNet_LinkNet_DeepLapv3+.ipynb`  
- `LinkNet and UNet Trained Model.ipynb`  

3. Follow the notebook cells to:

1. **Load Dataset** — Import satellite or aerial images along with their segmentation masks.  
2. **Train Models** — Train architectures such as U-Net, LinkNet, or DeepLabV3+ on your dataset.  
3. **Evaluate Segmentation Performance** — Compute metrics like IoU, Dice Score, and Accuracy.  
4. **Visualize Prediction Masks** — Generate and display predicted masks over the original images for qualitative analysis.

## 🧩 Features

- Multiple segmentation architectures: U-Net, LinkNet, and DeepLabV3+  
- End-to-end training and testing pipeline  
- Easy visualization of segmentation results  
- Metrics: IoU, Dice Coefficient, Accuracy  
- Custom dataset compatibility

   git clone https://github.com/omroy07/Geospatial-Image-Segmentation.git
   cd Geospatial-Image-Segmentation
