# AR Predictive Coding for Color Images

[![IHT3](https://img.shields.io/badge/Course-IHT3-blue.svg)](https://github.com)
[![Language](https://img.shields.io/badge/Language-MATLAB-orange.svg)](https://www.mathworks.com/products/matlab.html)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Project Overview

This project implements **Auto-Regressive (AR) Predictive Coding for RGB Color Images** using inter-plane prediction and causal windows. The implementation exploits spatial and spectral correlations between RGB channels to achieve efficient image compression through prediction.

### 🎯 Key Objectives

- **Inter-plane AR prediction**: Exploit correlations between R, G, B channels using optimal coefficients
- **Causal window processing**: Use symmetric boundary extension and raster-scan order
- **Optimal coefficient calculation**: Compute AR parameters using covariance matrices and least squares
- **Performance analysis**: Measure prediction accuracy and entropy reduction

### 🔧 Core Features

- ✅ **Optimal AR Coefficients**: Automatic calculation using covariance-based least squares
- ✅ **Inter-plane Prediction**: Advanced correlation exploitation between RGB channels
- ✅ **Symmetric Boundary Extension**: Proper handling of image borders using mirroring
- ✅ **Entropy Analysis**: Built-in entropy calculation for performance evaluation
- ✅ **Error Matrix Computation**: Detailed prediction error analysis

---

## 🗺️ Implementation Structure

### 🧮 **Phase 1: AR Coefficient Calculation (`Cal_para.m`)**
- **Covariance Matrix Construction**: Build autocorrelation matrices for each channel
- **Inter-channel Correlation**: Calculate cross-correlation terms between R, G, B
- **Linear System Solution**: Solve `K × coefficients = Y` for optimal AR parameters
- **Channel-specific Models**:
  - **R Channel**: 6 coefficients (2 spatial × 3 spectral)
  - **G Channel**: 7 coefficients (6 + current R pixel)  
  - **B Channel**: 8 coefficients (6 + current R + current G pixels)

### 🔍 **Phase 2: RGB Prediction (`Predict_RGB.m`)**
- **Boundary Extension**: Symmetric padding for causal window extraction
- **Pixel-by-pixel Prediction**: Raster-scan traversal with causal neighbors
- **Inter-plane Dependencies**:
  - R prediction: Uses spatial neighbors from all RGB channels
  - G prediction: Adds dependency on current R pixel
  - B prediction: Adds dependencies on current R and G pixels
- **Range Limitation**: Clamp predicted values to [0, 255]

### 📊 **Phase 3: Error Analysis (`erreur.m`)**
- **Error Matrix Calculation**: Pixel-wise difference between original and predicted
- **Entropy Computation**: Information-theoretic analysis of prediction quality
- **Performance Metrics**: Quantitative assessment of compression potential

---

## 🛠️ Technologies Used

### Programming Language
- **MATLAB R2020b+**: Core implementation and matrix operations
- **Image Processing Toolbox**: Image I/O and visualization

### Mathematical Framework
- **Linear Algebra**: Covariance matrices and least squares solving
- **Information Theory**: Entropy calculation for compression analysis  
- **Signal Processing**: Symmetric boundary extension and causal filtering

---

## 📁 Project Structure

```
AR-Predictive-Coding-for-Color-Images/
├── Cal_para.m              # AR coefficient calculation
├── Predict_RGB.m           # RGB inter-plane prediction  
├── erreur.m                # Error analysis and entropy calculation
├── hh.md                   # Mathematical formulation (LaTeX)
├── docs/                   # Project documentation
│   ├── subject.pdf         # Course assignment details
│   └── roadmap.pdf         # Implementation roadmap
├── images/                 # Test images and results
│   ├── test/              # Input test images
│   └── results/           # Prediction outputs
├── LICENSE                 # MIT License
└── README.md              # This file
```

---

## 🔬 Mathematical Formulation

### 1. AR Coefficient Matrices

**R Channel (6×6 matrix):**
```matlab
Kr = [RR00 RR11 RG00 RG11 RB00 RB11;
      RR11 RR00 GR11 RG00 BR11 RB00;
      RG00 GR11 GG00 GG11 GB00 GB11;
      RG11 RG00 GG11 GG00 BG11 GB00;
      RB00 BR11 GB00 BG11 BB00 BB11;
      RB11 RB00 GB11 GB00 BB11 BB00];
```

**G Channel (7×7 matrix):** Extends R matrix with additional R(i,j) terms

**B Channel (8×8 matrix):** Extends G matrix with additional R(i,j) and G(i,j) terms

### 2. Prediction Equations

```matlab
% R prediction (6 coefficients)
R̂(i,j) = r₁×R(i-1,j) + r₂×R(i,j-1) + r₃×G(i-1,j) + r₄×G(i,j-1) + r₅×B(i-1,j) + r₆×B(i,j-1)

% G prediction (7 coefficients)  
Ĝ(i,j) = g₁×R(i-1,j) + g₂×R(i,j-1) + g₃×G(i-1,j) + g₄×G(i,j-1) + g₅×B(i-1,j) + g₆×B(i,j-1) + g₇×R(i,j)

% B prediction (8 coefficients)
B̂(i,j) = b₁×R(i-1,j) + b₂×R(i,j-1) + b₃×G(i-1,j) + b₄×G(i,j-1) + b₅×B(i-1,j) + b₆×B(i,j-1) + b₇×R(i,j) + b₈×G(i,j)
```

### 3. Covariance Terms
The implementation calculates spatial and spectral covariance terms:
- **Spatial**: `(0,0)`, `(-1,0)`, `(0,-1)`, `(-1,1)`  
- **Spectral**: All combinations between R, G, B channels

---

## 🚀 Getting Started

### Prerequisites
- MATLAB R2020b or later
- Image Processing Toolbox
- Test RGB images (recommended: natural images, 256×256 or larger)

### Usage

#### 1. Calculate AR Coefficients
```matlab
% Load and analyze an image to get optimal coefficients
[r_coeffs, g_coeffs, b_coeffs] = Cal_para('path/to/your/image.jpg');

% Display coefficient values
fprintf('R coefficients: '); disp(r_coeffs');
fprintf('G coefficients: '); disp(g_coeffs');  
fprintf('B coefficients: '); disp(b_coeffs');
```

#### 2. Perform RGB Prediction
```matlab
% Use calculated coefficients to predict image
[R_pred, G_pred, B_pred] = Predict_RGB('path/to/your/image.jpg', r_coeffs, g_coeffs, b_coeffs);

% The function automatically displays the predicted image
% Predicted channels are returned as separate matrices
```

#### 3. Analyze Prediction Quality
```matlab
% Load original image
original = imread('path/to/your/image.jpg');
predicted = cat(3, R_pred, G_pred, B_pred);

% Calculate prediction error matrix  
error_matrix = calculerMatriceErreur(original, predicted);

% Calculate entropy of original vs predicted
H_original = calc_entropie(original);
H_predicted = calc_entropie(uint8(predicted));
H_error = calc_entropie(error_matrix);

fprintf('Original entropy: %.3f bits\n', H_original);
fprintf('Predicted entropy: %.3f bits\n', H_predicted);
fprintf('Error entropy: %.3f bits\n', H_error);
fprintf('Compression potential: %.1f%%\n', (1-H_error/H_original)*100);
```

---

## 🔬 Key Implementation Details

### **Symmetric Boundary Extension**
```matlab
% Mirror padding for causal window extraction
Rp = padarray(R, [1 1], 'symmetric');
Gp = padarray(G, [1 1], 'symmetric'); 
Bp = padarray(B, [1 1], 'symmetric');
```

### **Causal Neighbor Extraction**
```matlab
% Extract causal neighbors (top, left) for each channel
R_t = Rp(x-1,y);   R_l = Rp(x,y-1);     % R spatial neighbors
G_t = Gp(x-1,y);   G_l = Gp(x,y-1);     % G spatial neighbors  
B_t = Bp(x-1,y);   B_l = Bp(x,y-1);     % B spatial neighbors
R_c = Rp(x,y);     G_c = Gp(x,y);       % Current pixel values (for G, B prediction)
```

### **AR Coefficient Calculation**
```matlab
% Solve linear system: K × coefficients = Y
r = Kr \ Yr;  % R channel coefficients
g = Kg \ Yg;  % G channel coefficients  
b = Kb \ Yb;  % B channel coefficients
```

---

## 📊 Performance Analysis

### Expected Results
- **Prediction Accuracy**: High correlation between original and predicted images
- **Entropy Reduction**: 20-40% reduction in error entropy vs original
- **Visual Quality**: Predicted images should closely match originals
- **Coefficient Stability**: Consistent coefficients across similar image regions

### Evaluation Metrics
1. **Prediction Error**: Mean Squared Error (MSE) between original and predicted
2. **Entropy Analysis**: Information content of prediction errors
3. **Visual Inspection**: Qualitative assessment of prediction accuracy
4. **Coefficient Analysis**: Magnitude and stability of AR parameters

---

## 📖 Academic References

### Fundamental Theory
1. **Makhoul, J. (1975)** - "Linear Prediction: A Tutorial Review", Proc. IEEE
2. **Jain, A.K. (1981)** - "Image data compression: A review", Proc. IEEE
3. **Netravali, A.N. & Limb, J.O. (1980)** - "Picture coding: A review", Proc. IEEE

### RGB Predictive Coding
4. **"Interplane prediction for RGB video coding"** - IEEE Conference
   - [IEEE Xplore](https://ieeexplore.ieee.org/document/1419415)
5. **"High-Fidelity RGB Video Coding Using Adaptive Inter-Plane Weighted Prediction"** - IEEE Journals  
   - [IEEE Xplore](https://ieeexplore.ieee.org/document/4811977/)
6. **"A lossless image coding technique exploiting spectral correlation on the RGB space"** - IEEE Conference
   - [IEEE Xplore](https://ieeexplore.ieee.org/document/7079737)

### Advanced Applications
7. **"Linear prediction image coding using iterated function systems"** - ScienceDirect
   - [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S026288569800153X)
8. **"Predictive Coding - Overview"** - ScienceDirect Topics
   - [ScienceDirect](https://www.sciencedirect.com/topics/computer-science/predictive-coding)

---

## ⚠️ Implementation Notes

### 🔴 Causal Window Constraint
The implementation strictly respects causality by using only previously processed pixels (top and left neighbors) in raster-scan order.

### 🔶 Symmetric Boundary Handling  
Border pixels are handled using symmetric extension, which provides better prediction accuracy than zero-padding.

### 🔵 Matrix Conditioning
The covariance matrices are generally well-conditioned for natural images, but numerical stability should be monitored for synthetic or highly regular images.

### 🟡 Inter-plane Dependencies
The hierarchical prediction structure (R → G → B) captures the natural correlation structure of RGB color spaces effectively.

---

## 🤝 Contributing

This is an academic project for the IHT3 course on 2D/3D Visual Data Compression. The implementation serves as a foundation for understanding AR-based predictive coding in color images.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Course**: IHT3 - 2D and 3D Visual Data Compression  
**Project**: Auto-Regressive Predictive Coding for Color Images  
**Implementation**: MATLAB with Inter-plane Prediction  
**Academic Year**: 2024-2025