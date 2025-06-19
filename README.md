# Project 8: Color Image Predictive Coding with Feedback Loop

[![IHT3](https://img.shields.io/badge/Course-IHT3-blue.svg)](https://github.com)
[![Language](https://img.shields.io/badge/Language-C%2B%2B-orange.svg)](https://isocpp.org/)
[![License](https://img.shields.io/badge/License-Academic-green.svg)](LICENSE)

## 📋 Project Overview

This project implements a **predictive coding system for RGB color images** using causal windows and feedback loops. The main objective is to reduce signal entropy by exploiting spatial and inter-color correlations in digital images.

### 🎯 Key Objectives

- **Inter-plane prediction**: Exploit correlations between R, G, B channels
- **Causal window processing**: Use only previously processed pixels (raster-scan order)  
- **Feedback loop implementation**: Prediction based on reconstructed values
- **AR modeling**: Calculate optimal coefficients using least squares method
- **Performance analysis**: Compare global vs local strategies and measure entropy reduction

### 🔧 Core Features

- ✅ **RGB Inter-plane Prediction**: Advanced correlation exploitation between color channels
- ✅ **Feedback Loop**: Critical implementation using reconstructed values instead of originals
- ✅ **Dual Strategy**: Global (image-wide) and Local (32×32 blocks) prediction approaches
- ✅ **Optimal Coefficients**: AR model-based calculation using covariance matrices
- ✅ **Performance Metrics**: Entropy reduction, PSNR, MSE analysis

---

## 🗺️ Project Roadmap

### 📁 Phase 1: Preparation and Base Structure
- **Analysis & Understanding**
  - Study prediction windows for each RGB plane
  - Analyze function prototypes and dependencies
- **Code Architecture**
  - File structure setup (headers, sources)
  - Data structures definition (images, coefficients, errors)
  - Memory management utilities

### 🔍 Phase 2: Causal Windows Implementation  
- **Neighbor Extraction**
  - R plane: 9 neighbors (3 from each RGB plane)
  - G plane: 10 neighbors (9 + 1 additional from R)
  - B plane: 11 neighbors (9 + 2 additional from R,G)
- **Border Handling**
  - Edge case management and raster-scan traversal

### 🧮 Phase 3: Optimal Coefficient Calculation
- **AR Modeling**
  - Autocorrelation matrix computation
  - Linear equation systems construction
- **System Resolution**
  - LU/Cholesky decomposition implementation
  - Numerical stability validation

### 🎛️ Phase 4: Prediction Strategies
- **Global Strategy**: Single coefficient set for entire image
- **Local Strategy**: Block-specific coefficients (32×32 pixels)
- **Adaptive Management**: Transition handling between blocks

### 🔄 Phase 5: Encoding and Decoding
- **Encoding Procedure**
  - Mean calculation and signal centering
  - Prediction error computation
  - Uniform quantization with δ parameter
- **Decoding Procedure**
  - Reconstruction from quantized errors
  - Mean restoration and feedback loop maintenance

### 📊 Phase 6: Evaluation and Comparison
- **Performance Metrics**: Entropy, PSNR, MSE analysis
- **Strategy Comparison**: Global vs Local approaches
- **Window Size Impact**: Reduced window testing
- **Cross vs Band-by-Band**: Inter-plane vs single-plane prediction

### 🚀 Phase 7: Optimization and Finalization
- **Code Optimization**: Matrix calculations and memory management
- **Documentation**: Technical reports and visual demonstrations
- **Testing**: Unit tests and validation suites

---

## 🛠️ Technologies Used

### Programming Languages
- **C++**: Core implementation language

### Development Tools
- **IDE**: Visual Studio / CLion / Code::Blocks
- **Compiler**: GCC 9.0+ 
- **Build System**: CMake 3.15+
- **Version Control**: Git

### Libraries and Dependencies
- **Linear Algebra**: BLAS/LAPACK (optional, for matrix operations)
- **Image I/O**: OpenCV / STB_image / Custom implementations
- **Testing**: Google Test / Catch2

### Mathematical Tools
- **Matrix Operations**: Custom 2D array implementations
- **Statistical Analysis**: Covariance and correlation calculations
- **Optimization**: Least squares solvers
- **Entropy Calculation**: Information theory metrics

---

## 📁 Project Structure

```
project8-predictive-coding/
├── src/
│   ├── core/
│   │   ├── predictor.cpp          # Main prediction algorithms
│   │   ├── ar_model.cpp           # AR coefficient calculation  
│   │   ├── quantizer.cpp          # Quantization/dequantization
│   │   └── entropy.cpp            # Entropy calculation
│   ├── utils/
│   │   ├── image_io.cpp           # Image loading/saving
│   │   ├── matrix_ops.cpp         # Matrix operations
│   │   └── memory_mgmt.cpp        # Memory management
│   └── main.cpp                   # Main application
├── include/
│   ├── predictor.h                # Prediction interfaces
│   ├── ar_model.h                 # AR modeling
│   └── common.h                   # Common definitions
├── tests/
│   ├── test_predictor.cpp         # Unit tests
│   └── test_ar_model.cpp          # AR model tests  
├── data/
│   ├── images/                    # Test images
│   └── results/                   # Output results
├── docs/
│   ├── report.pdf                 # Technical report
│   └── roadmap.pdf                # Project roadmap
├── CMakeLists.txt                 # Build configuration
└── README.md                      # This file
```

---

## 🔬 Key Algorithms Implemented

### 1. Inter-Plane Prediction Equations

```math
R̂(i,j) = Σ(k=1 to 9) r_k × neighbors_k
Ĝ(i,j) = Σ(k=1 to 10) g_k × neighbors_k  
B̂(i,j) = Σ(k=1 to 11) b_k × neighbors_k
```

### 2. Feedback Loop (Critical Implementation)
```cpp
// ✅ CORRECT: Use reconstructed values
prediction = predict_from_neighbors(reconstructed_image, i, j);
error = original[i][j] - prediction;
quantized_error = quantize(error, delta);
reconstructed_image[i][j] = prediction + quantized_error; // Feedback!

// ❌ WRONG: Using original values breaks decoder synchronization
```

### 3. AR Coefficient Calculation
```math
R × coefficients = cross_correlation_vector
```
Where R is the autocorrelation matrix solved using LU decomposition.

---

## 📖 Academic References

### Fundamental Papers
1. **"Interplane prediction for RGB video coding"** - IEEE Conference Publication
   - Efficient RGB space video coding with up to 40% efficiency gains
   - [IEEE Xplore](https://ieeexplore.ieee.org/document/1419415)

2. **"High-Fidelity RGB Video Coding Using Adaptive Inter-Plane Weighted Prediction"** - IEEE Journals
   - Adaptive inter-plane-weighted prediction algorithm for RGB signals
   - [IEEE Xplore](https://ieeexplore.ieee.org/document/4811977/)

3. **"Region Adaptive Inter-Color Prediction Approach to RGB 4:4:4 Intra Coding"** - IEEE Conference
   - Region-adaptive inter-color prediction with 26-30% bit-rate savings
   - [IEEE Xplore](https://ieeexplore.ieee.org/document/5673985)

### Theoretical Background
4. **"Predictive Coding - Overview"** - ScienceDirect Topics
   - Comprehensive overview of predictive approaches for image compression
   - [ScienceDirect](https://www.sciencedirect.com/topics/computer-science/predictive-coding)

5. **"A lossless image coding technique exploiting spectral correlation on the RGB space"** - IEEE Conference
   - Spectral correlation exploitation in RGB space for lossless coding
   - [IEEE Xplore](https://ieeexplore.ieee.org/document/7079737)

6. **"Image Data Compression by Predictive Coding I: Prediction Algorithms"** (1974)
   - Foundational paper on predictive coding techniques
   - [ResearchGate](https://www.researchgate.net/publication/224104452_Image_Data_Compression_by_Predictive_Coding_I_Prediction_Algorithms)

### Modern Applications
7. **"Linear prediction image coding using iterated function systems"** - ScienceDirect
   - Hybrid LP-IFS system combining linear prediction with 2D AR models
   - [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S026288569800153X)

8. **"Lossless image compression via predictive coding of discrete Radon projections"** - ScienceDirect
   - Advanced predictive coding methods in transform domains
   - [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0923596508000192)

### Classic References
- **Makhoul (1975)** - "Linear Prediction: A Tutorial Review" (Proceedings IEEE)
- **Jain (1981)** - "Image data compression: A review" (Proceedings IEEE)
- **Gonzalez & Woods** - "Digital Image Processing" (Pearson)

---

## 🚀 Getting Started

### Prerequisites
- C++ compiler with C++11 support
- CMake 3.15 or higher
- Git for version control

### Building the Project
```bash
# Clone the repository
git clone <repository-url>
cd project8-predictive-coding

# Create build directory
mkdir build && cd build

# Configure and build
cmake ..
make

# Run tests
make test
```

### Usage Example
```bash
# Run global prediction strategy
./predictive_coding --input image.ppm --strategy global --delta 5

# Run local prediction strategy  
./predictive_coding --input image.ppm --strategy local --block-size 32

# Performance analysis
./predictive_coding --input image.ppm --analyze --output results.csv
```

---

## ⚠️ Critical Implementation Notes

### 🔴 Feedback Loop - CRUCIAL
**The most critical aspect**: Always use reconstructed values for prediction, never original values. This maintains perfect synchronization between encoder and decoder.

### 🔶 Causal Windows
Strictly respect raster-scan order. Never use "future" pixels that haven't been processed yet.

### 🔵 Numerical Stability  
Handle ill-conditioned matrices carefully and use appropriate data types (double precision).

---

## 📈 Expected Results

### Performance Metrics
- **Entropy Reduction**: 30-50% compared to original image
- **Compression Ratio**: 2:1 to 4:1 depending on image content
- **Quality**: Lossless reconstruction with δ=0
- **Speed**: Real-time processing for images up to 1024×1024

### Comparison Outcomes
- Local strategy typically outperforms global for complex images
- Inter-plane prediction shows significant gains over single-plane
- Optimal window size varies with image texture complexity

---

## 🤝 Contributing

This is an academic project for the IHT3 course. For questions or discussions, please refer to the course documentation or contact the development team.

---

## 📄 License

This project is developed for academic purposes as part of the IHT3 - 2D/3D Visual Data Compression course.

---

**Course**: IHT3 - 2D and 3D Visual Data Compression  
**Project**: 8 - Color Image Predictive Coding with Feedback Loop  
**Academic Year**: 2024-2025
