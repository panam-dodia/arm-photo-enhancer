# Arm Photo Enhancer - Multi-Degradation Image Restoration

## Project Overview

Arm Photo Enhancer is an on-device Android application that combines multiple state-of-the-art machine learning models to enhance and restore degraded photographs. The application leverages Arm's NNAPI acceleration to deliver high-quality image processing directly on mobile devices without requiring cloud connectivity or external servers.

### What Makes This Project Interesting

This project demonstrates the power of on-device AI by implementing a complete multi-degradation restoration pipeline on mobile hardware. Unlike simple filter-based enhancement apps, this application uses deep learning models to intelligently analyze and restore images affected by various degradation types including noise, blur, low light, and compression artifacts. The application showcases how modern Arm-based mobile processors can handle computationally intensive AI workloads that were previously only possible on desktop workstations or cloud infrastructure.

### Why This Project Should Win

1. **Complete ML Pipeline**: Implements three distinct neural network models (Zero-DCE for enhancement, DA-CLIP encoder for degradation analysis, and UNet for multi-step restoration) working in concert to deliver professional-grade results.

2. **True On-Device Processing**: All computation happens locally on the device, ensuring user privacy, eliminating network latency, and enabling offline functionality.

3. **Production-Ready Implementation**: Features a polished user interface with real-time before/after comparison, background processing with progress notifications, and seamless integration with Android's media system.

4. **Advanced Image Processing**: Implements Stochastic Differential Equation-based diffusion for restoration, representing cutting-edge research in computational photography.

5. **Optimized for Arm**: Fully utilizes Arm NNAPI acceleration and implements careful memory management to run complex models efficiently on mobile hardware.

## Functionality

### Core Features

#### 1. Quick Enhancement (Zero-DCE)
The application provides instant low-light image enhancement using Zero-DCE (Zero-Reference Deep Curve Estimation). This model intelligently adjusts image exposure and contrast without requiring reference images, making it ideal for improving photos taken in challenging lighting conditions.

#### 2. Multi-Degradation Restoration (DA-CLIP + UNet)
The restoration pipeline uses a two-stage approach:
- **Stage 1 - Degradation Analysis**: DA-CLIP (Degradation-Aware CLIP) encoder analyzes the input image to identify specific degradation types and generates context embeddings that guide the restoration process.
- **Stage 2 - Iterative Restoration**: A UNet model performs multi-step diffusion-based restoration, progressively removing noise, blur, and artifacts while preserving image details. The model uses the degradation context from DA-CLIP to adapt its restoration strategy to the specific problems present in each image.

#### 3. Interactive Comparison
A custom-built comparison slider allows users to drag between before and after versions of their images, providing immediate visual feedback on the enhancement results.

#### 4. Background Processing
Long-running restoration operations execute as foreground services, allowing users to minimize the application or use other apps while processing continues. Progress updates appear in the notification bar.

#### 5. Seamless Gallery Integration
Enhanced images save directly to the Android media library with proper metadata, and users receive tappable notifications to view their saved images immediately.

### Technical Implementation

**Model Architecture:**
- Zero-DCE: Lightweight convolutional network for exposure correction
- DA-CLIP Encoder: Vision transformer-based degradation analyzer
- UNet: Modified U-Net architecture with attention mechanisms for restoration

**Image Processing Pipeline:**
1. EXIF orientation correction ensures proper image display
2. Intelligent resizing with high-quality anti-aliasing preserves edge detail
3. Aspect ratio preservation with FIT_CENTER scaling
4. 32-bit floating-point tensor processing for maximum quality
5. Iterative refinement using 100-step diffusion process

**Memory Management:**
- Strategic model loading: Enhancement model loads at startup, restoration models load on-demand
- Explicit memory cleanup between operations
- Automatic garbage collection before intensive processing
- Careful bitmap management with proper disposal

**User Interface:**
- Material Design 3 with dark theme optimized for viewing photographs
- Clean status updates without intrusive popups
- Horizontal progress bar for restoration tracking
- Responsive layout adapting to different screen sizes

## Setup Instructions

### Prerequisites
- Android device with Arm64 (AArch64) processor
- Android 8.0 (API level 26) or higher
- Android Studio Koala or later
- 4GB+ device RAM recommended for optimal performance

### Building the Project

1. **Clone the Repository**
```bash
git clone https://github.com/yourusername/Arm_Hackthon.git
cd Arm_Hackthon
```

2. **Open in Android Studio**
- Launch Android Studio
- Select "Open an Existing Project"
- Navigate to the cloned directory and select it
- Wait for Gradle sync to complete

3. **Prepare Model Files**
Ensure the following ONNX model files are present in `app/src/main/assets/models/`:
- `zerodce_iter8.onnx` (Zero-DCE enhancement model)
- `daclip_encoder.onnx` (DA-CLIP degradation encoder)
- `unet.onnx` (UNet restoration model)

4. **Build the APK**
- Select "Build" > "Build Bundle(s) / APK(s)" > "Build APK(s)"
- Wait for build completion
- APK will be generated in `app/build/outputs/apk/debug/`

### Installing on Device

#### Option 1: Direct Installation via Android Studio
1. Enable Developer Options on your Arm Android device:
   - Go to Settings > About Phone
   - Tap "Build Number" seven times
   - Return to Settings > Developer Options
   - Enable "USB Debugging"

2. Connect device via USB and authorize the computer

3. Click "Run" in Android Studio and select your device

#### Option 2: Manual APK Installation
1. Transfer the APK to your device
2. Enable "Install from Unknown Sources" in device settings
3. Open the APK file and follow installation prompts

### Running the Application

1. **Grant Permissions**
   - On first launch, grant storage and notification permissions
   - These are required for image access and progress notifications

2. **Select an Image**
   - Tap "SELECT IMAGE" button
   - Choose a photo from your gallery
   - Image will load with proper orientation correction

3. **Choose Enhancement Type**
   - **QUICK ENHANCE**: Fast enhancement for low-light images
   - **RESTORE IMAGE**: Multi-degradation restoration for heavily degraded photos

4. **Compare Results**
   - Drag the slider left and right to compare before and after
   - Pinch to zoom for detail inspection

5. **Save Enhanced Image**
   - Tap "SAVE" to export to gallery
   - Tap the notification to view saved image
   - Find saved images in Pictures/ArmEnhancer folder

### Performance Notes

- Enhancement operations process in real-time
- Restoration operations run as background services
- Maximum input resolution is automatically managed for optimal memory usage
- All processing occurs on-device using Arm NNAPI acceleration
- No internet connection required

### Troubleshooting

**App crashes on launch:**
- Verify device has sufficient free RAM
- Ensure all model files are correctly placed in assets/models/
- Check that device meets minimum API level 26

**Images appear rotated:**
- The app automatically corrects EXIF orientation
- If issues persist, check source image metadata

**Slow processing:**
- Close other applications to free memory
- Processing speed varies by device capabilities
- Restoration is designed as a background operation

**Save fails:**
- Ensure storage permissions are granted
- Verify sufficient storage space available
- Check that media library is accessible

## Project Structure

```
Arm_Hackthon/
├── app/
│   ├── src/
│   │   ├── main/
│   │   │   ├── assets/
│   │   │   │   └── models/          # ONNX model files
│   │   │   ├── java/
│   │   │   │   └── com/panam/arm_hackthon/
│   │   │   │       ├── MainActivity.kt
│   │   │   │       ├── RestorationService.kt
│   │   │   │       ├── ml/
│   │   │   │       │   ├── ZeroDCE.kt
│   │   │   │       │   ├── DAClipEncoder.kt
│   │   │   │       │   └── UNet.kt
│   │   │   │       └── views/
│   │   │   │           └── ImageComparisonSlider.kt
│   │   │   ├── res/
│   │   │   │   ├── layout/
│   │   │   │   │   └── activity_main.xml
│   │   │   │   └── values/
│   │   │   │       └── themes.xml
│   │   │   └── AndroidManifest.xml
│   │   └── build.gradle.kts
│   └── ...
└── README.md
```

## Technologies Used

- **Language**: Kotlin
- **ML Framework**: ONNX Runtime with Arm NNAPI
- **UI Framework**: Android Material Design 3
- **Architecture**: MVVM with Kotlin Coroutines
- **Image Processing**: Android Bitmap API with custom high-quality scaling
- **Background Processing**: Android Foreground Services
- **Notifications**: Android NotificationCompat

## License

This project was developed for the Arm Hackathon 2025.

---

This project demonstrates that sophisticated AI-powered image processing can run efficiently on Arm-based mobile devices, bringing professional-grade computational photography tools directly to users' hands while maintaining privacy through on-device processing.
