# Slicer UI for BioSAM-2

# Important
- Still in development, use it at your own risk.
- Currently only the image prediction is available.
- Only default resolution (1024x1024)

# Installation
```shell
cd slicerUI
pip install -r requirements.txt
```
### ONNX Runtime
For Nvidia GPU computers:
`pip install onnxruntime-gpu`

Otherwise:
`pip install onnxruntime`

# ONNX model
- ONNX model is available at [LINK](https://drive.google.com/drive/folders/1SVdFWAK6AyU2OnKfU_kSIu2bQsuNgKQF?usp=sharing)
- Place the models in the `models` folder

## **BioSAM-2 Annotation App**:
 ```shell
 python annotation_app.py
 ```
Annotation Controls
- **Left click**: Adds a positive point, click one more time to delete it.
- **Right click**: Adds a negative point
- **Left click and drag**: Draws a rectangle
- **Add label button**: Adds a new label for annotation
- **Delete label button**: Deletes the last label
- **Save label button**: Save the mask & image

# References:
* SAM2 Repository: [https://github.com/facebookresearch/segment-anything-2](https://github.com/facebookresearch/segment-anything-2)
* ONNX-SAM2-Segment-Anything: [https://github.com/ibaiGorordo/ONNX-SAM2-Segment-Anything](https://github.com/ibaiGorordo/ONNX-SAM2-Segment-Anything)

