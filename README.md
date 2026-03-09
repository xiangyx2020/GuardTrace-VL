<h1 align="center">  <em>GuardTrace-VL: </em> Detecting Unsafe Multimodel Reasoning via Iterative Safety Supervision</h1>

<div align="center" style="line-height: 1; ">
  <!-- Huggingface Model -->
  <a href="https://huggingface.co/DloadingX/GuardTrace-VL-3B" target="_blank" style="margin: 2px;">
    <img alt="Huggingface Model" src="https://img.shields.io/badge/%F0%9F%A4%97%20Model-GuardTrace--VL--3B-green&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <!-- Huggingface Dataset -->
  <a href="https://huggingface.co/datasets/DloadingX/GuardTrace-VL-Dataset" target="_blank" style="margin: 2px;">
    <img alt="Huggingface Dataset" src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-GuardTrace--VL--Dataset-blue" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://www.apache.org/licenses/LICENSE-2.0 " target="_blank">
    <img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-yellow.svg? ">
  </a>
  <a href="https://arxiv.org/abs/2511.20994" target="_blank">
    <img alt="Dataset License" src="https://img.shields.io/badge/Paper-arxiv.2511.20994-red" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

## 🗞️ News
- **`2026/02/23`**:Our work got accepted in CVPR'26 :partying_face:
- **`2026/03/05`**:We released our model and dataset.

## 🛠️ Install
### Environment Setup
- Python >= 3.10.0
- Install required libraries:
  ```bash
  pip install -r requirements.txt
  ```

### Model & Dataset Download

1.**Download Model**

Download our GuardTrace-VL-3B model from Hugging Face to the `./model` directory:
```bash
# Create model directory
mkdir -p ./model
# Download model (requires git-lfs)
git lfs install
git clone https://huggingface.co/DloadingX/GuardTrace-VL-3B ./model/GuardTrace-VL-3B
```

2.**Download Test Dataset**

Download our GuardTrace-VL-Dataset from Hugging Face to the `./data` directory:

```bash
# Create data directory
mkdir -p ./data
# Download dataset
git clone https://huggingface.co/datasets/DloadingX/GuardTrace-VL-Dataset ./data/GuardTrace-VL-Dataset
# Organize test dataset structure (match eval code expected path)
mkdir -p ./data/test/images
cp -r ./data/GuardTrace-VL-Dataset/test/* ./data/test/
```
