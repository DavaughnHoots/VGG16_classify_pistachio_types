# VGG16 Pistachio Classification

This repository contains the implementation and paper for pistachio variety classification (Kirmizi vs. Siirt) using VGG16 transfer learning, achieving 99.23% ± 0.11% accuracy.

## Paper

The main paper is in `VGG16.tex`, prepared for IEEE conference submission. Key improvements include:
- Statistical validation across 5 independent runs
- Comprehensive metrics reporting (precision, recall, F1-score, specificity)
- Comparison with state-of-the-art (previous best: 98.84%)

## Results

Our method achieves:
- **Accuracy**: 99.23% ± 0.11% (95% CI: [99.10%, 99.36%])
- **Precision**: 99.27% ± 0.07%
- **Recall**: 99.22% ± 0.15%
- **F1-score**: 99.24% ± 0.09%

See `statistical_validation_results/` for detailed results.

## Dataset

The Pistachio Image Dataset (2,148 images) is available from:
- [Mendeley Data](https://data.mendeley.com/datasets/6spnxjr72c/1)
- Kirmizi variety: 1,232 images
- Siirt variety: 916 images

Download and extract to `Pistachio_Dataset/` directory.

## Implementation

### Requirements
- TensorFlow 2.4.1
- Keras 2.4.3
- NumPy
- scikit-learn
- CUDA 11.2 (for GPU support)

### Running Statistical Validation
```bash
python statistical_validation.py
```

This will:
1. Load the pistachio dataset
2. Run 5 independent training runs with different seeds
3. Generate statistical results and LaTeX table
4. Save results to `statistical_validation_results/`

### Hardware Used
- NVIDIA GeForce RTX 3070 GPU (8GB VRAM)
- Training time: ~15 minutes per run

## Citation

If you use this work, please cite:
```
[Paper citation to be added upon publication]
```

## License

[Add your preferred license]