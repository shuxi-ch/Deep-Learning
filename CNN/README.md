# CNN Image Classification on CIFAR-10 & CIFAR-100

### Overview  
This project implements and benchmarks convolutional neural networks (CNNs) on the CIFAR-10 and CIFAR-100 datasets. We explore various architectural choices—depth, filter sizes, data augmentation, and regularization—to optimize test accuracy and inspect model interpretability via feature‐map visualizations.

### Directory layout  
```bash
.
├── README.md
├── requirements.txt
├── data
│   ├── cifar10/    
│   └── cifar100/   
├── notebooks
│   └── CNN Image Classification on CIFAR-10 & CIFAR-100.ipynb
├── models
│   ├── cnn_cifar10.pth
│   └── cnn_cifar100.pth
└── outputs
    ├── training_curves.png
    ├── confusion_matrix_10.png
    ├── confusion_matrix_100.png
    └── feature_maps/           

```

---

### Problem Statement

Classify 32×32 color images into their correct categories on two benchmarks:

* **CIFAR-10:** 10 classes, 50 K train / 10 K test images.
* **CIFAR-100:** 100 classes, 50 K train / 10 K test images.

Challenges include small image size, fine-grained classes (CIFAR-100), and the need for regularization to prevent overfitting.

---

### Data Sources

* **CIFAR-10 & CIFAR-100:** Downloaded via `torchvision.datasets` or processed from binary files into PyTorch `Dataset` objects.

---

### Methodology 

1. **Data Preprocessing & Augmentation**

   * Normalization (mean/std per channel).
   * Random crops, flips, and color jitter to improve generalization.

2. **Model Architectures**

   * Baseline: 4–6 convolutional layers with ReLU & BatchNorm.
   * Variants: deeper networks, varying kernel sizes, dropout layers.

3. **Training Regimen**

   * Optimizer: SGD with momentum (or Adam), learning‐rate schedule.
   * Loss: Cross‐entropy.
   * Regularization: Weight decay, dropout.
   * Early stopping based on validation loss/accuracy.

4. **Evaluation & Visualization**

   * Tracked train/val loss and accuracy over epochs.
   * Generated confusion matrices.
   * Visualized intermediate feature maps for sample images.

---

### Key Results

| Dataset   | Test Accuracy | Test Loss | Comments                                  |
| --------- | ------------: | --------: | ----------------------------------------- |
| CIFAR-10  |    **XX.X %** | **X.XXX** | Best model: \[architecture name / config] |
| CIFAR-100 |    **XX.X %** | **X.XXX** | Best model: \[architecture name / config] |

*Baseline (simple CNN):*

* CIFAR-10: XX.X % accuracy
* CIFAR-100: XX.X % accuracy

---

### Key Learnings & Challenges

* **Depth vs. Overfitting:** Increasing layers improved CIFAR-10 but risked overfitting on CIFAR-100.
* **Augmentation Impact:** Random cropping & flips boosted test accuracy by \~X pp.
* **Feature‐Map Insights:** Early-layer filters capture edges/colors; deeper layers show class-specific textures.
* **Hyperparameter Sensitivity:** Learning‐rate schedule and weight decay had the largest effect on stability.

---

### Future Enhancements

1. **ResNet or DenseNet Backbones:** Replace custom CNN with residual blocks to push accuracy higher.
2. **Advanced Augmentation:** Incorporate CutMix, MixUp, or AutoAugment.
3. **Learning‐Rate Finding & Scheduling:** Use OneCycleLR or cosine annealing for faster convergence.
4. **Model Pruning & Quantization:** Reduce model size and improve inference speed for deployment.
