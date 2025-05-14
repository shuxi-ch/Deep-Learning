# Deep Learning Exercises Collection

### Overview  
This repository aggregates three self-contained deep-learning exercises—spanning image classification, sequence modeling, and transformer fine-tuning—each in its own folder with a detailed README:

1. **CNN Image Classification – CIFAR-10 & CIFAR-100**  
   - Custom convolutional architectures with data augmentation, regularization, and batch normalization.  
   - Benchmarked on CIFAR-10 and CIFAR-100 with training curves, confusion matrices, and feature‐map visualizations.

2. **RNN Text Generation – Shakespeare**  
   - Vanilla RNN, LSTM, and GRU models (including bidirectional layers) trained on Shakespeare’s corpus.  
   - Advanced generation via temperature‐controlled sampling and beam search, with perplexity and qualitative assessments.

3. **Transformer Text Classification – 20 Newsgroups**  
   - Fine-tuned BERT-base-uncased and RoBERTa-base using HuggingFace’s Trainer.  
   - Logged metrics, interpretability via attention rollout and LIME, and exported an ONNX model for deployment.


### Directory layout  

```bash
.
├── .gitignore
├── README.md                  # this overview
│
├── CNN/                       # CIFAR-10 & CIFAR-100 image classification
│   ├── README.md
│   ├── requirements.txt
│   ├── data
│   │   ├── cifar10/
│   │   └── cifar100/
│   ├── notebooks
│   │   └── CNN Image Classification on CIFAR-10 & CIFAR-100.ipynb
│   ├── models
│   │   ├── cnn_cifar10.pth
│   │   └── cnn_cifar100.pth
│   └── outputs
│       ├── training_curves.png
│       ├── confusion_matrix_10.png
│       ├── confusion_matrix_100.png
│       └── feature_maps/
│
├── RNN/                       # Shakespeare text-generation with RNNs
│   ├── README.md
│   ├── requirements.txt
│   ├── data
│   │   └── shakespeare.txt
│   ├── notebooks
│   │   └── Shakespeare Text Generation with RNNs.ipynb
│   ├── tokenizer_models
│   │   ├── shakespeare_bpe_tokenizer-merges.txt
│   │   └── shakespeare_bpe_tokenizer-vocab.json
│   ├── training_checkpoints_SimpleRNN
│   │   └── shakespeare_SimpleRNN_bidirTrue.weights.h5
│   └── outputs
│       ├── samples_temperature.txt
│       └── samples_beam.txt
│
└── Transformer/               # 20 Newsgroups transformer fine-tuning
    ├── README.md
    ├── requirements.txt
    ├── data
    │   └── 20newsgroups/
    ├── notebooks
    │   └── transformer_finetune.ipynb
    ├── baseline_model
    │   ├── config.json
    │   ├── model.safetensors
    │   ├── special_tokens_map.json
    │   ├── tokenizer.json
    │   ├── tokenizer_config.json
    │   ├── training_args.bin
    │   └── vocab.txt
    ├── logs
    │   └── events.out.tfevents.*
    ├── results
    │   ├── checkpoint-943/
    │   ├── checkpoint-1886/
    │   ├── checkpoint-2300/
    │   └── checkpoint-2829/
    └── models
        └── transformer.onnx
```
