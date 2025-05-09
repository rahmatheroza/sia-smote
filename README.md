# SIA-SMOTE: Siamese Network-Enhanced SMOTE for High-Dimensional Imbalanced Data

**SIA-SMOTE** is an advanced oversampling technique designed to address the challenges of class imbalance in high-dimensional datasets. By integrating the traditional SMOTE method with a Siamese network, SIA-SMOTE enhances the quality of synthetic samples, leading to improved model performance, especially in scenarios where data is scarce or imbalanced.

## 📚 Overview

Class imbalance is a prevalent issue in machine learning, often leading to biased models that underperform on minority classes. While SMOTE (Synthetic Minority Over-sampling Technique) has been a popular solution, it can inadvertently introduce noise by generating synthetic samples from outliers or noisy data points.

**SIA-SMOTE** addresses this limitation by:

* Utilizing a Siamese network to assess the similarity between samples, ensuring that synthetic data is generated from reliable and representative instances.
* Focusing on the decision boundary to better capture the distribution of the minority class.

This approach has demonstrated superior performance on datasets like MNIST, FMNIST, and various medical image datasets, outperforming traditional methods such as Random Oversampling, SMOTE, and ASN-SMOTE.

## 🧠 Methodology

1. **Data Preprocessing**: Normalize and prepare the dataset, ensuring it's suitable for training.
2. **Siamese Network Training**: Train a Siamese network to learn the similarity between data points, helping to identify reliable samples for oversampling.
3. **Sample Selection**: Use the trained Siamese network to select high-quality minority class samples that are suitable for generating synthetic data.
4. **Synthetic Sample Generation**: Apply the SMOTE technique on the selected samples to generate new, synthetic instances.
5. **Model Training**: Train your machine learning model using the augmented dataset.

## 🛠️ Installation

To use SIA-SMOTE in your project, follow these steps:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/yourusername/sia-smote.git
   cd sia-smote
   ```



2. **Create a Virtual Environment** (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```



3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```



## 📈 Usage

Here's a basic example of how to apply SIA-SMOTE to your dataset:

```python
from sia_smote import SiaSmote

# Initialize the SIA-SMOTE object
sia_smote = SiaSmote()

# Fit the model on your data
sia_smote.fit(X_train, y_train)

# Generate synthetic samples
X_resampled, y_resampled = sia_smote.transform()
```



For detailed examples and advanced usage, refer to the `examples/` directory in the repository.

## 🧪 Experiments

The effectiveness of SIA-SMOTE has been validated on multiple datasets:

* **MNIST**: Handwritten digit recognition dataset.
* **FMNIST**: Fashion product images dataset.
* **Medical Image Datasets**: Various datasets focusing on medical imaging tasks.

In all experiments, SIA-SMOTE consistently outperformed baseline methods in terms of accuracy, precision, recall, and F1-score.

## 📂 Project Structure

```plaintext
sia-smote/
├── sia_smote/
│   ├── __init__.py
│   ├── siamese_network.py
│   ├── smote.py
│   └── utils.py
├── examples/
│   └── mnist_example.py
├── tests/
│   └── test_sia_smote.py
├── requirements.txt
├── README.md
└── LICENSE
```



## 🤝 Contributing

We welcome contributions! If you'd like to improve SIA-SMOTE or fix any issues:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature-name`.
3. Make your changes and commit them: `git commit -m 'Add new feature'`.
4. Push to the branch: `git push origin feature-name`.
5. Open a pull request.

Please ensure your code adheres to the project's coding standards and includes relevant tests.

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 📖 Citation

If you use SIA-SMOTE in your research, please cite the following paper:

```bibtex
@inproceedings{heroza2023sia,
  title={SIA-SMOTE: A SMOTE-Based Oversampling Method with Better Interpolation on High-Dimensional Data by Using a Siamese Network},
  author={Heroza, Rahmat Izwan and Gan, John Q. and Raza, Haider},
  booktitle={Advances in Computational Intelligence},
  pages={448--460},
  year={2023},
  publisher={Springer}
}
```



## 🔗 References

* [SIA-SMOTE Springer Chapter](https://link.springer.com/chapter/10.1007/978-3-031-43085-5_35)
* [SMOTE: Synthetic Minority Over-sampling Technique](https://arxiv.org/abs/1106.1813)
* [Siamese Neural Networks for One-shot Image Recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)

---

For any questions or feedback, please open an issue or contact the maintainers directly.

---
