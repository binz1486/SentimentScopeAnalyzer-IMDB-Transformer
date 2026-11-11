<div align="center">

# 🎬 SentimentScope

### IMDB Sentiment Analysis with a Transformer Built from Scratch

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E?logo=huggingface&logoColor=black)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

*Is this movie review 👍 positive or 👎 negative? Let a transformer decide.*

</div>

---

## 📑 Table of Contents

- [📖 Project Overview](#-project-overview)
- [🏆 Key Results](#-key-results)
- [📊 Dataset](#-dataset)
- [🔍 Data Exploration](#-data-exploration)
- [🧠 Model Architecture](#-model-architecture)
- [⚙️ Training Setup](#️-training-setup)
- [📁 Repository Structure](#-repository-structure)
- [🚀 Getting Started](#-getting-started)
- [🔮 Usage: Running Inference](#-usage-running-inference)
- [💡 Key Takeaways](#-key-takeaways)
- [🛠️ Possible Improvements](#️-possible-improvements)
- [🧰 Tech Stack](#-tech-stack)
- [👤 Author](#-author)

---

## 📖 Project Overview

Sentiment analysis helps a platform understand how audiences feel about movies and shows. **SentimentScope** was built for the fictional entertainment company *CineScope*, which wants to improve its recommendation system by understanding user sentiment.

The task is framed as **binary classification**: given the text of an IMDB review, the model outputs `1` (positive 😊) or `0` (negative 😞).

Unlike text-generation transformers, which predict the next token, this model reads an entire review and produces **one prediction for the whole sequence**. That required a few changes:

- 🗂️ **Data:** each review is one sample with one label, with no sliding window.
- ✂️ **Tokenization:** subword tokenization (`bert-base-uncased`) instead of character-level tokens.
- 🎯 **Output head:** token embeddings are mean-pooled into one vector, then a linear layer produces two logits.
- 🔁 **Training:** a standard epoch-based loop with a shuffling `DataLoader` and per-epoch validation.

---

## 🏆 Key Results

| Metric | Value |
|---|---|
| ✅ Validation accuracy | **78.56%** |
---

## 📊 Dataset

The project uses the [Large Movie Review Dataset (IMDB)](https://ai.stanford.edu/~amaas/data/sentiment/) from Stanford.

| Split | Size | Purpose |
|---|---|---|
| 🏋️ Train | 22,500 reviews | Updating model weights |
| 🧪 Validation | 2,500 reviews | Tracking overfitting each epoch |
| 🏁 Test | 25,000 reviews | Final, one-time evaluation |

- 🏷️ **Labels:** positive = `1`, negative = `0`
- ⚖️ **Balance:** both classes are evenly represented
- 🔀 The training data is shuffled before splitting so the validation set is representative

The raw dataset is not stored in this repository because of its size. See [Getting Started](#-getting-started) to download it.

---

## 🔍 Data Exploration

Before modeling, the dataset was explored to understand label balance and review lengths.

### Label Distribution

![Label Distribution](images/label_distribution.png)

The dataset is perfectly balanced, so plain accuracy is a fair metric.

### Review Length Distribution

![Review Length Distribution](images/review_length.png)

Many reviews are longer than the 128-token context window, so long reviews are truncated. This is a trade-off between speed and context.

---

## 🧠 Model Architecture

The model, `DemoGPT`, is a stack of transformer blocks followed by a pooling step and a classification head.

```mermaid
flowchart LR
    A["📝 Review text"] --> B["✂️ BERT tokenizer<br/>(128 tokens)"]
    B --> C["🔢 Token + position<br/>embeddings"]
    C --> D["🧱 4 × Transformer block<br/>(attention + feed-forward)"]
    D --> E["📏 Layer norm"]
    E --> F["📉 Mean pooling"]
    F --> G["🎯 Linear head<br/>(2 logits)"]
    G --> H["😊 Positive / 😞 Negative"]
```

**How it works, step by step:**

1. 🔢 **Embeddings:** token embeddings plus learned positional embeddings
2. 🧱 **Transformer blocks:** multi-head self-attention and a feed-forward network, each with residual connections and layer normalization
3. 📏 **Final layer normalization**
4. 📉 **Mean pooling:** averages all token vectors into one review vector
5. 🎯 **Classification head:** a linear layer mapping that vector to two logits

**Configuration:**

| Parameter | Value |
|---|---|
| 📚 Vocabulary size | 30,522 (`bert-base-uncased`) |
| 📐 Embedding dimension | 128 |
| 📏 Context length | 128 tokens |
| 🧱 Transformer layers | 4 |
| 👀 Attention heads | 4 (head size 32) |
| 💧 Dropout | 0.1 |
| 🏷️ Output classes | 2 |

---

## ⚙️ Training Setup

- 📉 **Loss function:** Cross-entropy loss
- 🚀 **Optimizer:** AdamW, learning rate `3e-4`
- 📦 **Batch size:** 32
- 🔁 **Epochs:** X *(update with your final value)*
- 💻 **Hardware:** NVIDIA GPU
- 🧪 **Evaluation:** validation accuracy after every epoch; the test set is used once at the end

### Training Progress

![Training Curves](images/training_curves.png)

---

## 📁 Repository Structure

```
SentimentScope/
├── 📓 SentimentScope.ipynb        # Full code: data, model, training, evaluation
├── 💾 sentimentscope_model.pt     # Trained model checkpoint
├── 🖼️ images/                     # Plots used in this README
│   ├── label_distribution.png
│   ├── review_length.png
│   └── training_curves.png
└── 📄 README.md                   # Project documentation
```

---

## 🚀 Getting Started

### 1️⃣ Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2️⃣ Install dependencies

```bash
pip install torch transformers pandas numpy matplotlib jupyter
```

### 3️⃣ Download the dataset

```bash
wget https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xzf aclImdb_v1.tar.gz
```

### 4️⃣ Run the notebook

```bash
jupyter notebook SentimentScope.ipynb
```

> ⚡ A GPU is strongly recommended for training. Free options include Google Colab and Kaggle Notebooks.

---

## 🔮 Usage: Running Inference

Load the saved checkpoint to classify new reviews. Run the notebook's class and config definitions first (or import them), then:

```python
import torch
from transformers import AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

model = DemoGPT(config).to(device)
model.load_state_dict(torch.load("sentimentscope_model.pt", map_location=device))
model.eval()

def predict(texts):
    enc = tokenizer(texts, truncation=True, padding="max_length",
                    max_length=128, return_tensors="pt")
    with torch.no_grad():
        logits = model(enc["input_ids"].to(device))
    preds = torch.argmax(logits, dim=1).tolist()
    return ["positive" if p == 1 else "negative" for p in preds]

print(predict(["I loved every moment of this movie!",
               "A complete waste of time."]))
```

**Example output:**

```
['positive', 'negative']
```

---

## 💡 Key Takeaways

- 🎯 **Classification needs a different output stage than generation.** Mean pooling plus a small linear head turns per-token embeddings into one review-level prediction.
- ✂️ **Subword tokenization with padding and truncation keeps inputs uniform and efficient.** The trade-off is that reviews longer than 128 tokens lose their ending, which may hold useful sentiment cues.
- 📈 **Tracking validation accuracy each epoch is essential.** It shows when the model starts to overfit and when to stop or adjust hyperparameters.
- 🏗️ **Training from scratch has limits.** Pretrained transformers usually reach higher accuracy on this task, so these results reflect a small model trained without pretraining.

---

## 🛠️ Possible Improvements

- 📏 Increase the context length to capture more of each review
- ⏱️ Train for more epochs, with early stopping and learning-rate scheduling
- 📦 Scale up the model (larger embedding size, more layers or heads)
- 🤖 Fine-tune a pretrained model such as BERT for higher accuracy
- 📊 Add a confusion matrix, precision, recall and F1 score alongside accuracy

---

## 🧰 Tech Stack

| Area | Tools |
|---|---|
| 🐍 Language | Python |
| 🔥 Deep learning | PyTorch |
| 🤗 NLP | Hugging Face Transformers (`bert-base-uncased` tokenizer) |
| 📊 Data and visualization | Pandas, NumPy, Matplotlib |
| 📓 Environment | Jupyter Notebook |

---

## 👤 Author

**Binyameen**

- 🐙 GitHub: [@binz1486](https://github.com/binz1486)
- 💼 LinkedIn: [Binz1486](https://www.linkedin.com/in/binz1486)

---

<div align="center">

⭐ If you found this project useful, consider giving it a star!

</div>
