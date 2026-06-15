# Tanglish_Sentiment_Analysis
# Tamil-English Code-Switched Sentiment & Intent Analyzer

This repository contains a complete pipeline to fine-tune a Llama model (via QLoRA and Unsloth) on code-switched Tamil-English (Tanglish) text. The fine-tuned model predicts sentiment, emotion, intent, and translates Tamil slang words into English in a structured JSON response format.

---

## 📂 Project Directory Structure

Your local workspace contains the following files:
```
Sentiment_Analysis/
├── data/
│   └── sample_dataset.json      # 100 enriched training examples
├── scripts/
│   ├── prepare_data.py          # Data script that maps raw data to enriched dataset
│   ├── train.py                 # QLoRA fine-tuning script optimized for Google Colab (T4)
│   ├── inference.py             # Inference validation script verifying JSON output structure
│   └── export_model.py          # Merges adapters and exports model to GGUF format
└── README.md                    # This documentation and setup guide
```

---

## 🚀 Google Colab Setup (T4 GPU Free Tier)

Follow these steps to fine-tune your model on Google Colab:

### Step 1: Open a Google Colab Notebook
1. Go to [Google Colab](https://colab.research.google.com/).
2. Create a new notebook.
3. Go to **Runtime** > **Change runtime type**, select **T4 GPU**, and click **Save**.

### Step 2: Upload Files to Google Colab
Click on the **Folder icon** 📂 on the left sidebar of Google Colab, and upload the files as follows:
1. Create a folder named `data` by right-clicking in the file explorer and selecting **New folder**. Upload [sample_dataset.json](file:///c:/Users/User/Documents/Sentiment_Analysis/data/sample_dataset.json) inside the `data/` folder.
2. Create a folder named `scripts` and upload:
   * [train.py](file:///c:/Users/User/Documents/Sentiment_Analysis/scripts/train.py)
   * [inference.py](file:///c:/Users/User/Documents/Sentiment_Analysis/scripts/inference.py)
   * [export_model.py](file:///c:/Users/User/Documents/Sentiment_Analysis/scripts/export_model.py)

### Step 3: Install Unsloth and Dependencies
Run the following commands in the first cell of your Colab Notebook to install Unsloth and required Hugging Face packages:
```bash
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
```

---

## 🏋️ Fine-Tuning Execution

Run the scripts in Colab using python commands in cells:

### 1. Start Fine-Tuning
Run the training script to load the Llama-3 model in 4-bit, set up LoRA adapters, map the dataset to instruction chat format, and start training:
```bash
!python scripts/train.py
```
* **What it does**: It runs for 60 steps using a learning rate of $2 \times 10^{-4}$ and saves the resulting adapter weights into a folder named `lora_model`.

### 2. Verify Inference
Test the trained adapter model to verify that it outputs correctly structured JSON for custom Tanglish inputs:
```bash
!python scripts/inference.py
```
* **What it does**: It loads the saved adapters and runs inference on several Tanglish comments, outputting clean JSON blocks.

### 3. Merge & Export to GGUF
Merge the adapter weights with the base Llama-3-8B model and export directly to Q4_K_M GGUF format:
```bash
!python scripts/export_model.py
```
* **What it does**: It generates the GGUF file `model_gguf/model-unsloth-Q4_K_M.gguf` and a local `Modelfile` inside your Colab file viewer.

---

## 🦙 Running the Model Locally with Ollama

Once the GGUF export is finished:
1. Download `model-unsloth-Q4_K_M.gguf` from the Colab file view (`model_gguf/` directory) and the `Modelfile` to the **same directory** on your local Windows machine.
2. Open PowerShell or Command Prompt in that directory and run:
   ```powershell
   ollama create tanglish-analyzer -f Modelfile
   ```
3. Run the model in command line:
   ```powershell
   ollama run tanglish-analyzer
   ```
4. Test it by typing a custom message:
   ```
   >>> Macha Padam vera level! Marana waiting for FDFS!
   ```
   **Output:**
   ```json
   {
     "sentiment": "Positive",
     "emotion": "Excited",
     "intent": "Praise",
     "slang_meaning": "Macha: friend / brother-in-law; Vera level: next level / extraordinary; Marana waiting: waiting eagerly/intensely; FDFS: First Day First Show"
   }
   ```

---

## 📊 Hyperparameters Explained

* **`load_in_4bit = True`**: Loads the model weights in 4-bit double quantization, reducing memory from 16GB down to just ~5.5GB VRAM.
* **LoRA Rank (`r = 16`) and Alpha (`lora_alpha = 16`)**: Determines the width and scaling of the low-rank adaptation matrix. A rank of 16 strikes the ideal balance between learning capacity and memory.
* **`learning_rate = 2e-4`**: The optimal rate for training adapters on text classification/structured text generation.
* **`max_steps = 60`**: Since we have 100 high-quality training sentences, 60 training steps (with a batch size of 2 and gradient accumulation steps of 4, meaning 8 samples per step) equals about 5 complete epochs, which prevents overfitting while ensuring the model learns the output structure.

---

## ⚠️ Troubleshooting & Common Errors

### 1. Out of Memory (OOM) error on CUDA
* **Cause**: Trying to use standard FP16 or loading the model in full 16-bit without 4-bit quantization, or having too large a batch size.
* **Fix**: Ensure `load_in_4bit = True` in `FastLanguageModel.from_pretrained` and that `per_device_train_batch_size` is set to 2.

### 2. JSON Decoding Failure during Inference
* **Cause**: LLMs occasionally output surrounding markdown blocks (like ` ```json ` ... ` ``` `) or introductory sentences instead of pure JSON.
* **Fix**: The code in `scripts/inference.py` automatically strips markdown wrapper syntax and isolates raw brackets. Keep `temperature = 0.1` during inference to make JSON syntax highly deterministic.

### 3. Unsloth Import Errors on Local Machines
* **Cause**: Unsloth is highly optimized for Linux CUDA kernels (Google Colab, RunPod, Lambda Labs) and does not natively support Windows local installations easily.
* **Fix**: Run training and exporting exclusively on Google Colab or similar Linux GPU servers. Local Windows execution is meant for **Ollama inference** using the exported `.gguf` file, which is natively supported.
