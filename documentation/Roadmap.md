# 🫁 Chest X-Ray Pneumonia ML Project Roadmap (Clinical Optimization Edition)

**Dataset:** Chest X-Ray Images (Pneumonia) by Paul Mooney
**Goal:** Build an end-to-end classification model optimized for clinical triage (minimizing False Negatives) with an explainable web interface.

---

## Phase 1: Setup & Clinical EDA (Days 1-3)
*The goal here is to load the images, fix the validation split, and understand the clinical cost of errors.*

* **Step 1: Environment Setup**
    * Install Python, Jupyter Notebook, and core libraries: `numpy`, `pandas`, `matplotlib`, `opencv-python`, `torch` (PyTorch is recommended based on your logs), and `torchvision`.
* **Step 2: Fix Dataset Splits & Load Data**
    * Combine the existing `train` and `val` folders (since the default `val` only has 16 images) and create a robust 80/20 split.
* **Step 3: Exploratory Data Analysis & Clinical Context**
    * Analyze the natural class imbalance, which sits at approximately 3:1 in favor of pneumonia images.
    * Define the business metric: In a medical setting, a False Negative (sending a sick patient home) is a severe health risk, while a False Positive (testing a healthy patient further) is a manageable cost.

---

## Phase 2: Model Training & Experimentation (Days 4-8)
*Instead of just building a model, you will document the journey of finding the right loss function to handle the clinical asymmetry.* 

* **Step 1: Baseline Model (Experiment A)**
    * Train a base ResNet18 model using Transfer Learning (frozen base, training only the `fc` layer) with standard `CrossEntropyLoss` and neutral weights.
    * Observe that while accuracy is high (~95.99%), the number of False Negatives (19) is clinically unacceptable.
* **Step 2: The Pitfall of Class Weights (Experiments B & C)**
    * Experiment with mathematical class balancing (e.g., weights of `[4.0, 1.0]` and `[1.0, 1.5]`).
    * Document how heavily penalizing the minority class breaks the gradients or makes the model too conservative, paradoxically *increasing* False Negatives.
* **Step 3: Implementing Focal Loss (Experiment F)**
    * Replace `CrossEntropyLoss` with Focal Loss ($\gamma=2.0$) to dynamically penalize "hard" examples (historical False Negatives) rather than using static weights.
* **Step 4: Extended Training**
    * Train the ResNet18 model with Focal Loss for 15 epochs, allowing the network enough time to resolve complex radiological patterns and minimize False Positives.

---

## Phase 3: Post-Processing & Inferencia (Days 9-11)
*This is where your project shines. You will separate mathematical training from clinical decision-making.*

* **Step 1: Threshold Tuning (Experiment D & G)**
    * Extract the raw Softmax probabilities.
    * Lower the decision threshold for Pneumonia from the default 0.50 to 0.30 or 0.20 to artificially force a drop in False Negatives.
    * Document the trade-off: Achieving 100% Recall (0 FN) at a 0.30 threshold results in high False Positives, representing the "paranoia" limit of the model.
* **Step 2: Test-Time Augmentation (Experiment E & H)**
    * Implement TTA during inference. For every patient image, generate 3 variants in real-time: Original, Rotated (10 degrees), and Zoomed (CenterCrop). 
    * Implement "Hard Voting": If the sum of positive votes is >= 2, the final diagnosis is Pneumonia.
* **Step 3: Final Model Selection & Explainability**
    * Combine the Focal Loss model (15 epochs) with the TTA pipeline, which yields the best clinical balance: 94.08% Accuracy, 99.6% Recall, and only 3 False Negatives.
    * Implement Grad-CAM to generate heatmaps showing where the model "looked" to make its decision.

---

## Phase 4: Web Application / UI (Days 12-14)
*Getting the triage system into a usable app for a hypothetical Radiology Department.*

* **Step 1: Create a Streamlit App**
    * Build a simple UI where a clinician can upload a `.jpg` or `.png` X-ray.
* **Step 2: Integrate the TTA Pipeline**
    * Load your saved ResNet18 (Focal Loss) `.pth` model.
    * Implement the 3-way TTA logic directly in the `app.py` file so the web app uses the safest clinical pipeline.
* **Step 3: Clinical Sliders & Explainability**
    * Add a slider in the UI allowing the user to adjust the decision threshold (e.g., from conservative to high-sensitivity triage) based on your Threshold Tuning findings.
    * Display the Grad-CAM heatmap side-by-side with the uploaded X-ray.

---

## Phase 5: Portfolio Packaging (Days 15-16)
*Making your extensive experimentation stand out for recruiters.*

* **Step 1: Clean the Code**
    * Organize your Jupyter notebook into clear sections matching your experiment logs (Baseline, Focal Loss, TTA, Thresholding).
* **Step 2: Write a Data Science `README.md`**
    * Highlight the **Business/Clinical Value**: Explain why you optimized for Recall over pure Accuracy.
    * Showcase the final metrics clearly: **94.08% Accuracy | 99.6% Recall | Only 3 False Negatives**.
    * Include a table comparing the Baseline model to the final Focal Loss + TTA model.
* **Step 3: Upload to GitHub**
    * Push the code, the `README.md`, and the `requirements.txt`.