# 🫁 Chest X-Ray Pneumonia ML Project Roadmap

**Dataset:** [Chest X-Ray Images (Pneumonia) by Paul Mooney](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
**Goal:** Build an end-to-end classification model with an explainable web interface.

---

## Phase 1: Setup & Data Preparation (Days 1-3)
*The goal here is to load the images and fix a known issue with this dataset (the tiny validation set).*

- [ ] **Step 1: Environment Setup**
  - Install Python, Jupyter Notebook (or use Google Colab).
  - Install core libraries: `numpy`, `pandas`, `matplotlib`, `opencv-python`, `tensorflow` (or `pytorch`).
- [ ] **Step 2: Load the Data**
  - Write a script to iterate through the folders (`train`, `test`, `val`) and count the images.
- [ ] **Step 3: Fix the Dataset Splits**
  - *Dataset quirk:* The `val` folder only has 16 images.
  - Combine `train` and `val` folders, then use a library like `scikit-learn` to create a new 80/20 train/validation split.
- [ ] **Step 4: Exploratory Data Analysis (EDA)**
  - Plot a bar chart showing the number of "NORMAL" vs "PNEUMONIA" images (Note the class imbalance).
  - Plot a grid of 4 Normal and 4 Pneumonia X-rays to see what they look like.

> **🤖 AI Prompt to use:** *"I am working with the Paul Mooney Chest X-Ray dataset in Python using [TensorFlow/PyTorch]. The default 'val' folder only has 16 images. Can you write a script to combine the train and val directories, and then split the combined data into an 80% training and 20% validation set using standard Python libraries?"*

---

## Phase 2: Model Training & Transfer Learning (Days 4-7)
*Instead of building a model from scratch, you will use a pre-trained model (Transfer Learning) which is faster and more accurate.*

- [ ] **Step 1: Data Augmentation**
  - Apply transformations to the training data (slight rotations, zooming, flipping) to prevent overfitting.
- [ ] **Step 2: Handle Class Imbalance**
  - Calculate "class weights" so the model pays more attention to the "NORMAL" cases (since there are fewer of them).
- [ ] **Step 3: Build the Model**
  - Load a pre-trained model (e.g., `ResNet50` or `MobileNetV2`) without the top classification layers.
  - Add your own dense layers on top to classify 2 categories (Normal vs. Pneumonia).
- [ ] **Step 4: Train & Save**
  - Train the model using your augmented data and class weights.
  - Save the best model as a `.h5` or `.pth` file.

> **🤖 AI Prompt to use:** *"Write a script using [TensorFlow/PyTorch] to load a pre-trained ResNet50 model for binary classification. Include an image data generator with basic data augmentation (rotation, zoom), and show me how to apply class weights to handle an imbalanced dataset."*

---

## Phase 3: Evaluation & Explainability (Days 8-10)
*This is where you make the project stand out for recruiters.*

- [ ] **Step 1: Evaluate Metrics**
  - Run the model on the `test` folder.
  - Generate a Confusion Matrix.
  - Calculate Accuracy, Precision, Recall, and F1-Score. Focus heavily on **Recall** (you don't want to miss a pneumonia diagnosis).
- [ ] **Step 2: Add Grad-CAM (Explainability)**
  - Implement Gradient-weighted Class Activation Mapping (Grad-CAM).
  - Generate heatmaps over the X-rays showing where the model "looked" to find the pneumonia.
  - Save 5 examples of successful predictions with their heatmaps.

> **🤖 AI Prompt to use:** *"I have trained a ResNet50 model for binary classification. I want to generate a Grad-CAM heatmap for a test image to see what the model is focusing on. Can you provide the Python code to generate and overlay a Grad-CAM heatmap on the original X-ray image?"*

---

## Phase 4: Web Application / UI (Days 11-14)
*Getting the model out of the notebook and into a usable app.*

- [ ] **Step 1: Create a Streamlit App**
  - Create a new file called `app.py`.
  - Build a simple UI where a user can upload a `.jpg` or `.png` file.
- [ ] **Step 2: Integrate the Model**
  - Add code to `app.py` to load your saved model.
  - Pass the uploaded image to the model and display the prediction ("Pneumonia Detected" or "Normal").
- [ ] **Step 3: Display Explainability**
  - Integrate the Grad-CAM code into the app.
  - Display the heatmap side-by-side with the original uploaded image.

> **🤖 AI Prompt to use:** *"I want to build a Streamlit web app for my image classification model. Write a Streamlit script that allows a user to upload an image, resizes the image to 224x224, passes it to a saved Keras/PyTorch model, and displays the predicted class on the screen."*

---

## Phase 5: Portfolio Packaging (Days 15-16)
*Making it ready for your CV.*

- [ ] **Step 1: Clean the Code**
  - Remove unnecessary print statements and messy cells from your Jupyter notebooks.
- [ ] **Step 2: Create a `requirements.txt`**
  - List all the libraries needed to run the app.
- [ ] **Step 3: Write a Kick-Ass `README.md`**
  - Include: Project title, what it does, a screenshot of the Streamlit app showing the heatmap, how to run it locally, and the results (Recall/Accuracy).
- [ ] **Step 4: Upload to GitHub**
  - Push the code, README, and `requirements.txt`. (Don't upload the whole dataset, link to Kaggle instead).

> **🤖 AI Prompt to use:** *"I am finishing my machine learning project. Based on the following steps I took [paste brief summary], can you generate a professional README.md template for my GitHub repository? Include sections for Overview, Tech Stack, How to Run, and Results."*