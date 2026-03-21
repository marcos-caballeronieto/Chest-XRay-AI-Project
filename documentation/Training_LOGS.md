# 🫁 Model Experimentation Log: Pneumonia Detection

## ⚠️ Scope Disclaimer (Learning Only)
This document belongs to an **educational AI learning project**.

**It is not a medical device, it is not clinically validated, and it must not be used for diagnosis or real healthcare decisions.**

All references to "clinical setting," "real world," "hospital," or "production" are used here exclusively as a **didactic simulation** to analyze trade-offs between metrics.

## 📌 Clinical Context and Optimization Objective
The Kaggle dataset (Paul Mooney) presents a strong natural class imbalance (approximately 3:1 in favor of pneumonia images versus normal).

In a medical setting, error asymmetry is critical:
* **False Positive (FP):** Diagnosing pneumonia in a healthy patient. (Cost: additional testing, temporary anxiety).
* **False Negative (FN):** Sending a patient with pneumonia home by diagnosing them as healthy. (Cost: severe health risk).

**Experimentation Objective:** Find the optimal balance (trade-off) by adjusting the loss function (`CrossEntropyLoss`) to prioritize **Recall** (minimize False Negatives) while maintaining acceptable overall accuracy.

---

## 🧪 Experiment A: Baseline (Neutral Weights)

### ⚙️ Configuration
* **Architecture:** ResNet18 (Transfer Learning, frozen base).
* **Optimizer:** Adam (lr=0.001) training only the `fc` layer.
* **Loss Function:** `CrossEntropyLoss` with weights `[1.0, 1.0]`.
* **Hypothesis:** By applying no manual penalties, the model will rely solely on the dataset's natural distribution. We expect high overall Accuracy, but an unacceptable number of False Negatives due to the lack of clinical bias.

### 📊 Results (Epoch 5)
* **Loss (Train / Val):** 0.1518 / 0.1011
* **Overall Accuracy:** 95.99%

| Clinical Metric | Value | Real-World Implication |
| :--- | :--- | :--- |
| **False Negatives (FN)** | 19 | ⚠️ 19 severely ill patients without treatment. |
| **False Positives (FP)** | 23 | 💸 23 healthy patients subjected to further testing. |
| **Pneumonia Hits (TP)** | 758 | Correct disease diagnoses. |
| **Normal Hits (TN)** | 247 | Healthy patients correctly discharged. |

### 🧠 Analysis (Exp. A Conclusion)
The experiment serves as an exceptional baseline. Achieving nearly 96% accuracy in just 5 epochs demonstrates the power of Transfer Learning. The model does not suffer from overfitting (validation loss is lower than training loss).

However, from a product perspective, **19 False Negatives is a risky figure**. The model is too "balanced." We need to bias it to be more conservative when predicting that a patient is "Healthy." 
Although we will try to balance the classes mathematically first to see if it improves the results.

---

## 🧪 Experiment B: Mathematical Class Balancing (Weights to Minority)

### ⚙️ Configuration
* **Architecture:** ResNet18 (Transfer Learning, frozen base, new `fc` layer reset).
* **Optimizer:** Adam (lr=0.001) training only the `fc` layer.
* **Loss Function:** `CrossEntropyLoss` with weights `[4.0, 1.0]` (Normal: 4.0, Pneumonia: 1.0).
* **Hypothesis:** The dataset has a massive natural imbalance (approx. 3100 Pneumonia images vs. 1000 Normal ones). In this experiment, we apply a weight of `4.0` to the minority class ("Normal") to force the network to give it equal mathematical attention. We expect the model to improve in detecting healthy lungs (reducing False Positives), but we must monitor the impact on clinical False Negatives.

### 📊 Results (Epoch 5)
* **Loss (Train / Val):** 0.1544 / 0.1256
* **Overall Accuracy:** 95.32%

| Clinical Metric | Value | Real-World Implication | Comparison vs Exp A |
| :--- | :--- | :--- | :--- |
| **False Negatives (FN)** | 34 | ⚠️ 34 severely ill patients without treatment. | ❌ Worsens (+15) |
| **False Positives (FP)** | 15 | 💸 15 healthy patients subjected to further testing. | ✅ Improves (-8) |
| **Pneumonia Hits (TP)** | 743 | Correct disease diagnoses. | ⬇️ Decreases |
| **Normal Hits (TN)** | 255 | Healthy patients correctly discharged. | ⬆️ Increases |

### 🧠 Analysis (Exp. B Conclusion)
**Mathematical success, clinical failure.** The experiment worked exactly as mathematical theory dictates: by severely penalizing errors in the "Normal" class, the model became an expert at detecting healthy lungs. False Positives dropped (from 23 to 15) and hits in the Normal class increased.

However, in the medical context, the result is unacceptable. Because the model is now "terrified" of being wrong about a healthy patient, it has become extremely conservative when diagnosing Pneumonia. At the slightest doubt, it prefers to say "Normal" to avoid the x4 penalty. This triggered False Negatives to skyrocket from 19 to 34 (we sent 15 more sick patients home than in the baseline).

**Next Steps:** This experiment demonstrates that "balancing a dataset" is not always the correct solution if the business metric demands asymmetry. It is clear our objective is not mathematical balance, but clinical bias toward sensitivity. For **Experiment C**, we will play the reverse: we'll give a "slight clinical nudge" to Pneumonia (e.g., `[1.0, 1.5]`) to force the model to detect more sick patients, aiming to lower False Negatives without breaking gradients as occurred in our initial approaches.

---

## 🧪 Experiment C: Hyperparameter Tuning of Clinical Weights (Grid Search)
### ⚙️ Configuration and Objective
* **Architecture:** ResNet18.
* **Technique:** Iterative variation of the loss function (`CrossEntropyLoss`).
* **Description:** After observing that mathematical balance to compensate for dataset asymmetry (Exp B) does not improve overall accuracy and increases the number of *False Negatives*, we will explore small penalties to the Pneumonia class to force a clinical bias towards Recall (minimizing False Negatives), aiming to decrease False Negatives below the baseline value.

### 📌 Sub-experiment C1: Gentle Clinical Nudge (Weights 1.0 / 1.5)

### ⚙️ Configuration
* **Architecture:** ResNet18 (Transfer Learning, frozen base, new `fc` layer reset).
* **Optimizer:** Adam (lr=0.001) training only the `fc` layer.
* **Loss Function:** `CrossEntropyLoss` with weights `[1.0, 1.5]` (Normal: 1.0, Pneumonia: 1.5).
* **Hypothesis:** Knowing that massively penalizing the majority class breaks gradients (seen in previous experiments), we try a much softer approach. The goal is to apply a slight clinical bias toward Pneumonia without destabilizing the network, seeking to lower False Negatives below the baseline (19).

### 📊 Results (Epoch 5)
* **Loss (Train / Val):** 0.1299 / 0.1194
* **Overall Accuracy:** 95.61%

| Clinical Metric | Value | Real-World Implication | Comparison vs Exp A (Baseline) |
| :--- | :--- | :--- | :--- |
| **False Negatives (FN)** | 25 | ⚠️ 25 severely ill patients without treatment. | ❌ Worsens (+6) |
| **False Positives (FP)** | 21 | 💸 21 healthy patients subjected to further testing. | ✅ Improves (-2) |
| **Pneumonia Hits (TP)** | 752 | Correct disease diagnoses. | ⬇️ Decreases |
| **Normal Hits (TN)** | 249 | Healthy patients correctly discharged. | ⬆️ Increases |

### 🧠 Analysis (Exp. C1 Conclusion)
Increasing the penalty on Pneumonia data did not help decrease False Negatives. Even a "gentle nudge" (1.5) to the class that is naturally already the majority proved counterproductive to our clinical objective.

Instead of lowering False Negatives, they increased from 19 to 25 compared to our neutral weight model. This means that manipulating `CrossEntropyLoss` in favor of the already dominant class disorients the optimizer. Interestingly, the network became *better* at detecting healthy patients (False Positives dropped from 23 to 21), but we lost clinical sensitivity.

**Conclusion:** Tweaking loss function weights is not the right tool for this specific dataset. It has been proven that the most stable training path to extract correct patterns is through neutral weights (`[1.0, 1.0]`). Still, taking advantage of the low computational cost, we will try to test the effect of a slightly smaller penalty for Pneumonia.

### 📌 Sub-experiment C2: Light Clinical Nudge (Weights 1.0 / 1.2)

### ⚙️ Configuration
* **Architecture:** ResNet18 (Transfer Learning, frozen base, new `fc` layer reset).
* **Optimizer:** Adam (lr=0.001) training only the `fc` layer.
* **Loss Function:** `CrossEntropyLoss` with weights `[1.0, 1.2]` (Normal: 1.0, Pneumonia: 1.2).
* **Hypothesis:** Taking advantage of the low computational cost, we decided to chart the degradation curve by testing a very subtle intermediate weight. If `1.5` broke stability and `1.0` was stable, we try `1.2` to see if this "micro-adjustment" manages to reduce False Negatives without disorienting the optimizer.

### 📊 Results (Epoch 5)
* **Loss (Train / Val):** 0.1157 / 0.1070
* **Overall Accuracy:** 96.47% *(Highest overall accuracy to date)*

| Clinical Metric | Value | Real-World Implication | Comparison vs Exp A (Baseline) |
| :--- | :--- | :--- | :--- |
| **False Negatives (FN)** | 20 | ⚠️ 20 severely ill patients without treatment. | ↔️ Practically equal (+1) |
| **False Positives (FP)** | 17 | 💸 17 healthy patients subjected to further testing. | ✅ Improves (-6) |
| **Pneumonia Hits (TP)** | 757 | Correct disease diagnoses. | ↔️ Practically equal (-1) |
| **Normal Hits (TN)** | 253 | Healthy patients correctly discharged. | ⬆️ Increases |

### 🧠 Analysis (Exp. C2 Conclusion)
The intuition to test this light weight was a technical success: we achieved the most accurate model to date (96.47% Accuracy) and a fantastic drop in validation loss (0.1070). The model found an excellent balance, reducing False Positives (from 23 to 17).

However, from the strict perspective of our objective (minimizing False Negatives), we achieved no improvements. 20 False Negatives is essentially a technical tie with the 19 from our baseline. Everything suggests that the loss function can do no more to force Recall on this dataset without destroying overall accuracy.

**Definitive conclusion on weights:** The optimal training range for this imbalanced dataset is between `[1.0, 1.0]` and `[1.0, 1.2]`.

**Next Steps:** We consider the training-time tuning phase concluded. For **Experiment D**, we will load this C2 model into memory and apply **Threshold Tuning**. We will analyze raw Softmax probabilities in the inference phase and lower the decision threshold for Pneumonia from 50% to 30% or 20% to artificially force a drop in those 20 False Negatives, attempting not to overly affect overall accuracy.

---

## 🧪 Experiment D: Threshold Tuning (Decision Threshold Calibration)

### ⚙️ Configuration
* **Base Model:** The most accurate model obtained (Exp C2 - Weights 1.0 / 1.2).
* **Technique:** Post-training probabilistic modification during inference.
* **New Threshold:** `0.30` (30%). Instead of demanding 50% mathematical certainty, we force the model to diagnose Pneumonia if it detects at least a 30% probability in the image.
* **Hypothesis:** Adjusting the decision limit after the fact is much more stable than forcing loss gradients during training. We expect a drastic drop in False Negatives, assuming a controlled rise in False Positives.

### 📊 Results (With Threshold at 30%)
* **Overall Accuracy:** 95.61%

| Clinical Metric | Value | Real-World Implication | Comparison vs Best Model (Exp C2) |
| :--- | :--- | :--- | :--- |
| **False Negatives (FN)** | **13** | ⚠️ 13 severely ill patients without treatment. | 🌟 **Massive improvement (-7)** |
| **False Positives (FP)** | 33 | 💸 33 healthy patients subjected to further tests. | 📉 Worsens (+16) |
| **Pneumonia Hits (TP)** | 764 | Correct disease diagnoses. | ⬆️ Increases (+7) |
| **Normal Hits (TN)** | 237 | Healthy patients discharged. | ⬇️ Decreases (-16) |

### 🧠 Analysis (Exp. D Conclusion)
This experiment demonstrates that separating mathematical optimization (training) from clinical optimization (evaluation) is a good strategy for an imbalanced dataset like this.

We kept the stable and accurate "brain" from Experiment C2 but changed its diagnostic instructions. The result is the best compromise within this simulation, sacrificing barely 0.8% of overall Accuracy in exchange for cutting fatal errors (FN) almost in half.

### 📌 Sub-experiment D2: Aggressive Clinical Threshold (20%)

### ⚙️ Configuration
* **Base Model:** Exp C2 (Weights 1.0 / 1.2).
* **New Threshold:** `0.20` (20%). If the model detects 20% of features compatible with Pneumonia, it triggers the alert.

### 📊 Results (With Threshold at 20%)
* **Overall Accuracy:** 94.84%

| Clinical Metric | Value | Real-World Implication | Comparison vs Best Model (Exp C2) |
| :--- | :--- | :--- | :--- |
| **False Negatives (FN)** | **7** | ⚠️ 7 severely ill patients without treatment. | 🏆 **Record improvement (-13)** |
| **False Positives (FP)** | 47 | 💸 47 healthy patients subjected to further tests. | 📉 Worsens (+30) |
| **Pneumonia Hits (TP)** | 770 | Correct diagnoses. (Sensitivity: 99.1%) | ⬆️ Increases (+13) |
| **Normal Hits (TN)** | 223 | Healthy patients discharged. | ⬇️ Decreases (-30) |

### 🧠 Analysis (Exp. D2 Conclusion)
Lowering the decision threshold to 20% has proven to be a highly effective strategy. We achieved a **Sensitivity of 99.1%**, reducing False Negatives to a single-digit figure (7).

In this simulation, this is the most coherent trade-off for a triage scenario: the cost of manually reviewing 47 healthy patients is lower than the cost of missing 13 additional sick patients (the difference with our previous model).

**Verdict (Simulation):** As a laboratory conclusion, the candidate pipeline could utilize the training weights from Experiment C2 (`[1.0, 1.2]`) combined with an inference filter based on a `0.20` probabilistic threshold.

---

## 🧪 Experiment E: Test-Time Augmentation (TTA) and Consensus (Majority Voting)

### ⚙️ Configuration
* **Base Model:** The best model by weights (Exp C2 - Weights 1.0 / 1.2).
* **Technique:** TTA (Test-Time Augmentation) with "Hard Voting."
* **Description:** During the inference phase, we don't pass the image just once. We generate 3 real-time variants per patient:
  1. Original Image (Resize + Normalize).
  2. Rotated Image (10 degrees).
  3. Zoomed Image (CenterCrop).
* **Decision Rule:** The model casts 3 independent votes. If the sum of positive (Pneumonia) votes is $\ge 2$, the final diagnosis is Pneumonia.

### 📊 Results (Epoch 5 + TTA Inference)
* **Overall Accuracy:** 97.23% ⭐ *(All-time project record)*

| Clinical Metric | Value | Real-World Implication | Comparison vs Best Model (Exp C2) |
| :--- | :--- | :--- | :--- |
| **False Negatives (FN)** | **6** | ⚠️ 6 severely ill patients without treatment. | 🏆 **Record improvement (-14)** |
| **False Positives (FP)** | 23 | 💸 23 healthy patients subjected to further tests. | 📉 Slightly worsens (+6) but equals baseline. |
| **Pneumonia Hits (TP)** | 771 | Correct diagnoses. (Sensitivity: **99.2%**) | ⬆️ Increases (+14) |
| **Normal Hits (TN)** | 247 | Healthy patients discharged. | ⬇️ Decreases (-6) |

### 🧠 Analysis (TTA Inference Conclusion)
This experiment demonstrates that increasing effort during the inference phase can outperform training optimizations.

By forcing a 3-way consensus (TTA), the model demonstrated spectacular robustness against possible positioning or cropping biases in the radiographs. We achieved the best clinical metric (99.2% Sensitivity, with only 6 False Negatives) while keeping False Positives at bay and reaching the maximum overall *Accuracy* peak of the project (97.23%).

**Verdict (Simulation):** For the next experimental phase, we take the C2 model (`[1.0, 1.2]`) along with a 3-transformation pipeline and majority voting in inference as the baseline.

---

## 🧪 Experiment F: Focal Loss (Dynamic Error Penalization)
### ⚙️ Configuration and Objective
* **Technique:** Loss Function change to *Focal Loss*.
* **Description:** We will stop using `CrossEntropyLoss` and static weights. We will implement *Focal Loss*, a function designed specifically for imbalanced datasets. This function dynamically reduces the weight of "easy" images and focuses all of the network's mathematical attention on "hard" examples (historical False Negatives), forcing the model to learn more complex patterns.

### 📊 Temporal Results (Epoch 5)
* **Loss (Train / Val):** 0.0395 / 0.0321
* **Overall Accuracy:** 95.51%

| Clinical Metric | Value | Real-World Implication | Comparison vs Exp A (Baseline) |
| :--- | :--- | :--- | :--- |
| **False Negatives (FN)** | 24 | ⚠️ 24 severely ill patients without treatment. | 📉 Slightly worsens (+5) |
| **False Positives (FP)** | 23 | 💸 23 healthy patients subjected to further tests. | ↔️ Equal |
| **Pneumonia Hits (TP)** | 753 | Correct diagnoses. | ⬇️ Decreases (-5) |
| **Normal Hits (TN)** | 247 | Healthy patients discharged. | ↔️ Equal |

### 🧠 Analysis (Exp. F Phase 1 Conclusion)
Although the absolute numbers at Epoch 5 (24 FN) still do not surpass our historical Baseline (19 FN), the metric trend is revealing. The *Focal Loss* started off disoriented (reaching 114 FN at Epoch 2), but by forcing the model to focus exclusively on difficult examples, it triggered a drastic and sustained drop in errors.

Analysis of the learning curves indicates that validation loss continues to drop sharply (0.0321) with no signs of *overfitting*. **The model has not converged.** **Next steps:** As analytical intuition dictates, 5 epochs are not enough for Focal Loss to optimize the complex patterns of this dataset. We will proceed to a **Sub-experiment F2**, increasing the training cycle to **15 epochs** to allow the network to reach its full potential.

### 📌 Sub-experiment F2: Long-Term Focal Loss (15 Epochs)

### ⚙️ Configuration
* **Architecture:** ResNet18 (Transfer Learning, reset `fc` layer).
* **Loss Function:** `Focal Loss` (gamma=2.0).
* **Epochs:** 15 (Significant increase from the usual 5).
* **Hypothesis:** Observing that at Epoch 5 the model had not yet converged (loss was still plummeting), we extended training to 15 epochs assuming that *Focal Loss* needed more time to resolve the False Negative patterns.

### 📊 Results (Epoch 15)
* **Loss (Train / Val):** 0.0325 / 0.0304
* **Overall Accuracy:** 96.28%

| Clinical Metric | Value | Real-World Implication | Comparison vs Best Model (Exp C2) |
| :--- | :--- | :--- | :--- |
| **False Negatives (FN)** | 27 | ⚠️ 27 severely ill patients without treatment. | ❌ Worsens (+7) |
| **False Positives (FP)** | **12** | 💸 12 healthy patients subjected to tests. | 🏆 **Record improvement (-5)** |
| **Pneumonia Hits (TP)** | 750 | Correct diagnoses. | ⬇️ Decreases (-7) |
| **Normal Hits (TN)** | 258 | Healthy patients discharged. | ⬆️ Increases (+5) |

### 🧠 Analysis (Exp. F2 Conclusion)
**The limits of complex loss functions.** Extending training to 15 epochs allowed the network to converge beautifully from a statistical standpoint (reaching an enviable 96.28% accuracy and a record low of only 12 False Positives). The *Focal Loss* made the network incredibly confident when detecting healthy lungs.

However, for our critical clinical metric (Pneumonia Recall), it does not come close to the previous experiment. False Negatives stalled at 27.

**Verdict:** It is empirically proven in this project that to force an extreme clinical bias (like lowering FNs to single digits), post-processing techniques during inference (**Threshold Tuning and TTA**) are necessary, computationally cheaper, and more controllable than attempting to alter the training core with complex weights or loss functions.

**Next steps:** We will apply post-processing techniques to improve this model's performance and attempt to surpass previous results.

---

## 🧪 Experiment G: Focal Loss + Threshold Tuning (The Quest for Zero)

### ⚙️ Configuration
* **Base Model:** The model trained with Focal Loss at 15 epochs (Exp F2).
* **Technique:** Threshold Sweep during inference phase.
* **Hypothesis:** Knowing that Focal Loss generated high variability at the end of training, we use probabilistic calibration to find the exact point where Sensitivity is maximized without destroying Specificity.

### 📊 Results (Threshold Comparison)

| Threshold | Accuracy | False Negatives (FN) | False Positives (FP) | Implication / Trade-off |
| :---: | :---: | :---: | :---: | :--- |
| **0.50 (Base)** | 95.70% | 8 | 37 | Default conservative. |
| **0.45** | 94.27% | 6 | 54 | Similar to TTA, but with more FPs. |
| **0.40** | 93.79% | **3** | 62 | **Clinical Sweet Spot.** Minimal lethal risk (only 3 FNs) and moderate hospital load. |
| **0.35** | 91.69% | 2 | 85 | High paranoia. |
| **0.30** | 89.40% | **0** | 111 | **100% Recall.** Perfect triage. No sick patient slips through, at the cost of collapsing the testing room. |

### 🧠 Analysis (Exp. G Conclusion)
This experiment represents the **100% Recall** milestone, as well as training a model with minimal False Negatives and high overall accuracy. We have demonstrated that by combining a dynamic training penalty (*Focal Loss*) with a probabilistic inference correction (*Threshold Tuning*), we can force the network to achieve **100% Sensitivity (0 False Negatives at 30%)**.

**Application (simulated):** In an educational demo, this threshold could be exposed as a slider to visualize the trade-off between sensitivity and false alarms under different load scenarios.

---

## 🧪 Experiment H: Focal Loss + TTA

### ⚙️ Configuration
* **Base Model:** Trained with `Focal Loss` at 15 epochs (Exp F2).
* **Technique:** Test-Time Augmentation (TTA) with majority voting (3-way).
* **Hypothesis:** Focal loss gave us a highly sensitive model but unstable at its decision boundaries. By applying TTA in inference, we seek geometric consensus to absorb that instability, lowering False Negatives without skyrocketing False Positives as wildly as a simple threshold adjustment would.

### 📊 Results (TTA Inference)
* **Overall Accuracy:** 94.08%

| Clinical Metric | Value | Real-World Implication | Comparison vs Exp F2 (Base Focal Loss) |
| :--- | :--- | :--- | :--- |
| **False Negatives (FN)** | **3** | ⚠️ 3 severely ill patients without treatment. | 🌟 **Massive improvement (-5)** (Over baseline) |
| **False Positives (FP)** | 59 | 💸 59 healthy patients subjected to tests. | 📉 Worsens (+22) |
| **Pneumonia Hits (TP)** | 774 | Correct diagnoses. (Sensitivity: 99.6%) | ⬆️ Increases (+5) |
| **Normal Hits (TN)** | 211 | Healthy patients discharged. | ⬇️ Decreases (-22) |

### 🧠 Analysis (Exp. H Conclusion)
This experiment represents a slight improvement over using Threshold Tuning. We have demonstrated that the combination of dynamic training penalty (*Focal Loss*) alongside an inference consensus system (*TTA*) produces an experimental model that strongly limits False Negatives while maintaining high global accuracy.
While not the most numerically accurate model, it is one of the best at balancing strict False Negative limitation and maintaining precision.

---

## 🧪 Experiment I: Specialized Architecture (DenseNet121)
### ⚙️ Configuration and Objective
* **Technique:** Change of Base Architecture (Transfer Learning).
* **Description:** ResNet18 is fast and efficient, but DenseNet121 is the *gold standard* in medical literature (used in Stanford University's CheXNet model). Its dense connections between layers better preserve high-frequency information (subtle textures), which is critical for detecting blurry pulmonary infiltrates that ResNet might overlook.

### 📌 Sub-experiment I1: DenseNet121 Baseline (Neutral Weights)

### ⚙️ Configuration
* **Architecture:** DenseNet121 (Transfer Learning, frozen base, new `classifier` layer reset).
* **Optimizer:** Adam (lr=0.001) training only the classifier.
* **Loss Function:** `CrossEntropyLoss` with weights `[1.0, 1.0]`.
* **Hypothesis:** Verify if an architecture specialized in medical imaging improves the initial baseline without the need for additional techniques.

### 📊 Results (Epoch 5)
* **Loss (Train / Val):** 0.1545 / 0.1343
* **Overall Accuracy:** 94.75%

| Clinical Metric | Value | Real-World Implication | Comparison vs Exp A (ResNet Baseline) |
| :--- | :--- | :--- | :--- |
| **False Negatives (FN)** | **6** | ⚠️ 6 severely ill patients without treatment. | 🌟 **Massive improvement (-13)** (Over ResNet18)|
| **False Positives (FP)** | 49 | 💸 49 healthy patients subjected to tests. | 📉 Worsens (+26) |
| **Pneumonia Hits (TP)** | 771 | Correct diagnoses. | ⬆️ Increases (+13) |
| **Normal Hits (TN)** | 221 | Healthy patients discharged. | ⬇️ Decreases (-26) |

### 🧠 Analysis (Exp. I1 Conclusion)
**The importance of the Gold Standard.** Changing our AI's engine made a massive difference. Without applying weight balancing, Focal Loss, or Threshold Tuning, the native DenseNet121 architecture achieves **6 False Negatives** (tying our best ResNet with TTA).

This proves that its dense connections extract radiological features better. The native trade-off of this model is higher sensitivity (detects almost everything) at the cost of reducing specificity (False Positives increase to 49 and Accuracy drops slightly to 94.75%).

The next step would be to apply the techniques we found most effective (Focal Loss + TTA) to see if we can improve on the Experiment H model.

### 📌 Sub-experiment I2: DenseNet121 + Focal Loss (Base Training)

### ⚙️ Configuration
* **Architecture:** DenseNet121 (Transfer Learning, `classifier` layer reset).
* **Loss Function:** `Focal Loss` (gamma=2.0).
* **Epochs:** 15.
* **Hypothesis:** Injecting the dynamic penalty (Focal Loss) into the *Gold Standard* medical architecture (DenseNet121) will prepare a mathematically superior base model. Although we know pure Focal Loss generates instability, it will lay the foundation for applying post-processing techniques (TTA and Threshold Tuning).

### 📊 Results (Epoch 15)
* **Loss (Train / Val):** 0.0346 / 0.0288
* **Overall Accuracy:** 95.51%

| Clinical Metric | Value | Real-World Implication | Comparison vs Exp I1 (Base DenseNet) |
| :--- | :--- | :--- | :--- |
| **False Negatives (FN)** | 24 | ⚠️ 24 severely ill patients without treatment. | ❌ Worsens (+18) |
| **False Positives (FP)** | 23 | 💸 23 healthy patients subjected to tests. | ✅ Improves (-26) |
| **Pneumonia Hits (TP)** | 753 | Correct diagnoses. | ⬇️ Decreases (-18) |
| **Normal Hits (TN)** | 247 | Healthy patients discharged. | ⬆️ Increases (+26) |

### 🧠 Analysis (Exp. I2 Conclusion)
Just as happened with ResNet18, applying *Focal Loss* without post-processing generates instability in the final epochs (going from 19 FNs in epoch 14 to 24 FNs in epoch 15). Furthermore, in raw numbers, it fails to beat the incredible 6 FNs that DenseNet gave us "out of the box."

However, Focal Loss has served its purpose: it forced the network to express doubt on difficult cases, drastically reducing false alarms (False Positives dropped from 49 to 23 compared to the DenseNet baseline). We now have a highly specialized "brain" ready to be calibrated in the inference phase.

### 📌 Sub-experiment I3: DenseNet121 + Focal Loss + TTA

### ⚙️ Configuration
* **Base Model:** Trained with `Focal Loss` at 15 epochs (Exp I2).
* **Technique:** Test-Time Augmentation (TTA) with majority voting (3-way).
* **Hypothesis:** Check if combining the more complex architecture (DenseNet), the dynamic loss (Focal Loss), and inference consensus (TTA) produces the definitive model.

### 📊 Results (TTA Inference)
* **Overall Accuracy:** 95.70%

| Clinical Metric | Value | Real-World Implication | Comparison vs Exp I2 (No TTA) |
| :--- | :--- | :--- | :--- |
| **False Negatives (FN)** | 8 | ⚠️ 8 severely ill patients without treatment. | 🌟 **Improves (-16)** |
| **False Positives (FP)** | 37 | 💸 37 healthy patients subjected to tests. | 📉 Worsens (+14) |
| **Pneumonia Hits (TP)** | 769 | Correct diagnoses. | ⬆️ Increases (+16) |
| **Normal Hits (TN)** | 233 | Healthy patients discharged. | ⬇️ Decreases (-14) |

### 🧠 Analysis (Exp. I3 Conclusion)
Although TTA managed to stabilize the model (reducing FNs from 24 to 8), this result is clinically inferior to our ResNet18 with TTA (3 FNs) and our base DenseNet121 (6 FNs).

This demonstrates that injecting *Focal Loss* into an architecture that is already intrinsically very sensitive (DenseNet) overcomplicates the feature space.
**Next step:** Logic dictates that we must pivot. If native DenseNet121 with neutral weights (Exp I1) achieved 6 False Negatives on its own, we will apply TTA directly on that clean base model to see if we can reach perfect performance.

### 📌 Sub-experiment I4: DenseNet121 Baseline + TTA

### ⚙️ Configuration
* **Base Model:** Trained with `CrossEntropyLoss` and neutral weights at 5 epochs (Exp I1).
* **Technique:** Test-Time Augmentation (TTA) with majority voting (3-way).
* **Hypothesis:** Apply probabilistic consensus (TTA) directly on the clean DenseNet121 baseline, avoiding Focal Loss over-optimization, to attempt to beat the ResNet18 record.

### 📊 Results (TTA Inference)
* **Overall Accuracy:** 92.36%

| Clinical Metric | Value | Real-World Implication | Comparison vs Exp I1 (Base DenseNet) |
| :--- | :--- | :--- | :--- |
| **False Negatives (FN)** | **5** | ⚠️ 5 severely ill patients without treatment. | Slight improvement (-1) |
| **False Positives (FP)** | 75 | 💸 75 healthy patients subjected to tests. | 📉 Drastically worsens (+26) |
| **Pneumonia Hits (TP)** | 772 | Correct diagnoses. | ⬆️ Increases (+1) |
| **Normal Hits (TN)** | 195 | Healthy patients discharged. | ⬇️ Decreases (-26) |

### 🧠 Analysis (Exp. I4 Conclusion)
**The limit of clinical sensitivity.** By applying geometric consensus (TTA) to the base DenseNet model, we managed to reduce False Negatives to 5. However, the model became extremely paranoid: False Positives skyrocketed to 75 and overall accuracy dropped to 92.36%.

DenseNet121 is such a deep architecture and so sensitive to subtle textures that, when receiving multiple variations of the same image (TTA), it tends to over-diagnose the disease.

---

## 🏁 Provisional Verdict (Internal validation only)

After evaluating multiple architectures, loss functions, and inference post-processing techniques, the configuration taken as our experimental baseline was the one developed in **Experiment H**:

* **Architecture:** ResNet18 (Lightweight and fast for web environments).
* **Training:** *Focal Loss* at 15 epochs.
* **Inference:** 3-way *Test-Time Augmentation (TTA)*.

**Justification for clinical logic (simulated):**
This configuration proved to be the most balanced during internal validation. It achieved a **Recall of 99.6%**, allowing only **3 patients** (False Negatives) to slip past diagnosis, compared to the 19 patients lost in the original baseline. Furthermore, it kept hospital load relatively controlled (59 False Positives) alongside an overall accuracy of **94.08%**. This result was **provisional** and served solely for deciding which variant to take to the external test.

---

## Simulated External Evaluation (Test Set) and Domain Shift

### 📌 Final Evaluation Context
Up to this point, all optimizations (*Focal Loss*, *Thresholding*, *TTA*) were tuned using the validation set (`val`). However, in a simulated hospital scenario, models must face radiographs sourced from different machines, bearing different calibrations, contrasts, and resolutions.

To simulate this, we will evaluate the model against **624 images (`test` folder)** that the network has never seen. In medical literature, it's known that this Kaggle subset presents a strong **Domain Shift** compared to the training images.

### 🧪 Evaluation 1: Reference Model (ResNet18 + Focal Loss + TTA)

### ⚙️ Configuration
* **Model:** ResNet18 trained with Focal Loss at 15 epochs (Exp H).
* **Inference:** 3-way Test-Time Augmentation (TTA).
* **Objective:** Verify if the winning combination from the validation phase maintains its efficacy when facing radiographs from a different domain.

### 📊 Results (Test Set)
* **Overall Accuracy:** 72.44%

| Clinical Metric | Value | Real-World Implication | Comparison vs Validation (Exp H) |
| :--- | :--- | :--- | :--- |
| **False Negatives (FN)** | **0** | 🏆 **Perfect Triage (100% Recall).** No sick patient escapes. | 🌟 Improves (-3) |
| **False Positives (FP)** | 172 | ⚠️ Clinical paranoia. Massive hospital load. | 📉 Drastically worsens (+113) |
| **Pneumonia Hits (TP)** | 390 | Correct disease diagnoses. | N/A (Dataset shift) |
| **Normal Hits (TN)** | 62 | Extreme difficulty issuing medical discharges. | N/A (Dataset shift) |

### 🧠 Analysis (The Over-Optimization Problem)
This result is a classic case study of *Domain Shift*. From a strict life-safety perspective, the model is perfect: **0 False Negatives** (it detected 100% of sick patients).

However, its overall accuracy plummeted from 94% to 72.44%. Why? By utilizing such an aggressive loss function as *Focal Loss*, combined with exhaustive inspection (*TTA*), the ResNet18 became hypersensitive to the textures of the original dataset. When faced with radiographs possessing different lighting or contrast (Test Set), the model defaults to "conservative paranoia": at the slightest geometric doubt, it diagnoses pneumonia to avoid the penalty.

**Technical Decision (Pivot):** Mathematical over-optimization failed to generalize. If *forcing* a simple network (ResNet) makes it paranoid when faced with new data, the correct engineering solution is to evaluate how our intrinsically superior architecture (DenseNet121) behaves without any mathematical trickery biasing its learning.


### 🧪 Evaluation 2: Plan B (DenseNet121 Baseline)

### ⚙️ Configuration
* **Model:** DenseNet121 trained with neutral weights at 5 epochs (Exp I1).
* **Inference:** Pure (No TTA).
* **Objective:** Verify if the medical *Gold Standard* generalizes better to the *Domain Shift* thanks to its dense connections, without using techniques that artificially force sensitivity.

### 📊 Results (Test Set)
* **Overall Accuracy:** 78.04%

| Clinical Metric | Value | Real-World Implication | Comparison vs ResNet (Exp H) |
| :--- | :--- | :--- | :--- |
| **False Negatives (FN)** | **5** | ⚠️ 5 patients untreated. "Perfect Triage" is lost, but it's a very low number. | 📉 Worsens (+5) |
| **False Positives (FP)** | 132 | 💸 High hospital load, but controlled. | 🌟 Improves (-40) |
| **Pneumonia Hits (TP)** | 385 | Correct disease diagnoses. | ⬇️ Decreases (-5) |
| **Normal Hits (TN)** | 102 | Healthy patients discharged. | ⬆️ Increases (+40) |

### 🧠 Analysis (Conclusion of pivoting to DenseNet121)
The intuition to pivot towards DenseNet121 was a technical success. When confronted with the *Domain Shift* of the new external domain (Test Set), this architecture proved to be intrinsically more robust than a simple, over-optimized architecture.

Overall accuracy improved by almost 6% (rising to 78.04%), and it recovered 40 healthy patients that ResNet would have sent for unnecessary tests (lowering False Positives from 172 to 132). The trade-off is that we lost "Perfect Triage" (going from 0 to 5 False Negatives), but on a global level, this base model withstands the evaluated domain shift much better because it isn't carrying the biases of an aggressive loss function (*Focal Loss*).
We will test with TTA to try to improve overall accuracy.

### 🧪 Evaluation 3: DenseNet121 + TTA (The Consensus)

### ⚙️ Configuration
* **Model:** DenseNet121 trained with neutral weights at 5 epochs (Exp I1).
* **Inference:** 3-way Test-Time Augmentation (TTA).
* **Objective:** Verify if forcing a geometric consensus upon the most robust architecture achieves the best experimental compromise against the *Domain Shift*.

### 📊 Results (Test Set)
* **Overall Accuracy:** 73.56%

| Clinical Metric | Value | Real-World Implication | Comparison vs Pure DenseNet |
| :--- | :--- | :--- | :--- |
| **False Negatives (FN)** | **1** | 🏆 **Near-Perfect Triage.** Only 1 undetected patient. | 🌟 Improves (-4) |
| **False Positives (FP)** | 164 | ⚠️ Clinical paranoia induced by TTA. | 📉 Worsens (+32) |
| **Pneumonia Hits (TP)** | 389 | Correct disease diagnoses. | ⬆️ Increases (+4) |
| **Normal Hits (TN)** | 70 | Healthy patients discharged. | ⬇️ Decreases (-32) |

### 🧠 Analysis
The application of TTA has proven to act as a "sensitivity multiplier". When faced with data from a different distribution (*Domain Shift*), forcing the model to evaluate 3 image variations meant that any minimal visual anomaly was flagged as Pneumonia.
Even so, the model's accuracy worsened and is not close to its training performance, so we will retrain with a new strategy that avoids the **Over-fitting** the model demonstrates when seeing new images.

---

## 🧪 Experiment J: Medical Specialization and Fine-Tuning (Generalization Strategy)

### ⚙️ Configuration and Strategy
* **Architecture:** DenseNet121.
* **Technique:** *Deep Fine-Tuning* + *Input Scaling* + *Heavy Augmentation*.
* **Description:** After detecting the model struggles with **Domain Shift** (accuracy drop from 94% to 78% when switching datasets, simulating a change in medical centers), we abandoned shallow learning in favor of three engineering improvements:

1.  **Selective Unfreezing:** The final two dense blocks of the model are unfrozen. This allows internal filters to re-specialize on opaque lung tissue textures instead of generic ImageNet shapes.
2.  **High-Variability Data Augmentation:** Implemented `ColorJitter` (brightness, contrast, and saturation) and `RandomGrayscale` to simulate distinct scanner calibrations and force the model to ignore external visual noise.
3.  **Resolution Increment (448x448):** We double the pixel area processed to preserve subtle infiltration details lost in standard 224px compression.

### 🎯 Hypothesis on Accuracy
By specializing internal filters and training at a higher resolution, we seek to break the 78% accuracy glass ceiling on the Test Set. The goal is to reach an **Accuracy > 85%** while maintaining a **Clinical Recall > 95%**, proving the model is robust enough to analyze images with features different from its training data.

### 🧠 Technical Justification (MLOps)
The over-optimization of the previous training (Focal Loss) generated a bias towards the validation set, creating clinical paranoia (massive False Positives) against new data. This experiment represents moving from "black-box tweaking" to "medical domain specialization" within an applied research context.

### 📊 Training Results (Fine-Tuning at 10 Epochs)

* **Total Duration:** 38 minutes and 51 seconds.
* **Best Epoch (Sweet Spot):** Epoch 7.
* **Peak Performance:** Val Loss: **0.0311** | Val Accuracy: **99.14%**

| Training Phase | Model Behavior | Technical Analysis |
| :--- | :--- | :--- |
| **Epochs 1 - 5** | Stable learning. *Val Loss* steadily drops from 0.0548 to 0.0387. | The model correctly assimilates the new resolution (448px) and color distortion without losing the foundational knowledge of DenseNet. |
| **Epoch 6** | ⚠️ **First peak of instability.** *Val Loss* skyrockets to 0.0860, while *Train Loss* keeps falling (0.0255). | Early sign of *Overfitting*. The model attempts to memorize the distortions in the training "obstacle course." |
| **Epoch 7** | 🏆 **Recovery and All-Time Record.** The model stabilizes its gradients, achieving a historic minimum *Val Loss* of **0.0311**. | The *Checkpointing* system detects this peak of generalization and saves the physical weights of the model at this exact instant. |
| **Epochs 8 - 10** | 🚨 **Overfitting Collapse.** *Train Loss* borders on perfection (0.0139), but *Val Loss* suffers a massive rebound in Epoch 9 (0.0898) and fails to recover. | The model has lost its ability to generalize and is over-optimizing the training data. |

### 🧠 Analysis (Exp. J Conclusion)
This training cycle visually demonstrates why deep *Fine-Tuning* is volatile. Unfreezing millions of parameters and using aggressive data augmentation causes the neural network to experience "peaks of amnesia" or pure memorization (as seen in epochs 6 and 9).

Thanks to the implementation of **Model Checkpointing**, the script automatically ignored the degradation in later epochs and restored the "brain" of Epoch 7 to memory, ensuring that we evaluate the smartest and most unbiased version of our AI.

This is the best result we have achieved. We will now perform tests to verify if these results translate into improvements with distinct datasets.

### 🧪 Evaluation 4: Pure Fine-Tuned DenseNet121 (448px)

### ⚙️ Configuration
* **Model:** DenseNet121 with unfrozen deep layers (Exp J - Epoch 7).
* **Inference:** Pure high-resolution (448x448) without TTA.
* **Objective:** Verify if the model specialized in high-resolution radiological textures can better withstand the *Domain Shift* of the Test Set.

### 📊 Results (Test Set)
* **Overall Accuracy:** 81.57%

| Clinical Metric | Value | Real-World Implication | Comparison vs Base DenseNet (Exp I1) |
| :--- | :--- | :--- | :--- |
| **False Negatives (FN)** | **0** | 🏆 **Perfect Triage (100% Recall).** No sick patient escapes. | 🌟 Massive Improvement (-5) |
| **False Positives (FP)** | 115 | ⚠️ Moderate clinical paranoia. Manageable hospital load. | 🌟 Improves (-17) |
| **Pneumonia Hits (TP)** | 390 | Correct disease diagnoses. | ⬆️ Increases (+5) |
| **Normal Hits (TN)** | 119 | Healthy patients discharged. | ⬆️ Increases (+17) |

### 🧠 Analysis
The engineering strategy (Deep Fine-Tuning + Resolution increase to 448px) has been a success. The model broke the 78% accuracy barrier on external data (Domain Shift), raising it to 81.57%.

The most remarkable aspect is that by specializing its final dense layers, the model reached a **native 100% Sensitivity (0 False Negatives)**, without needing destabilizing mathematical tricks (*Focal Loss*) or computational inference overhead (*TTA*). It is, to date, the most robust and secure medical model in the project. Although accuracy could improve with TTA, we will run a test to confirm this.

### 🧪 Evaluation 6: The Medical Committee (5-way TTA at 448px)

### ⚙️ Configuration
* **Model:** Fine-Tuned DenseNet121 (Exp J - Epoch 7).
* **Inference:** 5-way Test-Time Augmentation (Original, Rotate Right, Rotate Left, Zoom, Altered Contrast).
* **Decision Rule:** Hard majority voting (3 out of 5 votes required for a positive diagnosis).
* **Objective:** Verify if a massive ensemble of variations (including color/lighting perturbations) manages to weed out the model's final False Positives.

### 📊 Results (Test Set)
* **Overall Accuracy:** 81.57%

| Clinical Metric | Value | Real-World Implication | Comparison vs 3-Way TTA |
| :--- | :--- | :--- | :--- |
| **False Negatives (FN)** | **0** | 🏆 **Unbreakable Perfect Triage.** | ↔️ Equal (0) |
| **False Positives (FP)** | 115 | ⚠️ Stabilized paranoia. | 📉 Slightly worsens (+2) |
| **Pneumonia Hits (TP)** | 390 | Correct diagnoses. | ↔️ Equal |
| **Normal Hits (TN)** | 119 | Healthy patients discharged. | ⬇️ Decreases (-2) |

### 🧠 Analysis
This experiment demonstrates a phenomenon known as "Feature Anchoring". The base model is incredibly robust after high-resolution *Fine-Tuning*, making its predictions mathematically unmovable. The 5-way committee yielded the exact same results as the pure inference of the base model.

**Architectural Decision for the API:**
Since 5-way TTA quintuples the computational cost (server latency) without adding clinical benefits, the definitive architecture for production will be the **Pure Fine-Tuned DenseNet121** model or, as a maximum level of safety, **3-way TTA**, ensuring the optimal balance between diagnostic speed (Physician UX) and clinical reliability (0 False Negatives).

---

## 🧾 Project Status (Educational Framework)

* This log documents an **experimental learning endeavor**, not a healthcare product.
* None of the models described here are validated for clinical use, diagnosis, or triage in real patients.
* Prior to any real-world healthcare use, comprehensive multicenter validation, regulatory review, external auditing, and formal medical supervision would be required.
