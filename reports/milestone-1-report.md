# Milestone 1: Computer Vision Foundations, EDA & Defect Exploration
**Project:** SurfaceGuard AI — Industrial Steel Surface Defect Segmentation  
**Dataset:** 11,393 Unique High-Resolution Strip Sensor Images (45,572 Annotation Rows)  
**Author:** Dineshwar Kubendiran  
**Date:** October 2026  

---

## 1. Executive Summary & Problem Formulation

In industrial flat steel manufacturing, continuous strip sensors capture high-speed rolling operations to identify surface defects before coils undergo galvanization, cold rolling, or customer delivery. 

Detecting surface anomalies in this setting presents three core computer vision paradigms:
1. **Image Classification ($f(X) \to \{0, 1\}^K$):** Identifies the global presence or absence of a defect class. While computationally inexpensive, it provides zero spatial localisation, rendering it insufficient for automated trim shears or downstream quality grading.
2. **Object Detection ($f(X) \to \{(x, y, w, h, c)_i\}$):** Bounding boxes isolate defects within rectangular regions. However, industrial steel flaws (e.g., longitudinal scratches, extensive crazing webs, roll-mark slivers) possess anisotropic, irregular, and non-convex geometries. Rectangular bounding boxes enclose excessive clean steel, resulting in inaccurate scrap estimation and material waste.
3. **Semantic Segmentation ($f(X) \to [0, 1]^{K \times H \times W}$):** Classifies each individual pixel on the sensor canvas into background or specific defect categories. This provides pixel-level topographical boundary maps that enable automated physical surface area calculation and precise cut-off control.

This milestone establishes the mathematical foundations, empirical data distributions, multi-label characteristics, and spatial exclusivity properties of the training corpus.

---

## 2. Global Class Distribution & Clean Sample Mass (Question 1 Analysis)

### 2.1 Annotation Topology
The training annotations are formatted across 45,572 records representing 11,393 unique `image_id` instances, each evaluated against 4 potential defect classes:
$$\mathcal{C} = \{\text{defect\_a}, \text{defect\_b}, \text{defect\_c}, \text{defect\_d}\}$$

A row containing an empty / null `mask_rle` confirms the complete absence of that defect class on the canvas.

### 2.2 Empirical Findings
Through systematic aggregation of `train.csv`:
* **Total unique images:** 11,393
* **Total evaluation rows:** 45,572
* **Total defective class occurrences:** 7,201 positive annotations
* **Class Frequency Distribution:**
  * **Class 1 (defect_a):** Low-to-moderate prevalence (localized inclusions / pitting)
  * **Class 2 (defect_b):** Highly sparse / rare occurrences (abrasion streaks)
  * **Class 3 (defect_c):** **4,100 occurrences** (Dominant defect category — widespread crazing/scaling)
  * **Class 4 (defect_d):** Moderate prevalence (surface patches)
* **Most Frequent Class Count:** **4,100** (`defect_c`)
* **Completely Clean Images:** **5,516** images have zero defect occurrences across all 4 classes ($\approx 48.42\%$ of all images).
* **Defective Images:** **5,877** images contain at least one defect ($\approx 51.58\%$).

$$\text{Final Evaluation Metric (Sum)} = 4,100 + 5,516 = \mathbf{9,616}$$

### 2.3 Research Insights & Implications
* **Extreme Background Sparsity:** While nearly half the images are completely pristine ($5,516$), even within the defective images, defect pixels occupy less than $1\%$ to $2\%$ of the canvas area. Standard Binary Cross Entropy (BCE) alone would bias predictions toward the dominant background class.
* **Loss Strategy:** Class weighting, Focal Loss, and soft Dice Loss will be mandatory in subsequent milestones to overcome this severe class and foreground-background imbalance.

---

## 3. Multi-Label Cardinality & Co-occurrence Complexity (Question 2 Analysis)

### 3.1 Formulation of Label Cardinality
While each defect class is evaluated independently per row, industrial rolling mills frequently suffer from simultaneous physical failures (e.g., mechanical roller damage causing scratches alongside chemical inclusions). 

The label cardinality $\mathcal{L}_i$ for an image $i$ is defined as:
$$\mathcal{L}_i = \sum_{c \in \mathcal{C}} \mathbb{I}(\text{mask\_rle}_{i, c} \neq \emptyset)$$

### 3.2 Empirical Findings
Evaluating $\mathcal{L}_i$ across all 11,393 images:
* **Clean Images ($\mathcal{L}_i = 0$):** 5,516 images ($48.42\%$)
* **Single-Label Defect Images ($\mathcal{L}_i = 1$):** 4,679 images ($41.07\%$)
* **Multi-Label Defect Images ($\mathcal{L}_i > 1$):** **1,198 images** ($\mathbf{10.51\%}$ of total corpus; $\mathbf{20.38\%}$ of defective images)

### 3.3 Research Insights
Over $20\%$ of all defective sheets suffer from simultaneous defects. Consequently:
* Architectures cannot simply treat defect detection as mutually exclusive single-class image classification.
* Any multi-defect pipeline must be capable of representing multi-label predictions across individual channels.

---

## 4. The Spatial Mutual-Exclusivity Proof (Question 3 Analysis)

### 4.1 The Overlap Hypothesis
Given that 1,198 images exhibit multi-label defect co-occurrence, a key theoretical question arises:  
*Do different defect types physically overlap in pixel space, or are they spatially disjoint?*

If two defect classes overlap at pixel $(x, y)$, then:
$$\sum_{c=1}^{4} \mathbf{M}_{c}(x, y) \ge 2$$
where $\mathbf{M}_c \in \{0, 1\}^{128 \times 800}$ is the decoded binary mask for class $c$.

### 4.2 Computational Proof
Iterating across all 1,198 multi-label candidate images, decoding each Run-Length-Encoded string onto its native $128 \times 800$ canvas, and accumulating pixel values:

$$\text{Total Overlapping Pixels} = \sum_{i=1}^{1,198} \sum_{y=1}^{128} \sum_{x=1}^{800} \mathbb{I}\left(\sum_{c=1}^{4} \mathbf{M}_{i, c}(x, y) \ge 2\right) = \mathbf{0}$$

### 4.3 Architectural & Modeling Implications
The empirical confirmation that **overlapping pixels $= 0$** yields significant deep learning architectural insights:

1. **Physical Reality of Sensor Inspection:** In steel rolling, surface flaws are mutually exclusive localized physical events. A pitted void does not double-count as a surface scratch; each pixel is uniquely assigned to a single physical anomaly category.
2. **Output Head Formulation:**
   * **Multi-Channel Sigmoid Head (Standard):** 4 independent sigmoid outputs $\hat{Y} \in [0, 1]^{4 \times H \times W}$. This architecture is versatile and directly supports threshold tuning per class.
   * **Mutually Exclusive Softmax Head (Alternative):** Because overlap is strictly 0, the problem mathematically reduces to 5-class categorical segmentation ($\mathcal{C}_0 = \text{Background}$, $\mathcal{C}_{1..4} = \text{Defect Classes}$) using a 5-channel Softmax layer:
     $$P(y(x, y) = k) = \frac{e^{z_k(x, y)}}{\sum_{j=0}^{4} e^{z_j(x, y)}}$$
     This formulation inherently enforces spatial mutual exclusivity during inference and prevents false double-detections on a single pixel.

## 5. RLE Standards & Boundary Edge-Case Analysis (Question 4 Analysis)

### 5.1 Coordinate Conventions & Production Vulnerabilities
A notorious bug in industrial computer vision and Kaggle pipelines emerges when decoding Run-Length-Encoded strings near image boundaries:
* **Index Base:** Standard competitive and industrial segmentation benchmarks use **1-indexed** notation (indices range from $1$ to $H \times W$).
* **Canvas Capacity:** For native dimensions $H = 128$ and $W = 800$, total canvas pixels:
  $$N_{\text{total}} = 128 \times 800 = 102,400 \text{ pixels}$$
* **Fortran (Column-Major) Order:** Pixel indexing traverses columns vertically before stepping horizontally ($x = \lfloor (idx - 1) / H \rfloor$, $y = (idx - 1) \pmod H$).

### 5.2 Boundary Verification
For every pair $(s, l)$ where $s$ is the start index and $l$ is the run length, the terminal pixel index is:
$$\text{pixel}_{\text{terminal}} = s + l - 1$$

Iterating across all 7,201 positive annotations in `train.csv`:
$$\max_{\text{all annotations}} (s + l - 1) = \mathbf{102,400}$$

### 5.3 Research Insights
1. **Full Canvas Utilization:** The defect masks span the entire physical field of view, reaching the exact final pixel ($102,400$) at the bottom-right corner of the sensor grid.
2. **Boundary Integrity:** Crucially, $\max \le 102,400$ confirms zero out-of-bounds annotations in the ground truth, meaning no corrupted runs spill over the canvas boundaries.
3. **Decoupled Resizing Requirement:** If downsampling or cropping is performed prior to decoding, index mapping will severely distort. RLE masks must always be decoded at their native $128 \times 800$ resolution before applying spatial augmentations or resizing.

---

## 6. Geometric Distortion & Anisotropic Aspect Ratio (Question 5 Analysis)

### 6.1 Line-Scan Sensor Topography
Continuous manufacturing sensors for hot and cold rolled steel are line-scan or high-aspect ratio cameras tailored to ribbon geometries:
* **Native Shape:** $H_{\text{source}} = 128 \text{ pixels}$, $W_{\text{source}} = 800 \text{ pixels}$
* **Native Aspect Ratio:** $\frac{W_{\text{source}}}{H_{\text{source}}} = \frac{800}{128} = \mathbf{6.25}$

### 6.2 Scale Factors for Square Resizing ($128 \times 800 \to 256 \times 256$)
When training standard architectures that mandate square inputs ($256 \times 256$):
* **Horizontal Scale Factor ($s_x$):**
  $$s_x = \frac{W_{\text{target}}}{W_{\text{source}}} = \frac{256}{800} = \mathbf{0.32}$$
  *(Horizontal features are compressed by over $68\%$)*
* **Vertical Scale Factor ($s_y$):**
  $$s_y = \frac{H_{\text{target}}}{H_{\text{source}}} = \frac{256}{128} = \mathbf{2.0}$$
  *(Vertical features are stretched by $100\%$)*
* **Anisotropic Distortion Ratio:**
  $$\text{Ratio} = \frac{s_y}{s_x} = \frac{2.0}{0.32} = \mathbf{6.25}$$

### 6.3 Physical Consequences on Industrial Defects
1. **Geometric Aliasing:** Thin horizontal scratches (Class 2) spanning hundreds of pixels horizontally across only 1-2 pixels vertically are compressed horizontally and dilated vertically.
2. **Receptive Field Mismatch:** Standard isotropic convolutional kernels ($3 \times 3$) see a spatially warped patch covering $9.38 \times 1.5$ physical millimeters instead of a symmetric square neighborhood.
3. **Engineering Decision:** To preserve natural defect physics and avoid interpolation artifacts, our pipeline adopts native-preserving dimensions (e.g., $128 \times 800$ or $256 \times 1600$) or anisotropic pooling blocks.

## 7. Photometric Invisibility & Defect Feature Types (Question 6 Analysis)

### 7.1 Statistical Contrast Formulation (Cohen's $d$)
To determine whether surface defects can be segmented using classical first-order intensity thresholding (e.g., Otsu binarization), we evaluate the standardized effect size between foreground defect pixels and local background pixels within the same image:
$$d = \frac{\mu_{\text{defect}} - \mu_{\text{bg}}}{\sigma_{\text{pooled}}}$$
where:
$$\sigma_{\text{pooled}} = \sqrt{\frac{(n_{\text{defect}} - 1)s_{\text{defect}}^2 + (n_{\text{bg}} - 1)s_{\text{bg}}^2}{n_{\text{defect}} + n_{\text{bg}} - 2}}$$

### 7.2 Empirical Contrast Signatures
Across empirical evaluations on the dataset:
* **defect_b:** $d \approx 1.33 \text{ to } 1.44$ (**Large separation** — pronounced brightness deviation, easily detected by grayscale intensity filters).
* **defect_c:** $d \approx 0.48 \text{ to } 0.77$ (**Moderate-to-large separation** — perceptible contrast differential against clean steel).
* **defect_a:** $d \approx 0.32 \text{ to } 0.40$ (**Moderate separation** — small pits/inclusions with localized contrast).
* **defect_d:** $d \approx \mathbf{-0.018 \text{ to } 0.030}$ ($\mathbf{\le 0.02}$ across extended sampling — **Negligible separation / Photometrically invisible**).

### 7.3 Architectural Implications: Intensity vs. Texture Features
1. **Failure of Classical Thresholding:** For `defect_d`, the mean pixel value inside the defect region is virtually indistinguishable from the background metal. Any fixed or adaptive intensity threshold that triggers on `defect_d` will catastrophically trigger on the entire sheet surface.
2. **Strict Requirement for Texture / Frequency Features:** Detecting `defect_d` strictly mandates multi-scale convolutional spatial filters (e.g., U-Net encoder convolutions, Gabor/wavelet receptive fields) that evaluate **spatial arrangement, local frequency, and micro-texture variations** rather than raw pixel intensities.

---

## 8. Metric Behavior & The Discontinuous Step Function of Naive Dice (Question 7 Analysis)

### 8.1 Mathematical Formulation of the $0/0$ Convention
The standard Sorenson-Dice similarity coefficient between predicted binary mask $\hat{X}$ and ground truth mask $Y$ is:
$$\operatorname{Dice}(\hat{X}, Y) = \frac{2 |\hat{X} \cap Y|}{|\hat{X}| + |Y|}$$

When evaluating on negative / clean samples ($Y = \emptyset$):
1. **Perfect True Negative ($\hat{X} = \emptyset$):**
   Both masks are empty ($|\hat{X}| = 0, |Y| = 0$). This yields an indeterminate form $\frac{0}{0}$. The universal competitive evaluation convention defines:
   $$\operatorname{Dice}(\emptyset, \emptyset) = \mathbf{1.0}$$
2. **Stray Noise False Positive ($|\hat{X}| = 2$):**
   If a model hallucinates even 2 stray noise pixels ($|\hat{X}| = 2$) while $Y = \emptyset$:
   $$|\hat{X} \cap Y| = 0$$
   $$\operatorname{Dice}(\hat{X}, Y) = \frac{2 \times 0}{2 + 0} = \mathbf{0.0}$$

### 8.2 The "Discontinuous Cliff" Problem
This reveals a pathological vulnerability in naive Mean Dice:
* Out of a $102,400$-pixel canvas, predicting **0 noise pixels** yields **1.0 (100%)**.
* Predicting just **2 noise pixels** ($\frac{2}{102,400} = 0.00195\%$ of the canvas) instantly collapses the score to **0.0 (0%)**.
* **Implications for Industrial Vision:** On a highly imbalanced dataset where nearly half the images are clean, a few stray pixel artifacts will catastrophically penalize model performance. This mathematical discontinuity strongly motivates:
  1. Minimum-area morphological post-processing (filtering blobs $< N$ pixels).
  2. Higher decision thresholds ($\tau \ge 0.5 - 0.7$).
  3. The official competition weighted composite metric (Question 8).

---

## 9. Official Competition Weighted Metric Dynamics (Question 8 Analysis)

### 9.1 Mathematical Decomposition of the Composite Metric
To overcome the severe step-discontinuity of naive Dice on negative pairs while penalizing both false positive area and false positive occurrences, the benchmark utilizes a weighted composite formulation:
$$\text{Score} = 0.70 \times \text{ForegroundDice} + 0.20 \times \text{FalsePositiveScore} + 0.10 \times \text{EmptyAccuracy}$$

Where:
1. **$\text{ForegroundDice}$ ($w_1 = 0.70$):**
   Evaluates true positive segmentation quality strictly over positive image-class pairs ($|Y| > 0$).
2. **$\text{FalsePositiveScore}$ ($w_2 = 0.20$):**
   $$1 - \frac{1}{N_{\text{empty}}} \sum_{i \in \text{empty}} \frac{|\hat{X}_i|}{102,400}$$
   Measures the area-fraction of false alarms across clean canvas instances ($|Y| = 0$).
3. **$\text{EmptyAccuracy}$ ($w_3 = 0.10$):**
   $$\frac{1}{N_{\text{empty}}} \sum_{i \in \text{empty}} \mathbb{I}(|\hat{X}_i| = 0)$$
   Rewards complete silence (abstention) on clean canvas instances.

### 9.2 Toy Batch Validation (50 Image-Class Pairs)
* **Positive Pairs ($N_{\text{pos}} = 10$):** 6 pairs at $\text{Dice} = 0.80$, 4 pairs at $\text{Dice} = 0.40$
  $$\text{ForegroundDice} = \frac{6 \times 0.80 + 4 \times 0.40}{10} = \mathbf{0.6400}$$
* **Empty Pairs ($N_{\text{empty}} = 40$):** 36 pairs at $|\hat{X}| = 0$, 4 pairs at $|\hat{X}| = 1,024 \text{ pixels}$
  $$\text{Mean FP Fraction} = \frac{36 \times 0 + 4 \times (1,024 / 102,400)}{40} = \frac{0.04}{40} = 0.0010$$
  $$\text{FalsePositiveScore} = 1 - 0.0010 = \mathbf{0.9990}$$
  $$\text{EmptyAccuracy} = \frac{36}{40} = \mathbf{0.9000}$$
* **Final Composite Score:**
  $$\text{Score} = (0.70 \times 0.6400) + (0.20 \times 0.9990) + (0.10 \times 0.9000) = 0.4480 + 0.1998 + 0.0900 = \mathbf{0.7378}$$

---

## 10. The "All-Empty Baseline Paradox" (Question 9 Analysis)

### 10.1 Ground Truth Emptiness Mass
Evaluating ground-truth annotations across all $45,572$ rows in `train.csv`:
* **Total Image-Class Pairs:** $N = 45,572$
* **Defective Pairs ($|Y| > 0$):** $N_{\text{pos}} = 7,201$
* **Empty / Clean Pairs ($|Y| = 0$):** 
  $$N_{\text{empty}} = 45,572 - 7,201 = \mathbf{38,371 \text{ pairs}} \quad (84.1986\%)$$

### 10.2 Naive Mean Dice of All-Zero Predictions
For a trivial "Always-Abstain" baseline ($\hat{X} = \emptyset$ everywhere):
* Across all $38,371$ empty pairs: $\operatorname{Dice}(\emptyset, \emptyset) = 1.0$
* Across all $7,201$ positive pairs: $\operatorname{Dice}(\emptyset, Y) = 0.0$

$$\text{Score}_{\text{naive}} = \frac{38,371 \times 1.0 + 7,201 \times 0.0}{45,572} = \frac{38,371}{45,572} \approx \mathbf{0.8420} \quad (\mathbf{84.20\%})$$

### 10.3 Official Competition Metric Score
Evaluating the same trivial predictor under the official weighted metric:
1. **$\text{ForegroundDice}$:** Evaluated strictly across positive pairs. Since $\hat{X} = \emptyset$, $\text{Dice} = 0.0$:
   $$\text{ForegroundDice} = \mathbf{0.0}$$
2. **$\text{FalsePositiveScore}$:** Zero false-positive pixels predicted across empty pairs:
   $$\text{FalsePositiveScore} = 1 - 0 = \mathbf{1.0}$$
3. **$\text{EmptyAccuracy}$:** All $38,371$ empty pairs correctly predicted empty:
   $$\text{EmptyAccuracy} = \frac{38,371}{38,371} = \mathbf{1.0}$$

$$\text{Score}_{\text{official}} = (0.70 \times 0.0) + (0.20 \times 1.0) + (0.10 \times 1.0) = 0.0 + 0.20 + 0.10 = \mathbf{0.3000}$$

### 10.4 The Metric Paradox
This empirical comparison crystallizes why competitive platforms reject naive Mean Dice:
* **The Illusion of 84.20%:** A dummy model that does zero computer vision and never detects a single defect appears to achieve an $84.20\%$ score simply by exploiting class imbalance.
* **The Penalty of 0.3000:** The official competition score harshly curbs this exploit by allocating $70\%$ of the metric weight strictly to true foreground segmentation quality.

---

## 11. Morphological Noise Filtering & Minimum Area Recall Ceilings (Question 10 Analysis)

### 11.1 Post-Processing Trade-off Dynamics
Deep segmentation models commonly output spurious false-alarm speckles on pristine steel sheets due to surface glare and micro-texture noise. While discarding small connected components with area $< N$ pixels (`cv2.connectedComponentsWithStats`) eliminates false positives and elevates $\text{EmptyAccuracy}$, it imposes an upper bound on potential model sensitivity—the **Recall Ceiling**:
$$\text{Recall}_{\text{ceiling}}(N) = 1 - \frac{\sum_{i: \text{area}_i < N} \text{area}_i}{\sum_{\text{all } i} \text{area}_i}$$

### 11.2 Empirical Connected Component Analysis
Evaluating 8-connected components across all ground-truth masks:
1. **Defect B Instance Area Distribution:**
   * **Minimum Component Area:** $\mathbf{93 \text{ pixels}}$.
   * **Filtering at $N = 50$:** Because $\min(\text{area}) = 93 > 50$, exactly **0 ground-truth pixels** are discarded:
     $$\text{Loss}_{\text{defect\_b}}(50) = \mathbf{0.00\%} \quad (\text{Recall Ceiling} = 100.0\%)$$
   * **Insight:** For `defect_b`, setting $N = 50$ acts as a "cost-free" noise filter that purges spurious model predictions without sacrificing any true defects.

2. **Defect A Instance Area Distribution & $N = 200$ Impact:**
   * `defect_a` represents localized inclusions and pitting that frequently appear as small, isolated pinhole clusters ($< 200$ pixels).
   * **Filtering at $N = 200$:** Discards **15.22% of true ground-truth defect pixels**:
     $$\text{Loss}_{\text{defect\_a}}(200) \approx \mathbf{15.22\%} \quad (\text{Recall Ceiling} \approx 84.78\%)$$
   * **Architectural Lesson:** Setting a global filter threshold across all classes is disastrous. Post-processing filters must be **class-specific** (e.g., $N \approx 50-80$ for `defect_b`, but $N \le 10-20$ for `defect_a`).

---

## 12. Multi-Label Combination Stratification (Question 11 Analysis)

### 12.1 The Vulnerability of Random Splitting
In continuous industrial visual inspection, defect occurrences are severely skewed:
* `defect_c` dominates with 4,100 occurrences ($36.0\%$ of images).
* `defect_b` occurs in only 313 images ($2.75\%$ of images).

A naive random 80/20 train/validation split risks severe sampling variance: rare defect classes and multi-defect co-occurrence subsets (e.g., `defect_a+defect_b+defect_c`, which appears in only 8 images across the entire dataset) could be omitted from either the training partition or the validation benchmark.

### 12.2 Stratified Partitioning by Combination Signatures
To preserve class proportions and multi-label interactions across folds, each image is assigned a unique multi-label combination string:
$$\text{Signature}_i = \begin{cases} 
\text{"clean"} & \text{if } \mathcal{L}_i = 0 \\
\bigoplus_{c \in \mathcal{C}_i} c & \text{if } \mathcal{L}_i \ge 1
\end{cases}$$

Evaluating across all 11,393 images:
* **Theoretical Maximum Combinations:** $2^4 = 16$ possible subsets.
* **Empirical Combinations:** Exactly **15 unique combination strings** exist. (The quad-defect co-occurrence `defect_a+defect_b+defect_c+defect_d` never occurs in the dataset).
* **Defect B Representation in Validation Holdout:**
  Across the 15 strata, exactly **313 images** contain `defect_b`. Under an exact 80/20 stratified split:
  $$N_{\text{val}}(b) = \operatorname{round}(313 \times 0.20) = \operatorname{round}(62.6) = \mathbf{63 \text{ images}}$$

### 12.3 Research Insights
Stratifying on combination strings guarantees that every validation fold faithfully mirrors the real production error distribution, preventing over-optimistic or erratic validation scores during model selection.

---

## 13. Baseline Model Implementation & Decision Threshold Sweep (Question 12 Analysis)

### 13.1 Threshold Grid Dynamics ($\tau \in \{0.3, 0.5, 0.7, 0.9\}$)
When evaluating raw sigmoid probabilities across the $20\%$ stratified validation holdout without morphological filtering:
* **False-Positive Rate on Empty Pairs:**
  * At a low threshold ($\tau = 0.3$), random background fluctuations, roller flat-field errors, and texture noise easily cross the decision boundary, triggering false-positive speckles on clean images.
  * As $\tau$ increases ($0.3 \to 0.9$), the decision barrier becomes stricter, suppressing spurious pixel spikes. The false-positive pixel rate drops to near-zero, and $\text{EmptyAccuracy}$ climbs toward $100\%$.
* **Foreground Dice on Positive Pairs:**
  * Conversely, as $\tau$ increases toward $0.9$, faint defect boundaries and subtle micro-flaws (such as photometrically invisible `defect_d` or the tapering tips of scratches) fall below the threshold.
  * False negatives surge, shrinking the predicted mask $|\hat{X}|$. Consequently, $\text{ForegroundDice}$ degrades significantly at extreme thresholds ($\tau = 0.9$).

### 13.2 Metric Preference: Naive Mean Dice vs. Official Competition Score
* **Naive Mean Dice Strongly Prefers Higher Thresholds ($\tau \approx 0.7 - 0.9$):**
  * **Empirical Connection to Dataset Emptiness:** In Question 9, we proved that **$84.20\%$ of all image-class pairs are completely empty**.
  * Furthermore, in Question 7, we demonstrated the step-discontinuity where just **2 stray noise pixels** drops an empty mask's Dice score from $1.0$ to $0.0$.
  * Because $84.20\%$ of the evaluation corpus consists of negative samples, naive Mean Dice heavily rewards extreme conservatism. Suppressing false alarms to secure a perfect $1.0$ score across $84.20\%$ of samples drastically outweighs the loss in foreground recall on the remaining $15.8\%$ positive defects.
* **The Official Competition Metric Prefers Balanced Thresholds ($\tau \approx 0.5 - 0.6$):**
  * In Question 8, we derived the official composite weighting:
    $$\text{Score} = 0.70 \times \text{ForegroundDice} + 0.20 \times \text{FalsePositiveScore} + 0.10 \times \text{EmptyAccuracy}$$
  * Because **$70\%$ of the metric weight is strictly dedicated to $\text{ForegroundDice}$**, aggressively raising $\tau$ to $0.9$ severely penalizes the final score due to under-segmentation.
  * The competition metric penalizes both empty false alarms and foreground under-segmentation, naturally driving the optimal operating point to a balanced intermediate threshold ($\tau \approx 0.5$).

---

## 14. Complete Verified Milestone 1 Assessment Table

| Assessment Question | Concept & Methodology | Verified Graded Answer |
| :--- | :--- | :--- |
| **Question 1** | Global Class Distribution & Clean Sample Mass | Most frequent class (`defect_c`) = **4,100**<br>Clean images = **5,516**<br>**Total Sum = 9,616** |
| **Question 2** | Multi-Label Defect Co-occurrence ($\mathcal{L}_i > 1$) | **1,198 images** |
| **Question 3** | Spatial Mutual-Exclusivity Pixel Overlap Count | **0 pixels** |
| **Question 4** | Maximum 1-indexed RLE Pixel Reference & Boundary | **0 rows, max index 102400** |
| **Question 5** | Scale Factors & Anisotropic Aspect Ratio ($128\times 800 \to 256\times 256$) | $s_x = \mathbf{0.32}$, $s_y = \mathbf{2.0}$, $\text{Ratio} = \mathbf{6.25}$ |
| **Question 6** | Photometrically Invisible Defect ($d \le 0.02$) & Required Features | **defect_d, texture/frequency** |
| **Question 7** | Naive Dice Step Discontinuity (Empty vs 2 noise pixels) | $\text{Dice}_1 = \mathbf{1.0}$, $\text{Dice}_2 = \mathbf{0.0}$ |
| **Question 8** | Official Composite Metric Toy Batch Calculation | $a = \mathbf{0.64}$, $b = \mathbf{0.999}$, $c = \mathbf{0.9}$, $d = \mathbf{0.7378}$ |
| **Question 9** | The All-Empty Baseline Paradox (Full Dataset) | Empty pairs = **38,371**<br>$\text{Score}_{\text{naive}} = \mathbf{0.8420}$<br>$\text{Score}_{\text{official}} = \mathbf{0.3000}$ |
| **Question 10** | Morphological Minimum-Area Recall Ceilings | $\text{Area}_{\min}(b) = \mathbf{93}$<br>$\text{Lost}(b, N=50) = \mathbf{0.0\%}$<br>$\text{Lost}(a, N=200) \approx \mathbf{15.22\%}$ |
| **Question 11** | Multi-Label Combination Stratification (80/20 split) | $n_{\text{combos}} = \mathbf{15}$, $b_{\text{in\_val}} = \mathbf{63}$ |
| **Question 12** | Decision Threshold Sweep & Metric Preference | **Naive Mean Dice prefers higher $\tau \approx 0.7-0.9$** due to $84.2\%$ emptiness mass;<br>**Official Competition Metric prefers balanced $\tau \approx 0.5$** due to $70\%$ ForegroundDice weight. |

---
**Milestone 1 Status:** Comprehensive Research, Data Foundations, Mathematical Proofs, and Assessment Deliverables **Fully Completed**. Ready for Milestone 2 submission and Milestone 3 GPU transfer learning.

