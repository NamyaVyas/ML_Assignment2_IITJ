# Assignment 2: Bagging, Boosting and Bias–Variance on Face Recognition

**Name:** [Your name]  
**Student number:** [Your number]  
**Repository:** [GitHub URL]  
**Final commit:** [commit hash]

## Abstract
This report compares a decision-tree baseline, bootstrap aggregation, and multiclass SAMME boosting on the Olivetti faces dataset. All methods use the same fixed identity-stratified split. A central-patch masking test evaluates robustness to missing image information. Bias squared and variance are estimated from 30 resampled training sets.

## 1. Dataset and experimental setup
The dataset contains 40 identities with 10 grayscale images per identity, each of size 64×64. For each identity, six images were assigned to training, two to validation, and two to the clean test set using a fixed random seed. The saved split was reused for all experiments. Models were trained only on clean training images. For the masked test set, rows 24–39 and columns 24–39 were set to zero. No PCA or other dimensionality reduction was used.

**Insert actual settings:** seed [ ], selected tree depth [ ], minimum leaf size [ ], selected bagging size [ ], selected SAMME depth [ ], selected boosting rounds [ ].

## 2. Decision tree and bagging
### 2.1 Single-tree baseline
Hyperparameters were selected by validation accuracy. Report the selected depth, minimum samples per leaf, train/validation/test accuracy, number of leaves, average samples per leaf, and the fraction of leaves containing one training example. Describe whether deeper trees increased training accuracy while validation performance plateaued or fell.

### 2.2 Bagging from scratch
Each tree was trained on a bootstrap sample of the original training-set size. The ensemble prediction was the majority vote. For each ensemble size B, report validation and clean/masked test accuracy, mean distinct examples per bootstrap sample, pairwise disagreement, and mean vote margin. The theoretical expected number of distinct examples in a bootstrap sample of size n is n[1−(1−1/n)^n], approximately 0.632n.

### 2.3 Clean versus masked test
Insert `outputs/q1_clean_masked.csv`. Discuss confidence as the maximum class probability for a tree and the average tree confidence for bagging. Include two cases where bagging corrected the tree and two cases where both failed; use actual predictions from your run and explain the likely error type.

## 3. Multiclass SAMME boosting
SAMME trains weak learners on weighted examples. After each learner, incorrectly classified examples receive increased weight. The learner weight is α = log((1−ε)/ε)+log(K−1), where ε is its weighted error and K is the number of classes. A learner with error at least 1−1/K is not better than random guessing and is rejected/stops the process. The ensemble predicts the class with the largest weighted vote score.

Insert the actual grid results and selected settings. Plot training/validation accuracy over rounds and report training time. Discuss how a large learner weight can make one training example influential, and inspect its prediction history before drawing a conclusion.

## 4. Bias and variance
Thirty new training sets were created by sampling six training faces with replacement within each identity. Models were refit without tuning. For each test example, the mean predicted class-probability vector was calculated across resamples. Bias squared is the mean squared distance between this mean vector and the one-hot true label; variance is the mean squared distance of each resample's probability vector from the mean vector.

Insert `outputs/q3_bias_variance.csv` and the standard deviation of test accuracy. Explain the observed trade-off. Do not assume the deepest tree or lowest-variance model must have the best test accuracy.

## 5. Conclusion
Summarize the measured clean-test performance, the impact of masking, and the relationship between model complexity, bagging, boosting, bias, and variance. State the main limitation: results depend on this small dataset and fixed split.

## References
[1] AT&T Laboratories Cambridge, The Database of Faces / Olivetti faces dataset.  
[2] L. Breiman, “Bagging Predictors,” Machine Learning, 1996.  
[3] Y. Freund and R. Schapire, “A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting,” 1997.
