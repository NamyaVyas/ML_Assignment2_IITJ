import os
import sys
import json
import time
import platform
import subprocess
import traceback
from importlib import metadata

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from sklearn.datasets import fetch_olivetti_faces
from sklearn.tree import DecisionTreeClassifier

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
SEED = 26006 

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "outputs")
FIG = os.path.join(BASE, "figures")

N_JOBS = max(1, (os.cpu_count() or 2) - 1)

TREE_DEPTHS = [2, 4, 8, 16, None]
TREE_LEAVES = [1, 2, 4]
B_VALUES = [1, 5, 10, 25, 50, 100, 200]
BOOST_DEPTHS = [1, 2, 3, 4, 10]
BOOST_ROUNDS = [25, 50, 100]
N_RESAMPLES = 30
MASK_LO, MASK_HI = 24, 40       # rows/cols 24..39 inclusive -> 16x16 block
EPS = 1e-10


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------
class Tee:
    """Send console output to the screen and to a log file at the same time."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            s.write(text)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def rng_for(*tags):
    """Independent, reproducible generator derived from the roll-number seed."""
    return np.random.default_rng([SEED] + [int(t) for t in tags])


def seed_from(rng):
    return int(rng.integers(0, 2 ** 31 - 1))


def depth_label(depth):
    return "None" if depth is None else int(depth)


def depth_code(depth):
    return 0 if depth is None else int(depth)


def save_plot(filename):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, filename), dpi=160, bbox_inches="tight")
    plt.close()


def save_csv(df, filename):
    df.to_csv(os.path.join(OUT, filename), index=False)


def show(title, df):
    print("\n--- " + title + " ---")
    print(df.to_string(index=False, float_format=lambda v: "%.4f" % v))


def run_parallel(function, argument_list):
    if N_JOBS == 1 or len(argument_list) == 1:
        return [function(*a) for a in argument_list]
    return Parallel(n_jobs=N_JOBS)(delayed(function)(*a) for a in argument_list)


def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def line_plot(x, series, xlabel, ylabel, filename, logx=False):
    plt.figure(figsize=(5.2, 3.6))
    for values, label in series:
        plt.plot(x, values, marker="o", label=label)
    if logx:
        plt.xscale("log")
        plt.xticks(x, [str(v) for v in x])
        plt.minorticks_off()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    if len(series) > 1 or series[0][1]:
        plt.legend()
    save_plot(filename)


def face_grid(items, filename, ncols=4):
    """items = list of (2D image, title). Saves a small gallery of faces."""
    if not items:
        return
    nrows = int(np.ceil(len(items) / ncols))
    plt.figure(figsize=(2.6 * ncols, 2.9 * nrows))
    for k, (img, title) in enumerate(items):
        plt.subplot(nrows, ncols, k + 1)
        plt.imshow(img, cmap="gray", vmin=0, vmax=1)
        plt.title(title, fontsize=7)
        plt.axis("off")
    save_plot(filename)


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
def load_data():
    """Olivetti faces, one fixed 6/2/2 split per identity (saved and reused)."""
    data = fetch_olivetti_faces()
    X = data.images.reshape(len(data.images), -1)
    y = data.target
    classes = np.unique(y)
    assert np.array_equal(classes, np.arange(len(classes))), "labels must be 0..K-1"

    split_file = os.path.join(OUT, "split_indices.npz")
    if os.path.exists(split_file):
        saved = np.load(split_file)
        tr, va, te = saved["train"], saved["val"], saved["test"]
        all_ids = np.concatenate([tr, va, te])
        if (len(all_ids) != len(y) or len(np.unique(all_ids)) != len(y)
                or all_ids.min() < 0 or all_ids.max() >= len(y)):
            raise ValueError(
                "outputs/split_indices.npz does not match this dataset. "
                "Back it up, remove it, and run again to create the fixed split."
            )
    else:
        rng = np.random.default_rng(SEED)
        tr, va, te = [], [], []
        for person in classes:
            ids = np.where(y == person)[0].copy()
            rng.shuffle(ids)
            tr.extend(ids[:6])
            va.extend(ids[6:8])
            te.extend(ids[8:10])
        tr = np.asarray(tr, dtype=int)
        va = np.asarray(va, dtype=int)
        te = np.asarray(te, dtype=int)
        np.savez(split_file, train=tr, val=va, test=te)

    for name, part, expected in [("train", tr, 6), ("val", va, 2), ("test", te, 2)]:
        counts = np.bincount(y[part], minlength=len(classes))
        assert np.all(counts == expected), "split %s is not %d per identity" % (name, expected)
    return X, y, classes, tr, va, te


def masked(X):
    """Set rows 24:40 and columns 24:40 of each 64x64 image to zero."""
    images = X.reshape(-1, 64, 64).copy()
    images[:, MASK_LO:MASK_HI, MASK_LO:MASK_HI] = 0.0
    return images.reshape(-1, 4096)


def patch_feature_flags():
    rows, cols = np.divmod(np.arange(4096), 64)
    return (rows >= MASK_LO) & (rows < MASK_HI) & (cols >= MASK_LO) & (cols < MASK_HI)


PATCH_FLAGS = patch_feature_flags()


def split_patch_share(tree):
    """Fraction of a tree's internal splits that use a pixel inside the masked patch."""
    feat = tree.tree_.feature
    feat = feat[feat >= 0]
    return float(PATCH_FLAGS[feat].mean()) if len(feat) else 0.0


# ----------------------------------------------------------------------
# Voting / tree utilities
# ----------------------------------------------------------------------
def vote_counts(pred_matrix, classes):
    """pred_matrix is (n_trees, n_samples). Returns (n_samples, n_classes) vote counts."""
    return np.stack([(pred_matrix == c).sum(axis=0) for c in classes], axis=1)


def winners(counts, classes):
    """Majority vote (ties go to the smallest class id)."""
    return classes[np.argmax(counts, axis=1)]


def vote_summary(counts):
    """Winning vote fraction and margin (winning fraction minus runner-up fraction)."""
    fractions = counts / counts.sum(axis=1, keepdims=True)
    ordered = np.sort(fractions, axis=1)
    return fractions, ordered[:, -1], ordered[:, -1] - ordered[:, -2]


def per_example_disagreement(counts):
    """Fraction of tree pairs that disagree, for every example, from vote counts."""
    n_trees = int(counts[0].sum())
    if n_trees < 2:
        return np.zeros(len(counts))
    agree_pairs = (counts * (counts - 1) / 2.0).sum(axis=1)
    return 1.0 - agree_pairs / (n_trees * (n_trees - 1) / 2.0)


def leaf_occupancy(model, X_fit, example_ids):
    """
    Leaf statistics in terms of DISTINCT training examples.
    example_ids[i] says which original training example row i of X_fit is
    (a bootstrap duplicate is the same example and is counted once).
    """
    leaf = model.apply(X_fit)
    pairs = np.unique(np.stack([leaf, np.asarray(example_ids)], axis=1), axis=0)
    _, per_leaf = np.unique(pairs[:, 0], return_counts=True)
    n_leaves = int(model.get_n_leaves())
    return {
        "number_of_leaves": n_leaves,
        "mean_samples_per_leaf": float(per_leaf.sum() / n_leaves),
        "fraction_leaves_with_one_training_example": float(np.mean(per_leaf == 1)),
    }


def aligned_probabilities(model, X, classes):
    """predict_proba columns arranged in the order of all K identities."""
    raw = model.predict_proba(X)
    aligned = np.zeros((len(X), len(classes)))
    for j, class_id in enumerate(model.classes_):
        aligned[:, np.where(classes == class_id)[0][0]] = raw[:, j]
    return aligned


def fit_bagging(Xr, yr, B, depth, leaf, rng):
    """Bagging from scratch: B trees, each on a bootstrap sample of the same size."""
    n = len(yr)
    models, boots = [], []
    for _ in range(B):
        ids = rng.integers(0, n, size=n)
        tree = DecisionTreeClassifier(
            max_depth=depth, min_samples_leaf=leaf, random_state=seed_from(rng)
        )
        tree.fit(Xr[ids], yr[ids])
        models.append(tree)
        boots.append(ids)
    return models, boots


def model_outputs(kind, model, X, classes):
    """Class-probability vector per sample: tree proba / bagging vote fractions /
    boosting normalised weighted votes. Also prediction and winning confidence."""
    if kind == "bagging":
        preds = np.asarray([m.predict(X) for m in model])
        prob = vote_counts(preds, classes) / float(len(model))
    else:
        prob = aligned_probabilities(model, X, classes)
    return {
        "pred": classes[np.argmax(prob, axis=1)],
        "conf": prob.max(axis=1),
        "prob": prob,
    }


# ----------------------------------------------------------------------
# Q1: tree baseline and bagging
# ----------------------------------------------------------------------
def tree_and_bagging(X, y, classes, tr, va, te):
    Xtr, ytr = X[tr], y[tr]
    Xva, yva = X[va], y[va]
    Xte, yte = X[te], y[te]
    Xmask = masked(Xte)
    n = len(tr)
    K = len(classes)

    # ---- Q1(a): choose the tree using VALIDATION accuracy only ----------
    settings, models, rows = [], [], []
    for depth in TREE_DEPTHS:
        for leaf in TREE_LEAVES:
            m = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf, random_state=SEED)
            m.fit(Xtr, ytr)
            occ = leaf_occupancy(m, Xtr, np.arange(n))
            settings.append((depth, leaf))
            models.append(m)
            rows.append({
                "max_depth": depth_label(depth),
                "min_samples_leaf": leaf,
                "actual_depth": int(m.get_depth()),
                "train_accuracy": accuracy(ytr, m.predict(Xtr)),
                "val_accuracy": accuracy(yva, m.predict(Xva)),
                "clean_test_accuracy": accuracy(yte, m.predict(Xte)),
                **occ,
            })
    tree_grid = pd.DataFrame(rows)
    # Ties in validation accuracy: prefer the simpler tree (fewer leaves).
    order = tree_grid.sort_values(
        ["val_accuracy", "number_of_leaves", "min_samples_leaf"],
        ascending=[False, True, True], kind="stable",
    )
    best_idx = int(order.index[0])
    depth, leaf = settings[best_idx]
    tree = models[best_idx]
    tree_grid["selected"] = False
    tree_grid.loc[best_idx, "selected"] = True
    save_csv(tree_grid, "q1_tree_grid.csv")
    show("Q1(a) tree grid (selected tree marked)", tree_grid)

    selected_summary = tree_grid.loc[[best_idx]].copy()
    selected_summary["masked_test_accuracy"] = accuracy(yte, tree.predict(Xmask))
    save_csv(selected_summary, "q1_selected_tree_summary.csv")
    print("Frozen tree: max_depth=%s, min_samples_leaf=%d" % (depth_label(depth), leaf))

    # ---- Q1(b),(c): bagging from scratch for every B ---------------------
    expected_distinct = n * (1 - (1 - 1 / n) ** n)
    bag_rows, bag_store = [], {}
    for B in B_VALUES:
        rng = rng_for(1, B)
        in_bag = np.zeros((B, n), dtype=bool)
        distinct = []
        models_b = []
        start = time.time()
        for b in range(B):
            ids = rng.integers(0, n, size=n)
            t = DecisionTreeClassifier(
                max_depth=depth, min_samples_leaf=leaf, random_state=seed_from(rng)
            )
            t.fit(Xtr[ids], ytr[ids])
            models_b.append(t)
            in_bag[b, ids] = True
            distinct.append(len(np.unique(ids)))
        training_seconds = time.time() - start  # training only

        P_tr = np.asarray([m.predict(Xtr) for m in models_b])
        P_va = np.asarray([m.predict(Xva) for m in models_b])
        P_te = np.asarray([m.predict(Xte) for m in models_b])
        P_mk = np.asarray([m.predict(Xmask) for m in models_b])
        C_tr = vote_counts(P_tr, classes)
        C_va = vote_counts(P_va, classes)
        C_te = vote_counts(P_te, classes)
        C_mk = vote_counts(P_mk, classes)

        # Genuine out-of-bag accuracy: for training example i, vote only among
        # trees whose bootstrap sample did not contain i.
        oob_correct, oob_n = 0, 0
        for i in range(n):
            eligible = ~in_bag[:, i]
            if eligible.any():
                votes = np.bincount(P_tr[eligible, i], minlength=K)
                oob_correct += int(np.argmax(votes) == ytr[i])
                oob_n += 1

        _, _, val_margin = vote_summary(C_va)
        _, test_conf, _ = vote_summary(C_te)
        bag_rows.append({
            "B": B,
            "train_accuracy": accuracy(ytr, winners(C_tr, classes)),
            "oob_accuracy": oob_correct / oob_n if oob_n else np.nan,
            "oob_examples_evaluated": oob_n,
            "val_accuracy": accuracy(yva, winners(C_va, classes)),
            "clean_test_accuracy": accuracy(yte, winners(C_te, classes)),
            "masked_test_accuracy": accuracy(yte, winners(C_mk, classes)),
            "training_seconds": training_seconds,
            "mean_distinct_training_examples": float(np.mean(distinct)),
            "theoretical_expected_distinct": expected_distinct,
            "mean_pairwise_disagreement": float(np.mean(per_example_disagreement(C_va))),
            "mean_vote_margin": float(np.mean(val_margin)),
            "mean_test_confidence": float(np.mean(test_conf)),
        })
        bag_store[B] = models_b
        print("  bagging B=%d done (%.1fs)" % (B, training_seconds))

    bag_df = pd.DataFrame(bag_rows)
    save_csv(bag_df, "q1_bagging.csv")
    show("Q1(b,c) bagging results", bag_df)

    # Final ensemble size: validation accuracy only; ties -> smaller B.
    best_row = bag_df.sort_values(["val_accuracy", "B"], ascending=[False, True], kind="stable").iloc[0]
    chosen_B = int(best_row["B"])
    final_bag = bag_store[chosen_B]
    print("Selected bagging size: B=%d" % chosen_B)

    # plots: OOB, disagreement, margin against B (+ the usual extras)
    Bs = bag_df["B"].tolist()
    line_plot(Bs, [(bag_df["oob_accuracy"], "")], "Number of trees (B)", "OOB accuracy", "q1_oob_accuracy.png", True)
    line_plot(Bs, [(bag_df["mean_pairwise_disagreement"], "")], "Number of trees (B)", "Mean pairwise disagreement", "q1_disagreement.png", True)
    line_plot(Bs, [(bag_df["mean_vote_margin"], "")], "Number of trees (B)", "Mean vote margin", "q1_vote_margin.png", True)
    line_plot(Bs, [(bag_df["clean_test_accuracy"], "clean test"), (bag_df["val_accuracy"], "validation")],
              "Number of trees (B)", "Accuracy", "q1_bagging_accuracy.png", True)
    line_plot(Bs, [(bag_df["training_seconds"], "")], "Number of trees (B)", "Training time (s)", "q1_bagging_time.png", True)
    plt.figure(figsize=(12, 3.4))
    for k, (col, lab) in enumerate([("oob_accuracy", "OOB accuracy"),
                                    ("mean_pairwise_disagreement", "Mean pairwise disagreement"),
                                    ("mean_vote_margin", "Mean vote margin")]):
        plt.subplot(1, 3, k + 1)
        plt.plot(Bs, bag_df[col], marker="o")
        plt.xscale("log")
        plt.xticks(Bs, [str(v) for v in Bs], fontsize=7)
        plt.minorticks_off()
        plt.xlabel("Number of trees (B)")
        plt.ylabel(lab)
        plt.grid(alpha=0.3)
    save_plot("q1_oob_disagreement_margin.png")

    # ---- Q1(d): frozen tree vs final bagged model, clean and masked ------
    cmp_rows, case_frames = [], []
    for test_name, Xin in [("clean", Xte), ("masked", Xmask)]:
        t_out = model_outputs("tree", tree, Xin, classes)
        b_out = model_outputs("bagging", final_bag, Xin, classes)
        for name, o in [("single_tree", t_out), ("bagging", b_out)]:
            ok = o["pred"] == yte
            cmp_rows.append({
                "model": name, "test_type": test_name,
                "accuracy": float(ok.mean()),
                "mean_confidence": float(o["conf"].mean()),
                "mean_confidence_when_correct": float(o["conf"][ok].mean()) if ok.any() else np.nan,
                "mean_confidence_when_wrong": float(o["conf"][~ok].mean()) if (~ok).any() else np.nan,
            })
        counts = (b_out["prob"] * len(final_bag)).round().astype(int)
        rows_c = []
        for j in range(len(yte)):
            t_ok = bool(t_out["pred"][j] == yte[j])
            b_ok = bool(b_out["pred"][j] == yte[j])
            rows_c.append({
                "test_type": test_name,
                "test_position": j,
                "original_dataset_index": int(te[j]),
                "true_identity": int(yte[j]),
                "tree_prediction": int(t_out["pred"][j]),
                "tree_confidence": float(t_out["conf"][j]),
                "bag_prediction": int(b_out["pred"][j]),
                "n_trees": len(final_bag),
                "bag_winning_votes": int(counts[j].max()),
                "bag_votes_for_true_identity": int(counts[j, yte[j]]),
                "bag_votes_for_tree_prediction": int(counts[j, t_out["pred"][j]]),
                "tree_correct": t_ok,
                "bag_correct": b_ok,
                "bagging_corrected_tree": (not t_ok) and b_ok,
                "both_wrong": (not t_ok) and (not b_ok),
            })
        case_frames.append(pd.DataFrame(rows_c))
    cmp_df = pd.DataFrame(cmp_rows)
    save_csv(cmp_df, "q1_clean_masked.csv")
    show("Q1(d) single tree vs bagging, clean and masked", cmp_df)

    cases = pd.concat(case_frames, ignore_index=True)
    save_csv(cases, "q1_test_cases.csv")

    # automatically pick two "bagging corrects the tree" and two "both wrong" faces
    selected = []
    for test_name in ["clean", "masked"]:
        d = cases[cases["test_type"] == test_name]
        fixed = d[d["bagging_corrected_tree"]].sort_values(
            ["bag_votes_for_true_identity", "test_position"], ascending=[False, True]).head(2)
        wrong = d[d["both_wrong"]].sort_values(
            ["bag_winning_votes", "test_position"], ascending=[False, True]).head(2)
        if len(fixed) < 2:
            print("WARNING: only %d 'bagging corrected the tree' faces on %s test set" % (len(fixed), test_name))
        if len(wrong) < 2:
            print("WARNING: only %d 'both wrong' faces on %s test set" % (len(wrong), test_name))
        selected.append(fixed.assign(case_type="bagging_corrected_tree"))
        selected.append(wrong.assign(case_type="both_wrong"))
    selected = pd.concat(selected, ignore_index=True)
    save_csv(selected, "q1_selected_cases.csv")
    show("Q1(d) selected example faces (tree prediction and bagging vote counts)",
         selected[["test_type", "case_type", "original_dataset_index", "true_identity",
                   "tree_prediction", "tree_confidence", "bag_prediction", "bag_winning_votes",
                   "bag_votes_for_true_identity", "bag_votes_for_tree_prediction", "n_trees"]])
    clean_imgs = Xte.reshape(-1, 64, 64)
    mask_imgs = Xmask.reshape(-1, 64, 64)
    items = []
    for _, r in selected.iterrows():
        img = clean_imgs[r["test_position"]] if r["test_type"] == "clean" else mask_imgs[r["test_position"]]
        items.append((img, "%s | %s\ntrue %d, tree %d, bag %d (true-votes %d/%d)" % (
            r["test_type"], r["case_type"].replace("_", " "), r["true_identity"],
            r["tree_prediction"], r["bag_prediction"], r["bag_votes_for_true_identity"], r["n_trees"])))
    face_grid(items, "q1_selected_faces.png", ncols=4)

    return {
        "tree": tree, "depth": depth, "leaf": leaf,
        "chosen_B": chosen_B, "final_bag": final_bag, "bag_df": bag_df,
        "q1_compare": cmp_df,
    }


# ----------------------------------------------------------------------
# Q2: SAMME AdaBoost from scratch
# ----------------------------------------------------------------------
class SimpleSAMME:
    """Multiclass SAMME with decision-tree weak learners.
    alpha_t = log((1-err)/err) + log(K-1);  S_k(x) = sum_t alpha_t * 1[h_t(x)=k];
    confidence q_k(x) = S_k(x) / sum_t alpha_t."""

    def __init__(self, n_estimators=50, max_depth=2, random_state=SEED):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

    def fit(self, X, y, keep_details=False):
        self.classes_ = np.unique(y)
        n = len(y)
        K = len(self.classes_)
        weights = np.ones(n, dtype=float) / n
        self.models_, self.alphas_ = [], []
        self.history_, self.weight_history_, self.details_ = [], [], []
        self.stop_reason_ = "completed all requested rounds"

        for t in range(1, self.n_estimators + 1):
            learner = DecisionTreeClassifier(
                max_depth=self.max_depth, random_state=self.random_state + t
            )
            learner.fit(X, y, sample_weight=weights)
            prediction = learner.predict(X)
            wrong = prediction != y
            error = float(np.sum(weights * wrong))

            # Explicit handling of a learner that is not better than random
            # K-class guessing (error >= 1 - 1/K): reject it and stop boosting.
            if error >= 1.0 - 1.0 / K:
                self.stop_reason_ = (
                    "round %d: weighted error %.4f >= 1-1/K = %.4f (not better than "
                    "random guessing); learner rejected and boosting stopped"
                    % (t, error, 1.0 - 1.0 / K))
                break

            perfect = error <= 0.0
            safe_error = max(error, EPS)  # avoid log(0) for a perfect learner
            alpha = float(np.log((1.0 - safe_error) / safe_error) + np.log(K - 1))

            before = weights.copy()
            weights = weights * np.exp(alpha * wrong)
            weights = weights / weights.sum()

            self.models_.append(learner)
            self.alphas_.append(alpha)
            self.weight_history_.append(weights.copy())
            top_n = min(10, n)
            max_index = int(np.argmax(weights))
            self.history_.append({
                "round": t,
                "weighted_training_error": error,
                "learner_weight": alpha,
                "max_sample_weight_after_update": float(weights.max()),
                "max_weight_sample_index": max_index,
                "max_weight_true_identity": int(y[max_index]),
                "top_10_sample_weight_sum_after_update": float(np.sort(weights)[-top_n:].sum()),
            })
            if keep_details:
                self.details_.append({
                    "round": t, "initial_weights": before, "predictions": prediction,
                    "is_wrong": wrong, "weighted_error": error, "alpha": alpha,
                    "updated_weights": weights.copy(),
                })
            if perfect:
                # Weights no longer change, so more rounds would repeat this learner.
                self.stop_reason_ = ("round %d: weak learner has zero weighted error; "
                                     "sample weights cannot change any more, boosting stopped" % t)
                break

        if not self.models_:
            raise ValueError("No weak learner beat random guessing: " + self.stop_reason_)
        return self

    def learner_predictions(self, X):
        return np.asarray([m.predict(X) for m in self.models_])

    def decision_scores(self, X):
        P = self.learner_predictions(X)
        scores = np.zeros((len(X), len(self.classes_)))
        for t, alpha in enumerate(self.alphas_):
            scores += alpha * (P[t][:, None] == self.classes_[None, :])
        return scores

    def staged_predictions(self, X):
        """Ensemble prediction after 1, 2, ..., T learners. Shape (T, n)."""
        P = self.learner_predictions(X)
        scores = np.zeros((len(X), len(self.classes_)))
        out = []
        for t, alpha in enumerate(self.alphas_):
            scores += alpha * (P[t][:, None] == self.classes_[None, :])
            out.append(self.classes_[np.argmax(scores, axis=1)])
        return np.asarray(out)

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_scores(X), axis=1)]

    def predict_proba(self, X):
        scores = self.decision_scores(X)
        total = float(np.sum(self.alphas_))
        if total <= 0:
            return np.full_like(scores, 1.0 / len(self.classes_))
        return scores / total


def toy_samme_demo():
    """Q2(a): three-class toy problem (9 samples), one complete SAMME round,
    run through the SAME SimpleSAMME class that is used on the face data."""
    Xtoy = np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8]], dtype=float)
    ytoy = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    model = SimpleSAMME(n_estimators=1, max_depth=1, random_state=SEED).fit(Xtoy, ytoy, keep_details=True)
    d = model.details_[0]
    K = len(np.unique(ytoy))
    save_csv(pd.DataFrame({
        "sample": np.arange(len(ytoy)), "x": Xtoy[:, 0], "true_class": ytoy,
        "initial_weight": d["initial_weights"], "weak_learner_prediction": d["predictions"],
        "is_wrong": d["is_wrong"], "updated_weight": d["updated_weights"],
    }), "q2_toy_example.csv")
    with open(os.path.join(OUT, "q2_toy_summary.txt"), "w") as f:
        f.write("Number of classes K = %d\n" % K)
        f.write("Random-guess error level 1-1/K = %.6f\n" % (1 - 1.0 / K))
        f.write("Weighted error = %.6f\n" % d["weighted_error"])
        f.write("Learner weight alpha = log((1-err)/err) + log(K-1) = %.6f\n" % d["alpha"])
        f.write("Initial weights sum to %.6f\n" % d["initial_weights"].sum())
        f.write("Updated (normalised) weights sum to %.6f\n" % d["updated_weights"].sum())
    print("Toy SAMME round: error=%.4f alpha=%.4f" % (d["weighted_error"], d["alpha"]))


def boosting(X, y, classes, tr, va, te):
    Xtr, ytr = X[tr], y[tr]
    Xva, yva = X[va], y[va]
    Xte, yte = X[te], y[te]
    Xmask = masked(Xte)

    # ---- Q2(b): grid, selection by clean VALIDATION accuracy only --------
    rows, fitted = [], []
    for depth in BOOST_DEPTHS:
        for rounds in BOOST_ROUNDS:
            start = time.time()
            try:
                m = SimpleSAMME(n_estimators=rounds, max_depth=depth, random_state=SEED).fit(Xtr, ytr)
                elapsed = time.time() - start
                row = {
                    "depth": depth, "requested_rounds": rounds,
                    "actual_learners": len(m.models_),
                    "train_accuracy": accuracy(ytr, m.predict(Xtr)),
                    "val_accuracy": accuracy(yva, m.predict(Xva)),
                    "clean_test_accuracy": accuracy(yte, m.predict(Xte)),
                    "training_seconds": elapsed,
                    "stop_reason": m.stop_reason_,
                }
                fitted.append((row, m))
            except ValueError as exc:
                row = {
                    "depth": depth, "requested_rounds": rounds, "actual_learners": 0,
                    "train_accuracy": np.nan, "val_accuracy": np.nan,
                    "clean_test_accuracy": np.nan,
                    "training_seconds": time.time() - start, "stop_reason": str(exc),
                }
            rows.append(row)
            print("  boosting depth=%d rounds=%d done" % (depth, rounds))
    grid = pd.DataFrame(rows)
    if not fitted:
        raise RuntimeError("All SAMME configurations failed; see q2_boosting_grid.csv")
    best_row, model = sorted(
        fitted, key=lambda p: (-p[0]["val_accuracy"], p[0]["requested_rounds"], p[0]["depth"]))[0]
    depth = int(best_row["depth"])
    rounds = int(best_row["requested_rounds"])
    grid["selected"] = (grid["depth"] == depth) & (grid["requested_rounds"] == rounds)
    save_csv(grid, "q2_boosting_grid.csv")
    show("Q2(b) boosting grid", grid.drop(columns=["stop_reason"]))
    print("Selected SAMME: depth=%d, requested rounds=%d, actual learners=%d" % (depth, rounds, len(model.models_)))
    print("Stop reason: " + model.stop_reason_)

    # per-round history of the selected model
    stage_tr = model.staged_predictions(Xtr)
    stage_va = model.staged_predictions(Xva)
    history_df = pd.DataFrame(model.history_)
    history_df["max_learner_weight_so_far"] = np.maximum.accumulate(model.alphas_)
    history_df["total_learner_weight_so_far"] = np.cumsum(model.alphas_)
    history_df["train_accuracy"] = [accuracy(ytr, p) for p in stage_tr]
    history_df["val_accuracy"] = [accuracy(yva, p) for p in stage_va]
    save_csv(history_df, "q2_selected_history.csv")

    rr = history_df["round"]
    plt.figure(figsize=(5.2, 3.6))
    plt.plot(rr, history_df["train_accuracy"], label="Training")
    plt.plot(rr, history_df["val_accuracy"], label="Validation")
    plt.xlabel("Boosting round")
    plt.ylabel("Accuracy")
    plt.grid(alpha=0.3)
    plt.legend()
    save_plot("q2_boosting_history.png")

    plots = [
        ("weighted_training_error", "Weighted training error", "q2_weighted_error.png"),
        ("learner_weight", "Learner weight alpha", "q2_learner_weight.png"),
        ("max_sample_weight_after_update", "Maximum sample weight", "q2_max_sample_weight.png"),
        ("top_10_sample_weight_sum_after_update", "Weight of the 10 heaviest samples", "q2_top10_weight.png"),
    ]
    for column, ylabel, filename in plots:
        plt.figure(figsize=(5.2, 3.6))
        plt.plot(rr, history_df[column], marker=".")
        plt.xlabel("Boosting round")
        plt.ylabel(ylabel)
        plt.grid(alpha=0.3)
        save_plot(filename)
    plt.figure(figsize=(10, 7))
    for k, (column, ylabel, _) in enumerate(plots):
        plt.subplot(2, 2, k + 1)
        plt.plot(rr, history_df[column], marker=".")
        plt.xlabel("Boosting round")
        plt.ylabel(ylabel)
        plt.grid(alpha=0.3)
    save_plot("q2_boosting_diagnostics.png")

    # ---- Q2(c): one training face that receives a very large weight ------
    W = np.asarray(model.weight_history_)          # (rounds, n) weights after each update
    round_no, sample = np.unravel_index(np.argmax(W), W.shape)
    P_tr = model.learner_predictions(Xtr)
    hard_rows = []
    for i in range(len(model.models_)):
        hard_rows.append({
            "round": i + 1,
            "training_row_position": int(sample),
            "original_dataset_index": int(tr[sample]),
            "true_identity": int(ytr[sample]),
            "sample_weight_after_update": float(W[i, sample]),
            "weight_rank_among_training_images": int((W[i] > W[i, sample]).sum() + 1),
            "learner_prediction": int(P_tr[i, sample]),
            "learner_correct": bool(P_tr[i, sample] == ytr[sample]),
            "times_misclassified_so_far": int((P_tr[:i + 1, sample] != ytr[sample]).sum()),
            "ensemble_prediction_so_far": int(stage_tr[i, sample]),
            "ensemble_correct_so_far": bool(stage_tr[i, sample] == ytr[sample]),
        })
    hard_df = pd.DataFrame(hard_rows)
    save_csv(hard_df, "q2_large_weight_example.csv")
    print("Heaviest training face: dataset index %d (identity %d), max weight %.4f at round %d" % (
        tr[sample], ytr[sample], W[round_no, sample], round_no + 1))
    plt.figure(figsize=(5.2, 3.6))
    plt.plot(hard_df["round"], hard_df["sample_weight_after_update"], marker=".")
    plt.xlabel("Boosting round")
    plt.ylabel("Weight of the heaviest training face")
    plt.grid(alpha=0.3)
    save_plot("q2_hard_sample_weight.png")
    face_grid([(Xtr[sample].reshape(64, 64), "Heaviest training face\nidentity %d, max weight %.3f" % (
        ytr[sample], W[round_no, sample]))], "q2_hard_sample_face.png", ncols=1)

    return {"model": model, "depth": depth, "rounds": rounds, "grid": grid,
            "history": history_df, "hard_sample": int(sample)}


def boosting_vs_bagging(grid, bag_df, boost_depth, tree_depth):
    """Q2(c): 25 / 50 / 100 learners, boosting (selected depth) vs bagging (frozen tree)."""
    rows = []
    for n_learners in [25, 50, 100]:
        g = grid[(grid["depth"] == boost_depth) & (grid["requested_rounds"] == n_learners)].iloc[0]
        rows.append({
            "method": "boosting (SAMME, depth %d)" % boost_depth, "learners": n_learners,
            "actual_learners": int(g["actual_learners"]),
            "train_accuracy": g["train_accuracy"], "val_accuracy": g["val_accuracy"],
            "clean_test_accuracy": g["clean_test_accuracy"], "training_seconds": g["training_seconds"],
        })
        b = bag_df[bag_df["B"] == n_learners].iloc[0]
        rows.append({
            "method": "bagging (frozen tree, depth %s)" % tree_depth, "learners": n_learners,
            "actual_learners": n_learners,
            "train_accuracy": b["train_accuracy"], "val_accuracy": b["val_accuracy"],
            "clean_test_accuracy": b["clean_test_accuracy"], "training_seconds": b["training_seconds"],
        })
    df = pd.DataFrame(rows)
    save_csv(df, "q2c_boosting_vs_bagging.csv")
    show("Q2(c) boosting vs bagging", df)


def final_model_comparison(X, y, classes, te, tree, final_bag, boost):
    """Q2(d): frozen single tree, final bagged model, selected boosting model
    on clean and masked test sets, plus a face where bagging and boosting differ."""
    Xte, yte = X[te], y[te]
    Xmask = masked(Xte)
    K = len(classes)

    outs = {}
    rows = []
    for test_name, Xin in [("clean", Xte), ("masked", Xmask)]:
        for name, kind, model in [("single_tree", "tree", tree), ("bagging", "bagging", final_bag),
                                  ("boosting", "boosting", boost)]:
            o = model_outputs(kind, model, Xin, classes)
            outs[(name, test_name)] = o
            ok = o["pred"] == yte
            rows.append({
                "model": name, "test_type": test_name,
                "accuracy": float(ok.mean()),
                "mean_winning_vote_or_probability": float(o["conf"].mean()),
                "mean_confidence_when_correct": float(o["conf"][ok].mean()) if ok.any() else np.nan,
                "mean_confidence_when_wrong": float(o["conf"][~ok].mean()) if (~ok).any() else np.nan,
            })
    cmp_df = pd.DataFrame(rows)
    save_csv(cmp_df, "q2d_all_models_clean_masked.csv")
    show("Q2(d) all three models, clean vs masked", cmp_df)

    # How much do the models rely on the masked patch? (share of splits using patch pixels)
    alpha = np.asarray(boost.alphas_)
    boost_shares = np.asarray([split_patch_share(m) for m in boost.models_])
    usage = pd.DataFrame([
        {"model": "single_tree", "share_of_splits_in_masked_patch": split_patch_share(tree)},
        {"model": "bagging", "share_of_splits_in_masked_patch": float(np.mean([split_patch_share(m) for m in final_bag]))},
        {"model": "boosting", "share_of_splits_in_masked_patch": float(boost_shares.mean()),
         "alpha_weighted_share": float((alpha * boost_shares).sum() / alpha.sum())},
    ])
    usage["patch_pixel_fraction"] = float(PATCH_FLAGS.mean())
    save_csv(usage, "q2d_patch_usage.csv")
    show("Q2(d) share of tree splits that use the masked patch", usage)

    # per-face comparison of bagging and boosting
    bc, bm = outs[("bagging", "clean")], outs[("bagging", "masked")]
    sc, sm = outs[("boosting", "clean")], outs[("boosting", "masked")]
    idx = np.arange(len(yte))
    face = pd.DataFrame({
        "test_position": idx, "original_dataset_index": te, "true_identity": yte,
        "tree_clean_pred": outs[("single_tree", "clean")]["pred"],
        "tree_masked_pred": outs[("single_tree", "masked")]["pred"],
        "bag_clean_pred": bc["pred"], "bag_masked_pred": bm["pred"],
        "bag_clean_true_vote_fraction": bc["prob"][idx, yte], "bag_masked_true_vote_fraction": bm["prob"][idx, yte],
        "bag_clean_winning_fraction": bc["conf"], "bag_masked_winning_fraction": bm["conf"],
        "boost_clean_pred": sc["pred"], "boost_masked_pred": sm["pred"],
        "boost_clean_true_prob": sc["prob"][idx, yte], "boost_masked_true_prob": sm["prob"][idx, yte],
        "boost_clean_winning_prob": sc["conf"], "boost_masked_winning_prob": sm["conf"],
    })
    face["bag_clean_ok"] = face["bag_clean_pred"] == face["true_identity"]
    face["bag_masked_ok"] = face["bag_masked_pred"] == face["true_identity"]
    face["boost_clean_ok"] = face["boost_clean_pred"] == face["true_identity"]
    face["boost_masked_ok"] = face["boost_masked_pred"] == face["true_identity"]
    differs = face["bag_masked_ok"] != face["boost_masked_ok"]
    both_clean_ok = face["bag_clean_ok"] & face["boost_clean_ok"]
    face["priority"] = np.where(both_clean_ok & differs, 0, np.where(differs, 1, 2))
    face["change_gap"] = np.abs(
        (face["bag_masked_true_vote_fraction"] - face["bag_clean_true_vote_fraction"])
        - (face["boost_masked_true_prob"] - face["boost_clean_true_prob"]))
    face = face.sort_values(["priority", "change_gap"], ascending=[True, False]).reset_index(drop=True)
    save_csv(face, "q2d_face_level.csv")
    save_csv(face.head(5), "q2d_candidate_faces.csv")

    pick = face.iloc[0]
    j = int(pick["test_position"])
    cb_clean = vote_counts(np.asarray([m.predict(Xte[j:j + 1]) for m in final_bag]), classes)[0]
    cb_mask = vote_counts(np.asarray([m.predict(Xmask[j:j + 1]) for m in final_bag]), classes)[0]
    T = len(final_bag)
    dis_clean = float(per_example_disagreement(cb_clean[None, :])[0])
    dis_mask = float(per_example_disagreement(cb_mask[None, :])[0])
    P_boost_clean = boost.learner_predictions(Xte[j:j + 1])[:, 0]
    P_boost_mask = boost.learner_predictions(Xmask[j:j + 1])[:, 0]
    true_id = int(yte[j])

    def top3_counts(c):
        order = np.argsort(-c, kind="stable")[:3]
        return ", ".join("id %d: %d votes" % (k, c[k]) for k in order)

    def top3_probs(p):
        order = np.argsort(-p, kind="stable")[:3]
        return ", ".join("id %d: %.3f" % (k, p[k]) for k in order)

    lines = [
        "Selected face (test position %d, dataset index %d, true identity %d)" % (j, te[j], true_id),
        "Selection rule: prefer a face where bagging and boosting differ on the masked image while both were right on the clean image.",
        "",
        "BAGGING (%d trees)" % T,
        "  clean : prediction %d | top votes: %s" % (pick["bag_clean_pred"], top3_counts(cb_clean)),
        "  masked: prediction %d | top votes: %s" % (pick["bag_masked_pred"], top3_counts(cb_mask)),
        "  pairwise tree disagreement on this face: clean %.3f, masked %.3f" % (dis_clean, dis_mask),
        "BOOSTING (%d learners, sum alpha = %.2f)" % (len(boost.models_), float(alpha.sum())),
        "  clean : prediction %d | top normalised weighted votes: %s" % (pick["boost_clean_pred"], top3_probs(sc["prob"][j])),
        "  masked: prediction %d | top normalised weighted votes: %s" % (pick["boost_masked_pred"], top3_probs(sm["prob"][j])),
        "  learners voting for the true identity: clean %d, masked %d (alpha mass: clean %.2f, masked %.2f)" % (
            int((P_boost_clean == true_id).sum()), int((P_boost_mask == true_id).sum()),
            float(alpha[P_boost_clean == true_id].sum()), float(alpha[P_boost_mask == true_id].sum())),
        "  largest single learner weight alpha = %.3f (learner %d)" % (float(alpha.max()), int(np.argmax(alpha)) + 1),
    ]
    with open(os.path.join(OUT, "q2d_selected_face_details.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    face_grid([(Xte[j].reshape(64, 64), "clean | true %d\nbag %d, boost %d" % (true_id, pick["bag_clean_pred"], pick["boost_clean_pred"])),
               (Xmask[j].reshape(64, 64), "masked | true %d\nbag %d, boost %d" % (true_id, pick["bag_masked_pred"], pick["boost_masked_pred"]))],
              "q2d_selected_face.png", ncols=2)
    return cmp_df


# ----------------------------------------------------------------------
# Q3: bias and variance
# ----------------------------------------------------------------------
def make_resamples(tr, y, classes):
    """30 new training sets: within each identity draw 6 faces WITH replacement
    from that identity's original 6 training faces."""
    rng = rng_for(3)
    resamples = []
    for _ in range(N_RESAMPLES):
        ids = []
        for person in classes:
            pool = tr[y[tr] == person]
            ids.extend(rng.choice(pool, size=len(pool), replace=True))
        resamples.append(np.asarray(ids, dtype=int))
    return resamples


def bias_variance_numbers(P, onehot):
    """P: (R, N, K) probability vectors. Bias^2 = mean_i ||pbar_i - e_i||^2,
    Variance = mean_{i,r} ||p_r(x_i) - pbar_i||^2."""
    mean_p = P.mean(axis=0)
    bias2 = float(np.mean(np.sum((mean_p - onehot) ** 2, axis=1)))
    variance = float(np.mean(np.sum((P - mean_p[None, :, :]) ** 2, axis=2)))
    return bias2, variance


def _refit_all(rep, ids, X, y, Xclean, Xmask, depth, leaf, B, boost_depth, boost_rounds, classes):
    Xr, yr = X[ids], y[ids]
    rng = rng_for(4, rep)
    tree = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf, random_state=seed_from(rng))
    tree.fit(Xr, yr)
    bag, _ = fit_bagging(Xr, yr, B, depth, leaf, rng)
    try:
        boost = SimpleSAMME(n_estimators=boost_rounds, max_depth=boost_depth,
                            random_state=seed_from(rng)).fit(Xr, yr)
    except ValueError:
        boost = None
    result = {}
    for name, kind, model in [("single_tree", "tree", tree), ("bagging", "bagging", bag),
                              ("boosting", "boosting", boost)]:
        if model is None:
            result[name] = None
            continue
        result[name] = {tag: model_outputs(kind, model, Xin, classes)["prob"]
                        for tag, Xin in [("clean", Xclean), ("masked", Xmask)]}
    return result


def bias_variance_main(X, y, classes, tr, te, depth, leaf, B, boost_depth, boost_rounds, resamples):
    """Q3(a): selected single tree, final bagged model, selected boosting model."""
    Xte, yte = X[te], y[te]
    Xmask = masked(Xte)
    onehot = np.eye(len(classes))[yte]
    results = run_parallel(_refit_all, [
        (rep, ids, X, y, Xte, Xmask, depth, leaf, B, boost_depth, boost_rounds, classes)
        for rep, ids in enumerate(resamples)])

    rows, acc_rows = [], []
    for name in ["single_tree", "bagging", "boosting"]:
        valid = [r[name] for r in results if r[name] is not None]
        row = {"model": name, "resamples_used": len(valid),
               "bagging_trees": B if name == "bagging" else np.nan}
        for tag in ["clean", "masked"]:
            P = np.asarray([v[tag] for v in valid])
            b2, var = bias_variance_numbers(P, onehot)
            accs = [float(np.mean(np.argmax(p, axis=1) == yte)) for p in P]
            row[tag + "_bias_squared"] = b2
            row[tag + "_variance"] = var
            row[tag + "_mean_test_accuracy"] = float(np.mean(accs))
            row[tag + "_std_test_accuracy"] = float(np.std(accs, ddof=1)) if len(accs) > 1 else np.nan
            for r, a in enumerate(accs):
                acc_rows.append({"model": name, "test_type": tag, "refit": r, "accuracy": a})
        rows.append(row)
    df = pd.DataFrame(rows)
    save_csv(df, "q3_bias_variance.csv")
    save_csv(pd.DataFrame(acc_rows), "q3_refit_accuracies.csv")
    show("Q3(a) bias^2, variance and accuracy over %d refits" % N_RESAMPLES, df)

    plt.figure(figsize=(8, 3.4))
    for k, (col, lab) in enumerate([("clean_bias_squared", "Bias squared"), ("clean_variance", "Variance")]):
        plt.subplot(1, 2, k + 1)
        plt.bar(df["model"], df[col])
        plt.ylabel(lab)
        plt.xticks(rotation=15)
    save_plot("q3_bias_variance_models.png")
    return df


def _depth_worker(depth, rep, ids, X, y, Xte, leaf, classes):
    Xr, yr = X[ids], y[ids]
    rng = rng_for(5, rep, depth_code(depth))
    tree = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf, random_state=seed_from(rng))
    tree.fit(Xr, yr)
    tree_occ = leaf_occupancy(tree, Xr, ids)
    tree_prob = aligned_probabilities(tree, Xte, classes)

    models, boots = fit_bagging(Xr, yr, 100, depth, leaf, rng)
    preds = np.asarray([m.predict(Xte) for m in models])
    counts = vote_counts(preds, classes)
    bag_prob = counts / 100.0
    occs = [leaf_occupancy(m, Xr[b], ids[b]) for m, b in zip(models, boots)]
    bag_occ = {k: float(np.mean([o[k] for o in occs])) for k in occs[0]}
    return {
        "tree_prob": tree_prob, "bag_prob": bag_prob,
        "tree_occ": tree_occ, "bag_occ": bag_occ,
        "tree_depth": int(tree.get_depth()),
        "bag_depth": float(np.mean([m.get_depth() for m in models])),
        "disagreement": float(np.mean(per_example_disagreement(counts))),
    }


def depth_experiment(X, y, classes, tr, te, selected_depth, leaf, resamples):
    """Q3(b),(c): single tree and 100-tree bagging at a shallow, the selected and a much deeper depth."""
    Xte, yte = X[te], y[te]
    onehot = np.eye(len(classes))[yte]
    if selected_depth is None:
        depths = [2, 8, None]
    else:
        shallow = 1 if selected_depth <= 2 else 2
        deeper = None if selected_depth >= 8 else 16
        depths = [shallow, selected_depth, deeper]
    depths = list(dict.fromkeys(depths))
    print("Depth experiment uses depths:", [depth_label(d) for d in depths])

    tasks = [(d, rep, ids, X, y, Xte, leaf, classes) for d in depths for rep, ids in enumerate(resamples)]
    results = run_parallel(_depth_worker, tasks)

    rows = []
    for di, d in enumerate(depths):
        chunk = results[di * len(resamples):(di + 1) * len(resamples)]
        for name, key, occ_key, depth_key in [("single_tree", "tree_prob", "tree_occ", "tree_depth"),
                                              ("bagging_100", "bag_prob", "bag_occ", "bag_depth")]:
            P = np.asarray([c[key] for c in chunk])
            b2, var = bias_variance_numbers(P, onehot)
            accs = [float(np.mean(np.argmax(p, axis=1) == yte)) for p in P]
            rows.append({
                "depth": depth_label(d), "is_selected_depth": d == selected_depth,
                "model": name, "resamples": len(chunk),
                "bagging_trees_per_resample": 100 if name == "bagging_100" else 0,
                "bias_squared": b2, "variance": var,
                "mean_test_accuracy": float(np.mean(accs)),
                "std_test_accuracy": float(np.std(accs, ddof=1)),
                "mean_actual_tree_depth": float(np.mean([c[depth_key] for c in chunk])),
                "mean_leaf_count": float(np.mean([c[occ_key]["number_of_leaves"] for c in chunk])),
                "mean_samples_per_leaf": float(np.mean([c[occ_key]["mean_samples_per_leaf"] for c in chunk])),
                "fraction_leaves_with_one_training_example": float(np.mean(
                    [c[occ_key]["fraction_leaves_with_one_training_example"] for c in chunk])),
                "mean_pairwise_disagreement": 0.0 if name == "single_tree" else float(np.mean([c["disagreement"] for c in chunk])),
            })
    result = pd.DataFrame(rows)
    save_csv(result, "q3_depth_comparison.csv")
    show("Q3(b) depth comparison", result)

    # Q3(b)/(c): what bagging removes at each depth, and where the accuracy gain is largest
    summ = []
    for d in depths:
        s = result[(result["depth"] == depth_label(d)) & (result["model"] == "single_tree")].iloc[0]
        b = result[(result["depth"] == depth_label(d)) & (result["model"] == "bagging_100")].iloc[0]
        summ.append({
            "depth": depth_label(d),
            "single_bias_squared": s["bias_squared"], "bagging_bias_squared": b["bias_squared"],
            "bias_change": b["bias_squared"] - s["bias_squared"],
            "single_variance": s["variance"], "bagging_variance": b["variance"],
            "variance_removed": s["variance"] - b["variance"],
            "variance_removed_percent": 100.0 * (s["variance"] - b["variance"]) / s["variance"],
            "single_accuracy": s["mean_test_accuracy"], "bagging_accuracy": b["mean_test_accuracy"],
            "accuracy_gain": b["mean_test_accuracy"] - s["mean_test_accuracy"],
            "single_mean_leaves": s["mean_leaf_count"],
            "single_fraction_singleton_leaves": s["fraction_leaves_with_one_training_example"],
            "bagging_disagreement": b["mean_pairwise_disagreement"],
        })
    summary = pd.DataFrame(summ)
    save_csv(summary, "q3c_summary.csv")
    show("Q3(b,c) what bagging removes at each depth", summary)
    best = summary.sort_values("accuracy_gain", ascending=False).iloc[0]
    deepest = summary.iloc[-1]["depth"]
    print("Largest bagging accuracy gain is at depth %s (gain %.4f); deepest tested depth is %s -> %s" % (
        best["depth"], best["accuracy_gain"], deepest,
        "the deepest tree DOES give the largest gain" if best["depth"] == deepest
        else "the deepest tree does NOT give the largest gain"))

    labels = [str(depth_label(d)) for d in depths]
    plt.figure(figsize=(8, 3.4))
    for k, (col, lab) in enumerate([("bias_squared", "Bias squared"), ("variance", "Variance")]):
        plt.subplot(1, 2, k + 1)
        for name in ["single_tree", "bagging_100"]:
            part = result[result["model"] == name]
            plt.plot(labels, part[col], marker="o", label=name)
        plt.xlabel("Maximum tree depth")
        plt.ylabel(lab)
        plt.grid(alpha=0.3)
        plt.legend(fontsize=7)
    save_plot("q3_depth_comparison.png")
    return result, summary


def final_table(frozen_cmp, bv_df, run_note):
    """Q3(d): one table with clean accuracy, masked accuracy, bias^2 and variance."""
    rows = []
    for name in ["single_tree", "bagging", "boosting"]:
        c = frozen_cmp[(frozen_cmp["model"] == name) & (frozen_cmp["test_type"] == "clean")].iloc[0]
        m = frozen_cmp[(frozen_cmp["model"] == name) & (frozen_cmp["test_type"] == "masked")].iloc[0]
        b = bv_df[bv_df["model"] == name].iloc[0]
        rows.append({
            "model": name,
            "clean_test_accuracy": c["accuracy"],
            "masked_test_accuracy": m["accuracy"],
            "accuracy_drop_from_mask": c["accuracy"] - m["accuracy"],
            "bias_squared": b["clean_bias_squared"],
            "variance": b["clean_variance"],
            "refit_mean_clean_accuracy": b["clean_mean_test_accuracy"],
            "refit_std_clean_accuracy": b["clean_std_test_accuracy"],
            "masked_bias_squared": b["masked_bias_squared"],
            "masked_variance": b["masked_variance"],
            "refit_mean_masked_accuracy": b["masked_mean_test_accuracy"],
            "refit_std_masked_accuracy": b["masked_std_test_accuracy"],
        })
    table = pd.DataFrame(rows)
    save_csv(table, "q3d_final_table.csv")
    show("Q3(d) FINAL TABLE (accuracy columns: frozen models; bias/variance: 30 refits)", table)
    return table


# ----------------------------------------------------------------------
# Environment record (reproducibility)
# ----------------------------------------------------------------------
def record_environment():
    packages = {}
    for p in ["numpy", "pandas", "scikit-learn", "matplotlib", "joblib", "scipy"]:
        try:
            packages[p] = metadata.version(p)
        except Exception:
            pass
    info = {"python": sys.version, "platform": platform.platform(), "packages": packages}
    with open(os.path.join(OUT, "environment.json"), "w") as f:
        json.dump(info, f, indent=2)
    try:
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True,
                                text=True, timeout=120)
        if freeze.returncode == 0 and freeze.stdout.strip():
            with open(os.path.join(BASE, "requirements.txt"), "w") as f:
                f.write(freeze.stdout)
    except Exception:
        pass


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def run_everything():
    start = time.time()
    record_environment()
    X, y, classes, tr, va, te = load_data()
    print("Loaded %d images. Seed = %d" % (len(X), SEED))
    print("Split sizes: train=%d, validation=%d, test=%d" % (len(tr), len(va), len(te)))
    print("Parallel workers for Q3:", N_JOBS)

    print("\n=== Q2(a): three-class SAMME example ===")
    toy_samme_demo()

    print("\n=== Q1: decision tree and bagging ===")
    q1 = tree_and_bagging(X, y, classes, tr, va, te)

    print("\n=== Q2: SAMME boosting ===")
    q2 = boosting(X, y, classes, tr, va, te)
    boosting_vs_bagging(q2["grid"], q1["bag_df"], q2["depth"], depth_label(q1["depth"]))
    frozen_cmp = final_model_comparison(X, y, classes, te, q1["tree"], q1["final_bag"], q2["model"])

    print("\n=== Q3: bias and variance (30 refits) ===")
    resamples = make_resamples(tr, y, classes)
    bv = bias_variance_main(X, y, classes, tr, te, q1["depth"], q1["leaf"], q1["chosen_B"],
                            q2["depth"], q2["rounds"], resamples)
    depth_experiment(X, y, classes, tr, te, q1["depth"], q1["leaf"], resamples)
    final_table(frozen_cmp, bv, "")

    summary = {
        "seed": SEED, "train_count": int(len(tr)), "validation_count": int(len(va)),
        "test_count": int(len(te)),
        "tree_max_depth": depth_label(q1["depth"]), "tree_min_samples_leaf": int(q1["leaf"]),
        "bagging_B": int(q1["chosen_B"]),
        "boosting_depth": int(q2["depth"]), "boosting_requested_rounds": int(q2["rounds"]),
        "boosting_actual_learners": int(len(q2["model"].models_)),
        "boosting_stop_reason": q2["model"].stop_reason_,
        "n_resamples": N_RESAMPLES, "mask": "rows and columns %d..%d set to 0" % (MASK_LO, MASK_HI - 1),
        "B_values": B_VALUES, "tree_depth_grid": [depth_label(d) for d in TREE_DEPTHS],
        "tree_leaf_grid": TREE_LEAVES, "boost_depth_grid": BOOST_DEPTHS, "boost_rounds_grid": BOOST_ROUNDS,
        "elapsed_seconds": float(time.time() - start),
    }
    with open(os.path.join(OUT, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nAll experiments finished.")
    print(json.dumps(summary, indent=2))
    print("Check the outputs/ and figures/ folders before writing the report.")


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    log = open(os.path.join(OUT, "final_log.txt"), "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log)
    try:
        run_everything()
    except Exception:
        traceback.print_exc(file=log)
        raise
    finally:
        sys.stdout = sys.__stdout__
        log.close()


if __name__ == "__main__":
    main()