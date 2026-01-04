import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    recall_score,
    precision_score,
    f1_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix
)


def metrics_comparison(
    y_true,
    y_pred,
    y_proba,
    y_pred_bench,
    y_proba_bench,
    model_name="Model",
    benchmark_name="Benchmark"
):
    metrics_model = {
        "ROC_AUC": roc_auc_score(y_true, y_proba),
        "PR_AUC": average_precision_score(y_true, y_proba),
        "Recall_1": recall_score(y_true, y_pred),
        "Precision_1": precision_score(y_true, y_pred),
        "F1_1": f1_score(y_true, y_pred),
    }

    metrics_bench = {
        "ROC_AUC": roc_auc_score(y_true, y_proba_bench),
        "PR_AUC": average_precision_score(y_true, y_proba_bench),
        "Recall_1": recall_score(y_true, y_pred_bench),
        "Precision_1": precision_score(y_true, y_pred_bench),
        "F1_1": f1_score(y_true, y_pred_bench),
    }

    df = pd.DataFrame({
        model_name: metrics_model,
        benchmark_name: metrics_bench
    })

    df["Δ"] = df[model_name] - df[benchmark_name]

    print(f"\n===== {model_name} vs {benchmark_name} — TEST =====\n")
    print(df.round(4).to_string())

    return df


def plot_roc_comparison(
    y_true,
    y_proba_dict,
    title="ROC Curve Comparison"
):
    plt.figure(figsize=(7, 6))

    for model_name, y_proba in y_proba_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = roc_auc_score(y_true, y_proba)

        plt.plot(
            fpr,
            tpr,
            label=f"{model_name} (AUC = {roc_auc:.3f})"
        )

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_pr_comparison(
    y_true,
    y_proba_dict,
    title="Precision–Recall Curve Comparison",
    show_baseline=True
):
    plt.figure(figsize=(7, 6))

    for model_name, y_proba in y_proba_dict.items():
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = average_precision_score(y_true, y_proba)

        plt.plot(
            recall,
            precision,
            label=f"{model_name} (PR-AUC = {pr_auc:.3f})"
        )

    if show_baseline:
        baseline = y_true.mean()
        plt.hlines(
            baseline,
            xmin=0,
            xmax=1,
            linestyles="dashed",
            colors="gray",
            label="Baseline (prevalence)"
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()




def plot_confusion_matrices(
    y_true,
    y_pred_1,
    y_pred_2,
    model_names=("Model 1", "Model 2"),
    normalize=False,
    cmap="Blues"
):
   

    if normalize:
        cm1 = confusion_matrix(y_true, y_pred_1, normalize="true")
        cm2 = confusion_matrix(y_true, y_pred_2, normalize="true")
        fmt = ".2f"
    else:
        cm1 = confusion_matrix(y_true, y_pred_1)
        cm2 = confusion_matrix(y_true, y_pred_2)
        fmt = "d"

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, cm, name in zip(axes, [cm1, cm2], model_names):
        cm_df = pd.DataFrame(
            cm,
            index=["Actual_0", "Actual_1"],
            columns=["Pred_0", "Pred_1"]
        )

        sns.heatmap(
            cm_df,
            annot=True,
            fmt=fmt,
            cmap=cmap,
            ax=ax
        )
        ax.set_title(f"{name}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    plt.show()