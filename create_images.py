from functions import save_figure

# dfs = [
#     "train_baseline_InceptionV3_added_metrics",
#     "train_baseline_ResNet50V2_added_metrics",
#     "supcon_encoder_InceptionV3_added_metrics",
#     "supcon_encoder_ResNet50V2_added_metrics",
# ]

dfs = [
    "train_baseline_InceptionV3_added_metrics_incl_tabular",
    "train_baseline_ResNet50V2_added_metrics_incl_tabular",
    "supcon_encoder_InceptionV3_added_metrics_incl_tabular",
    "supcon_encoder_ResNet50V2_added_metrics_incl_tabular",
]

for df in dfs:
    # Loss and accuracy
    save_figure(
        hist_filelocation=f"CSVLogger/{df}.csv",
        column1="loss",
        column1_name="Loss",
        column2=None,
        column2_name="",
        img_width=1000,
        img_height=600,
        img_name=f"{df}_loss",
    )
    save_figure(
        hist_filelocation=f"CSVLogger/{df}.csv",
        column1=None,
        column1_name="",
        column2="accuracy",
        column2_name="Accuracy",
        img_width=1000,
        img_height=600,
        img_name=f"{df}_accuracy",
    )
    # F1 and AUC
    # Loss and accuracy
    save_figure(
        hist_filelocation=f"CSVLogger/{df}.csv",
        column1=None,
        column1_name="",
        column2="auc",
        column2_name="AUC",
        img_width=1000,
        img_height=600,
        img_name=f"{df}_auc",
    )
    save_figure(
        hist_filelocation=f"CSVLogger/{df}.csv",
        column1="f1",
        column1_name="F1",
        column2=None,
        column2_name="",
        img_width=1000,
        img_height=600,
        img_name=f"{df}_f1",
    )
    # Sensitivity and Specifity
    save_figure(
        hist_filelocation=f"CSVLogger/{df}.csv",
        column1="recall",
        column1_name="Sensitivity",
        column2="specificity",
        column2_name="Specificity",
        img_width=1000,
        img_height=600,
        img_name=f"{df}_SS",
    )
    # Recall and Precision
    save_figure(
        hist_filelocation=f"CSVLogger/{df}.csv",
        column1="recall",
        column1_name="Recall",
        column2="precision",
        column2_name="Precision",
        img_width=1000,
        img_height=600,
        img_name=f"{df}_PR",
    )
