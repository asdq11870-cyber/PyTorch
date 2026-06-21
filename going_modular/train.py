import torch
import os
import yaml
from going_modular import utils, transforms, engine, data_setup, model_builder
device = "cuda" if torch.cuda.is_available() else "cpu"

#Setup Hyperparameters
with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

data_transform = transforms.create_data_transforms(
    height=CONFIG["IMG_HEIGHT"],
    width=CONFIG["IMG_WIDTH"],
    augmentation=CONFIG["AUGMENTATION"],
    aug_scale=CONFIG["AUG_SCALE"],
    normalise=CONFIG["NORMALISE"],
    mean=CONFIG["MEAN"],
    std=CONFIG["STD"]
)

train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=CONFIG["TRAIN_DIR"],
    test_dir=CONFIG["TEST_DIR"],
    transform=data_transform,
    batch_size=CONFIG["BATCH_SIZE"]
)

# Create model with help from model_builder.py
model = model_builder.TinyVGG(
    input_shape=3,
    hidden_units=CONFIG["HIDDEN_UNITS"],
    output_shape=len(class_names)
).to(device)

# Set loss and optimizer

optimizer_map = {
    "adadelta": torch.optim.Adadelta,
    "adagrad": torch.optim.Adagrad,
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "adamax": torch.optim.Adamax,
    "asgd": torch.optim.ASGD,
    "lbfgs": torch.optim.LBFGS,
    "nadam": torch.optim.NAdam,
    "radam": torch.optim.RAdam,
    "rmsprop": torch.optim.RMSprop,
    "rprop": torch.optim.Rprop,
    "sgd": torch.optim.SGD,
    "sparseadam": torch.optim.SparseAdam,
}

loss_fn_map = {
    "l1": torch.nn.L1Loss,
    "mse": torch.nn.MSELoss,
    "cross_entropy": torch.nn.CrossEntropyLoss,
    "nll": torch.nn.NLLLoss,
    "poisson_nll": torch.nn.PoissonNLLLoss,
    "gaussian_nll": torch.nn.GaussianNLLLoss,
    "kl_div": torch.nn.KLDivLoss,
    "bce": torch.nn.BCELoss,
    "bce_logits": torch.nn.BCEWithLogitsLoss,
    "hinge_embedding": torch.nn.HingeEmbeddingLoss,
    "multilabel_margin": torch.nn.MultiLabelMarginLoss,
    "smooth_l1": torch.nn.SmoothL1Loss,
    "huber": torch.nn.HuberLoss,
    "soft_margin": torch.nn.SoftMarginLoss,
    "multilabel_soft_margin": torch.nn.MultiLabelSoftMarginLoss,
    "cosine_embedding": torch.nn.CosineEmbeddingLoss,
    "margin_ranking": torch.nn.MarginRankingLoss,
    "multimargin": torch.nn.MultiMarginLoss,
    "triplet_margin": torch.nn.TripletMarginLoss,
    "triplet_margin_distance": torch.nn.TripletMarginWithDistanceLoss,
    "ctc": torch.nn.CTCLoss,
}

loss_fn = loss_fn_map[CONFIG["LOSS_FN"]]()

optimiser = optimizer_map[CONFIG["OPTIMISER"]](
    model.parameters(),
    lr=CONFIG["LEARNING_RATE"]
)


# Start training with help from engine.py
engine.batch_train(model=model,
             train_data_loader=train_dataloader,
             test_data_loader=test_dataloader,
             loss_fn=loss_fn,
             optimiser=optimiser,
             epochs=CONFIG["NUM_EPOCHS"],
             device=device,
             loss_curves=CONFIG["LOSS_CURVES"],
             divisor=CONFIG["DIVISOR"]
             )

# Save the model with help from utils.py
utils.save_model(model=model,
                 target_dir=CONFIG["TARGET_DIR"],
                 model_name=CONFIG["MODEL_NAME"])
