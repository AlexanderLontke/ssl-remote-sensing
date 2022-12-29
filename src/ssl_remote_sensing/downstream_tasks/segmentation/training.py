from tqdm import tqdm
import torch
import wandb
from torchmetrics import Accuracy
from torchmetrics.functional.classification import multiclass_jaccard_index
import pandas as pd
import numpy as np
from ssl_remote_sensing.downstream_tasks.segmentation.constants import DFC2020_LABELS


def train(
    model,
    train_config,
    train_loader,
    val_loader,
    loss_fn,
    device,
    model_path,
    wandb=wandb,
    n_classes = 8
):
    # Initialise the optimizer
    if train_config.optim == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=train_config.lr)
    elif train_config.optim == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=train_config.lr)

    model = model.to(device)

    # Create lists for logging losses and evalualtion metrics:
    train_losses = []
    train_accs = []
    train_accs_perclass = []
    train_ious = []

    val_losses = []
    val_accs = []
    val_accs_perclass = []
    val_ious = []


    # IoU
    # jaccard = JaccardIndex(task="multiclass", num_classes=9).to(device)

    # accuracy
    accuracy = Accuracy(task="multiclass", num_classes=n_classes).to(device)
    accuracy_perclass = Accuracy(task="multiclass", num_classes=n_classes,average = None).to(device)

    # For every epoch
    for epoch in range(train_config.epochs):
        epoch_loss = 0
        progress = tqdm(
            enumerate(train_loader), desc="Train Loss: ", total=len(train_loader), position=0,
            leave=True,
        )

        # Specify you are in training mode
        model.train()

        epoch_train_loss = 0
        epoch_val_loss = 0

        epoch_train_ious = 0
        epoch_val_ious = 0

        epoch_train_accs = 0
        epoch_val_accs = 0

        epoch_train_accs_class = [0] * n_classes
        epoch_val_accs_class = [0] * n_classes

        for i, batch in progress:
            # Transfer data to GPU if available
            data = batch["image"].float().to(device)
            label = batch["label"].long().to(device)

            # Make a forward pass
            output = model(data)
            output_multi = torch.nn.functional.softmax(output, dim=1)
            output_multi = torch.argmax(output_multi, dim=1)

            # Compute IoU
            epoch_train_ious += multiclass_jaccard_index(
                output_multi.to(device), label, num_classes=n_classes
            ) / len(train_loader)

            # Compute pixel accuracies
            epoch_train_accs += accuracy(output_multi.to(device), label.int()) / len(
                train_loader
            )

            # Compute class-wise pixel accuracies
            train_acc_class = accuracy_perclass(output_multi.to(device), label.int()) / len(
                train_loader
            )
            # print("Debug: ", train_acc_class)
            epoch_train_accs_class = [sum(x) for x in zip(train_acc_class,epoch_train_accs_class)]
            # print("Debug: ", epoch_train_accs_class)

            # Compute the loss
            loss = loss_fn(output, label)

            # Clear the gradients
            optimizer.zero_grad()

            # Calculate gradients
            loss.backward()

            # Update Weights
            optimizer.step()

            # Accumulate the loss over the eopch
            epoch_train_loss += loss / len(train_loader)

            progress.set_description(
                "Epoch = {}, Train Loss: {:.4f}".format(epoch + 1, epoch_train_loss)
            )

        progress = tqdm(
            enumerate(val_loader),
            desc="val Loss: ",
            total=len(val_loader),
            position=0,
            leave=True,
        )

        # Specify you are in evaluation mode
        model.eval()

        # Deactivate autograd engine (no backpropagation allowed)
        with torch.no_grad():
            epoch_val_loss = 0
            for j, batch in progress:
                # Transfer Data to GPU if available
                data = batch["image"].float().to(device)
                label = batch["label"].long().to(device)

                # Make a forward pass
                output = model(data)
                output_multi = torch.nn.functional.softmax(output, dim=1)
                output_multi = torch.argmax(output_multi, dim=1)
                

                # Compute IoU
                epoch_val_ious += multiclass_jaccard_index(
                    output_multi.to(device), label, num_classes=n_classes
                ) / len(val_loader)

                # Compute pixel accuracies
                epoch_val_accs += accuracy(output_multi.to(device), label.int()) / len(
                    val_loader
                )

                # Compute class-wise pixel accuracies
                val_acc_class = accuracy_perclass(output_multi.to(device), label.int()) / len(
                    val_loader
                )
                epoch_val_accs_class = [sum(x) for x in zip(val_acc_class,epoch_val_accs_class)]

                # Compute the loss
                val_loss = loss_fn(output, label)

                # Accumulate the loss over the epoch
                epoch_val_loss += val_loss / len(val_loader)

                progress.set_description(
                    "Validation Loss: {:.4f}".format(epoch_val_loss)
                )

        if epoch == 0:
            best_val_loss = epoch_val_loss
        else:
            if epoch_val_loss <= best_val_loss:
                best_val_loss = epoch_val_loss
                # Save only the best model
                torch.save(model.state_dict(), model_path)
                print("Saving Model...")

        # save result to wandb
        wandb.log(
            {
                "train_loss_segmentation": epoch_train_loss,
                "val_loss_segmentation": epoch_val_loss,
                "train_iou_segmentation": epoch_train_ious,
                "val_iou_segmentation": epoch_val_ious,
                "train_acc_segmentation": epoch_train_accs,
                "val_acc_segmentation": epoch_val_accs,
            }
        )

        # Save losses in list, so that we can visualise them later.
        train_losses.append(epoch_train_loss.cpu().detach().numpy())
        val_losses.append(epoch_val_loss.cpu().detach().numpy())

        # Save IoUs in list, so that we can visualise them later.
        train_ious.append(epoch_train_ious.cpu().detach().numpy())
        val_ious.append(epoch_val_ious.cpu().detach().numpy())
        print(f"train_iou is {epoch_train_ious:.4f}, val_iou is {epoch_val_ious:.4f}")

        # Save accuracies in list, so that we can visualise them later.
        train_accs.append(epoch_train_accs.cpu().detach().numpy())
        val_accs.append(epoch_val_accs.cpu().detach().numpy())
        print(f"train_acc is {epoch_train_accs:.4f}, val_acc is {epoch_val_accs:.4f}")

        # Test print
        epoch_train_accs_class = [x.cpu().detach() for x in epoch_train_accs_class]
        epoch_val_accs_class = [x.cpu().detach() for x in epoch_val_accs_class]
        # print(f"\ntrain acc per class is {epoch_train_accs_class}, \nval acc per class is {epoch_val_accs_class}")

        # Save accuracies per class in list, so that we can visualise them later.
        train_accs_perclass.append(epoch_train_accs_class)
        val_accs_perclass.append(epoch_val_accs_class)


    # Create table for accuracies per class
    train_table_accs_class = wandb.Table(columns = DFC2020_LABELS, data = np.array(train_accs_perclass))
    val_table_accs_class = wandb.Table(columns = DFC2020_LABELS, data = np.array(val_accs_perclass))
    # table_accs_class.index = [print(f"Epoch {x}") for x in range(train_config.epochs+1)]
    wandb.log({"Train accuracy per class": train_table_accs_class, "Val accuracy per class": val_table_accs_class,})

    print("Finished Training")
