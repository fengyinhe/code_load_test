import time
import math
import logging
import torch
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from Model import CSTE

plt.switch_backend('agg')


# ------------------------- Model Saving -------------------------
def save_checkpoint(model, name="CSTEformer_model_dict_vst.pt"):
    """Save model weights only."""
    torch.save(model.state_dict(), name)


# ------------------------- Main Training Launcher -------------------------
def run_training(train_loader, test_loader, device, args):

    torch.manual_seed(1234)

    # ----- Model configuration -----
    input_dim = 13
    hidden_dim = 6
    lstm_layers = 2
    num_outputs = 1

    # Instantiate model
    model = CSTE.DINO_EXP(input_dim, hidden_dim, lstm_layers, num_outputs).to(device)

    # Optim & Scheduler
    optim = Adam(model.parameters(), lr=args.lr)
    lr_sched = MultiStepLR(optim, milestones=args.milestones, gamma=args.gamma)

    logging.info(f"Optimizer = {optim}")
    logging.info(f"Scheduler = {lr_sched}")

    loss_mse = torch.nn.MSELoss()
    loss_mae = torch.nn.L1Loss()

    # Logs
    best_rmse = float("inf")
    best_train = 0
    best_mae = float("inf")
    best_ccc = 0

    epoch_axis = []
    rmse_records = []
    mae_records = []

    # --------------------- Training Loop ---------------------
    for epoch in range(1, args.Epochs + 1):
        t0 = time.time()

        # --- Train one epoch ---
        epoch_train_loss = train_one_epoch(
            train_loader=train_loader,
            model=model,
            optimizer=optim,
            loss_fn=loss_mse,
            scheduler=lr_sched,
            device=device
        )

        # --- Validation ---
        mse_v, mae_v, ccc_v = evaluate_model(
            data_loader=test_loader,
            model=model,
            mse_loss=loss_mse,
            mae_loss=loss_mae,
            device=device
        )

        rmse_v = math.sqrt(mse_v)
        dt = time.time() - t0

        print(f"Epoch {epoch:3d} | {dt:5.2f}s | Train {epoch_train_loss:5.2f} | "
              f"RMSE {rmse_v:5.2f} | MAE {mae_v:5.2f} | CCC {ccc_v:.4f} | LR {optim.param_groups[0]['lr']:.6f}")

        logging.info(f"Epoch {epoch:3d} | {dt:5.2f}s | Train {epoch_train_loss:5.2f} | "
                     f"RMSE {rmse_v:5.2f} | MAE {mae_v:5.2f} | CCC {ccc_v:.4f} | LR {optim.param_groups[0]['lr']:.6f}")

        # Save best checkpoint
        if rmse_v < best_rmse:
            save_checkpoint(model)
            best_rmse = rmse_v
            best_mae = mae_v
            best_ccc = ccc_v
            best_train = epoch_train_loss

        epoch_axis.append(epoch)
        rmse_records.append(rmse_v)
        mae_records.append(mae_v)

        torch.cuda.empty_cache()

    # ----- Save RMSE & MAE curve -----
    plt.plot(epoch_axis, rmse_records, label='RMSE', linewidth=1)
    plt.plot(epoch_axis, mae_records, label='MAE', linewidth=1)
    plt.legend()
    plt.savefig(args.imgload)

    logging.info("-" * 75)
    logging.info(f"Best → Train {best_train:.4f} | RMSE {best_rmse:.4f} | "
                 f"MAE {best_mae:.4f} | CCC {best_ccc:.4f}")

    print("-" * 75)
    print(f"Best → Train {best_train:.4f} | RMSE {best_rmse:.4f} | "
          f"MAE {best_mae:.4f} | CCC {best_ccc:.4f}")


# ===============================================================
#                  Training & Evaluation Functions
# ===============================================================

def train_one_epoch(train_loader, model, optimizer, loss_fn, scheduler, device):
    model.train()
    total_loss = 0

    for batch_idx, (x, y) in enumerate(train_loader):

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        preds = model(x).squeeze()

        # MSE → RMSE
        loss = torch.sqrt(loss_fn(preds, y.float()))
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(train_loader.dataset)


def evaluate_model(data_loader, model, mse_loss, mae_loss, device):
    model.eval()

    mse_sum = 0
    mae_sum = 0

    preds_list = []
    labels_list = []

    eps = 1e-6

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)

            out = model(x).squeeze()

            preds_list.append(out.cpu())
            labels_list.append(y.cpu())

            mse_sum += mse_loss(out, y).item() * x.size(0)
            mae_sum += mae_loss(out, y).item() * x.size(0)

    mse_avg = mse_sum / len(data_loader.dataset)
    mae_avg = mae_sum / len(data_loader.dataset)

    # ---- CCC Calculation ----
    preds_all = torch.cat(preds_list).float()
    labels_all = torch.cat(labels_list).float()

    mean_t = labels_all.mean()
    mean_p = preds_all.mean()

    var_t = labels_all.var(unbiased=False)
    var_p = preds_all.var(unbiased=False)

    cov = torch.mean((labels_all - mean_t) * (preds_all - mean_p))

    numerator = 2 * cov
    denominator = var_t + var_p + (mean_t - mean_p) ** 2

    ccc = numerator / (denominator + eps)

    return mse_avg, mae_avg, ccc.item()
