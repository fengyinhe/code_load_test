import argparse
import time
import logging
import torch
from torch.utils.data import DataLoader

from dataloader.train_dataload import load_train
from dataloader.dataload import load_data
from train import run_training


def parse_arguments():
    parser = argparse.ArgumentParser(description="Training Script for CSTEformer (Modified Version)")

    # ------- Training settings -------
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--lr_milestones', type=int, nargs='+', default=[100, 200, 300, 400])
    parser.add_argument('--lr_gamma', type=float, default=0.9)
    parser.add_argument('--image_name', type=str, default='test_CSTEformer.jpg')

    # ------- Dataset settings -------
    parser.add_argument('--frame_size', type=int, default=126)
    parser.add_argument('--clip_length', type=int, default=24)
    parser.add_argument('--csv_marks', type=str, nargs='+', default=[100, 200, 300, 400])

    return parser.parse_args()


def setup_logging():
    timestamp = time.strftime('%m-%d_%H-%M')
    logfile = f'./log/{timestamp}.log'
    logging.basicConfig(
        filename=logfile,
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.info("==== Modified CSTEformer Training Session ====")


def prepare_dataloaders(data_root, csv_source, args):
    train_set = load_data(root=data_root, csv_load=csv_source, mode='train', args=args)
    test_set = load_data(root=data_root, csv_load=csv_source, mode='test', args=args)

    return {
        "train": DataLoader(train_set, batch_size=args.batch, shuffle=True,
                            num_workers=8, drop_last=True),
        "test": DataLoader(test_set, batch_size=args.batch, shuffle=True,
                           num_workers=8, drop_last=True)
    }


def main():
    args = parse_arguments()
    setup_logging()

    # Display configuration
    logging.info(f"Batch Size = {args.batch}")
    logging.info(f"Total Epochs = {args.epochs}")
    logging.info(f"Learning Rate = {args.learning_rate}")

    # ------- Device selection -------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------- Data loading -------
    csv_file = ""           # CSV path kept consistent with original script
    dataset_dir = "../datasets/select_clips/"

    loader_dict = prepare_dataloaders(dataset_dir, csv_file, args)

    # ------- Start training process -------
    run_training(loader_dict['train'], loader_dict['test'], device, args)


if __name__ == "__main__":
    main()