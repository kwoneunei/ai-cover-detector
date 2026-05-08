"""
Main script that trains, validates, and evaluates
various models including AASIST.

AASIST
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import argparse
import json
import os
import sys
import warnings
from importlib import import_module
from pathlib import Path
from shutil import copy
from typing import Dict, List, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchcontrib.optim import SWA
from tqdm import tqdm

from data.dataloader import Dataset_SingFake, Dataset_SingFake_mert_w2v
from utils.eval_metrics import compute_eer
from utils.utils import create_optimizer, seed_worker, set_seed
from model.SingGraph import Wav2Vec2Model

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def main(args: argparse.Namespace) -> None:
    with open(args.config, "r") as f_json:
        config = json.loads(f_json.read())

    model_config = config["model_config"]
    optim_config = config["optim_config"]
    optim_config["epochs"] = config["num_epochs"]

    track = config["track"]
    assert track in ["LA", "PA", "DF"], "Invalid track given"

    if "eval_all_best" not in config:
        config["eval_all_best"] = "True"
    if "freq_aug" not in config:
        config["freq_aug"] = "False"

    set_seed(args.seed, config)

    output_dir = Path(args.output_dir)

    model_tag = "{}_{}_ep{}_bs{}".format(
        track,
        os.path.splitext(os.path.basename(args.config))[0],
        config["num_epochs"],
        config["batch_size"]
    )
    if args.comment:
        model_tag = model_tag + "_{}".format(args.comment)

    model_tag = output_dir / model_tag
    model_save_path = model_tag / "weights"
    writer = SummaryWriter(model_tag)

    os.makedirs(model_save_path, exist_ok=True)
    copy(args.config, model_tag / "config.conf")

    gpu_id = args.gpu
    device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device))
    if device == "cpu":
        raise ValueError("GPU not detected!")

    model = get_wav2vec2_model(model_config, device).to(device)

    dataset_loaders = get_singfake_loaders(args.seed, args, config)

    if args.eval:
        weight_path = args.eval_model_weights if args.eval_model_weights else config["model_path"]
        model.load_state_dict(torch.load(weight_path, map_location=device))
        print("Model loaded : {}".format(weight_path))
        print("Evaluating...")

        for data_key in dataset_loaders:
            eer = evaluate(dataset_loaders[data_key], model, device)
            print("{} EER: {:.2f} %".format(data_key, eer * 100))

        sys.exit(0)

    optim_config["steps_per_epoch"] = len(dataset_loaders["train"])
    optimizer, scheduler = create_optimizer(model.parameters(), optim_config)

    best_dev_eer = 1.0
    best_epoch = -1

    f_log = open(model_tag / "metric_log.txt", "a")
    f_log.write("=" * 5 + "\n")

    metric_path = model_tag / "metrics"
    os.makedirs(metric_path, exist_ok=True)

    for epoch in range(config["num_epochs"]):
        print("Start training epoch{:03d}".format(epoch))
        running_loss = train_epoch(
            dataset_loaders["train"],
            model,
            optimizer,
            device,
            scheduler,
            config
        )

        dev_eer = evaluate(dataset_loaders["dev"], model, device)

        print("DONE.\nLoss:{:.5f}, dev_eer: {:.2f} %".format(
            running_loss, dev_eer * 100
        ))
        writer.add_scalar("loss", running_loss, epoch)
        writer.add_scalar("dev_eer", dev_eer, epoch)

        if best_dev_eer >= dev_eer:
            print("best model find at epoch", epoch)
            best_dev_eer = dev_eer
            best_epoch = epoch

            torch.save(
                model.state_dict(),
                model_save_path / "epoch_{}_{:03.3f}.pth".format(epoch, dev_eer)
            )
            torch.save(model.state_dict(), model_save_path / "best.pth")

            log_text = "epoch{:03d}, dev_eer: {:.4f}%".format(epoch, dev_eer * 100)
            print(log_text)
            f_log.write(log_text + "\n")
            f_log.flush()

        writer.add_scalar("best_dev_eer", best_dev_eer, epoch)

    print("Start final evaluation")

    best_weight_path = model_save_path / "best.pth"
    model.load_state_dict(torch.load(best_weight_path, map_location=device))
    print("Loaded best model from:", best_weight_path)
    print("Best epoch: {}, best dev EER: {:.2f} %".format(best_epoch, best_dev_eer * 100))

    test_eer = evaluate(dataset_loaders["test_A"], model, device)

    f_log.write("=" * 5 + "\n")
    f_log.write("Best epoch: {}\n".format(best_epoch))
    f_log.write("Best dev EER: {:.3f} %\n".format(best_dev_eer * 100))
    f_log.write("Final test_A EER (best.pth): {:.3f} %\n".format(test_eer * 100))
    f_log.close()

    print("Exp FIN. test_A EER: {:.3f} %".format(test_eer * 100))


def get_model(model_config: Dict, device: torch.device):
    module = import_module("models.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    model = _model(model_config).to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("no. model params:{}".format(nb_params))
    return model


def get_wav2vec2_model(model_config: Dict, device: torch.device):
    model = Wav2Vec2Model(model_config, device)
    return model


def get_singfake_loaders(seed: int, args: argparse.Namespace, config: dict) -> List[torch.utils.data.DataLoader]:
    base_dir = "./dataset/"
    target_sr = float(config["target_sr"])

    dataset_keys = ["train", "dev", "test_A"]
    datasets = {}

    common_settings = {
        "batch_size": config["batch_size"],
        "num_workers": 4,
        "pin_memory": True
    }

    gen = torch.Generator()
    gen.manual_seed(seed)

    for key in dataset_keys:
        dataset_path = os.path.join(base_dir, key)
        shuffle = True if key == "train" else False
        drop_last = True if key == "train" else False
        worker_init_fn = seed_worker if key == "train" else None
        generator = gen if key == "train" else None

        dataset = Dataset_SingFake_mert_w2v(
            args,
            config,
            base_dir=dataset_path,
            algo=args.algo,
            state="train" if key == "train" else "test",
            target_sr=target_sr
        )

        datasets[key] = DataLoader(
            dataset,
            shuffle=shuffle,
            drop_last=drop_last,
            worker_init_fn=worker_init_fn,
            generator=generator,
            **common_settings
        )

    return datasets


def evaluate(loader, model, device: torch.device):
    model.eval()
    target_scores = []
    nontarget_scores = []
    debug = False
    count = 0

    with torch.no_grad():
        for batch_x, batch_x2, batch_y in tqdm(loader, total=len(loader)):
            batch_x, batch_x2 = batch_x.to(device), batch_x2.to(device)
            batch_out = model(batch_x, batch_x2)
            batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
            batch_y = batch_y.data.cpu().numpy().ravel()

            for i in range(len(batch_y)):
                if batch_y[i] == 1:
                    target_scores.append(batch_score[i])
                else:
                    nontarget_scores.append(batch_score[i])

            count += 1
            if count == 10 and debug:
                break

    eer, _ = compute_eer(target_scores, nontarget_scores)
    return eer


def train_epoch(
    trn_loader: DataLoader,
    model,
    optim: Union[torch.optim.SGD, torch.optim.Adam],
    device: torch.device,
    scheduler: torch.optim.lr_scheduler,
    config: argparse.Namespace
):
    running_loss = 0
    num_total = 0.0
    model.train()

    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    pbar = tqdm(trn_loader, total=len(trn_loader))

    for batch_x, batch_x2, batch_y in pbar:
        batch_size = batch_x.size(0)
        num_total += batch_size

        batch_x, batch_x2 = batch_x.to(device), batch_x2.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)

        batch_out = model(batch_x, batch_x2)
        batch_loss = criterion(batch_out, batch_y)
        running_loss += batch_loss.item() * batch_size

        pbar.set_description("loss: {:.5f}, running loss: {:.5f}".format(
            batch_loss.item(), running_loss / num_total
        ))

        optim.zero_grad()
        batch_loss.backward()
        optim.step()

        if config["optim_config"]["scheduler"] in ["cosine", "keras_decay"]:
            scheduler.step()
        elif scheduler is None:
            pass
        else:
            raise ValueError("scheduler error, got:{}".format(scheduler))

    running_loss /= num_total
    return running_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASVspoof detection system")
    parser.add_argument("--config", dest="config", type=str, help="configuration file", required=True)
    parser.add_argument("--output_dir", dest="output_dir", type=str, help="output directory for results", default="./exp_result")
    parser.add_argument("--seed", type=int, default=1234, help="random seed (default: 1234)")
    parser.add_argument("--eval", action="store_true", help="when this flag is given, evaluates given model and exit")
    parser.add_argument("--comment", type=str, default=None, help="comment to describe the saved model")
    parser.add_argument("--eval_model_weights", type=str, default=None, help="path to the model weight file")
    parser.add_argument("--gpu", type=int, default=0, help="gpu id to use (default: 0)")

    parser.add_argument(
        '--algo',
        type=int,
        default=3,
        help='Rawboost algos. 0: No augmentation 1: LnL_convolutive_noise, 2: ISD_additive_noise, 3: SSI_additive_noise, 4: series algo (1+2+3), 5: series algo (1+2), 6: series algo (1+3), 7: series algo(2+3), 8: parallel algo(1,2)'
    )

    parser.add_argument('--nBands', type=int, default=5)
    parser.add_argument('--minF', type=int, default=20)
    parser.add_argument('--maxF', type=int, default=8000)
    parser.add_argument('--minBW', type=int, default=100)
    parser.add_argument('--maxBW', type=int, default=1000)
    parser.add_argument('--minCoeff', type=int, default=10)
    parser.add_argument('--maxCoeff', type=int, default=100)
    parser.add_argument('--minG', type=int, default=0)
    parser.add_argument('--maxG', type=int, default=0)
    parser.add_argument('--minBiasLinNonLin', type=int, default=5)
    parser.add_argument('--maxBiasLinNonLin', type=int, default=20)
    parser.add_argument('--N_f', type=int, default=5)
    parser.add_argument('--P', type=int, default=10)
    parser.add_argument('--g_sd', type=int, default=2)
    parser.add_argument('--SNRmin', type=int, default=10)
    parser.add_argument('--SNRmax', type=int, default=40)

    main(parser.parse_args())