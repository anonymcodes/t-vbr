import sys

sys.path.append("../")

import argparse
import os
import math
from ray import tune
from beta_rec.train_engine import TrainEngine, print_dict_as_table
from beta_rec.models.tvbr import TVBREngine
from beta_rec.utils.monitor import Monitor
from beta_rec.utils.common_util import update_args
from beta_rec.utils.constants import MAX_N_UPDATE
from tqdm import tqdm
from beta_rec.train_engine import get_device


def parse_args():
    """ Parse args from command line

        Returns:
            args object.
    """
    parser = argparse.ArgumentParser(description="Run VBCAR..")
    parser.add_argument(
        "--config_file",
        nargs="?",
        type=str,
        default="../configs/tvbr_default.json",
        help="Specify the config file name. Only accept a file from ../configs/",
    )
    # If the following settings are specified with command line,
    # These settings will used to update the parameters received from the config file.
    parser.add_argument(
        "--dataset",
        nargs="?",
        type=str,
        help="Options are: tafeng, dunnhunmby and instacart",
    )
    parser.add_argument(
        "--data_split",
        nargs="?",
        type=str,
        help="Options are: leave_one_out and temporal",
    )
    parser.add_argument("--root_dir", nargs="?", type=str, help="working directory")
    parser.add_argument(
        "--percent",
        nargs="?",
        type=float,
        help="The percentage of the subset of the dataset, only availbe on instacart dataset.",
    )
    parser.add_argument(
        "--n_sample", nargs="?", type=int, help="Number of sampled triples."
    )
    parser.add_argument(
        "--time_step", nargs="?", type=int, help="Number of time steps."
    )
    parser.add_argument("--device", nargs="?", type=str, help="device.")
    parser.add_argument("--item_fea_type", nargs="?", type=str, help="device.")
    parser.add_argument(
        "--emb_dim", nargs="?", type=int, help="Dimension of the embedding."
    )
    parser.add_argument(
        "--late_dim", nargs="?", type=int, help="Dimension of the latent layers.",
    )
    parser.add_argument("--lr", nargs="?", type=float, help="Intial learning rate.")
    parser.add_argument("--max_epoch", nargs="?", type=int, help="Number of max epoch.")
    parser.add_argument(
        "--batch_size", nargs="?", type=int, help="Batch size for training."
    )
    parser.add_argument("--optimizer", nargs="?", type=str, help="OPTI")
    parser.add_argument("--activator", nargs="?", type=str, help="activator")
    parser.add_argument("--alpha", nargs="?", type=float, help="ALPHA")
    return parser.parse_args()


class TVBR_train(TrainEngine):
    """ An instance class from the TrainEngine base class

    """

    def __init__(self, config):
        """Constructor

                Args:
                    config (dict): All the parameters for the model
        """
        self.config = config
        super(TVBR_train, self).__init__(self.config)
        self.load_dataset()
        self.train_data = self.dataset.sample_triple_time()
        self.config["alpha_step"] = (1 - self.config["alpha"]) / (
            self.config["max_epoch"]
        )
        self.gpu_id, self.config["device_str"] = get_device() if self.config["device"] == "gpu" else (None, "cpu")
        self.engine = TVBREngine(self.config)

    def train(self):
        """Default train implementation

        """
        assert hasattr(self, "engine"), "Please specify the exact model engine !"
        self.monitor = Monitor(
            log_dir=self.config["run_dir"], delay=1, gpu_id=self.gpu_id
        )
        self.engine.data = self.dataset
        print("Start training... ")
        epoch_bar = tqdm(range(self.config["max_epoch"]), file=sys.stdout)
        for epoch in epoch_bar:
            print(f"Epoch {epoch} starts !")
            print("-" * 80)
            if epoch > 0 and self.eval_engine.n_no_update == 0:
                # previous epoch have already obtained better result
                self.engine.save_checkpoint(
                    model_dir=os.path.join(self.config["model_save_dir"], "model.cpk")
                )

            if self.eval_engine.n_no_update >= MAX_N_UPDATE:
                print(
                    "Early stop criterion triggered, no performance update for {:} times".format(
                        MAX_N_UPDATE
                    )
                )
                break
            self.engine.train_an_epoch(self.train_data, epoch_id=epoch)
            self.eval_engine.train_eval(
                self.dataset.valid[0], self.dataset.test[0], self.engine.model, epoch
            )
            # anneal alpha
            self.engine.model.alpha = min(
                self.config["alpha"] + math.exp(epoch - self.config["max_epoch"] + 20),
                1,
            )
            """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
            lr = self.config["lr"] * (0.5 ** (epoch // 10))
            for param_group in self.engine.optimizer.param_groups:
                param_group["lr"] = lr
        self.config["run_time"] = self.monitor.stop()
        return self.eval_engine.best_valid_performance


def tune_train(config):
    TVBR = TVBR_train(config)
    best_performance = TVBR.train()
    tune.track.log(best_ndcg=best_performance)
    TVBR.test()


if __name__ == "__main__":
    args = parse_args()
    config = {}
    update_args(config, args)
    tvbr = TVBR_train(config)
    tvbr.train()
    tvbr.test()
