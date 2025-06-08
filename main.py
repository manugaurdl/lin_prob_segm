from types import MethodType
from gitignore_parser import parse_gitignore
import logging
import torch
from lightning.pytorch import cli
from lightning.pytorch.callbacks import ModelSummary
from lightning.pytorch.loops.training_epoch_loop import _TrainingEpochLoop
from lightning.pytorch.loops.fetchers import _DataFetcher, _DataLoaderIterDataFetcher

from training.lightning_module import LightningModule
from datasets.lightning_data_module import LightningDataModule

import traceback
def print_stack():
    print("Call stack:")
    for line in traceback.format_stack():
        print(line.strip())

def _should_check_val_fx(self: _TrainingEpochLoop, data_fetcher: _DataFetcher) -> bool:
    if not self._should_check_val_epoch():
        return False

    is_infinite_dataset = self.trainer.val_check_batch == float("inf")
    is_last_batch = self.batch_progress.is_last_batch
    if is_last_batch and (
        is_infinite_dataset or isinstance(data_fetcher, _DataLoaderIterDataFetcher)
    ):
        return True

    if self.trainer.should_stop and self.trainer.fit_loop._can_stop_early:
        return True

    is_val_check_batch = is_last_batch
    if isinstance(self.trainer.limit_train_batches, int) and is_infinite_dataset:
        is_val_check_batch = (
            self.batch_idx + 1
        ) % self.trainer.limit_train_batches == 0
    elif self.trainer.val_check_batch != float("inf"):
        if self.trainer.check_val_every_n_epoch is not None:
            is_val_check_batch = (
                self.batch_idx + 1
            ) % self.trainer.val_check_batch == 0
        else:
            # added below to check val based on global steps instead of batches in case of iteration based val check
            is_val_check_batch = (
                self.global_step
            ) % self.trainer.val_check_batch == 0 and not self._should_accumulate()

    return is_val_check_batch


class LightningCLI(cli.LightningCLI):
    def __init__(self, *args, **kwargs):
        logging.getLogger().setLevel(logging.INFO)
        torch.set_float32_matmul_precision("medium")
        torch._dynamo.config.suppress_errors = True  # type: ignore
        super().__init__(*args, **kwargs)

    def add_arguments_to_parser(self, parser):
        parser.add_argument("--root", type=str)
        parser.link_arguments("root", "data.init_args.root")
        parser.link_arguments("root", "trainer.logger.init_args.save_dir")

        parser.add_argument("--no_compile", action="store_true")

        parser.link_arguments("trainer.devices", "data.init_args.devices")

        parser.link_arguments(
            "data.init_args.num_classes", "model.init_args.num_classes"
        )
        parser.link_arguments(
            "data.init_args.num_classes",
            "model.init_args.network.init_args.num_classes",
        )

        parser.link_arguments(
            "data.init_args.num_metrics", "model.init_args.num_metrics"
        )

        parser.link_arguments("data.init_args.ignore_idx", "model.init_args.ignore_idx")

        parser.link_arguments("data.init_args.img_size", "model.init_args.img_size")
        parser.link_arguments(
            "data.init_args.img_size", "model.init_args.network.init_args.img_size"
        )
        parser.add_argument('--wandb', action='store_true', help='log on wandb')
        parser.add_argument('--run_name', type = str, help='wandb run_name')

    def fit(self, model, **kwargs):
        if self.config.fit.wandb:
            self.trainer.logger._wandb_init['project'] = "lin_probe_ade20k"
            self.trainer.logger._wandb_init['name'] = self.config.fit.run_name
            self.trainer.logger.experiment

            is_gitignored = parse_gitignore(".gitignore")
            include_fn = lambda path: path.endswith(".py") or path.endswith(".yaml")
            self.trainer.logger.experiment.log_code(  # type: ignore
                ".", include_fn=include_fn, exclude_fn=is_gitignored
            )

        self.trainer.fit_loop.epoch_loop._should_check_val_fx = MethodType(
            _should_check_val_fx, self.trainer.fit_loop.epoch_loop
        )
        if not self.config[self.config["subcommand"]]["no_compile"]:  # type: ignore
            model = torch.compile(model)
        self.trainer.fit(model, **kwargs)  # type: ignore

def cli_main():
    LightningCLI(
        LightningModule,
        LightningDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_callback=None,
        seed_everything_default=0,
        trainer_defaults={
            "precision": "16-mixed",
            "log_every_n_steps": 1,
            "enable_model_summary": False,
            "callbacks": [ModelSummary(max_depth=2)],
            "devices": 1,
            "accumulate_grad_batches": 16,
        },
    )


if __name__ == "__main__":
    cli_main()
#python main.py fit -c configs/ade20k_linear_semantic.yaml --model.network.encoder_name vit_base_patch14_dinov2 --model.freeze_encoder True --data.num_workers 12 --root /storage/users/manugaur/benchmark-vfm-ss/