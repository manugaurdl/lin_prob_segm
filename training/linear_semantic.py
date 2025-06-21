import torch
import torch.nn as nn
import wandb
import torch.nn.functional as F
from torch.optim.lr_scheduler import PolynomialLR

from training.lightning_module import LightningModule

import traceback
from torchvision.transforms.v2.functional import resize


def print_stack():
    print("Call stack:")
    for line in traceback.format_stack():
        print(line.strip())

class LinearSemantic(LightningModule):
    def __init__(
        self,
        network: nn.Module,
        num_metrics: int,
        num_classes: int,
        text_conditioning: bool,
        ignore_idx: int,
        img_size: tuple[int, int],
        lr: float = 1e-4,
        weight_decay: float = 0.05,
        poly_lr_decay_power: float = 0.9,
        lr_multiplier_encoder: float = 0.1,
        freeze_encoder: bool = False,
    ):
        super().__init__(
            img_size=img_size,
            freeze_encoder=freeze_encoder,
            network=network,
            weight_decay=weight_decay,
            lr=lr,
            lr_multiplier_encoder=lr_multiplier_encoder,
        )
        self.save_hyperparameters()

        self.ignore_idx = ignore_idx
        self.poly_lr_decay_power = poly_lr_decay_power
        self.text_conditioning = text_conditioning
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_idx)
        self.init_metrics_semantic(num_classes, ignore_idx, num_metrics)
        self.obj_id = torch.tensor([-1]) #its overriden if text_conditioning
    
    def training_step(self, batch, batch_idx):

        imgs, targets = batch
        if self.text_conditioning:
            targets, self.obj_id = self.sampled_obj_to_per_pixel_targets_semantic(targets[0], self.ignore_idx)
        else:
            targets = self.to_per_pixel_targets_semantic(targets, self.ignore_idx) 

        logits = self(imgs, obj_label=self.obj_id)
        logits = F.interpolate(logits, self.img_size, mode="bilinear")
        loss_total = self.criterion(logits, targets)
        self.log("trainer/loss", loss_total, sync_dist=True, prog_bar=True)

        return loss_total
    
    def eval_step(
        self,
        batch,
        batch_idx=None,
        dataloader_idx=None,
        log_prefix=None,
        is_notebook=False,
    ):
        imgs, targets = batch
        img = resize(imgs[0], (512,512))

        if self.text_conditioning:       
            targets, self.obj_id = self.individual_obj_to_per_pixel_targets_semantic(targets[0], self.ignore_idx) #maybe preprocess this; skip the for loop
            img = img.unsqueeze(0).expand(len(self.obj_id),-1,-1,-1)
        else:
            targets = self.to_per_pixel_targets_semantic(targets, self.ignore_idx)        
        
        logits = self(img, obj_label=self.obj_id)
        logits = F.interpolate(logits, imgs[0].size()[-2:], mode="bilinear")

        # logits = self.revert_window_logits_semantic(crop_logits, origins, img_sizes)
        ##### sliding window eval
        # if self.text_conditioning:
        #     targets, self.obj_id = self.sampled_obj_to_per_pixel_targets_semantic(targets[0], self.ignore_idx)
        # else:
        #     targets = self.to_per_pixel_targets_semantic(targets, self.ignore_idx)
        
        # crops, origins, img_sizes = self.window_imgs_semantic(imgs)
        # crop_logits = self(crops, obj_label=self.obj_id.expand(crops.size(0)))
        # crop_logits = F.interpolate(crop_logits, self.img_size, mode="bilinear")
        # logits = self.revert_window_logits_semantic(crop_logits, origins, img_sizes)
        
        if is_notebook:
            return logits

        self.update_metrics(logits, targets, dataloader_idx)

        if batch_idx == 0:
            name = f"{log_prefix}_{dataloader_idx}_pred_{batch_idx}"
            plot = self.plot_semantic(
                imgs[0],
                targets[0],
                logits=logits[0],
            )
            self.trainer.logger.experiment.log({name: [wandb.Image(plot)]})  # type: ignore

    def on_validation_epoch_end(self):
        self._on_eval_epoch_end_semantic("val")

    def configure_optimizers(self):
        optimizer = super().configure_optimizers()

        lr_scheduler = {
            "scheduler": PolynomialLR(
                optimizer,
                int(self.trainer.estimated_stepping_batches),
                self.poly_lr_decay_power,
            ),
            "interval": "step",
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
