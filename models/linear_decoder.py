from models.encoder import Encoder
import torch.nn as nn
import torch
import os
from torch.nn.utils.rnn import pad_sequence
import traceback
def print_stack():
    print("Call stack:")
    for line in traceback.format_stack():
        print(line.strip())

def pad_text_feats(text_feats):
    padded_text_feats = pad_sequence(text_feats, batch_first = True, padding_value = 0)
    lengths = torch.tensor([feat.size(0) for feat in text_feats])
    max_len = padded_text_feats.size(1)
    padding_mask = torch.arange(max_len).expand(padded_text_feats.size(0), max_len) < lengths.unsqueeze(1)
    return padded_text_feats, padding_mask

class LinearDecoder(Encoder):
    def __init__(
        self,
        encoder_name,
        num_classes,
        text_conditioning,
        data_dir,
        img_size,
        original_res,
        sub_norm=False,
        patch_size=16,
        pretrained=True,
        ckpt_dir="",
        ckpt_name=""
    ):
        super().__init__(
            text_conditioning,
            encoder_name=encoder_name,
            img_size=img_size,
            original_res = original_res,
            sub_norm=sub_norm,
            patch_size=patch_size,
            pretrained=pretrained,
            ckpt_dir=ckpt_dir,
            ckpt_name=ckpt_name,
        )
        self.head = nn.Linear(self.embed_dim, num_classes)
        self.text_conditioning = text_conditioning
        
        if self.text_conditioning:
            #load roberta feats for class labels
            # {'feats': class_text_feats, 'pad_mask': padding_mask}
            if os.path.isfile(os.path.join(data_dir , "ade150class_roberta_feats.pt")):
                self.class_text_feats = torch.load(os.path.join(data_dir , "ade150class_roberta_feats.pt"))
            else:
                #compute roberta feats for class labels
                id2label = {int(_.split("\t")[0]) : _.split("\t")[-1][:-1].strip() for _ in open(os.path.join(data_dir, "objectInfo150.txt"), "r").readlines()[1:]}
                id2label[0] = 'background'
                #load model
                roberta = torch.hub.load('pytorch/fairseq', 'roberta.large').eval()
                roberta.cuda()

                # compute text feats
                with torch.no_grad():
                    class_text_feats, padding_mask = pad_text_feats([roberta.extract_features(roberta.encode(name.lower().strip())).squeeze(0) for name in list(id2label.values())])
                roberta = None ; torch.cuda.empty_cache()
                
                self.class_text_feats = {'feats': class_text_feats, 'pad_mask': padding_mask.to(class_text_feats.device)}
                torch.save(self.class_text_feats, os.path.join(data_dir, "ade150class_roberta_feats.pt"))
        
    def forward(self, x: torch.Tensor, obj_label) -> torch.Tensor:
        
        if self.text_conditioning:
            x = super().forward(x, text_cond=(self.class_text_feats['feats'][obj_label], self.class_text_feats['pad_mask'][obj_label]))
        else:
            x = super().forward(x, obj_label) #obj_label is just a placeholder

        x = self.head(x)
        x = x.transpose(1, 2)

        return x.reshape(x.shape[0], -1, *self.grid_size)