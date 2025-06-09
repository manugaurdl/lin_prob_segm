import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize


def int2mil(number):
    if abs(number) >= 100_000:
        formatted_number = "{:.1f}M".format(number / 1_000_000)
    else:
        formatted_number = str(number)
    return formatted_number

def trainable_params(model):
    return int2mil(sum(p.numel() for p in model.parameters() if p.requires_grad == True))

def freeze_all_but_name(model, string):
    for name, param in model.named_parameters():
        if string in name:
            continue
        else:
            param.requires_grad = False

def unfreeze_params_with_name(model, string):
    nparams_before = sum(p.numel() for p in model.parameters() if p.requires_grad == True)
    for name, param in model.named_parameters():
        if string in name:
            param.requires_grad = True
    nparams_after = sum(p.numel() for p in model.parameters() if p.requires_grad == True)
    print(f"{string} adds {int2mil(nparams_after - nparams_before)} params")

class LoRAParametrization(nn.Module):
    def __init__(self, features_in, features_out, rank, alpha, device = None):
        super().__init__()
        # device : same device as feats
        self.lora_B = nn.Parameter(torch.zeros((features_in, rank)).to(device))
        self.lora_A = nn.Parameter(torch.zeros((rank, features_out)).to(device))
        nn.init.normal_(self.lora_A, mean=0, std=1)
        
        # Section 4.1 of the paper: 
        #   We then scale ∆Wx by α/r , where α is a constant in r. 
        #   When optimizing with Adam, tuning α is roughly the same as tuning the learning rate if we scale the initialization appropriately. 
        #   As a result, we simply set α to the first r we try and do not tune it. 
        #   This scaling helps to reduce the need to retune hyperparameters when we vary r.
        self.scale = alpha / rank
        self.enabled = True

    def forward(self, original_weights):
        if self.enabled:
            # Return W + (B*A)*scale
            return original_weights + torch.matmul(self.lora_B, self.lora_A).view(original_weights.shape) * self.scale
        else:
            return original_weights


def linear_layer_parameterization(layer, device, rank, lora_alpha=16):
    features_in, features_out = layer.weight.shape
    return LoRAParametrization(
        features_in, features_out, rank=rank, alpha=lora_alpha, device=device
    )

def parameterize(layer,weight_name, rank):
    parametrize.register_parametrization(
layer, weight_name, linear_layer_parameterization(layer, layer.weight.device, rank), unsafe =True)


def lora_siglip(model, rank, last_n_blocks):
        print(f"Trainable params before LORA :{trainable_params(model.visual_encoder.trunk)}")

        # ----------------------------------------- Parameterize --------------------------------------------------------------------
        #ViT blocks
        
        n_blocks = len(model.visual_encoder.trunk.blocks)
        for _ in range(last_n_blocks):
        # for l_num in config['model']['cross_attn_layers']:
            l_num = n_blocks - _ - 1
            block = model.visual_encoder.trunk.blocks[l_num]

            parameterize(block.attn.qkv, "weight", rank)
            parameterize(block.attn.proj, "weight", rank)
            parameterize(block.mlp.fc1, "weight", rank)
            parameterize(block.mlp.fc2, "weight", rank)
        
        # MAP head
        # parameterize(model.visual_encoder.trunk.attn_pool.mlp.fc1, "weight", rank)
        # parameterize(model.visual_encoder.trunk.attn_pool.mlp.fc2, "weight", rank)
        
        # ----------------------------------------- FREEZE --------------------------------------------------------------------
        
        freeze_all_but_name(model.visual_encoder.trunk.blocks, "lora") #Blocks
        # freeze_all_but_name(model.visual_encoder.trunk.attn_pool.mlp, "lora") # MAP head
        unfreeze_params_with_name(model.visual_encoder.trunk.blocks, "cross_attn")

        for name, param in model.named_parameters(): 
            if "visual_encoder.trunk.patch_embed" in name or "visual_encoder.siglip.text" in name:
                param.requires_grad = False
    
        print(f"Trainable params after LORA :{trainable_params(model.visual_encoder.trunk)}")