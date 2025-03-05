import torch
import torch.nn as nn
import torchvision

class DinoV2(torch.nn.Module):
    AVAILABLE_MODELS = [
        'dinov2_vits14',
        'dinov2_vitb14',
        'dinov2_vitl14',
        'dinov2_vitg14'
    ]
    
    def __init__(
        self,
        backbone_name="dinov2_vitb14",
        unfreeze_n_blocks=2,
        reshape_output=True,
        normalize = True
    ):
        super().__init__()
        
        self.backbone_name = backbone_name
        self.unfreeze_n_blocks = unfreeze_n_blocks
        self.reshape_output = reshape_output
        self.normalize = normalize
        # make sure the backbone_name is in the available models
        if self.backbone_name not in self.AVAILABLE_MODELS:
            print(f"Backbone {self.backbone_name} is not recognized!, using dinov2_vitb14")
            self.backbone_name = "dinov2_vitb14"                             
                
        self.dino = torch.hub.load('facebookresearch/dinov2', self.backbone_name)
        
        # freeze all parameters
        for param in self.dino.parameters():
            param.requires_grad = False
        
        # unfreeze the last few blocks
        for block in self.dino.blocks[ -unfreeze_n_blocks : ]:
            for param in block.parameters():
                param.requires_grad = True
        
        # remove the output norm layer of dino
        if self.normalize != True:
            self.dino.norm = nn.Identity() # remove the normalization layer
        
        self.out_channels = self.dino.embed_dim
        
    @property
    def patch_size(self):
        return self.dino.patch_embed.patch_size[0]  # Assuming square patches
    
    def forward(self, x):
        B, _, H, W = x.shape
        # No need to compute gradients for frozen layers
        with torch.no_grad():
            x = self.dino.prepare_tokens_with_masks(x)
            for blk in self.dino.blocks[ : -self.unfreeze_n_blocks]:
                x = blk(x)

        # Last blocks are trained
        for blk in self.dino.blocks[-self.unfreeze_n_blocks : ]:
            x = blk(x)
            
        
        x = x[:, 1:] # remove the [CLS] token
        
        # reshape the output tensor to B, C, H, W
        if self.reshape_output:
            _, _, C = x.shape # or C = self.embed_dim
            patch_size = self.patch_size
            x = x.permute(0, 2, 1).view(B, C, H // patch_size, W // patch_size)
        return x