from torchvision.models.swin_transformer import _swin_transformer, SwinTransformerBlockV2, PatchMergingV2

def Swin(num_classes):
    return _swin_transformer(
        patch_size=[2, 2],
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[4, 4],
        stochastic_depth_prob=0.2,
        weights=None,
        progress=True,
        block=SwinTransformerBlockV2,
        downsample_layer=PatchMergingV2,
        num_classes=num_classes,
    )