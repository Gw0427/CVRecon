import torch

from cvrecon import transformer


class MVFusionMean(torch.nn.Module):
    def forward(self, features, valid_mask):
        return mv_fusion_mean(features, valid_mask)


class MVFusionTransformer(torch.nn.Module):
    def __init__(self, input_depth, n_layers, n_attn_heads, cv_cha=0):
        super().__init__()
        # self-attention transformer
        self.self_transformer = transformer.Transformer(
            input_depth,
            input_depth * 2,
            num_layers=n_layers,
            num_heads=n_attn_heads,
        )
        # cross attention transformer
        self.cross_transformer = transformer.CrossTransformer(
            input_depth,
            input_depth * 2,
            num_layers=n_layers,
            num_heads=n_attn_heads,
        )
        self.depth_mlp = torch.nn.Linear(1 + cv_cha + 56, input_depth, bias=True)
        self.proj_tsdf_mlp = torch.nn.Linear(input_depth, 1, bias=True)

        for mlp in [self.depth_mlp, self.proj_tsdf_mlp]:
            torch.nn.init.kaiming_normal_(mlp.weight)
            torch.nn.init.zeros_(mlp.bias)

        self.prev_output = None

    def forward(self, features, bp_depth, bp_mask, use_proj_occ, use_cache=True):
        '''
        features: [n_imgs, in_channels, n_voxels]
        bp_depth: [n_imgs, n_voxels]
        bp_mask: [n_imgs, n_voxels]
        use_cache: boolean
        '''
        device = features.device

        # mask
        attn_mask = bp_mask.transpose(0, 1)  # [n_voxels, n_imgs]
        if self.prev_output is None:
            # self-attention case
            attn_mask = ~attn_mask[:, None].repeat(1, attn_mask.shape[1], 1).contiguous()
            torch.diagonal(attn_mask, dim1=1, dim2=2)[:] = False
        else:
            # cross-attention case
            attn_mask = ~attn_mask  # [n_voxels, n_imgs]
            attn_mask = attn_mask.unsqueeze(1)  # [n_voxels, 1, n_imgs]

        # feature process
        im_z_norm = (bp_depth - 1.85) / 0.85
        features = torch.cat((features, im_z_norm[:, None]), dim=1)
        features = self.depth_mlp(features.transpose(1, 2))  # [n_imgs, n_voxels, in_channels]

        if self.prev_output is None:
            # use self-attention
            features = self.self_transformer(features, attn_mask)
        else:
            # use cross-attention，previous output as Q
            features = self.cross_transformer(
                query=self.prev_output,
                key=features,
                value=features,
                mask=attn_mask
            )

        # update
        if use_cache:
            print("Features before cache:",
                  "min:", torch.min(features).item(),
                  "max:", torch.max(features).item(),
                  "mean:", torch.mean(features).item()
                  )
            self.prev_output = features.detach()

        # Project features
        batchsize, nvoxels, _ = features.shape
        proj_occ_logits = self.proj_tsdf_mlp(
            features.reshape(batchsize * nvoxels, -1)
        ).reshape(batchsize, nvoxels)

        if use_proj_occ:
            weights = proj_occ_logits.masked_fill(~bp_mask, -9e3)
            weights = torch.cat(
                (
                    weights,
                    torch.zeros(
                        (1, weights.shape[1]),
                        device=device,
                        dtype=weights.dtype,
                    ),
                ),
                dim=0,
            )
            features = torch.cat(
                (
                    features,
                    torch.zeros(
                        (1, features.shape[1], features.shape[2]),
                        device=device,
                        dtype=features.dtype,
                    ),
                ),
                dim=0,
            )
            weights = torch.softmax(weights, dim=0)
            pooled_features = torch.sum(features * weights[..., None], dim=0)
        else:
            pooled_features = mv_fusion_mean(features, bp_mask)

        return pooled_features, proj_occ_logits

    def clear_cache(self):
        self.prev_output = None


def mv_fusion_mean(features, valid_mask):
    '''
    features: [n_imgs, n_voxels, n_channels]
    valid_mask: [n_imgs, n_voxels]

    return:
        pooled_features: each voxel's feature is the average of all seen pixels' feature
    '''
    weights = torch.sum(valid_mask, dim=0)
    weights[weights == 0] = 1
    pooled_features = (
        torch.sum(features * valid_mask[..., None], dim=0) / weights[:, None]
    )
    return pooled_features
