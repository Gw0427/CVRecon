import torch


class MlpBlock(torch.nn.Module):
    """Transformer Feed-Forward Block"""

    def __init__(self, in_dim, mlp_dim, out_dim, dropout_rate=0.0):
        super().__init__()

        # init layers
        self.fc1 = torch.nn.Linear(in_dim, mlp_dim)
        self.fc2 = torch.nn.Linear(mlp_dim, out_dim)
        self.act = torch.nn.ReLU(True)
        self.dropout1 = torch.nn.Dropout(dropout_rate)
        self.dropout2 = torch.nn.Dropout(dropout_rate)

        torch.nn.init.kaiming_normal_(self.fc1.weight)
        torch.nn.init.kaiming_normal_(self.fc2.weight)
        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.zeros_(self.fc2.bias)

    def forward(self, x):

        out = self.fc1(x)
        out = self.act(out)
        if self.dropout1:
            out = self.dropout1(out)

        out = self.fc2(out)
        out = self.dropout2(out)
        return out


class EncoderBlock(torch.nn.Module):
    def __init__(
        self, in_dim, num_heads, mlp_dim, dropout_rate=0.0, attn_dropout_rate=0.0
    ):
        super().__init__()

        self.norm1 = torch.nn.LayerNorm(in_dim)
        self.attn = torch.nn.MultiheadAttention(in_dim, num_heads)
        if dropout_rate > 0:
            self.dropout = torch.nn.Dropout(dropout_rate)
        else:
            self.dropout = None
        self.norm2 = torch.nn.LayerNorm(in_dim)
        self.mlp = MlpBlock(in_dim, mlp_dim, in_dim, dropout_rate)

    def forward(self, x, mask=None):
        residual = x
        x = self.norm1(x)
        x, attn_weights = self.attn(x, x, x, attn_mask=mask, need_weights=False)
        if self.dropout is not None:
            x = self.dropout(x)
        x += residual
        residual = x

        x = self.norm2(x)
        x = self.mlp(x)
        x += residual
        return x


class Transformer(torch.nn.Module):
    def __init__(
        self,
        emb_dim,
        mlp_dim,
        num_layers=1,
        num_heads=1,
        dropout_rate=0.0,
        attn_dropout_rate=0.0,
    ):
        super().__init__()

        in_dim = emb_dim
        self.encoder_layers = torch.nn.ModuleList()
        for i in range(num_layers):
            layer = EncoderBlock(
                in_dim, num_heads, mlp_dim, dropout_rate, attn_dropout_rate
            )
            self.encoder_layers.append(layer)

        self.num_heads = num_heads

    def forward(self, x, mask=None):
        '''
        x: [n_imgs, n_voxels, in_channels]
        torch.nn.MultiheadAttention: [seq_len, batchsize, dim] -> [seq_len, batchsize, dim]
        '''
        if self.num_heads > 1:
            b, s, t = mask.shape
            mask_n = mask.repeat(1, self.num_heads, 1).reshape(b * self.num_heads, s, t)
        else:
            mask_n = mask

        for layer in self.encoder_layers:
            x = layer(x, mask=mask_n)

        return x


class CrossTransformer(torch.nn.Module):
    def __init__(
        self,
        emb_dim,
        mlp_dim,
        num_layers=1,
        num_heads=1,
        dropout_rate=0.0,
        attn_dropout_rate=0.0,
    ):
        super().__init__()

        in_dim = emb_dim
        self.encoder_layers = torch.nn.ModuleList()
        for i in range(num_layers):
            layer = CrossAttentionBlock(
                in_dim, num_heads, mlp_dim, dropout_rate, attn_dropout_rate
            )
            self.encoder_layers.append(layer)

        self.num_heads = num_heads

    def forward(self, query, key, value, mask=None):
        """
        query: [1, n_voxels, in_channels] - Reference view features
        key: [n_imgs-1, n_voxels, in_channels] - Source view features
        value: [n_imgs-1, n_voxels, in_channels] - Source view features
        mask: [n_voxels, n_imgs-1] - Attention mask
        Returns: [1, n_voxels, in_channels] - Fused features
        """

        if self.num_heads > 1 and mask is not None:
            n_voxels = mask.shape[0]
            batch_size = mask.shape[2]

            # 1. mask dimention [n_voxels, batch_size, batch_size]
            mask = mask.squeeze(1)  # [n_voxels, batch_size]
            mask = mask.unsqueeze(2).repeat(1, 1, batch_size)  # [n_voxels, batch_size, batch_size]

            mask = mask.repeat_interleave(self.num_heads, dim=0)  # [n_voxels * num_heads, batch_size, batch_size]

        x = query
        for layer in self.encoder_layers:
            x = layer(x, key, value, mask=mask)

        return x

# query: (seq_len_q, batch_size, in_dim)
# key:  (seq_len_k, batch_size, in_dim)
# value: (seq_len_k, batch_size, in_dim)
class CrossAttentionBlock(torch.nn.Module):
    def __init__(
        self, in_dim, num_heads, mlp_dim, dropout_rate=0.0, attn_dropout_rate=0.0
    ):
        super().__init__()

        self.norm1 = torch.nn.LayerNorm(in_dim)
        self.cross_attn = torch.nn.MultiheadAttention(in_dim, num_heads)
        if dropout_rate > 0:
            self.dropout = torch.nn.Dropout(dropout_rate)
        else:
            self.dropout = None
        self.norm2 = torch.nn.LayerNorm(in_dim)
        self.mlp = MlpBlock(in_dim, mlp_dim, in_dim, dropout_rate)

    def forward(self, query, key, value, mask=None):
        residual = query
        query = self.norm1(query)
        # Cross attention between query (reference) and key/value (source) features
        x, attn_weights = self.cross_attn(
            query, key, value, 
            attn_mask=mask,
            need_weights=False
        )
        
        if self.dropout is not None:
            x = self.dropout(x)
        x += residual
        residual = x

        x = self.norm2(x)
        x = self.mlp(x)
        x += residual
        return x