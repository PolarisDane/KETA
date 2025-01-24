import torch
import torch.nn as nn
import pdb, time
# from torch_scatter import scatter_mean, scatter_add

import numpy as np
import torch.nn.functional as F

class CrossAttentionLayer(nn.Module):
    """Cross attention layer.

    Args:
        d_model (int): Model dimension.
        num_heads (int): Number of heads.
        dropout (float): Dropout rate.
    """

    def __init__(self, d_model, num_heads, dropout, fix=False):
        super().__init__()
        self.fix = fix
        self.num_heads = num_heads
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        """Init weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, sources, queries, attn_masks=None):
        """Forward pass.

        Args:
            sources (List[Tensor]): of len batch_size,
                each of shape (n_points_i, d_model).
            queries (List[Tensor]): of len batch_size,
                each of shape(n_queries_i, d_model).
            attn_masks (List[Tensor] or None): of len batch_size,
                each of shape (n_queries, n_points).
        
        Return:
            List[Tensor]: Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        """
        outputs = []
        # import pdb; pdb.set_trace()
        for query, source, attn_mask in zip(queries, sources, attn_masks):
            k = v = source
            # attn_mask = attn_masks[i] if attn_masks is not None else None
            output, _ = self.attn(query, k, v, attn_mask=attn_mask.repeat(self.num_heads, 1, 1))
            if self.fix:
                output = self.dropout(output)
            output = output + query
            if self.fix:
                output = self.norm(output)
            outputs.append(output)
        return outputs


class SelfAttentionLayer(nn.Module):
    """Self attention layer.

    Args:
        d_model (int): Model dimension.
        num_heads (int): Number of heads.
        dropout (float): Dropout rate.
    """

    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Forward pass.

        Args:
            x (List[Tensor]): Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        
        Returns:
            List[Tensor]: Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        """
        out = []
        for y in x:
            z, _ = self.attn(y, y, y)
            z = self.dropout(z) + y
            z = self.norm(z)
            out.append(z)
        return out


class FFN(nn.Module):
    """Feed forward network.

    Args:
        d_model (int): Model dimension.
        hidden_dim (int): Hidden dimension.
        dropout (float): Dropout rate.
        activation_fn (str): 'relu' or 'gelu'.
    """

    def __init__(self, d_model, hidden_dim, dropout, activation_fn):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU() if activation_fn == 'relu' else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """Forward pass.

        Args:
            x (List[Tensor]): Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        
        Returns:
            List[Tensor]: Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        """
        out = []
        for y in x:
            z = self.net(y)
            z = z + y
            z = self.norm(z)
            out.append(z)
        return out


class TextEncoder(nn.Module):
    """MLP-based text encoder.
    
    Args:
        d_models (int): Model dimensions.
        input_dim (int): Input dimension.
        output_dim (int): Text embedding dimension.
        activation_fn (str): 'relu' or 'gelu'.    
    """

    def __init__(self, d_models, input_dim, output_dim, activation_fn, device):
        super().__init__()
        dims = [input_dim] + d_models + [output_dim]
        nets = []
        for i in range(len(d_models) + 1):
            nets.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(d_models):
                nets.append(nn.ReLU() if activation_fn == 'relu' else nn.GELU())
        self.net = nn.Sequential(*nets)
        self.out_dim = output_dim
        self.device = device
        self.init_weights()
        
    def init_weights(self):
        """Init weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(self, x, max_spt, spt_len):
        """Forward pass.

        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_spt, input_dim).
        
        Returns:
            List[Tensor]: of len batch_size,
                each of shape(n_spt, output_dim).
        """      
        ret = torch.zeros((len(x), max_spt, 1, self.out_dim)).to(self.device)
        for i in range(len(x)):
            ret[i, :spt_len[i], :, :] = self.net(x[i, :spt_len[i]])
        return ret
    
class QueryDecoder(nn.Module):
    """Query decoder for SPFormer.

    Args:
        num_layers (int): Number of transformer layers.
        query_channels (int): Number of channels for query.
        in_channels (int): Number of input channels.
        d_model (int): Number of channels for model layers.
        num_heads (int): Number of head in attention layer.
        hidden_dim (int): Dimension of attention layer.
        dropout (float): Dropout rate for transformer layer.
        activation_fn (str): 'relu' of 'gelu'.
        iter_pred (bool): Whether to predict iteratively.
        attn_mask (bool): Whether to use mask attention.
    """

    def __init__(self, num_layers, query_channels, in_channels, d_model, num_heads, hidden_dim,
                 dropout, activation_fn, iter_pred, attn_mask, fix_attention,
                 device, domain_type
                 ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU()
        )
        self.query_proj = nn.Sequential(
            nn.Linear(query_channels, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.cross_attn_layers = nn.ModuleList([])
        self.self_attn_layers = nn.ModuleList([])
        self.ffn_layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.cross_attn_layers.append(
                CrossAttentionLayer(d_model, num_heads, dropout, fix_attention)
            )
            self.self_attn_layers.append(
                SelfAttentionLayer(d_model, num_heads, dropout)
            )
            self.ffn_layers.append(
                FFN(d_model, hidden_dim, dropout, activation_fn)
            )
            
        self.out_norm = nn.LayerNorm(d_model)
        self.out_cls = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2)
        )
        # self.x_mask = nn.Sequential(
        #     nn.Linear(in_channels, d_model),
        #     nn.ReLU(),
        #     nn.Linear(d_model, d_model)
        # )
        self.iter_pred = iter_pred
        self.attn_mask = attn_mask
        self.device = device
        self.domain_type = domain_type
        
    def _get_queries(self, queries=None, batch_size=None):
        """Get query tensor.

        Args:
            queries (List[Tensor], optional): of len batch_size,
                each of shape (n_queries_i, in_channels).
            batch_size (int, optional): batch size.
        
        Returns:
            List[Tensor]: of len batch_size, each of shape
                (n_queries_i, d_model).
        """
        if batch_size is None:
            batch_size = len(queries)
        
        result_queries = []
        for i in range(batch_size):
            result_queries.append(self.query_proj(queries[i]))
        return result_queries

    def get_mask(self, decoder_output, shape):      
        if self.domain_type == "gaussian":
            idx = torch.arange(shape[1]).to(self.device).unsqueeze(0).unsqueeze(0).repeat(shape[0], 1, 1) + 1/2
            mu = torch.sigmoid(decoder_output[:, :, :1]) * shape[1]
            sigma = torch.sigmoid(decoder_output[:, :, 1:]) * shape[1] / 6
            domain_weight = torch.exp(-0.5 * (idx - mu) ** 2 / (sigma ** 2))
            domain_weight = torch.nn.functional.normalize(domain_weight, dim=2, p=1)
            return domain_weight
        else:
            raise NotImplementedError
    
    def _forward_head(self, queries, mask_feats):
        """Prediction head forward.

        Args:
            queries (List[Tensor] | Tensor): List of len batch_size,
                each of shape (n_queries_i, d_model). Or tensor of
                shape (batch_size, n_queries, d_model).
            mask_feats (List[Tensor]): of len batch_size,
                each of shape (n_points_i, d_model).

        """
        cls_preds, pred_masks, attn_masks = [], [], []
        for i in range(len(queries)):
            norm_query = self.out_norm(queries[i])
            cls_preds.append(self.out_cls(norm_query))
            pred_mask = self.get_mask(cls_preds[-1], mask_feats[i].shape)
            if self.attn_mask:
                attn_mask = (pred_mask < torch.exp(torch.tensor(-2)) * torch.max(pred_mask, dim=2)[0].unsqueeze(-1)).bool()
                attn_mask[torch.where(
                    attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                attn_mask = attn_mask.detach()
                attn_masks.append(attn_mask)
            pred_masks.append(pred_mask)
        attn_masks = attn_masks if self.attn_mask else None
        return cls_preds, pred_masks, attn_masks

    def forward_simple(self, x, queries):
        """Simple forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_input, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_queries, query_channles).
        
        Returns:
            Dict: with labels, masks, and scores.
        """
        inst_feats = [self.input_proj(y) for y in x]
        mask_feats = [self.x_mask(y) for y in x]
        queries = self._get_queries(queries, len(x))
        for i in range(len(self.cross_attn_layers)):
            queries = self.cross_attn_layers[i](inst_feats, queries)
            queries = self.self_attn_layers[i](queries.squeeze(1))
            queries = self.ffn_layers[i](queries.unsqueeze(1))
            
        cls_preds, pred_masks, _ = self._forward_head(queries, mask_feats)
        return dict(
            cls_preds=cls_preds,
            masks=pred_masks,
        )

    def forward_iter_pred(self, x, queries):
        """Iterative forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_input, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_queries, in_channles).
        
        Returns:
            Dict: with labels, masks, scores, and aux_outputs.
        """
        # import pdb; pdb.set_trace()
        cls_preds, pred_scores, pred_masks = [], [], []
        inst_feats = [self.input_proj(y) for y in x]
        # mask_feats = [self.x_mask(y) for y in x]
        queries = self._get_queries(queries, len(x))
        cls_pred, pred_mask, attn_mask = self._forward_head(queries, inst_feats)
        cls_preds.append(cls_pred)
        pred_masks.append(pred_mask)
        for i in range(len(self.cross_attn_layers)):
            queries = self.cross_attn_layers[i](inst_feats, queries, attn_mask)
            # queries = self.self_attn_layers[i](queries)
            queries = self.ffn_layers[i](queries)
            cls_pred, pred_mask, attn_mask = self._forward_head(
                queries, inst_feats)
            cls_preds.append(cls_pred)
            pred_masks.append(pred_mask)

        return dict(
            cls_preds=cls_preds[-1],
            masks=pred_masks[-1])

    def forward(self, x, queries=None, query_len=None):
        """Forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with labels, masks, scores, and possibly aux_outputs.
        """
        queries = [q[:query_len[i]] for i, q in enumerate(queries)]
        if self.iter_pred:
            return self.forward_iter_pred(x, queries)
        else:
            return self.forward_simple(x, queries)

class TextKPAlignmentModel(nn.Module):
    def __init__(self, text_encoding_dims, text_embedding_dim, kp_dim, domain_type, activation_fn,
                 num_layers, d_model, num_heads, hidden_dim, dropout,
                 iter_pred=False, attn_mask=False, fix_attention=True,
                 device='cuda'
                ):
        """For alignment.
        
        """
        super().__init__()
        self.text_embedding_dim = text_embedding_dim
        self.kp_dim = kp_dim
        
        self.text_feature_extractor = TextEncoder(text_encoding_dims, text_embedding_dim, kp_dim, activation_fn, device).to(device)

        self.domain_model = QueryDecoder(num_layers, text_embedding_dim, kp_dim,
                                         d_model, num_heads, hidden_dim, dropout, activation_fn,
                                         iter_pred, attn_mask, fix_attention, device, domain_type
                                        ).to(device)
        
        self.domain_type = domain_type
        self.device = device
    
    def get_weight_distribution(self, decoder_output, batch_idx, shape):
        if self.domain_type == "uniform":
            domain_weight = torch.zeros((length, 1)).to(self.device)
            begin = decoder_output[:, 0]
            begin_index = begin - (begin - torch.round(begin)).detach()
            end = decoder_output[:, 1]
            end_index = end - (end - torch.round(end)).detach()
            domain_weight[begin_index:end_index] = 1. / (end - begin) / length
            raise NotImplementedError
        elif self.domain_type == "gaussian":
            # import pdb; pdb.set_trace()
            decoder_ret = decoder_output['masks'][batch_idx]
            return decoder_ret
        elif self.domain_type == "norm":
            decoder_ret = decoder_output['masks'][batch_idx]
            decoder_ret = torch.softmax(decoder_ret, dim=2)
            return decoder_ret
        else:
            raise NotImplementedError
       
    def forward(self, text_embedding, kp, max_spt, spt_len):
        """Forward pass.
        
        Args:
            text_embedding (Tensor): of len batch_size, each of shape
                (n_spt, text_embedding_dim).
            kp (List[Tensor]): of len batch_size, each of shape
                (n_points_i, kp_dim).
        
        """
        kp = [i.cuda() for i in kp]
        text_embedding = text_embedding.unsqueeze(2).cuda()
        text_features = self.text_feature_extractor(text_embedding, max_spt, spt_len)
        # import pdb; pdb.set_trace()
        decoder_ret = self.domain_model(kp, text_embedding, spt_len)
        weighted_kps = torch.zeros((len(kp), max_spt, 1, self.kp_dim)).to(self.device)
        for i in range(len(kp)):
            weight = self.get_weight_distribution(decoder_ret, i, kp[i].shape)
            weighted_kp = torch.einsum("nij,njk->nik", weight, kp[i])
            weighted_kps[i, :spt_len[i], :, :] = weighted_kp  

        loss = F.mse_loss(text_features, weighted_kps)

        return loss
    
    def calc_dist(self, text_embedding, kp, max_spt, spt_len):
        """Forward pass returning dist.
        
        Args:
            text_embedding (List[Tensor]): of len batch_size, each of shape
                (n_spt, text_embedding_dim).
            kp (List[Tensor]): of len batch_size, each of shape
                (n_points_i, kp_dim).
        
        """
        kp = [i.cuda() for i in kp]
        text_embedding = text_embedding.unsqueeze(2).cuda()
        text_features = self.text_feature_extractor(text_embedding, max_spt, spt_len)
        # import pdb; pdb.set_trace()
        decoder_ret = self.domain_model(kp, text_embedding, spt_len)
        weighted_kps = torch.zeros((len(kp), max_spt, 1, self.kp_dim)).to(self.device)
        for i in range(len(kp)):
            weight = self.get_weight_distribution(decoder_ret, i, kp[i].shape)
            weighted_kp = torch.einsum("nij,njk->nik", weight, kp[i])
            weighted_kps[i, :spt_len[i], :, :] = weighted_kp  

        dist = text_features - weighted_kps

        return dist
