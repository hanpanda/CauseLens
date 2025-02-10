import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch.conv import GATConv, GATv2Conv, GraphConv, DotGatConv

from layers import myGATConv, myDotGATConv


# ================================= Graph AE =================================
class GraphEncoder(nn.Module):

    def __init__(self, in_feats, out_feats, conv_type, num_layers, num_heads=3):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if conv_type == 'DotGATConv':
                if i != 0:
                    in_feats = num_heads * out_feats
                self.layers.append(DotGatConv(in_feats=in_feats, out_feats=out_feats, num_heads=num_heads))
            elif conv_type == 'GATConv':
                if i != 0:
                    in_feats = num_heads * out_feats
                self.layers.append(GATConv(in_feats=in_feats, out_feats=out_feats, num_heads=num_heads))
            elif conv_type == 'GraphConv':
                self.layers.append(GraphConv(in_feats=in_feats, out_feats=out_feats))

        if 'GAT' in conv_type:
            self.linear = nn.Linear(num_heads * out_feats, out_feats)

    def forward(self, graph, x):
        for layer in self.layers:
            x = layer(graph, x)
            x = x.view(x.shape[0], -1)

        if hasattr(self, 'linear'):
            x = self.linear(x)

        return x


class GraphAE(nn.Module):

    def __init__(
        self,
        conv_type,
        node_in_feats,
        in_feats,
        hidden_feats,
        num_enc_layers,
        num_dec_layers,
        num_heads,
        bidirected=False
    ):
        super().__init__()
        self.node_in_feats = node_in_feats
        self.in_projs = nn.ModuleDict()
        self.out_projs = nn.ModuleDict()
        for ntype, node_feats in node_in_feats.items():
            self.in_projs[ntype] = nn.Linear(node_feats, in_feats)
            self.out_projs[ntype] = nn.Linear(in_feats, node_feats)

        self.encoder = GraphEncoder(in_feats, hidden_feats, conv_type, num_enc_layers, num_heads)
        # self.decoder = GraphEncoder(hidden_feats, in_feats, conv_type, num_dec_layers, num_heads)
        self.decoder = MLP(hidden_feats, in_feats, act_last=True)
        self.bidirected = bidirected
        self.sort_keys = sorted(node_in_feats.keys())

    def forward(self, graph, x, device):
        """Convert graph to a homogeneous graph."""
        graph = dgl.to_homogeneous(graph)
        if self.bidirected:
            graph = dgl.to_bidirected(graph)

        graph = graph.to(device)
        for ntype, feat in x.items():
            x[ntype] = feat.to(device)

        x_dims = []
        x_proj = []
        for ntype in self.sort_keys:
            x_dims.append(x[ntype].shape[0])
            x_proj.append(self.in_projs[ntype](x[ntype]))
        x_proj = torch.concat(x_proj)

        x_hid = self.encoder(graph, x_proj)
        x_dec = self.decoder(x_hid)

        x_out = {}
        x_dec_list = torch.split(x_dec, x_dims, dim=0)
        for i, ntype in enumerate(self.sort_keys):
            x_out[ntype] = torch.sigmoid(self.out_projs[ntype](x_dec_list[i]))

        return x_out


# ================================= 通用模型 =================================
class MLP(nn.Module):

    def __init__(self, in_feats, out_feats, dropout=0, act_last=False):
        super().__init__()
        self.dropout = dropout
        self.act_last = act_last
        self.linears = nn.ModuleList()
        self.linears.append(nn.Linear(in_feats, out_feats * 3))
        self.linears.append(nn.Linear(out_feats * 3, int(out_feats * 2)))
        self.linears.append(nn.Linear(int(out_feats * 2), out_feats))

    def forward(self, x):
        # x = x.float()
        for i, linear in enumerate(self.linears):
            x = linear(x)
            if i != len(self.linears) - 1:
                x = F.dropout(x, self.dropout, self.training)
                x = F.relu(x)
        if self.act_last:
            x = F.relu(x)

        return x


class MLPAE(nn.Module):

    def __init__(self, in_feats, hidden_feats, dropout=0):
        super().__init__()
        self.encoder = MLP(in_feats, hidden_feats, dropout)
        self.decoder = MLP(hidden_feats, in_feats, dropout)

    def forward(self, g, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return g.adj().to_dense(), x


class GATEncoder(nn.Module):

    def __init__(
        self,
        in_feats,
        out_feats,
        num_etypes,
        etype_feats,
        edge_feats,
        gat_conv=GATConv,
        num_heads=3,
        num_layers=1,
        feat_drop=0,
        attn_drop=0,
        residual=False,
        alpha=0.0,
        use_etype_feats=True,
        use_edge_feats=True,
        emb_dim=0,
        embedding=False,
    ):
        super().__init__()

        self.gat_layers = nn.ModuleList()
        if gat_conv == myGATConv:
            self.gat_layers.append(
                gat_conv(
                    in_feats,
                    out_feats,
                    num_heads=num_heads,
                    activation=nn.LeakyReLU(),
                    feat_drop=feat_drop,
                    attn_drop=attn_drop,
                    num_etypes=num_etypes,
                    edge_feats=edge_feats,
                    etype_feats=etype_feats,
                    residual=residual,
                    alpha=alpha,
                    use_edge_feats=use_edge_feats,
                    use_etype_feats=use_etype_feats,
                    emb_dim=emb_dim,
                    embedding=embedding,
                )
            )
            for _ in range(num_layers - 1):
                self.gat_layers.append(
                    gat_conv(
                        out_feats * num_heads,
                        out_feats,
                        num_heads=num_heads,
                        activation=nn.LeakyReLU(),
                        feat_drop=feat_drop,
                        attn_drop=attn_drop,
                        num_etypes=num_etypes,
                        edge_feats=edge_feats,
                        etype_feats=etype_feats,
                        residual=residual,
                        alpha=alpha,
                        use_edge_feats=use_edge_feats,
                        use_etype_feats=use_etype_feats,
                    )
                )
        elif gat_conv == myDotGATConv:
            self.gat_layers.append(
                gat_conv(
                    in_feats,
                    out_feats,
                    num_heads=num_heads,
                    activation=nn.LeakyReLU(),
                    feat_drop=feat_drop,
                    attn_drop=attn_drop,
                    # num_etypes=num_etypes,
                    edge_feats=edge_feats,
                    # etype_feats=etype_feats,
                    residual=residual,
                    use_edge_feats=use_edge_feats,
                    # use_etype_feats=use_etype_feats,
                    emb_dim=emb_dim,
                    embedding=embedding
                )
            )
            for _ in range(num_layers - 1):
                self.gat_layers.append(
                    gat_conv(
                        out_feats * num_heads,
                        out_feats,
                        num_heads=num_heads,
                        activation=nn.LeakyReLU(),
                        feat_drop=feat_drop,
                        attn_drop=attn_drop,
                        edge_feats=edge_feats,
                        residual=residual,
                        use_edge_feats=use_edge_feats,
                    )
                )
        else:
            self.gat_layers.append(
                gat_conv(
                    in_feats,
                    out_feats,
                    num_heads=num_heads,
                    activation=nn.LeakyReLU(),
                    feat_drop=feat_drop,
                    attn_drop=attn_drop
                )
            )
            for _ in range(num_layers - 1):
                self.gat_layers.append(
                    gat_conv(
                        out_feats * num_heads,
                        out_feats,
                        num_heads=num_heads,
                        activation=nn.LeakyReLU(),
                        feat_drop=feat_drop,
                        attn_drop=attn_drop
                    )
                )
        self.linear = nn.Linear(num_heads * out_feats, out_feats)

    def forward(self, g, x, emb_dict, get_attention=False):
        if isinstance(x, dict):
            for gat in self.gat_layers:
                x = gat(g, x, emb_dict, get_attention)
                if isinstance(x, tuple):
                    x, attn = x
                for ntype, feat in x.items():
                    x[ntype] = feat.view(feat.shape[0], -1)
            for ntype, feat in x.items():
                feat = feat.view(feat.shape[0], -1)
                x[ntype] = self.linear(feat)
        else:
            for gat in self.gat_layers:
                x = gat(g, x)
                x = x.view(x.shape[0], -1)
            x = self.linear(x)

        if get_attention:
            return x, attn
        else:
            return x


class InnerProductDecoder(nn.Module):

    def __init__(self, act=True):
        super().__init__()
        self.act = act

    def forward(self, x, y):
        x = x @ y.T
        if self.act:
            x = torch.sigmoid(x)

        return x


# ================================= CustomGAE模型 =================================
class CustomGAE(nn.Module):

    def __init__(
        self,
        in_feats,
        hidden_feats,
        conv_type='GAT',
        num_heads=3,
        feat_drop=0,
        attn_drop=0,
        num_enc_layers=2,
        num_dec_layers=2,
        num_etypes=0,
        etype_feats=0,
        edge_feats=0,
        residual=False,
        alpha=0.0,
        node_in_feats={},
        decoder_type='gnn',
        use_etype_feats=True,
        use_edge_feats=True,
        num_nodes_dict={},
        embedding=False,
    ):
        super().__init__()
        if conv_type == 'GAT':
            gat_conv = GATConv
        elif conv_type == 'GATv2':
            gat_conv = GATv2Conv
        elif conv_type == 'myGAT':
            gat_conv = myGATConv
        elif conv_type == 'DotGAT':
            gat_conv = myDotGATConv

        self.embedding = embedding
        self.in_feats = in_feats
        # self.norm = nn.BatchNorm1d(in_feats)
        self.encoder = GATEncoder(
            in_feats=in_feats,
            out_feats=hidden_feats,
            num_etypes=num_etypes,
            etype_feats=etype_feats,
            edge_feats=edge_feats,
            gat_conv=gat_conv,
            num_heads=num_heads,
            num_layers=num_enc_layers,
            feat_drop=feat_drop,
            attn_drop=attn_drop,
            residual=residual,
            alpha=alpha,
            use_etype_feats=use_etype_feats,
            use_edge_feats=use_edge_feats,
            emb_dim=in_feats,
            embedding=embedding,
        )

        self.decoder_type = decoder_type
        if self.decoder_type == 'gnn':
            self.attr_decoder = GATEncoder(
                in_feats=hidden_feats,
                out_feats=in_feats,
                num_etypes=num_etypes,
                etype_feats=etype_feats,
                edge_feats=edge_feats,
                gat_conv=gat_conv,
                num_heads=num_heads,
                num_layers=num_dec_layers,
                feat_drop=feat_drop,
                attn_drop=attn_drop,
                residual=residual,
                alpha=alpha,
                use_edge_feats=use_etype_feats,
                use_etype_feats=use_etype_feats,
                emb_dim=in_feats,
                embedding=False,
            )
        elif self.decoder_type == 'mlp':
            self.attr_decoder = MLP(in_feats=hidden_feats, out_feats=in_feats, act_last=True)

        self.struct_decoder = InnerProductDecoder(act=True)

        # 对于不同类型节点特征投影到相同维度
        self.in_projs = nn.ModuleDict()
        for node_type, node_feats in node_in_feats.items():
            self.in_projs[node_type] = nn.Linear(node_feats, in_feats)

        # 不同类型节点特征投影到各自原来的维度
        self.out_projs = nn.ModuleDict()
        for node_type, node_feats in node_in_feats.items():
            self.out_projs[node_type] = nn.Linear(in_feats, node_feats)

        # all nodes embedding
        if embedding:
            self.num_node_dict = num_nodes_dict
            self.emb_layer_dict = nn.ModuleDict()
            for node_type in node_in_feats.keys():
                self.emb_layer_dict[node_type] = nn.Embedding(
                    num_embeddings=num_nodes_dict[node_type], embedding_dim=in_feats
                )

    def set_emb(self, init_emb_dict):
        for ntype in self.emb_layer_dict.keys():
            self.emb_layer_dict[ntype] = nn.Embedding.from_pretrained(init_emb_dict[ntype])

    def forward(self, g, X, get_attention=False):
        # 嵌入
        emb_dict = {}
        if self.embedding:
            for ntype, emb_layer in self.emb_layer_dict.items():
                emb_dict[ntype] = emb_layer(torch.arange(self.num_node_dict[ntype]).to(g.device)).to(g.device)

        # 投影
        X_proj = {}
        for ntype in X.keys():
            x_proj = self.in_projs[ntype](X[ntype])

            if torch.isnan(X[ntype]).any():
                raise Exception('Nan异常!')
            if torch.isnan(x_proj).any():
                raise Exception('Nan异常!')
            X_proj[ntype] = x_proj

        # 编码
        Z = self.encoder(g, X_proj, emb_dict, get_attention)
        if isinstance(Z, tuple):
            Z, attn = Z

        # 解码
        if self.decoder_type == 'gnn':
            X_hat = self.attr_decoder(g, Z, emb_dict)
        elif self.decoder_type == 'mlp':
            X_hat = {}
            for ntype, feat in Z.items():
                X_hat[ntype] = self.attr_decoder(feat)

        # 投影
        for ntype in X_hat.keys():
            X_hat[ntype] = self.out_projs[ntype](X_hat[ntype])

        # sigmoid output 约束输出 (0, 1)
        for ntype, feat in X_hat.items():
            X_hat[ntype] = 2 * torch.sigmoid(feat)
            # X_hat[ntype] = torch.relu(feat)

        if get_attention:
            return X_hat, attn
        else:
            return X_hat


# ================================= 损失函数 =================================
def loss_func_customgae(g, X, X_hat, is_mask=False, type='mse', recon_ntypes=['api']):
    feat_loss = {}

    if type == 'rmse':
        for ntype in recon_ntypes:
            if ntype == 'api' and is_mask:
                mask = g.ndata['mask']
                diff = X_hat['masked_{}'.format(ntype)] - X[ntype]
                diff = diff[mask[ntype] == 1]
            else:
                diff = X_hat[ntype] - X[ntype]
            feat_loss[ntype] = torch.mean(torch.sqrt(torch.sum(torch.pow(diff, 2), 1)))

    elif type == 'mse':
        for ntype in recon_ntypes:
            if ntype == 'api' and is_mask:
                mask = g.ndata['mask']
                diff = X_hat['masked_{}'.format(ntype)] - X[ntype]
                diff = diff[mask[ntype] == 1]
            else:
                diff = X_hat[ntype] - X[ntype]
            feat_loss[ntype] = torch.mean(torch.sum(torch.pow(diff, 2), 1))

    total_feat_loss = 0
    for value in feat_loss.values():
        total_feat_loss += value

    return total_feat_loss
