import torch as th
from torch import nn
from dgl import function as fn
from dgl.base import DGLError
from dgl.utils import expand_as_pair
from dgl.nn.functional import edge_softmax
from dgl.nn.pytorch.utils import Identity


class myGATConv(nn.Module):
    """
    """

    def __init__(
        self,
        in_feats,
        out_feats,
        num_heads,
        num_etypes,
        etype_feats=0,
        edge_feats=0,
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        residual=False,
        activation=None,
        allow_zero_in_degree=True,
        bias=True,
        alpha=0.0,  # 残余注意力比例
        use_etype_feats=True,
        use_edge_feats=True,
        emb_dim=0,
        embedding=False,
    ):
        super(myGATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self._use_etype_feat = use_etype_feats
        self._use_edge_feat = use_edge_feats
        self._embedding = embedding
        self._activation = activation
        self._bias = bias

        # 全连接层
        self.fc_src = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=bias)
        if self._embedding:
            self.fc_emb_src = nn.Linear(emb_dim, out_feats * num_heads, bias=bias)
            self.fc_emb_dst = nn.Linear(emb_dim, out_feats * num_heads, bias=bias)
        else:
            self.fc_dst = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=bias)

        self.fc_edge_1 = nn.Linear(etype_feats, out_feats * num_heads, bias=bias)
        self.fc_edge_2 = nn.Linear(edge_feats, out_feats * num_heads, bias=bias)
        # 新增edge embedding用于计算注意力分数
        self.etype_emb = nn.Parameter(th.zeros(size=(num_etypes, etype_feats)))

        self.alpha = alpha
        self.attn = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if self._in_dst_feats != out_feats * num_heads:
                self.res_fc = nn.Linear(self._in_dst_feats, num_heads * out_feats, bias=bias)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer("res_fc", None)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The fc weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The attention weights are using xavier initialization method.
        """
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_edge_1.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_edge_2.weight, gain=gain)
        if self._embedding:
            nn.init.xavier_normal_(self.fc_emb_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_emb_dst.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)

        if self._bias:
            nn.init.constant_(self.fc_src.bias, 0)
            nn.init.constant_(self.fc_edge_1.bias, 0)
            nn.init.constant_(self.fc_edge_2.bias, 0)
            if self._embedding:
                nn.init.constant_(self.fc_emb_src.bias, 0)
                nn.init.constant_(self.fc_emb_dst.bias, 0)
            else:
                nn.init.constant_(self.fc_dst.bias, 0)

        nn.init.xavier_normal_(self.attn, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
            if self._bias:
                nn.init.constant_(self.res_fc.bias, 0)

    def set_allow_zero_in_degree(self, set_value):
        r"""
        Description
        -----------
        Set allow_zero_in_degree flag.

        Parameters
        ----------
        set_value : bool
            The value to be set to the flag.
        """
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, emb_dict, get_attention=False):
        device = graph.device
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                for etype in graph.canonical_etypes:
                    if (graph.in_degrees(etype=etype) == 0).any():
                        raise DGLError(
                            "There are 0-in-degree nodes in the graph, "
                            "output for those nodes will be invalid. "
                            "This is harmful for some applications, "
                            "causing silent performance regression. "
                            "Adding self-loop on the input graph by "
                            "calling `g = dgl.add_self_loop(g)` will resolve "
                            "the issue. Setting ``allow_zero_in_degree`` "
                            "to be `True` when constructing this module will "
                            "suppress the check and let the code run."
                        )
            if isinstance(feat, dict):
                # step 1: feat dropout & fc
                h = {}
                for ntype, node_feat in feat.items():
                    h[ntype] = self.feat_drop(node_feat)
                feat_src, feat_dst = {}, {}
                for ntype, feat in h.items():
                    feat_src[ntype] = self.fc_src(feat).view(-1, self._num_heads, self._out_feats)
                    if self._embedding == False:
                        feat_dst[ntype] = self.fc_dst(feat).view(-1, self._num_heads, self._out_feats)
                graph.srcdata.update({"feat_src": feat_src})
                graph.srcdata.update({"feat_dst": feat_dst})

                # step 2: attention
                if self._embedding:
                    feat_emb_src, feat_emb_dst = {}, {}
                    for ntype, node_emb in emb_dict.items():
                        feat_emb_src[ntype] = self.fc_emb_src(node_emb
                                                             ).repeat(graph.batch_size,
                                                                      1).view(-1, self._num_heads, self._out_feats)
                        feat_emb_dst[ntype] = self.fc_emb_dst(node_emb
                                                             ).repeat(graph.batch_size,
                                                                      1).view(-1, self._num_heads, self._out_feats)
                    # (num_src_edge, num_heads, out_dim)
                    graph.srcdata.update({"emb_src": feat_emb_src})
                    graph.dstdata.update({"emb_dst": feat_emb_dst})
                if self._use_edge_feat:
                    feat_edge = {}
                    for etype, efeat in graph.edata['feat'].items():
                        feat_edge[etype] = self.fc_edge_2(efeat).view(-1, self._num_heads, self._out_feats)
                if self._use_etype_feat:
                    feat_etype = self.fc_edge_1(self.etype_emb).view(-1, self._num_heads, self._out_feats)

                # 求和计算e再LeakyReLU。
                for i, etype in enumerate(sorted(graph.canonical_etypes)):
                    if self._use_etype_feat and self._use_edge_feat:
                        graph.edata['e'] = {etype: feat_edge[etype] + feat_etype[i]}
                    elif self._use_edge_feat:
                        graph.edata['e'] = {etype: feat_edge[etype]}
                    elif self._use_etype_feat:
                        num_edges = graph.num_edges(etype)
                        graph.edata['e'] = {etype: feat_etype[i].repeat(num_edges, 1, 1)}
                    else:
                        num_edges = graph.num_edges(etype)
                        graph.edata['e'] = {etype: th.zeros(num_edges, self._out_feats).to(device)}
                if self._embedding:
                    graph.apply_edges(fn.e_add_u('e', 'emb_src', 'e'))
                    graph.apply_edges(fn.e_add_v('e', 'emb_dst', 'e'))
                else:
                    graph.apply_edges(fn.e_add_u('e', 'feat_src', 'e'))
                    graph.apply_edges(fn.e_add_v('e', 'feat_dst', 'e'))

                e = {}
                for etype in graph.canonical_etypes:
                    # (num_edge, num_heads, out_dim)
                    e[etype] = self.leaky_relu(graph.edata['e'][etype])

                    # 乘上参数向量a (num_edge, num_heads)
                    e[etype] = ((e[etype] * self.attn).sum(dim=-1).unsqueeze(dim=2))

                    # test: 对efeat值为0的边设置e为1e-9使得softmax结果趋于0
                    indices = th.nonzero(graph.edata['nan'][etype] == True).tolist()
                    e[etype][indices, :] = 1e-9

                # 计算 softmax & dropout {:(num_edge, num_heads)}
                a = edge_softmax(graph, e)
                for etype in graph.canonical_etypes:
                    # test: 对efeat值为0的边设置a值为0
                    indices = th.nonzero(graph.edata['nan'][etype] == True).tolist()
                    a_etype_clone = a[etype].clone()
                    a_etype_clone[indices, :] = a_etype_clone[indices, :] * 0
                    a[etype] = a_etype_clone
                    graph.edata["a"] = {etype: self.attn_drop(a[etype])}

                # step 3: message passing：注意力加权求和
                graph.update_all(fn.u_mul_e("feat_src", "a", "m"), fn.sum("m", "ft"))
                rst = graph.dstdata["ft"]  # {(num_node_types): (num_nodes_dst, num_heads, out_dim)}

                # step 4: residual
                if self.res_fc is not None:
                    for ntype in rst.keys():
                        resval = self.res_fc(h[ntype]).view(h[ntype].shape[0], -1, self._out_feats)
                        rst[ntype] = rst[ntype] + resval

                # step 5: activation
                if self._activation is not None:
                    for ntype in rst.keys():
                        rst[ntype] = self._activation(rst[ntype])

            if get_attention:
                return rst, graph.edata["a"]
            else:
                return rst


class myDotGATConv(nn.Module):
    r"""Apply dot product version of self attention in GATv2.

    """

    def __init__(
        self,
        in_feats,
        out_feats,
        edge_feats,
        num_heads,
        feat_drop,
        attn_drop,
        residual,
        bias=True,
        activation=None,
        allow_zero_in_degree=True,
        # use_etype_feats=True,
        use_edge_feats=True,
        emb_dim=0,
        embedding=False,
    ):
        super(myDotGATConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._edge_feats = edge_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self._num_heads = num_heads
        self.bias = bias
        self._embedding = embedding

        if self._embedding:
            self.fc_k = nn.Linear(
                emb_dim,
                self._out_feats * self._num_heads,
                bias=bias,
            )
            self.fc_q = nn.Linear(
                emb_dim,
                self._out_feats * self._num_heads,
                bias=bias,
            )
        else:
            self.fc_k = nn.Linear(
                self._in_feats,
                self._out_feats * self._num_heads,
                bias=bias,
            )
            self.fc_q = nn.Linear(
                self._in_feats,
                self._out_feats * self._num_heads,
                bias=bias,
            )
        self.fc_v = nn.Linear(
            self._in_feats,
            self._out_feats * self._num_heads,
            bias=bias,
        )
        self.fc_e = nn.Linear(self._edge_feats, self._out_feats * self._num_heads, bias=bias)
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        if residual:
            if self._in_feats != out_feats * num_heads:
                self.res_fc = nn.Linear(self._in_feats, num_heads * out_feats, bias=bias)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer("res_fc", None)
        self.activation = activation
        self.use_edge_feats = use_edge_feats

        self.reset_parameters()

    def reset_parameters(self):
        """
        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The fc weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The attention weights are using xavier initialization method.
        """
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.fc_q.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_k.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_v.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_e.weight, gain=gain)
        if self.bias:
            nn.init.constant_(self.fc_q.bias, 0)
            nn.init.constant_(self.fc_k.bias, 0)
            nn.init.constant_(self.fc_v.bias, 0)
            nn.init.constant_(self.fc_e.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
            if self.bias:
                nn.init.constant_(self.res_fc.bias, 0)

    def forward(self, graph, feat, emb_dict, get_attention=False):
        device = graph.device
        graph = graph.local_var()

        if not self._allow_zero_in_degree:
            for etype in graph.canonical_etypes:
                if (graph.in_degrees(etype=etype) == 0).any():
                    raise DGLError(
                        "There are 0-in-degree nodes in the graph, "
                        "output for those nodes will be invalid. "
                        "This is harmful for some applications, "
                        "causing silent performance regression. "
                        "Adding self-loop on the input graph by "
                        "calling `g = dgl.add_self_loop(g)` will resolve "
                        "the issue. Setting ``allow_zero_in_degree`` "
                        "to be `True` when constructing this module will "
                        "suppress the check and let the code run."
                    )

        # feat dropout
        h = {}
        for ntype, nfeat in feat.items():
            h[ntype] = self.feat_drop(nfeat)

        # q = W_q * h_i, k = W_k * h_j, e = W_e * h_e
        feat_q, feat_k, feat_v = {}, {}, {}
        for ntype, nfeat in h.items():
            if self._embedding:
                feat_q[ntype] = self.fc_q(emb_dict[ntype]).repeat(graph.batch_size,
                                                                  1).view(-1, self._num_heads, self._out_feats)
                # feat_k[ntype] = self.fc_k(emb_dict[ntype]).repeat(graph.batch_size, 1).view(
                #     -1, self._num_heads, self._out_feats
                # )
            else:
                feat_q[ntype] = self.fc_q(nfeat).view(-1, self._num_heads, self._out_feats)
                # feat_k[ntype] = self.fc_k(nfeat).view(
                #     -1, self._num_heads, self._out_feats
                # )
            # test: embedding
            feat_k[ntype] = self.fc_k(nfeat).view(-1, self._num_heads, self._out_feats)
            feat_v[ntype] = self.fc_v(nfeat).view(-1, self._num_heads, self._out_feats)

        feat_edge = {}
        if self.use_edge_feats:
            for etype, efeat in graph.edata['feat'].items():
                feat_edge[etype] = self.fc_e(efeat).view(-1, self._num_heads, self._out_feats)
        else:
            for etype in graph.canonical_etypes:
                num_edges = graph.num_edges(etype=etype)
                feat_edge[etype] = th.zeros((num_edges, self._num_heads, self._out_feats)).to(device)

        # Assign features to nodes
        graph.ndata.update({"q": feat_q})
        graph.ndata.update({"k": feat_k})
        graph.ndata.update({"v": feat_v})
        graph.edata.update({"e": feat_edge})

        # <q, k + e>
        graph.apply_edges(fn.u_add_e("k", "e", "ke"))
        graph.apply_edges(fn.v_dot_e("q", "ke", "a"))

        # Step 2. edge softmax to compute attention scores
        for etype, attn in graph.edata["a"].items():
            graph.edata["a"] = {etype: attn / self._out_feats**0.5}
            # test:
            indices = th.nonzero(graph.edata['nan'][etype] == True).tolist()
            graph.edata["a"][etype][indices, :] = 1e-9

        graph.edata["sa"] = edge_softmax(graph, graph.edata["a"])

        for etype in graph.canonical_etypes:
            # test: 对efeat值为0的边设置a值为0
            indices = th.nonzero(graph.edata['nan'][etype] == True).tolist()
            a_etype_clone = graph.edata["sa"][etype].clone()
            a_etype_clone[indices, :] = a_etype_clone[indices, :] * 0
            graph.edata["sa"][etype] = a_etype_clone
            # attention dropout
            graph.edata["sa"] = {etype: self.attn_drop(graph.edata["sa"][etype])}

        # Step 3. Broadcast softmax value to each edge, and aggregate dst node
        # graph.apply_edges(fn.u_add_e("v", "e", "v"))
        graph.update_all(fn.u_mul_e("v", "sa", "attn"), fn.sum("attn", "agg_u"))

        # output results to the destination nodes
        rst = graph.ndata["agg_u"]

        # residual
        if self.res_fc is not None:
            for ntype in rst.keys():
                resval = self.res_fc(h[ntype]).view(h[ntype].shape[0], -1, self._out_feats)
                rst[ntype] = rst[ntype] + resval

        # activation
        if self.activation is not None:
            for ntype in rst.keys():
                rst[ntype] = self.activation(rst[ntype])

        if get_attention:
            return rst, graph.edata["sa"]
        else:
            return rst
