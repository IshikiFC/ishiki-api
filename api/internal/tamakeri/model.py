import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from api.internal.tamakeri.utils import to_torch, to_numpy, map_r


def load_model(model, model_path):
    loaded_dict_ = torch.load(model_path)
    model_dict = model.state_dict()
    loaded_dict = {k: v for k, v in loaded_dict_.items() if k in model_dict}
    model_dict.update(loaded_dict)
    model.load_state_dict(model_dict)
    return model


class BaseModel(nn.Module):
    def __init__(self, env=None, args=None, action_length=None):
        super().__init__()
        self.action_length = env.action_length() if action_length is None else action_length

    def init_hidden(self, batch_size=None):
        return None

    def inference(self, x, hidden, **kwargs):
        # numpy array -> numpy array
        self.eval()
        with torch.no_grad():
            xt = to_torch(x, unsqueeze=0)
            ht = to_torch(hidden, unsqueeze=1)
            outputs = self.forward(xt, ht, **kwargs)

        return tuple(
            [(to_numpy(o).squeeze(0) if o is not None else None) for o in outputs[:-1]] + \
            [map_r(outputs[-1], lambda o: to_numpy(o).squeeze(1)) if outputs[-1] is not None else None]
        )


class Dense(nn.Module):
    def __init__(self, units0, units1, bnunits=0, bias=True):
        super().__init__()
        if bnunits > 0:
            bias = False
        self.dense = nn.Linear(units0, units1, bias=bias)
        self.bnunits = bnunits
        self.bn = nn.BatchNorm1d(bnunits) if bnunits > 0 else None

    def forward(self, x):
        h = self.dense(x)
        if self.bn is not None:
            size = h.size()
            h = h.view(-1, self.bnunits)
            h = self.bn(h)
            h = h.view(*size)
        return h


class MultiHeadAttention(nn.Module):
    # multi head attention for sets
    # https://github.com/akurniawan/pytorch-transformer/blob/master/modules/attention.py
    def __init__(self, in_dim, out_dim, out_heads, relation_dim=0,
                 residual=False, projection=True, layer_norm=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.out_heads = out_heads
        self.relation_dim = relation_dim
        assert self.out_dim % self.out_heads == 0
        self.query_layer = nn.Linear(self.in_dim + self.relation_dim, self.out_dim, bias=False)
        self.key_layer = nn.Linear(self.in_dim + self.relation_dim, self.out_dim, bias=False)
        self.value_layer = nn.Linear(self.in_dim, self.out_dim, bias=False)
        self.residual = residual
        self.projection = projection
        if self.projection:
            self.proj_layer = nn.Linear(self.out_dim, self.out_dim)
        self.layer_norm = layer_norm
        if self.layer_norm:
            self.ln = nn.LayerNorm(self.out_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.query_layer.weight, -0.1, 0.1)
        nn.init.uniform_(self.key_layer.weight, -0.1, 0.1)
        nn.init.uniform_(self.value_layer.weight, -0.1, 0.1)
        if self.projection:
            nn.init.uniform_(self.proj_layer.weight, -0.1, 0.1)

    def forward(self, query, key, relation=None, mask=None, key_mask=None, distance=None):
        """
        Args:
            query (torch.Tensor): [batch, query_len, in_dim]
            key (torch.Tensor): [batch, key_len, in_dim]
            relation (torch.Tensor): [batch, query_len, key_len, relation_dim]
            mask (torch.Tensor): [batch, query_len]
            key_mask (torch.Tensor): [batch, key_len]
        Returns:
            torch.Tensor: [batch, query_len, out_dim]
        """

        query_len = query.size(-2)
        key_len = key.size(-2)
        head_dim = self.out_dim // self.out_heads

        if key_mask is None:
            if torch.equal(query, key):
                key_mask = mask

        if relation is not None:
            relation = relation.view(-1, query_len, key_len, self.relation_dim)

            query_ = query.view(-1, query_len, 1, self.in_dim).repeat(1, 1, key_len, 1)
            query_ = torch.cat([query_, relation], dim=-1)

            key_ = key.view(-1, 1, key_len, self.in_dim).repeat(1, query_len, 1, 1)
            key_ = torch.cat([key_, relation], dim=-1)

            Q = self.query_layer(query_).view(-1, query_len * key_len, self.out_heads, head_dim)
            K = self.key_layer(key_).view(-1, query_len * key_len, self.out_heads, head_dim)

            Q = Q.transpose(1, 2).contiguous().view(-1, query_len, key_len, head_dim)
            K = K.transpose(1, 2).contiguous().view(-1, query_len, key_len, head_dim)

            attention = (Q * K).sum(dim=-1)
        else:
            Q = self.query_layer(query).view(-1, query_len, self.out_heads, head_dim)
            K = self.key_layer(key).view(-1, key_len, self.out_heads, head_dim)

            Q = Q.transpose(1, 2).contiguous().view(-1, query_len, head_dim)
            K = K.transpose(1, 2).contiguous().view(-1, key_len, head_dim)

            attention = torch.bmm(Q, K.transpose(1, 2))

        if distance is not None:
            attention = attention - torch.log1p(distance.repeat(self.out_heads, 1, 1))
        attention = attention * (float(head_dim) ** -0.5)

        if key_mask is not None:
            attention = attention.view(-1, self.out_heads, query_len, key_len)
            attention = attention + ((1 - key_mask) * -1e32).view(-1, 1, 1, key_len)
        attention = F.softmax(attention, dim=-1)
        if mask is not None:
            attention = attention * mask.view(-1, 1, query_len, 1)
            attention = attention.contiguous().view(-1, query_len, key_len)

        V = self.value_layer(key).view(-1, key_len, self.out_heads, head_dim)
        V = V.transpose(1, 2).contiguous().view(-1, key_len, head_dim)

        output = torch.bmm(attention, V).view(-1, self.out_heads, query_len, head_dim)
        output = output.transpose(1, 2).contiguous().view(*query.size()[:-2], query_len, self.out_dim)

        if self.projection:
            output = self.proj_layer(output)

        if self.residual:
            output = output + query

        if self.layer_norm:
            output = self.ln(output)

        if mask is not None:
            output = output * mask.unsqueeze(-1)
        attention = attention.view(*query.size()[:-2], self.out_heads, query_len, key_len).detach()

        return output, attention


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu'):
        super().__init__()
        self.in_channels, self.out_channels, self.activation = in_channels, out_channels, activation
        self.blocks = nn.Identity()
        self.activate = nn.ReLU()  # activation_func(activation)
        self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        if self.should_apply_shortcut: residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        x = self.activate(x)
        return x

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels


class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = (
            self.kernel_size[0] // 2, self.kernel_size[1] // 2)  # dynamic add padding based on the kernel_size


class ResNetResidualBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.expansion, self.downsampling, self.conv = expansion, downsampling, partial(Conv2dAuto, kernel_size=3,
                                                                                        bias=False)
        self.shortcut = nn.Sequential(
            nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1,
                      stride=self.downsampling, bias=False),
            nn.BatchNorm2d(self.expanded_channels)) if self.should_apply_shortcut else None

    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels


def activation_func(activation):
    return nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()]
    ])[activation]


def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False)
    return nn.Sequential(conv3x3(in_channels, out_channels, *args, **kwargs), nn.BatchNorm2d(out_channels))


class ResNetBasicBlock(ResNetResidualBlock):
    """
    Basic ResNet block composed by two layers of 3x3conv/batchnorm/activation
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.downsampling),
            activation_func(self.activation),
            conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False),
        )


class FootballNet(BaseModel):
    class FootballEncoder(nn.Module):
        def __init__(self, filters):
            super().__init__()
            self.player_embedding = nn.Embedding(32, 5, padding_idx=0)
            self.mode_embedding = nn.Embedding(8, 3, padding_idx=0)
            self.fc_teammate = nn.Linear(23, filters)
            self.fc_opponent = nn.Linear(23, filters)
            self.fc = nn.Linear(filters + 41, filters)

        def forward(self, x):
            bs = x['mode_index'].size(0)
            # scalar features
            m_emb = self.mode_embedding(x['mode_index']).view(bs, -1)
            ball = x['ball']
            s = torch.cat([ball, x['match'], x['distance']['b2o'].view(bs, -1), m_emb], dim=1)

            # player features
            p_emb_self = self.player_embedding(x['player_index']['self'])
            ball_concat_self = ball.view(bs, 1, -1).repeat(1, x['player']['self'].size(1), 1)
            p_self = torch.cat([x['player']['self'], p_emb_self, ball_concat_self], dim=2)

            p_emb_opp = self.player_embedding(x['player_index']['opp'])
            ball_concat_opp = ball.view(bs, 1, -1).repeat(1, x['player']['opp'].size(1), 1)
            p_opp = torch.cat([x['player']['opp'], p_emb_opp, ball_concat_opp], dim=2)

            # encoding linear layer
            p_self = self.fc_teammate(p_self)
            p_opp = self.fc_opponent(p_opp)

            p = F.relu(torch.cat([p_self, p_opp], dim=1))
            s_concat = s.view(bs, 1, -1).repeat(1, p.size(1), 1)
            p = torch.cat([p, x['distance']['p2bo'].view(bs, p.size(1), -1), s_concat], dim=2)

            h = F.relu(self.fc(p))

            # relation
            rel = None  # x['distance']['p2p']
            distance = None  # x['distance']['p2p']

            return h, rel, distance

    class FootballBlock(nn.Module):
        def __init__(self, filters, heads):
            super().__init__()
            self.attention = MultiHeadAttention(filters, filters, heads, relation_dim=0,
                                                residual=True, projection=True)

        def forward(self, x, rel, distance=None):
            h, _ = self.attention(x, x, relation=rel, distance=distance)
            return h

    class FootballControll(nn.Module):
        def __init__(self, filters, final_filters):
            super().__init__()
            self.filters = filters
            self.attention = MultiHeadAttention(filters, filters, 1, residual=False, projection=True)
            # self.fc_control = Dense(filters * 3, final_filters, bnunits=final_filters)
            self.fc_control = Dense(filters * 3, final_filters, bnunits=final_filters)

        def forward(self, x, e, control_flag):
            x_controled = (x * control_flag).sum(dim=1, keepdim=True)
            e_controled = (e * control_flag).sum(dim=1, keepdim=True)

            h, _ = self.attention(x_controled, x)

            h = torch.cat([x_controled, e_controled, h], dim=2).view(x.size(0), -1)
            # h = torch.cat([h, cnn_h.view(cnn_h.size(0), -1)], dim=1)
            h = self.fc_control(h)
            return h

    class FootballHead(nn.Module):
        def __init__(self, filters):
            super().__init__()
            self.head_p = nn.Linear(filters, 19, bias=False)
            self.head_p_special = nn.Linear(filters, 1 + 8 * 4, bias=False)
            self.head_v = nn.Linear(filters, 1, bias=True)
            self.head_r = nn.Linear(filters, 1, bias=False)

        def forward(self, x):
            p = self.head_p(x)
            p2 = self.head_p_special(x)
            v = self.head_v(x)
            r = self.head_r(x)
            return torch.cat([p, p2], -1), v, r

    class CNNModel(nn.Module):
        def __init__(self, final_filters):
            super().__init__()
            self.conv1 = nn.Sequential(
                nn.Conv2d(53, 128, kernel_size=1, stride=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 160, kernel_size=1, stride=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(160, 128, kernel_size=1, stride=1, bias=False),
                nn.ReLU(inplace=True)
            )
            self.pool1 = nn.AdaptiveAvgPool2d((1, 11))
            self.conv2 = nn.Sequential(
                nn.BatchNorm2d(128),
                nn.Conv2d(128, 160, kernel_size=(1, 1), stride=1, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(160),
                nn.Conv2d(160, 96, kernel_size=(1, 1), stride=1, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(96),
                nn.Conv2d(96, final_filters, kernel_size=(1, 1), stride=1, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(final_filters),
            )
            self.pool2 = nn.AdaptiveAvgPool2d((1, 1))
            self.flatten = nn.Flatten()

        def forward(self, x):
            x = x['cnn_feature']
            x = self.conv1(x)
            x = self.pool1(x)
            x = self.conv2(x)
            x = self.pool2(x)
            x = self.flatten(x)
            return x

    class SMMEncoder(nn.Module):
        class SMMBlock(nn.Module):
            def __init__(self, in_filters, out_filters, residuals=2):
                super().__init__()
                self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, bias=False)
                self.pool1 = nn.MaxPool2d(3, stride=2)
                self.blocks = nn.ModuleList([
                    ResNetBasicBlock(out_filters, out_filters)
                    for _ in range(residuals)])

            def forward(self, x):
                h = self.conv1(x)
                h = self.pool1(h)
                for block in self.blocks:
                    h = block(h)
                return h

        def __init__(self, filters):
            super().__init__()
            # 4, 72, 96 => filters, 1, 3
            self.blocks = nn.ModuleList([
                self.SMMBlock(4, filters),
                self.SMMBlock(filters, filters),
                self.SMMBlock(filters, filters),
                self.SMMBlock(filters, filters),
            ])

        def forward(self, x):
            x = x['smm']
            h = x
            for block in self.blocks:
                h = block(h)
            h = F.relu(h)
            return h

    class ActionHistoryEncoder(nn.Module):
        def __init__(self, input_size=19, hidden_size=64, num_layers=2, bidirectional=True):
            super().__init__()
            self.action_emd = nn.Embedding(19, 8)
            self.rnn = nn.GRU(8, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)

        def forward(self, x):
            h = self.action_emd(x['action_history'])
            h = h.squeeze(dim=2)
            self.rnn.flatten_parameters()
            h, _ = self.rnn(h)
            return h

    def __init__(self, env, args={}, action_length=None):
        super().__init__(env, args, action_length)
        blocks = 5
        filters = 96
        final_filters = 128
        smm_filters = 32
        self.encoder = self.FootballEncoder(filters)
        self.blocks = nn.ModuleList([self.FootballBlock(filters, 8) for _ in range(blocks)])
        self.control = self.FootballControll(filters, final_filters)  # to head

        self.cnn = self.CNNModel(final_filters)  # to control
        # self.smm = self.SMMEncoder(smm_filters)  # to control
        rnn_hidden = 64
        self.rnn = self.ActionHistoryEncoder(19, rnn_hidden, 2)

        self.head = self.FootballHead(final_filters + final_filters + rnn_hidden * 2)
        # self.head = self.FootballHead(19, final_filters)

    def init_hidden(self, batch_size=None):
        return None

    def forward(self, x, hidden):
        e, rel, distance = self.encoder(x)
        h = e
        for block in self.blocks:
            h = block(h, rel, distance)
        cnn_h = self.cnn(x)
        # smm_h = self.smm(x)
        # h = self.control(h, e, x['control_flag'], cnn_h, smm_h)
        h = self.control(h, e, x['control_flag'])
        rnn_h = self.rnn(x)

        #         p, v, r = self.head(torch.cat([h,
        #                                        cnn_h.view(cnn_h.size(0), -1),
        #                                        smm_h.view(smm_h.size(0), -1)], axis=-1))

        rnn_h_head_tail = rnn_h[:, 0, :] + rnn_h[:, -1, :]
        rnn_h_plus_stick = torch.cat([rnn_h_head_tail[:, :-4], x['control']], dim=1)
        p, v, r = self.head(torch.cat([h,
                                       cnn_h.view(cnn_h.size(0), -1),
                                       rnn_h_plus_stick,
                                       ], axis=-1))
        # p, v, r = self.head(h)

        return p, torch.tanh(v), torch.tanh(r), hidden
