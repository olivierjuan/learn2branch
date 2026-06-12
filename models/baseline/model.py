import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle


class PreNormException(Exception):
    pass


class PreNormLayer(nn.Module):
    """
    Our pre-normalization layer, whose purpose is to normalize an input layer
    to zero mean and unit variance to speed-up and stabilize GCN training. The
    layer's parameters are aimed to be computed during the pre-training phase.
    """

    def __init__(self, n_units, shift=True, scale=True):
        super().__init__()
        assert shift or scale
        self.n_units = n_units
        self.waiting_updates = False
        self.received_updates = False

        if shift:
            self.register_buffer('shift', torch.zeros(n_units, dtype=torch.float32))
        else:
            self.shift = None

        if scale:
            self.register_buffer('scale', torch.ones(n_units, dtype=torch.float32))
        else:
            self.scale = None

    def build(self, input_shapes):
        self.built = True

    def forward(self, x):
        if self.waiting_updates:
            self.update_stats(x)
            self.received_updates = True
            raise PreNormException

        if self.shift is not None:
            x = x + self.shift

        if self.scale is not None:
            x = x * self.scale

        return x

    def start_updates(self):
        """
        Initializes the pre-training phase.
        """
        self.avg = 0
        self.var = 0
        self.m2 = 0
        self.count = 0
        self.waiting_updates = True
        self.received_updates = False

    def update_stats(self, x):
        """
        Online mean and variance estimation. See: Chan et al. (1979) Updating
        Formulae and a Pairwise Algorithm for Computing Sample Variances.
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
        """
        assert self.n_units == 1 or x.shape[-1] == self.n_units, f"Expected input dimension of size {self.n_units}, got {x.shape[-1]}."

        device = x.device
        x = x.reshape(-1, self.n_units)
        sample_avg = x.mean(dim=0)
        sample_var = ((x - sample_avg) ** 2).mean(dim=0)
        sample_count = torch.tensor(x.numel() / self.n_units, dtype=torch.float32, device=device)

        if not hasattr(self, 'count') or self.count == 0:
            self.avg = torch.zeros(self.n_units, dtype=torch.float32, device=device)
            self.var = torch.zeros(self.n_units, dtype=torch.float32, device=device)
            self.m2 = torch.zeros(self.n_units, dtype=torch.float32, device=device)
            self.count = torch.tensor(0.0, dtype=torch.float32, device=device)

        delta = sample_avg - self.avg

        self.m2 = self.var * self.count + sample_var * sample_count + delta ** 2 * self.count * sample_count / (
                self.count + sample_count)

        self.count += sample_count
        self.avg += delta * sample_count / self.count
        self.var = self.m2 / self.count if self.count > 0 else torch.ones_like(self.var)

    def stop_updates(self):
        """
        Ends pre-training for that layer, and fixes the layers's parameters.        
        """
        assert self.count > 0
        if self.shift is not None:
            self.shift.copy_(-self.avg.to(self.shift.device))
        
        if self.scale is not None:
            var = self.var.to(self.scale.device)
            var = torch.where(var == 0, torch.ones_like(var), var)  # NaN check trick
            self.scale.copy_(1 / torch.sqrt(var))
        
        if hasattr(self, 'avg'): del self.avg
        if hasattr(self, 'var'): del self.var
        if hasattr(self, 'm2'): del self.m2
        if hasattr(self, 'count'): del self.count
        self.waiting_updates = False


class BipartiteGraphConvolution(nn.Module):
    """
    Partial bipartite graph convolution (either left-to-right or right-to-left).
    """

    def __init__(self, emb_size, activation, initializer, right_to_left=False):
        super().__init__()
        self.emb_size = emb_size
        self.activation = activation
        self.initializer = initializer
        self.right_to_left = right_to_left
        self.built = False

    def build(self, input_shapes):
        l_shape, ei_shape, ev_shape, r_shape = input_shapes

        self.feature_module_left = nn.Sequential(
            nn.Linear(l_shape[1], self.emb_size, bias=True)
        )
        self.feature_module_edge = nn.Sequential(
            nn.Linear(ev_shape[1], self.emb_size, bias=False)
        )
        self.feature_module_right = nn.Sequential(
            nn.Linear(r_shape[1], self.emb_size, bias=False)
        )
        self.feature_module_final = nn.Sequential(
            PreNormLayer(1, shift=False),  # normalize after summation trick
            nn.ReLU(),
            nn.Linear(self.emb_size, self.emb_size, bias=True)
        )

        self.post_conv_module = nn.Sequential(
            PreNormLayer(1, shift=False),  # normalize after convolution
        )

        # output_layers
        self.output_module = nn.Sequential(
            nn.Linear(self.emb_size + (l_shape[1] if self.right_to_left else r_shape[1]), self.emb_size, bias=True),
            nn.ReLU(),
            nn.Linear(self.emb_size, self.emb_size, bias=True),
        )

        # Apply orthogonal initialization to all Linear layers
        for module in [self.feature_module_left, self.feature_module_edge, self.feature_module_right, 
                       self.feature_module_final, self.post_conv_module, self.output_module]:
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=1.0)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0.0)

        self.built = True

    def forward(self, inputs, training=None):
        """
        Perfoms a partial graph convolution on the given bipartite graph.
        """
        left_features, edge_indices, edge_features, right_features, scatter_out_size = inputs

        if self.right_to_left:
            scatter_dim = 0
            prev_features = left_features
        else:
            scatter_dim = 1
            prev_features = right_features

        # compute joint features
        joint_features = self.feature_module_final(
            self.feature_module_left(left_features)[edge_indices[0].long()] +
            self.feature_module_edge(edge_features) +
            self.feature_module_right(right_features)[edge_indices[1].long()]
        )

        # perform convolution
        out_size = int(scatter_out_size)
        conv_output = torch.zeros(out_size, self.emb_size, dtype=joint_features.dtype, device=joint_features.device)
        conv_output.index_add_(0, edge_indices[scatter_dim].long(), joint_features)
        
        conv_output = self.post_conv_module(conv_output)

        # apply final module
        output = self.output_module(torch.cat([
            conv_output,
            prev_features,
        ], dim=1))

        return output

    def call(self, *args, **kwargs):
        return self(*args, **kwargs)


class BaseModel(nn.Module):
    """
    Our base model class, which implements basic save/restore and pre-training
    methods.
    """

    def pre_train_init(self):
        self.pre_train_init_rec(self)

    @staticmethod
    def pre_train_init_rec(model):
        for child in model.children():
            if isinstance(child, PreNormLayer):
                child.start_updates()
            else:
                BaseModel.pre_train_init_rec(child)

    def pre_train_next(self):
        return self.pre_train_next_rec(self, self.__class__.__name__)

    @staticmethod
    def pre_train_next_rec(model, name):
        for name_child, child in model.named_children():
            full_name = f"{name}/{name_child}"
            if isinstance(child, PreNormLayer) and child.waiting_updates and child.received_updates:
                child.stop_updates()
                return child, full_name
            else:
                result = BaseModel.pre_train_next_rec(child, full_name)
                if result is not None:
                    return result
        return None

    def pre_train(self, *args, **kwargs):
        try:
            self.forward(*args, **kwargs)
            return False
        except PreNormException:
            return True

    def save_state(self, path):
        torch.save(self.state_dict(), path)

    def restore_state(self, path):
        device = next(self.parameters()).device if any(self.parameters()) else torch.device('cpu')
        self.load_state_dict(torch.load(path, map_location=device))


class GCNPolicy(BaseModel):
    """
    Our bipartite Graph Convolutional neural Network (GCN) model.
    """

    def __init__(self):
        super().__init__()

        self.emb_size = 64
        self.cons_nfeats = 5
        self.edge_nfeats = 1
        self.var_nfeats = 19

        self.activation = torch.nn.functional.relu
        self.built = False

        # CONSTRAINT EMBEDDING
        self.cons_embedding = nn.Sequential(
            PreNormLayer(n_units=self.cons_nfeats),
            nn.Linear(self.cons_nfeats, self.emb_size, bias=True),
            nn.ReLU(),
            nn.Linear(self.emb_size, self.emb_size, bias=True),
            nn.ReLU(),
        )

        # EDGE EMBEDDING
        self.edge_embedding = nn.Sequential(
            PreNormLayer(self.edge_nfeats),
        )

        # VARIABLE EMBEDDING
        self.var_embedding = nn.Sequential(
            PreNormLayer(n_units=self.var_nfeats),
            nn.Linear(self.var_nfeats, self.emb_size, bias=True),
            nn.ReLU(),
            nn.Linear(self.emb_size, self.emb_size, bias=True),
            nn.ReLU(),
        )

        # GRAPH CONVOLUTIONS
        self.conv_v_to_c = BipartiteGraphConvolution(self.emb_size, self.activation, None, right_to_left=True)
        self.conv_c_to_v = BipartiteGraphConvolution(self.emb_size, self.activation, None)

        # OUTPUT
        self.output_module = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size, bias=True),
            nn.ReLU(),
            nn.Linear(self.emb_size, 1, bias=False),
        )

        # build model right-away
        self.build([
            (None, self.cons_nfeats),
            (2, None),
            (None, self.edge_nfeats),
            (None, self.var_nfeats),
            (None, ),
            (None, ),
        ])

        # Dummy signature to match TF1 imports without issues
        self.input_signature = None

    def build(self, input_shapes):
        c_shape, ei_shape, ev_shape, v_shape, nc_shape, nv_shape = input_shapes
        emb_shape = [None, self.emb_size]

        if not self.built:
            self.cons_embedding[0].build(c_shape)
            self.edge_embedding[0].build(ev_shape)
            self.var_embedding[0].build(v_shape)
            self.conv_v_to_c.build((emb_shape, ei_shape, ev_shape, emb_shape))
            self.conv_c_to_v.build((emb_shape, ei_shape, ev_shape, emb_shape))
            
            # Apply orthogonal initialization to all Linear layers
            for module in [self.cons_embedding, self.edge_embedding, self.var_embedding, self.output_module]:
                for layer in module.modules():
                    if isinstance(layer, nn.Linear):
                        nn.init.orthogonal_(layer.weight, gain=1.0)
                        if layer.bias is not None:
                            nn.init.constant_(layer.bias, 0.0)
            self.built = True

    @staticmethod
    def pad_output(output, n_vars_per_sample, pad_value=-1e8):
        if torch.is_tensor(n_vars_per_sample):
            n_vars_list = n_vars_per_sample.cpu().tolist()
        else:
            n_vars_list = list(n_vars_per_sample)

        n_vars_max = max(n_vars_list)

        chunks = torch.split(output, n_vars_list, dim=1)
        padded_chunks = []
        for x in chunks:
            padding_size = n_vars_max - x.shape[1]
            if padding_size > 0:
                padded_x = torch.nn.functional.pad(x, (0, padding_size), value=pad_value)
            else:
                padded_x = x
            padded_chunks.append(padded_x)

        return torch.cat(padded_chunks, dim=0)

    def forward(self, inputs, training=False):
        """
        Accepts stacked mini-batches, i.e. several bipartite graphs aggregated
        as one. In that case the number of variables per samples has to be
        provided, and the output consists in a padded dense tensor.
        """
        constraint_features, edge_indices, edge_features, variable_features, n_cons_per_sample, n_vars_per_sample = inputs
        n_cons_total = torch.sum(n_cons_per_sample)
        n_vars_total = torch.sum(n_vars_per_sample)

        # EMBEDDINGS
        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        # GRAPH CONVOLUTIONS
        constraint_features = self.conv_v_to_c((
            constraint_features, edge_indices, edge_features, variable_features, n_cons_total), training)
        constraint_features = self.activation(constraint_features)

        variable_features = self.conv_c_to_v((
            constraint_features, edge_indices, edge_features, variable_features, n_vars_total), training)
        variable_features = self.activation(variable_features)

        # OUTPUT
        output = self.output_module(variable_features)
        output = output.reshape(1, -1)

        if n_vars_per_sample.shape[0] > 1:
            output = self.pad_output(output, n_vars_per_sample)

        return output

    def call(self, *args, **kwargs):
        return self(*args, **kwargs)
