import math
import torch
import torch.nn as nn
from hydra.utils import instantiate
from rl_games.algos_torch.network_builder import A2CBuilder, NetworkBuilder


class DictObsNetwork(A2CBuilder.Network):
    def __init__(self, params, **kwargs):
        actions_num = kwargs.pop("actions_num")
        self.value_size = kwargs.pop("value_size", 1)
        self.num_seqs = kwargs.pop("num_seqs", 1)

        NetworkBuilder.BaseNetwork.__init__(self)
        self.load(params)
        self.actor_cnn = nn.Sequential()
        self.critic_cnn = nn.Sequential()
        self.actor_mlp = nn.Sequential()
        self.critic_mlp = nn.Sequential()

        self.dict_feature_encoder = instantiate(params["dict_feature_encoder"])
        mlp_input_shape = self.dict_feature_encoder.output_dim

        in_mlp_shape = mlp_input_shape
        if len(self.units) == 0:
            out_size = mlp_input_shape
        else:
            out_size = self.units[-1]

        if self.has_rnn:
            if not self.is_rnn_before_mlp:
                rnn_in_size = out_size
                out_size = self.rnn_units
                if self.rnn_concat_input:
                    rnn_in_size += in_mlp_shape
            else:
                rnn_in_size = in_mlp_shape
                in_mlp_shape = self.rnn_units

            if self.separate:
                self.a_rnn = self._build_rnn(
                    self.rnn_name, rnn_in_size, self.rnn_units, self.rnn_layers
                )
                self.c_rnn = self._build_rnn(
                    self.rnn_name, rnn_in_size, self.rnn_units, self.rnn_layers
                )
                if self.rnn_ln:
                    self.a_layer_norm = torch.nn.LayerNorm(self.rnn_units)
                    self.c_layer_norm = torch.nn.LayerNorm(self.rnn_units)
            else:
                self.rnn = self._build_rnn(
                    self.rnn_name, rnn_in_size, self.rnn_units, self.rnn_layers
                )
                if self.rnn_ln:
                    self.layer_norm = torch.nn.LayerNorm(self.rnn_units)

        mlp_args = {
            "input_size": in_mlp_shape,
            "units": self.units,
            "activation": self.activation,
            "norm_func_name": self.normalization,
            "dense_func": torch.nn.Linear,
            "d2rl": self.is_d2rl,
            "norm_only_first_layer": self.norm_only_first_layer,
        }
        self.actor_mlp = self._build_mlp(**mlp_args)
        if self.separate:
            self.critic_mlp = self._build_mlp(**mlp_args)

        self.value = self._build_value_layer(out_size, self.value_size)
        self.value_act = self.activations_factory.create(self.value_activation)

        if self.is_discrete:
            self.logits = torch.nn.Linear(out_size, actions_num)
        """
            for multidiscrete actions num is a tuple
        """
        if self.is_multi_discrete:
            self.logits = torch.nn.ModuleList(
                [torch.nn.Linear(out_size, num) for num in actions_num]
            )
        if self.is_continuous:
            self.mu = torch.nn.Linear(out_size, actions_num)
            self.mu_act = self.activations_factory.create(
                self.space_config["mu_activation"]
            )
            mu_init = self.init_factory.create(**self.space_config["mu_init"])
            self.sigma_act = self.activations_factory.create(
                self.space_config["sigma_activation"]
            )
            sigma_init = self.init_factory.create(**self.space_config["sigma_init"])

            if self.fixed_sigma:
                self.sigma = nn.Parameter(
                    torch.zeros(actions_num, requires_grad=True, dtype=torch.float32),
                    requires_grad=True,
                )
            else:
                self.sigma = torch.nn.Linear(out_size, actions_num)

        mlp_init = self.init_factory.create(**self.initializer)
        if self.has_cnn:
            cnn_init = self.init_factory.create(**self.cnn["initializer"])

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                cnn_init(m.weight)
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                mlp_init(m.weight)
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.zeros_(m.bias)

        if self.is_continuous:
            mu_init(self.mu.weight)
            if self.fixed_sigma:
                sigma_init(self.sigma)
            else:
                sigma_init(self.sigma.weight)

    def forward(self, obs_dict):
        obs = obs_dict["obs"]
        states = obs_dict.get("rnn_states", None)
        dones = obs_dict.get("dones", None)
        bptt_len = obs_dict.get("bptt_len", 0)

        obs = self.dict_feature_encoder(obs)

        if self.separate:
            a_out = c_out = obs
            a_out = self.actor_cnn(a_out)
            a_out = a_out.contiguous().view(a_out.size(0), -1)

            c_out = self.critic_cnn(c_out)
            c_out = c_out.contiguous().view(c_out.size(0), -1)

            if self.has_rnn:
                seq_length = obs_dict.get("seq_length", 1)

                if not self.is_rnn_before_mlp:
                    a_out_in = a_out
                    c_out_in = c_out
                    a_out = self.actor_mlp(a_out_in)
                    c_out = self.critic_mlp(c_out_in)

                    if self.rnn_concat_input:
                        a_out = torch.cat([a_out, a_out_in], dim=1)
                        c_out = torch.cat([c_out, c_out_in], dim=1)

                batch_size = a_out.size()[0]
                num_seqs = batch_size // seq_length
                a_out = a_out.reshape(num_seqs, seq_length, -1)
                c_out = c_out.reshape(num_seqs, seq_length, -1)

                a_out = a_out.transpose(0, 1)
                c_out = c_out.transpose(0, 1)
                if dones is not None:
                    dones = dones.reshape(num_seqs, seq_length, -1)
                    dones = dones.transpose(0, 1)

                if len(states) == 2:
                    a_states = states[0]
                    c_states = states[1]
                else:
                    a_states = states[:2]
                    c_states = states[2:]
                a_out, a_states = self.a_rnn(a_out, a_states, dones, bptt_len)
                c_out, c_states = self.c_rnn(c_out, c_states, dones, bptt_len)

                a_out = a_out.transpose(0, 1)
                c_out = c_out.transpose(0, 1)
                a_out = a_out.contiguous().reshape(
                    a_out.size()[0] * a_out.size()[1], -1
                )
                c_out = c_out.contiguous().reshape(
                    c_out.size()[0] * c_out.size()[1], -1
                )

                if self.rnn_ln:
                    a_out = self.a_layer_norm(a_out)
                    c_out = self.c_layer_norm(c_out)

                if type(a_states) is not tuple:
                    a_states = (a_states,)
                    c_states = (c_states,)
                states = a_states + c_states

                if self.is_rnn_before_mlp:
                    a_out = self.actor_mlp(a_out)
                    c_out = self.critic_mlp(c_out)
            else:
                a_out = self.actor_mlp(a_out)
                c_out = self.critic_mlp(c_out)

            value = self.value_act(self.value(c_out))

            if self.is_discrete:
                logits = self.logits(a_out)
                return logits, value, states

            if self.is_multi_discrete:
                logits = [logit(a_out) for logit in self.logits]
                return logits, value, states

            if self.is_continuous:
                mu = self.mu_act(self.mu(a_out))
                if self.fixed_sigma:
                    sigma = mu * 0.0 + self.sigma_act(self.sigma)
                else:
                    sigma = self.sigma_act(self.sigma(a_out))

                return mu, sigma, value, states
        else:
            out = obs
            out = self.actor_cnn(out)
            out = out.flatten(1)

            if self.has_rnn:
                seq_length = obs_dict.get("seq_length", 1)

                out_in = out
                if not self.is_rnn_before_mlp:
                    out_in = out
                    out = self.actor_mlp(out)
                    if self.rnn_concat_input:
                        out = torch.cat([out, out_in], dim=1)

                batch_size = out.size()[0]
                num_seqs = batch_size // seq_length
                out = out.reshape(num_seqs, seq_length, -1)

                if len(states) == 1:
                    states = states[0]

                out = out.transpose(0, 1)
                if dones is not None:
                    dones = dones.reshape(num_seqs, seq_length, -1)
                    dones = dones.transpose(0, 1)
                out, states = self.rnn(out, states, dones, bptt_len)
                out = out.transpose(0, 1)
                out = out.contiguous().reshape(out.size()[0] * out.size()[1], -1)

                if self.rnn_ln:
                    out = self.layer_norm(out)
                if self.is_rnn_before_mlp:
                    out = self.actor_mlp(out)
                if type(states) is not tuple:
                    states = (states,)
            else:
                out = self.actor_mlp(out)
            value = self.value_act(self.value(out))

            if self.central_value:
                return value, states

            if self.is_discrete:
                logits = self.logits(out)
                return logits, value, states
            if self.is_multi_discrete:
                logits = [logit(out) for logit in self.logits]
                return logits, value, states
            if self.is_continuous:
                mu = self.mu_act(self.mu(out))
                if self.fixed_sigma:
                    sigma = self.sigma_act(self.sigma)
                else:
                    sigma = self.sigma_act(self.sigma(out))
                return mu, mu * 0 + sigma, value, states

    def re_initialize_mu_gripper(self):
        # assume the last dim of self.mu.weight corresponds to gripper action
        nn.init.kaiming_uniform_(self.mu.weight[-1:, :], a=math.sqrt(5))
        if self.mu.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.mu.weight[-1:, :])
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.mu.bias[-1:], -bound, bound)


class DictObsBuilder(A2CBuilder):
    def build(self, name, **kwargs):
        net = DictObsNetwork(self.params, **kwargs)
        return net
