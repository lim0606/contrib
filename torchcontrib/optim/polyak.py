from collections import defaultdict
from itertools import chain
from torch.optim import Optimizer
import torch
import warnings


class Polyak(Optimizer):
    def __init__(self, optimizer, polyak_start=None, polyak_freq=None, polyak_lr=None, polyak_decay=1.0):
        r"""Implements Stochastic Weight Averaging (Polyak).

        Stochastic Weight Averaging was proposed in `Averaging Weights Leads to
        Wider Optima and Better Generalization`_ by Pavel Izmailov, Dmitrii
        Podoprikhin, Timur Garipov, Dmitry Vetrov and Andrew Gordon Wilson
        (UAI 2018).

        Polyak is implemented as a wrapper class taking optimizer instance as input
        and applying Polyak on top of that optimizer.

        Polyak can be used in two modes: automatic and manual. In the automatic
        mode Polyak running averages are automatically updated every
        :attr:`polyak_freq` steps after :attr:`polyak_start` steps of optimization. If
        :attr:`polyak_lr` is provided, the learning rate of the optimizer is reset
        to :attr:`polyak_lr` at every step starting from :attr:`polyak_start`. To use
        Polyak in automatic mode provide values for both :attr:`polyak_start` and
        :attr:`polyak_freq` arguments.

        Alternatively, in the manual mode, use :meth:`update_polyak` or
        :meth:`update_polyak_group` methods to update the Polyak running averages.

        In the end of training use `swap_buf_sgd` method to set the optimized
        variables to the computed averages.

        Args:
            optimizer (torch.optim.Optimizer): optimizer to use with Polyak
            polyak_start (int): number of steps before starting to apply Polyak in
                automatic mode; if None, manual mode is selected (default: None)
            polyak_freq (int): number of steps between subsequent updates of
                Polyak running averages in automatic mode; if None, manual mode is
                selected (default: None)
            polyak_lr (float): learning rate to use starting from step polyak_start
                in automatic mode; if None, learning rate is not changed
                (default: None)

        Examples:
            >>> # automatic mode
            >>> base_opt = torch.optim.SGD(model.parameters(), lr=0.1)
            >>> opt = torchcontrib.optim.Polyak(
            >>>                 base_opt, polyak_start=10, polyak_freq=5, polyak_lr=0.05, polyak_decay=0.995)
            >>> for _ in range(100):
            >>>     opt.zero_grad()
            >>>     loss_fn(model(input), target).backward()
            >>>     opt.step()
            >>> opt.swap_buf_sgd()
            >>> # manual mode
            >>> opt = torchcontrib.optim.Polyak(base_opt)
            >>> for i in range(100):
            >>>     opt.zero_grad()
            >>>     loss_fn(model(input), target).backward()
            >>>     opt.step()
            >>>     if i > 10 and i % 5 == 0:
            >>>         opt.update_polyak()
            >>> opt.swap_buf_sgd()

        .. note::
            Polyak does not support parameter-specific values of :attr:`polyak_start`,
            :attr:`polyak_freq` or :attr:`polyak_lr`. In automatic mode Polyak uses the
            same :attr:`polyak_start`, :attr:`polyak_freq` and :attr:`polyak_lr` for all
            parameter groups. If needed, use manual mode with
            :meth:`update_polyak_group` to use different update schedules for
            different parameter groups.

        .. note::
            Call :meth:`swap_buf_sgd` in the end of training to use the computed
            running averages.

        .. note::
            If you are using Polyak to optimize the parameters of a Neural Network
            containing Batch Normalization layers, you need to update the
            :attr:`running_mean` and :attr:`running_var` statistics of the
            Batch Normalization module. You can do so by using
            `torchcontrib.optim.polyak.bn_update` utility.

        .. note::
            See the blogpost
            https://pytorch.org/blog/stochastic-weight-averaging-in-pytorch/
            for an extended description of this Polyak implementation.

        .. note::
            The repo https://github.com/izmailovpavel/contrib_polyak_examples
            contains examples of using this Polyak implementation.

        .. _Averaging Weights Leads to Wider Optima and Better Generalization:
            https://arxiv.org/abs/1803.05407
        .. _Improving Consistency-Based Semi-Supervised Learning with Weight
            Averaging:
            https://arxiv.org/abs/1806.05594
        """
        self._auto_mode, (self.polyak_start, self.polyak_freq) = \
            self._check_params(self, polyak_start, polyak_freq)
        self.polyak_lr = polyak_lr
        self.polyak_decay = polyak_decay

        if self._auto_mode:
            if polyak_start < 0:
                raise ValueError("Invalid polyak_start: {}".format(polyak_start))
            if polyak_freq < 1:
                raise ValueError("Invalid polyak_freq: {}".format(polyak_freq))
        else:
            if self.polyak_lr is not None:
                warnings.warn(
                    "Some of polyak_start, polyak_freq is None, ignoring polyak_lr")
            # If not in auto mode make all polyak parameters None
            self.polyak_lr = None
            self.polyak_start = None
            self.polyak_freq = None

        if self.polyak_lr is not None and self.polyak_lr < 0:
            raise ValueError("Invalid Polyak learning rate: {}".format(polyak_lr))

        self.optimizer = optimizer

        self.defaults = self.optimizer.defaults
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.opt_state = self.optimizer.state
        for group in self.param_groups:
            group['n_avg'] = 0
            group['step_counter'] = 0

        self.is_swapped = False

    @staticmethod
    def _check_params(self, polyak_start, polyak_freq):
        params = [polyak_start, polyak_freq]
        params_none = [param is None for param in params]
        if not all(params_none) and any(params_none):
            warnings.warn(
                "Some of polyak_start, polyak_freq is None, ignoring other")
        for i, param in enumerate(params):
            if param is not None and not isinstance(param, int):
                params[i] = int(param)
                warnings.warn("Casting polyak_start, polyak_freq to int")
        return not any(params_none), params

    def _reset_lr_to_polyak(self):
        if self.polyak_lr is None:
            return
        for param_group in self.param_groups:
            if param_group['step_counter'] >= self.polyak_start:
                param_group['lr'] = self.polyak_lr

    def update_polyak_group(self, group):
        r"""Updates the Polyak running averages for the given parameter group.

        Arguments:
            param_group (dict): Specifies for what parameter group Polyak running
                averages should be updated

        Examples:
            >>> # automatic mode
            >>> base_opt = torch.optim.SGD([{'params': [x]},
            >>>             {'params': [y], 'lr': 1e-3}], lr=1e-2, momentum=0.9)
            >>> opt = torchcontrib.optim.Polyak(base_opt)
            >>> for i in range(100):
            >>>     opt.zero_grad()
            >>>     loss_fn(model(input), target).backward()
            >>>     opt.step()
            >>>     if i > 10 and i % 5 == 0:
            >>>         # Update Polyak for the second parameter group
            >>>         opt.update_polyak_group(opt.param_groups[1])
            >>> opt.swap_buf_sgd()
        """
        for p in group['params']:
            param_state = self.state[p]
            if 'polyak_buffer' not in param_state:
                param_state['polyak_buffer'] = torch.zeros_like(p.data)
            buf = param_state['polyak_buffer']
            diff = (p.data - buf) * (1.0 - self.polyak_decay)
            buf.add_(diff)
        group["n_avg"] += 1

    def update_polyak(self):
        r"""Updates the Polyak running averages of all optimized parameters.
        """
        for group in self.param_groups:
            self.update_polyak_group(group)

    def use_buf(self):
        if self.is_swapped:
            pass
        else:
            self.swap_buf_sgd()

    def use_sgd(self):
        if self.is_swapped:
            self.swap_buf_sgd()
        else:
            pass

    def swap_buf_sgd(self):
        r"""Swaps the values of the optimized variables and polyak buffers.

        It's meant to be called in the end of training to use the collected
        polyak running averages. It can also be used to evaluate the running
        averages during training; to continue training `swap_buf_sgd`
        should be called again.
        """
        if self.is_swapped:
            self.is_swapped = False
        elif not self.is_swapped:
            self.is_swapped = True

        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                if 'polyak_buffer' not in param_state:
                    # If polyak wasn't applied we don't swap params
                    #warnings.warn(
                    #    "Polyak wasn't applied to param {}; skipping it".format(p))
                    #continue
                    warnings.warn("Polyak wasn't applied")
                    self.is_swapped = False
                    break
                buf = param_state['polyak_buffer']
                tmp = torch.empty_like(p.data)
                tmp.copy_(p.data)
                p.data.copy_(buf)
                buf.copy_(tmp)

    def step(self, closure=None):
        r"""Performs a single optimization step.

        In automatic mode also updates Polyak running averages.
        """
        self._reset_lr_to_polyak()
        loss = self.optimizer.step(closure)
        for group in self.param_groups:
            group["step_counter"] += 1
            steps = group["step_counter"]
            if self._auto_mode:
                if steps > self.polyak_start and steps % self.polyak_freq == 0:
                    self.update_polyak_group(group)
        return loss

    def state_dict(self):
        r"""Returns the state of Polyak as a :class:`dict`.

        It contains three entries:
            * opt_state - a dict holding current optimization state of the base
                optimizer. Its content differs between optimizer classes.
            * polyak_state - a dict containing current state of Polyak. For each
                optimized variable it contains polyak_buffer keeping the running
                average of the variable
            * param_groups - a dict containing all parameter groups
        """
        opt_state_dict = self.optimizer.state_dict()
        polyak_state = {(id(k) if isinstance(k, torch.Tensor) else k): v
                     for k, v in self.state.items()}
        opt_state = opt_state_dict["state"]
        param_groups = opt_state_dict["param_groups"]
        return {"opt_state": opt_state, "polyak_state": polyak_state,
                "param_groups": param_groups}

    def load_state_dict(self, state_dict):
        r"""Loads the optimizer state.

        Args:
            state_dict (dict): Polyak optimizer state. Should be an object returned
                from a call to `state_dict`.
        """
        polyak_state_dict = {"state": state_dict["polyak_state"],
                          "param_groups": state_dict["param_groups"]}
        opt_state_dict = {"state": state_dict["opt_state"],
                          "param_groups": state_dict["param_groups"]}
        super(Polyak, self).load_state_dict(polyak_state_dict)
        self.optimizer.load_state_dict(opt_state_dict)
        self.opt_state = self.optimizer.state

    def add_param_group(self, param_group):
        r"""Add a param group to the :class:`Optimizer` s `param_groups`.

        This can be useful when fine tuning a pre-trained network as frozen
        layers can be made trainable and added to the :class:`Optimizer` as
        training progresses.

        Args:
            param_group (dict): Specifies what Tensors should be optimized along
            with group specific optimization options.
        """
        param_group['n_avg'] = 0
        param_group['step_counter'] = 0
        self.optimizer.add_param_group(param_group)

    @staticmethod
    def bn_update(loader, model, device=None):
        r"""Updates BatchNorm running_mean, running_var buffers in the model.

        It performs one pass over data in `loader` to estimate the activation
        statistics for BatchNorm layers in the model.

        Args:
            loader (torch.utils.data.DataLoader): dataset loader to compute the
                activation statistics on. Each data batch should be either a
                tensor, or a list/tuple whose first element is a tensor
                containing data.

            model (torch.nn.Module): model for which we seek to update BatchNorm
                statistics.

            device (torch.device, optional): If set, data will be trasferred to
                :attr:`device` before being passed into :attr:`model`.
        """
        if not _check_bn(model):
            return
        was_training = model.training
        model.train()
        momenta = {}
        model.apply(_reset_bn)
        model.apply(lambda module: _get_momenta(module, momenta))
        n = 0
        for input in loader:
            if isinstance(input, (list, tuple)):
                input = input[0]
            b = input.size(0)

            momentum = b / float(n + b)
            for module in momenta.keys():
                module.momentum = momentum

            if device is not None:
                input = input.to(device)

            model(input)
            n += b

        model.apply(lambda module: _set_momenta(module, momenta))
        model.train(was_training)


# BatchNorm utils
def _check_bn_apply(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def _check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn_apply(module, flag))
    return flag[0]


def _reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]
