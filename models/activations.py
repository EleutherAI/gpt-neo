import mesh_tensorflow as mtf
import tensorflow.compat.v1 as tf
import random


def get_activation_fn(params):
    activation_fn = params.get("activation_fn", "gelu")

    def _arcsinh(x):
        return mtf.log(x + mtf.sqrt(1 + x ** 2))

    def _var(x, init):
        return mtf.get_variable(x.mesh, f"activation-{random.randint(0, 2 ** 32):x}", [],
                                initializer=tf.constant_initializer(init), dtype=x.dtype)

    def _pos_var(x, val):
        return mtf.softplus(_var(x, 0)) + val

    if activation_fn == "gelu":  # https://arxiv.org/abs/1606.08415
        return mtf.gelu
    elif activation_fn == "relu":
        return mtf.relu
    elif activation_fn == "sigmoid":
        return mtf.sigmoid
    elif activation_fn == "tanh":
        return mtf.tanh
    elif activation_fn == "selu":  # https://arxiv.org/abs/1706.02515
        return mtf.selu
    elif activation_fn == "elu":  # https://arxiv.org/abs/1511.07289
        return mtf.elu
    elif activation_fn == "lrelu001":
        return lambda x: mtf.leaky_relu(x, alpha=0.01)
    elif activation_fn == "lrelu020":
        return lambda x: mtf.leaky_relu(x, alpha=0.20)

    elif activation_fn == "abs":
        return mtf.abs
    elif activation_fn == "id":
        return lambda x: x
    elif activation_fn == "sin":
        return mtf.sin
    elif activation_fn == "cos":
        return mtf.cos
    elif activation_fn == "sign":
        return mtf.sign
    elif activation_fn == "triangle_relax":
        return lambda x: mtf.sin(x) - mtf.sin(3 * x) / 9 + mtf.sin(5 * x) / 25 - mtf.sin(7 * x) / 49
    elif activation_fn == "square_relax":
        return lambda x: mtf.cos(x) - mtf.cos(3 * x) / 3 + mtf.cos(5 * x) / 5 - mtf.cos(7 * x) / 7
    elif activation_fn == "spike":
        return lambda x: 1 / (1 + x ** 2)
    elif activation_fn == "spike2":
        return lambda x: mtf.exp(-x ** 2)

    elif activation_fn == "tanhshrink":
        return lambda x: x - tanh(x)
    elif activation_fn == "softsign":
        return lambda x: x / (mtf.abs(x) + 1)
    elif activation_fn == "softmax":
        return lambda x: mtf.softmax(x, x.shape[-1])
    elif activation_fn == "logsoftmax":
        return lambda x: mtf.log_softmax(x, x.shape[-1])
    elif activation_fn == "bipolarsigmoid":
        return lambda x: mtf.sigmoid(x) * 2 - 1
    elif activation_fn == "rrelu":  # https://arxiv.org/abs/1505.00853
        def _rrelu_fn(x):
            negative_scale = random.random()
            return (negative_scale * mtf.abs(x) + x) / (1 + negative_scale)

        return _rrelu_fn
    elif activation_fn == "elish":  # https://arxiv.org/abs/1808.00783v1
        def _elish_fn(x):
            cond = mtf.cast(mtf.greater(x, 0), x.dtype)
            exp = mtf.exp(x)
            return cond * x / (1 + exp) + (1 - cond) * (exp - 1) / (1 / exp + 1)

        return _elish_fn

    elif activation_fn == "silu":  # https://arxiv.org/abs/1710.05941
        return mtf.swish

    elif activation_fn == "arcsinh":
        return _arcsinh


    # parametric
    elif activation_fn == "aria":  # https://arxiv.org/abs/1805.08878
        return lambda x: x * (_var(x, 0) + _var(x, 1) / (
                _pos_var(x, 0) + _var(x, 1) * mtf.exp(_var(x, -1) * x) ** (1 / _pos_var(x, 1))))
    elif activation_fn == "prelu":  # https://arxiv.org/abs/1502.01852
        return lambda x: mtf.leaky_relu(x, alpha=_var(x, 0.2))
    elif activation_fn == "parcsinh":
        return lambda x: _var(x, 1) * _arcsinh(x * _pos_var(x, 1))
    elif activation_fn == "psoftplus":
        return lambda x: _var(x, 1) * mtf.softplus(x * _var(x, 1)) + _var(x, 0)
    elif activation_fn == "proottanh":
        return lambda x: (x ** _pos_var(x, 2) + _pos_var(x, 1)) ** (1 / _pos_var(x, 3)) * mtf.tanh(x)

    # https://arxiv.org/abs/1710.05941, https://arxiv.org/abs/1901.02671
    elif activation_fn == "maxsig":
        return lambda x: mtf.maximum(x, mtf.sigmoid(x))
    elif activation_fn == "cosid":
        return lambda x: mtf.cos(x) - x
    elif activation_fn == "minsin":
        return lambda x: mtf.minimum(x, mtf.sin(x))
    elif activation_fn == "maxtanh":
        return lambda x: mtf.maximum(x, mtf.tanh(x))

    elif activation_fn == "softplus":
        return mtf.softplus
    elif activation_fn == "mish":  # https://arxiv.org/abs/1908.08681
        return lambda x: x * mtf.tanh(mtf.softplus(x))
    elif activation_fn == "tanhexp":  # https://arxiv.org/abs/2003.09855
        return lambda x: x * mtf.tanh(mtf.exp(x))
    elif activation_fn == "lisht":  # https://arxiv.org/abs/1901.05894
        return lambda x: x * mtf.tanh(x)
    elif activation_fn == "seagull":  # https://arxiv.org/abs/2011.11713
        return lambda x: mtf.log(1 + x ** 2)
    elif activation_fn == "snake":  # https://arxiv.org/abs/2006.08195
        return lambda x: x + mtf.sin(x) ** 2

    elif activation_fn == "roottanh":  # made up
        return lambda x: (x ** 2 + 1) ** (1 / 3) * mtf.tanh(x)
    elif activation_fn == "softplusmone":  # made up
        return lambda x: mtf.softplus(x) - 1

    else:
        raise ValueError('unknown activation function "activation_fn" in config')
