import mesh_tensorflow as mtf
import tensorflow.compat.v1 as tf
import random

BASE_FNS = {'gelu': mtf.gelu,
            'relu': mtf.relu,
            'sigmoid': mtf.sigmoid,
            'tanh': mtf.tanh,
            'selu': mtf.selu,
            'elu': mtf.elu,
            'abs': mtf.abs,
            'sin': mtf.sin,
            'cos': mtf.cos,
            'sign': mtf.sign,
            'silu': mtf.swish,
            'softplus': mtf.softplus
            }


def _arcsinh(x):
    return mtf.log(x + mtf.sqrt(1 + x ** 2))


def _var(x, init):
    return mtf.get_variable(x.mesh, f"activation-{random.randint(0, 2 ** 32):x}", [],
                            initializer=tf.constant_initializer(init), dtype=x.dtype)


def _pos_var(x, val):
    return mtf.softplus(_var(x, 0)) + val


def _rrelu(x):
    negative_scale = random.random()
    return (negative_scale * mtf.abs(x) + x) / (1 + negative_scale)


def _elish(x):
    cond = mtf.cast(mtf.greater(x, 0), x.dtype)
    exp = mtf.exp(x)
    return cond * x / (1 + exp) + (1 - cond) * (exp - 1) / (1 / exp + 1)


CUSTOM_FNS = {'lrelu001': lambda x: mtf.leaky_relu(x, alpha=0.01),
              'lrelu020': lambda x: mtf.leaky_relu(x, alpha=0.20),
              'id': lambda x: x,
              'triangle_relax': lambda x: mtf.sin(x) - mtf.sin(3 * x) / 9 + mtf.sin(5 * x) / 25 - mtf.sin(7 * x) / 49,
              'square_relax': lambda x: mtf.cos(x) - mtf.cos(3 * x) / 3 + mtf.cos(5 * x) / 5 - mtf.cos(7 * x) / 7,
              'spike': lambda x: 1 / (1 + x ** 2),
              'spike2': lambda x: mtf.exp(-x ** 2),
              'tanhshrink': lambda x: x - tanh(x),
              'softsign': lambda x: x / (mtf.abs(x) + 1),
              'softmax': lambda x: mtf.softmax(x, x.shape[-1]),
              'logsoftmax': lambda x: mtf.log_softmax(x, x.shape[-1]),
              'bipolarsigmoid': lambda x: mtf.sigmoid(x) * 2 - 1,
              'rrelu': _rrelu,
              'elish': _elish,
              'arcsinh': _arcsinh,
              'aria': lambda x: x * (_var(x, 0) + _var(x, 1) / (
                          _pos_var(x, 0) + _var(x, 1) * mtf.exp(_var(x, -1) * x) ** (1 / _pos_var(x, 1)))),
              'prelu': lambda x: mtf.leaky_relu(x, alpha=_var(x, 0.2)),
              'parcsinh': lambda x: _var(x, 1) * _arcsinh(x * _pos_var(x, 1)),
              'psoftplus': lambda x: _var(x, 1) * mtf.softplus(x * _var(x, 1)) + _var(x, 0),
              'proottanh': lambda x: (x ** _pos_var(x, 2) + _pos_var(x, 1)) ** (1 / _pos_var(x, 3)) * mtf.tanh(x),
              'maxsig': lambda x: mtf.maximum(x, mtf.sigmoid(x)),
              'cosid': lambda x: mtf.cos(x) - x,
              'minsin': lambda x: mtf.minimum(x, mtf.sin(x)),
              'maxtanh': lambda x: mtf.maximum(x, mtf.tanh(x)),
              'mish': lambda x: x * mtf.tanh(mtf.softplus(x)),
              'tanhexp': lambda x: x * mtf.tanh(mtf.exp(x)),
              'lisht': lambda x: x * mtf.tanh(x),
              'seagull': lambda x: mtf.log(1 + x ** 2),
              'snake': lambda x: x + mtf.sin(x) ** 2,
              'roottanh': lambda x: (x ** 2 + 1) ** (1 / 3) * mtf.tanh(x),
              'softplusmone': lambda x: mtf.softplus(x) - 1
              }


def get_activation_fn(params):
    if "activation_fn" in params:
        activation_fn = params["activation_fn"]
    else:
        print("Defaulting to GELU activation (see here: https://arxiv.org/abs/1606.08415)")
        activation_fn = "gelu"

    if activation_fn in BASE_FNS:
        return BASE_FNS[activation_fn]

    if activation_fn in CUSTOM_FNS:
        return CUSTOM_FNS[activation_fn]

    raise ValueError('unknown activation function "activation_fn" in config')



