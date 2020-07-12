# coding=utf-8
# Copyright 2020 The Mesh TensorFlow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A toy model using Mesh TensorFlow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mesh_tensorflow as mtf
import tensorflow.compat.v1 as tf
import os, json, math

from tensorflow.python.data.ops.dataset_ops import Dataset
from tensorflow.python.platform import flags
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import tpu_config  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.tpu import tpu_estimator  # pylint: disable=g-direct-tensorflow-import
from tensorflow_estimator.python.estimator import estimator as estimator_lib
import mesh_tensorflow.transformer as mtf_transformer

FLAGS = flags.FLAGS

tf.flags.DEFINE_integer('batch_size', 64, 'Training batch size.')
tf.flags.DEFINE_integer('sequence_size', 128, 'Sequence Len')
tf.flags.DEFINE_integer('hidden_size', 16, 'Size of each hidden layer.')
tf.flags.DEFINE_integer('num_hidden_layers', 1, 'Number of layers.')
tf.flags.DEFINE_string('master_dtype', 'float32', 'dtype for master vars.')
tf.flags.DEFINE_string('slice_dtype', 'float32', 'dtype for slice vars.')
tf.flags.DEFINE_string('activation_dtype', 'float32', 'dtype for activations.')
tf.flags.DEFINE_string('optimizer', 'Adafactor', 'optimizer (SGD or Adafactor).')
tf.flags.DEFINE_float('lr', 1e-4, 'Learning rate.')
tf.flags.DEFINE_string('mesh_shape', 'all:8', 'mesh shape')
tf.flags.DEFINE_string('layout', '', 'layout rules')
tf.flags.DEFINE_integer('iterations', 500,
                        'Number of iterations per training loop.')
tf.flags.DEFINE_integer('train_steps', 10000, 'max steps')
tf.flags.DEFINE_integer('steps_per_checkpoint', 200, 'steps_per_checkpoint')
tf.flags.DEFINE_string(
    'model_dir',
    default='gs://datasets_storage_1/models/GPTNeo_prettybig',
    help='The directory where the model will be stored.')
tf.flags.DEFINE_string(
    'data_path',
    default='gs://datasets_storage_1/datasets/bundestag',
    help='The directory where the data is stored.')
tf.flags.DEFINE_string('datasets', default='bundestag_*.tfrecords","",10,"random_sample",1.0', help="dataset details")

# need flags for: batch_size, iterations, n_ctx, datasets, data_path
tf.flags.DEFINE_integer('n_ctx', 128, ' ')

# Optimizer settings
tf.flags.DEFINE_float('weight_decay', 0.01, 'weight decay setting for Adam optimizer')  # beta1, beta2, epsilon
tf.flags.DEFINE_float('beta1', 0.9, 'beta1 setting for Adam optimizer')
tf.flags.DEFINE_float('beta2', 0.98, 'beta2 setting for Adam optimizer')
tf.flags.DEFINE_float('epsilon', 1e-9, 'epsilon setting for Adam optimizer')

tf.flags.DEFINE_bool('use_tpu', True, 'use TPU')
tf.flags.DEFINE_string('model_params', 'configs/GPT_NEO_TEST.json', help="path to model config")
# Cloud TPU Cluster Resolvers
tf.flags.DEFINE_string(
    'tpu',
    default=None,
    help='The Cloud TPU to use for training. This should be either the name '
         'used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.')

tf.flags.DEFINE_string(
    'gcp_project',
    default=None,
    help='Project name for the Cloud TPU-enabled project. If not specified, we '
         'will attempt to automatically detect the GCE project from metadata.')

tf.flags.DEFINE_string(
    'tpu_zone',
    default=None,
    help='GCE zone where the Cloud TPU is located in. If not specified, we '
         'will attempt to automatically detect the GCE project from metadata.')


# --------------------------------------------------------------------------------
# INPUT FNS:


def text_dataset(files, stitch, datatype, batch=True):
    dataset = tf.data.Dataset.from_tensor_slices(files)
    dataset = dataset.apply(
        tf.data.experimental.parallel_interleave(tf.data.TFRecordDataset, cycle_length=4, sloppy=False))

    if "sample" in datatype:
        def _parse_function(example_proto):
            features = {
                "hash": tf.VarLenFeature(tf.string),
                "text": tf.VarLenFeature(tf.int64)
            }
            parsed_features = tf.parse_single_example(example_proto, features)
            return parsed_features["text"], parsed_features["text"].dense_shape[0]
    else:
        def _parse_function(example_proto):
            features = {
                "text": tf.VarLenFeature(tf.int64)
            }
            parsed_features = tf.parse_single_example(example_proto, features)
            return parsed_features["text"]  # Assuming the text is not sparse

    dataset = dataset.map(_parse_function, num_parallel_calls=1)

    # Subsample method
    if "sample" in datatype:
        # Since samples can be less than the correct length, and TPUs don't like variable lengths, this function stitches together enough samples
        # to have a text at least 1024 tokens long. For this to work the stitch parameter must be correctly tuned so that
        # stitch * min(characters_in_text) >= amount
        def _stitch_text(x, y):
            x = tf.sparse.to_dense(x)

            def _get_x(i):
                return tf.gather(x[i], tf.range(y[i]))

            out = _get_x(0)
            for i in range(1, stitch):
                out = tf.concat([out, [50256], _get_x(i)], axis=0)  # text1<|endoftext|>text2

            return out

        # Hack-y way to stitch together multiple texts
        dataset = dataset.shuffle(1000 * stitch).batch(stitch, drop_remainder=True).map(_stitch_text,
                                                                                        num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # Sample 1024(+1) tokens from the stitched together text
        if datatype == "random_sample":
            def _sample_text(x):
                s = tf.size(x)
                r = tf.random.uniform([], maxval=s - (FLAGS.n_ctx + 1), dtype=tf.dtypes.int32)
                r1 = tf.range(r, r + FLAGS.n_ctx)
                r2 = tf.range(r + 1, (r + 1) + FLAGS.n_ctx)
                r1 = tf.reshape(r1, [FLAGS.n_ctx])  # Somehow, this makes the compiler happy
                r2 = tf.reshape(r2, [
                    FLAGS.n_ctx])  # TPUs want constant sized input, and these reshapes makes it recognize the shape of the input
                vals1 = tf.gather(x, r1)
                vals2 = tf.gather(x, r2)

                vals1 = tf.reshape(vals1, [FLAGS.n_ctx])
                vals2 = tf.reshape(vals2, [FLAGS.n_ctx])
                return vals1, vals2

        else:
            def _sample_text(x):
                vals1 = x[:FLAGS.n_ctx]
                vals2 = x[1:FLAGS.n_ctx + 1]

                vals1 = tf.reshape(vals1, [FLAGS.n_ctx])
                vals2 = tf.reshape(vals2, [FLAGS.n_ctx])
                return vals1, vals2

        dataset = dataset.map(_sample_text, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if batch:
        dataset = dataset.batch(FLAGS.batch_size, drop_remainder=True).prefetch(FLAGS.iterations * 2)

    dataset = dataset.repeat()

    return dataset


def generic_text(eval=False, dsets=[["bundestag_*.tfrecords", "", 10, "random_sample", 1.0]]):
    # params["datasets"] = [(train glob, eval_glob, stitch, ["random_sample", "sample", "chunk"] weight)]
    i = 0 if not eval else 1
    datasets = [text_dataset(tf.io.gfile.glob(os.path.join(FLAGS.data_path, dataset[i])), stitch=dataset[2],
                             datatype=dataset[3], batch=False)
                for dataset in dsets]
    weights = [dataset[4] for dataset in dsets]

    dataset = tf.data.experimental.sample_from_datasets(datasets, weights=weights)
    dataset = dataset.batch(FLAGS.batch_size, drop_remainder=True).prefetch(FLAGS.iterations * 2)

    return dataset


class TextInput(object):

    def __init__(self):
        self.dsets = [["bundestag_*.tfrecords", "", 10, "random_sample", 1.0]]

    def __call__(self, params):
        dset = generic_text(dsets=self.dsets)
        return dset


# --------------------------------------------------------------------------------
# LAYERS:

sentinel = object()


def expand_tile(value, newdim):
    """Add a new axis of given size."""
    return mtf.broadcast(value,
                         [newdim] + value.shape.dims)  # shape.dims gets us a list which we need in order to concat

def positions_for(tokens: mtf.Tensor, past_length: int, batch_dim: mtf.Dimension):
    nsteps = tokens.shape[1]
    return expand_tile(past_length + mtf.range(tokens.mesh, nsteps, dtype=tf.int32), batch_dim)


def expand_tile_alt(tensors, dim_name='stacked_dim', axis=0):
    """
    given a list of mtf.tensors, and a name for the new Dimension stack them along the chosen axis.

    :param tensors: list of mtf.Tensors
    :param name: str
    :param axis: int
    :return:
    """
    out = mtf.stack(tensors, dim_name=dim_name, axis=axis, name="expand_tile_stack")
    return out


def positions_for_alt(tokens: mtf.Tensor, past_length: int, batch_size: int, dim_name="stacked_dim"):
    nsteps = tokens.shape[1]
    r = past_length + mtf.range(tokens.mesh, nsteps, dtype=tf.int32)
    rs = []
    for i in range(batch_size):
        rs.append(r)
    return expand_tile_alt(rs, dim_name=dim_name)


def gelu(x):
    # return 0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))
    return mtf.gelu(x)


def norm(x, scope, *, axis=sentinel, epsilon=1e-5, params=None):
    """Normalize to mean = 0, std = 1, then do a diagonal affine transform."""
    if axis is sentinel:
        axis = x.shape[-1]

    with tf.variable_scope(scope):
        n_state = x.shape[-1]

        # assuming we never do fp16 training, only bf16 or fp32. change if we someday do GPU training
        dt = tf.float32

        g = mtf.get_variable(x.mesh, 'g', [n_state], initializer=tf.constant_initializer(1, dtype=dt), dtype=dt)
        b = mtf.get_variable(x.mesh, 'b', [n_state], initializer=tf.constant_initializer(0, dtype=dt), dtype=dt)

        u = mtf.reduce_mean(x, reduced_dim=axis, name="norm_reduce_mean_u")
        s = mtf.reduce_mean(mtf.square(x - u), reduced_dim=axis, name="norm_reduce_mean_s")

        u = mtf.broadcast(u, x.shape)
        s = mtf.broadcast(s, x.shape)

        x = (x - u) * mtf.rsqrt(s + epsilon)
        x = x * g + b
        return x


# TODO: this isnt actually a convolution, rename it to something more appropriate
def conv1d(x, scope, nf, *, w_init_stdev=0.02, params=None, scale=False):
    # nf = number of features

    if params[
        "scale_by_depth"] and scale:  # Scale by sqrt(num_layers), only happens at the final projection before a res block output
        w_init_stdev = w_init_stdev * (1. / math.sqrt(params["n_layer"]))
    if params["scale_by_in"]:  # Scale by sqrt(num_input_features)
        w_init_stdev = w_init_stdev * (1. / math.sqrt(x.shape[-1].size))  # Dimension is a namedtuple of (name, size)

    # assuming we never do fp16 training, only bf16 or fp32. change if we someday do GPU training
    dt = tf.float32

    # TODO: verify that this is actually right

    # rename the channels dim so we dont get a collision
    x = mtf.reshape(x, x.shape.rename_dimension(x.shape[-1].name, 'tmp_channels'))

    # not in the variable_scope because mtf already has a variable_scope in it
    c = mtf.layers.conv1d(x, nf, name=scope, filter_size=1, stride=1,
                          filter_initializer=tf.random_normal_initializer(stddev=w_init_stdev, dtype=dt))
    with tf.variable_scope(scope):

        b = mtf.get_variable(x.mesh, 'b', [nf], initializer=tf.constant_initializer(0, dtype=tf.bfloat16), dtype=dt)
        # NWC

        b = mtf.broadcast(b, c.shape)

        c += b
        return c


def visible_pos(mesh, nd, ns):
    """1's in the lower triangle, counting from the lower right corner.

    Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.

    UPDATE: modified for mtf
    """
    i = mtf.range(mesh, nd, tf.int32)[:, None]
    j = mtf.range(mesh, ns, tf.int32)
    m = i >= j - ns + nd
    return m


def attn(x, scope, n_state, *, past, params, block_offset=0, train=False):
    # n_state is the same as config['n_embd'], which is also the same as dim_embd.
    assert x.shape.ndims == 3  # Should be [batch, sequence, features]
    assert n_state.size % params["n_head"] == 0
    if past is not None:
        assert past.shape.ndims == 5  # Should be [batch, 2, heads, sequence, features], where 2 is [k, v]

    # TODO: implement proper past cache. in the meantime, don't pass a past if implementing local attention!!!
    # update: *shouldn't* be a problem anymore not that we've switched to meshtf, but tpus don't support pasts apparantly (?? - ask Daj).
    # we can remove this assert once we're sure we didnt break anything
    # assert not (params["local"] and past is not None)
    assert past is None

    # x :: [batch, seq, n_embd]
    x_shape = x.shape
    dim_batch = x_shape[0]
    dim_seq = x_shape[1]
    dim_embd = x_shape[2]

    dim_heads = mtf.Dimension("heads", params['n_head'])
    dim_features_per_head_key = mtf.Dimension("features_per_head_key", params['n_embd'] // params['n_head'])
    dim_features_per_head_value = mtf.Dimension("features_per_head_value", params['n_embd'] // params['n_head'])
    print("features_per_head", params['n_embd'] // params['n_head'])

    # input length is past seq + x seq because when sampling, subsequent x is only length 1
    # no longer needed in mtf because TPUs cant handle pasts anyways, apparently
    # inp_len = dim_seq + (tf.shape(past)[3] if past is not None else 0)

    def split_heads(x, last_dim):
        print('split heads shape', x.shape)
        # From [batch, sequence, features] to [batch, heads, sequence, features_per_head]
        # heads is split out of features!
        x = mtf.reshape(x, [dim_batch, dim_seq, dim_heads, last_dim], name="split_heads_reshape")
        x = mtf.transpose(x, [dim_batch, dim_heads, dim_seq, last_dim], name="split_heads_transpose")
        return x

    def merge_heads(x):
        print('merge heads shape', x.shape)
        # Reverse of split_heads
        # from [batch, heads, sequence, features_per_head] to [batch, sequence, features_per_head]
        x = mtf.transpose(x, [dim_batch, dim_seq, dim_heads, dim_features_per_head_value], name="merge_heads_transpose")
        x = mtf.reshape(x, [dim_batch, dim_seq, dim_embd], name="merge_heads_reshape")
        return x

    # the old mask_attn_weights applied directly to the QK; this returns a bias that the attention code from mtf adds to the attention matrix.
    def biasmask_attn_weights(mesh, dtype):
        # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.

        # n_src and n_dest are both the same, i.e equal to sequence length
        ns = dim_seq
        nd = dim_seq

        vis = visible_pos(mesh, nd, ns)
        # TODO: am I doing this right? trying to get to [1, 1, nd, ns]. not sure if a singleton dimension object is the right way.
        # and I'm assuming it gets broadcasted from there to [batch, heads, seq, seq]?

        vis = mtf.broadcast(vis, [dim_batch, dim_heads, nd, ns])
        return mtf_transformer.attention.visibility_mask_to_attention_bias(vis, dtype)

    with tf.variable_scope(scope):
        dim_qkv = mtf.Dimension("qkv", n_state.size * 3)
        c = conv1d(x, 'c_attn', dim_qkv, params=params)

        conv_output_channels = c.shape[2]  # should be equal to dim_qkv
        q, k, v = mtf.split(c, conv_output_channels, 3)
        q, k, v = split_heads(q, dim_features_per_head_key), split_heads(k, dim_features_per_head_key), split_heads(v, dim_features_per_head_value)

        # this is the "2" dim in pasts. probably presents are not needed until we get the pasts stuff working.
        present = mtf.stack([mtf.reshape(x, k.shape.rename_dimension('features_per_head_key', 'features_per_head_value')), v], "kv", axis=1, name="stack_presents_attn")

        if past is not None:
            # TODO: convert this code to mtf. Not neccessary until we start optimizing sampling.
            pk, pv = tf.unstack(past, axis=1)
            k = tf.concat([pk, k], axis=-2)
            v = tf.concat([pv, v], axis=-2)

        # TODO: control whether layer is local on a layer-by-layer basis, not as a global.
        if params["local"]:
            # `local_attention_1d` has built in autoregressive masking, so we don't need mask_attn_weights.
            a = mtf_transformer.attention.local_attention_1d(
                q, k, v,
                length_dim=dim_seq,
                key_dim=dim_features_per_head_key,
                value_dim=dim_features_per_head_value,
                length_dim_num_splits=1,
                attention_kwargs={}
                # mtf argument here should be **kwargs but is just kwargs! so we have to actually give a dict
                # TODO: we might need to split along length dimension at some point, when we do we'll need to wire this up as a param
            )

        else:
            print('qkv shape', q.shape, k.shape, v.shape)

            # HOWEVER, `attention` DOES NOT implement masking so we need to pass in `bias` on our own!
            a = mtf_transformer.attention.attention(
                q, k, v,
                memory_length_dim=dim_seq,
                key_dim=dim_features_per_head_key,
                value_dim=dim_features_per_head_value,
                bias=biasmask_attn_weights(q.mesh, q.dtype)
            )

        a = merge_heads(a)
        a = conv1d(a, 'c_proj', dim_embd, params=params)
        a = mtf.dropout(a, params["res_dropout"], name="attn_dropout")

        return a, present


def mlp(x, scope, n_state, *, params, train=False):
    with tf.variable_scope(scope):
        # TODO: nx will probably be the only thing that needs changing here
        # TODO: also n_state needs to be a Dimension. probably best if we standardize and make whatever calls this provide a Dimension in the first place.
        nx = x.shape[-1]
        h = gelu(conv1d(x, 'c_fc', n_state, params=params))
        h2 = conv1d(h, 'c_proj', nx, params=params, scale=True)
        h2 = mtf.dropout(h2, params["res_dropout"], name="mlp_dropout")
        return h2


def block(x, scope, *, past, params, train=False, block_offset=0):
    with tf.variable_scope(scope):
        nx = x.shape[-1]
        a, present = attn(norm(x, 'ln_1', params=params), 'attn', nx, past=past, params=params,
                          block_offset=block_offset)
        x = x + a

        dim_intermediate_expanded = mtf.Dimension('intermediate_expanded', nx.size * 4)
        m = mlp(norm(x, 'ln_2', params=params), 'mlp', dim_intermediate_expanded, params=params, train=train)
        x = x + m
        return x, present


def toy_model(features, labels, params, mesh, past=None):
    """A toy model implemented by mesh tensorlfow."""
    print('input details:')
    print(features.shape)
    results = {}

    # define master dtypes
    master_dtype = tf.as_dtype(FLAGS.master_dtype)
    slice_dtype = tf.as_dtype(FLAGS.slice_dtype)
    activation_dtype = tf.as_dtype(FLAGS.activation_dtype)

    # define mtf dims
    batch_dim = mtf.Dimension('batch', FLAGS.batch_size)
    sequence_dim = mtf.Dimension('sequence', FLAGS.sequence_size)
    embd_dim = mtf.Dimension("embd", params["n_embd"])
    vocab_dim = mtf.Dimension("vocab", params["n_vocab"])

    # convert input tensor to mtf tensor
    x = mtf.import_tf_tensor(mesh, features, mtf.Shape([batch_dim, sequence_dim]))
    #x = mtf.cast(x, activation_dtype)  # TODO: is this necessary

    wpe = mtf.get_variable(mesh, 'wpe', mtf.Shape([sequence_dim, embd_dim]),  # Position encoding
                           initializer=tf.random_normal_initializer(stddev=0.01))
    wte = mtf.get_variable(mesh, 'wte', mtf.Shape([vocab_dim, embd_dim]),  # Text encoding
                           initializer=tf.random_normal_initializer(stddev=0.02))

    past_length = 0 if past is None else mtf.Shape(past)[-2]

    # # if params["embed_dropout"] > 0:
    # #     TODO: if you get everything else working, add dropout
    # #     wpe = mtf.dropout(wpe, params["embed_dropout"])
    # #     wte = mtf.dropout(wte, params["embed_dropout"])
    h = mtf.gather(wte, x, vocab_dim) + mtf.gather(wpe, positions_for(x, past_length, batch_dim), vocab_dim)
    print('h shape:', h.shape)
    # # Transformer
    presents = []
    # # TODO: sanity check - pretty sure dim in unstack needs to be a Dimension - since we're passing in dim 1,
    # # just create a singleton?
    # # but it's none if pasts is none anyway... think it should be fine?
    # singleton = mtf.Dimension('singleton', 1)
    # pasts = mtf.unstack(past, dim=singleton) if past is not None else [None] * params["n_layer"]
    # assert len(pasts) == params["n_layer"]
    # print('PAST LENGTHS:')
    # print(len(pasts))
    pasts = [None] * params["n_layer"]

    for layer, past in enumerate(pasts):
        h, present = block(h, 'h%d' % layer, past=past, params=params,
                           block_offset=(layer * params["layer_offset"]) % params["fixed_attn_block_size"])
        presents.append(present)

    # dim_name = "results"
    # results['present'] = mtf.stack(presents, dim_name=dim_name, axis=1)

    # h = norm(h, 'ln_f', params=params)
    # dim_combined_batch_sequence = mtf.Dimension('combined_batch_sequence', batch_dim.size * sequence_dim.size)
    # h_flat = mtf.reshape(h, mtf.Shape([dim_combined_batch_sequence, embd_dim]))

    # # h_flat :: [batch*seq, embd]
    # # wte :: [vocab, embd]
    # print('H_FLAT / WTE SHAPES:')
    # print(h_flat.shape)
    # print(wte.shape)
    # print('OUTPUT SHAPE:')
    # print([dim_combined_batch_sequence, vocab_dim])

    #h = x
    #lnum = 0
    #dim = embd_dim
    #h = mtf.layers.dense(
    #    h, dim,
    #    use_bias=False,
    #    master_dtype=master_dtype,
    #    slice_dtype=slice_dtype,
    #    name='layer_%d' % lnum)
    y = h

    # dim_combined_batch_sequence = mtf.Dimension('combined_batch_sequence', batch_dim.size * sequence_dim.size)
    # h = expand_tile(h, embd_dim)
    # h_flat = mtf.reshape(h, mtf.Shape([dim_combined_batch_sequence, embd_dim]))

    # logits = mtf.einsum([h, y], output_shape=[batch_dim, sequence_dim])
    # to_stack = []
    # for i in range(params["n_vocab"]):
    #     to_stack.append(logits)
    # logits = mtf.stack(to_stack, 'stacked_dim', axis=2)
    # # logits = mtf.reshape(logits, [batch_dim, sequence_dim])
    # results['logits'] = logits

    # vdim = results["logits"].shape[2]
    # labels = mtf.import_tf_tensor(mesh, labels, mtf.Shape([batch_dim, sequence_dim]))

    # loss_batch = mtf.layers.softmax_cross_entropy_with_logits(logits=results["logits"], targets=labels, vocab_dim=vdim)
    loss = mtf.reduce_mean(y)

    return y, loss


def model_fn(features, labels, mode, params):
    """A model is called by TpuEstimator."""
    global_step = tf.train.get_global_step()
    graph = mtf.Graph()
    mesh_shape = mtf.convert_to_shape(FLAGS.mesh_shape)
    layout_rules = mtf.convert_to_layout_rules(FLAGS.layout)
    print('PARAMS:')
    print(params)
    print('Loading other params from file')
    # Read params of model
    with open(FLAGS.model_params, "r") as f:
        new_params = json.load(f)
    params.update(new_params)
    print('PARAMS AFTER LOAD FROM FILE:')
    print(params)
    if FLAGS.use_tpu:
        ctx = params['context']
        num_hosts = ctx.num_hosts
        host_placement_fn = ctx.tpu_host_placement_function
        device_list = [host_placement_fn(host_id=t) for t in range(num_hosts)]
        tf.logging.info('device_list = %s' % device_list, )
        # TODO(ylc): Better estimation of replica cache size?
        replica_cache_size = 300 * 1000000  # 300M per replica
        # Worker 0 caches all the TPU binaries.
        worker0_mem = replica_cache_size * ctx.num_replicas
        devices_memeory_usage = [worker0_mem] + [0] * (num_hosts - 1)
        var_placer = mtf.utils.BalancedVariablePlacer(device_list,
                                                      devices_memeory_usage)
        mesh_devices = [''] * mesh_shape.size
        mesh_impl = mtf.simd_mesh_impl.SimdMeshImpl(
            mesh_shape, layout_rules, mesh_devices, ctx.device_assignment)
    else:
        var_placer = None
        mesh_devices = [''] * mesh_shape.size
        mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(
            mesh_shape, layout_rules, mesh_devices)
    mesh = mtf.Mesh(graph, 'my_mesh', var_placer)

    with mtf.utils.outside_all_rewrites():
        logits, loss = toy_model(features, labels, params, mesh)

    # TRAIN mode
    if mode == tf.estimator.ModeKeys.TRAIN:
        var_grads = mtf.gradients([loss],
                                  [v.outputs[0] for v in graph.trainable_variables])
        optimizer = mtf.optimize.AdamWeightDecayOptimizer(
            learning_rate=FLAGS.lr,
            weight_decay_rate=FLAGS.lr * FLAGS.weight_decay,
            beta_1=FLAGS.beta1,
            beta_2=FLAGS.beta2,
            epsilon=FLAGS.epsilon)
        update_ops = optimizer.apply_grads(var_grads, graph.trainable_variables)
    else:
        # for now, we can only export fully-replicated tensors.
        fully_replicated_logits = mtf.anonymize(logits)

    lowering = mtf.Lowering(graph, {mesh: mesh_impl})

    tf_loss = tf.to_float(lowering.export_to_tf_tensor(loss))

    if mode == tf.estimator.ModeKeys.TRAIN:
        tf_update_ops = [lowering.lowered_operation(op) for op in update_ops]
        tf_update_ops.append(tf.assign_add(global_step, 1))
        tf.logging.info('tf_update_ops: {}'.format(tf_update_ops))
        train_op = tf.group(tf_update_ops)
    else:
        tf_logits = lowering.export_to_tf_tensor(fully_replicated_logits)

    with mtf.utils.outside_all_rewrites():
        # Copy master variables to slices. Must be called first.
        restore_hook = mtf.MtfRestoreHook(lowering)
        if mode == tf.estimator.ModeKeys.TRAIN:
            saver = tf.train.Saver(
                tf.global_variables(),
                sharded=True,
                max_to_keep=10,
                keep_checkpoint_every_n_hours=2,
                defer_build=False,
                save_relative_paths=True)
            tf.add_to_collection(tf.GraphKeys.SAVERS, saver)
            saver_listener = mtf.MtfCheckpointSaverListener(lowering)
            saver_hook = tf.train.CheckpointSaverHook(
                FLAGS.model_dir,
                save_steps=1000,
                saver=saver,
                listeners=[saver_listener])

            return tpu_estimator.TPUEstimatorSpec(
                tf.estimator.ModeKeys.TRAIN,
                loss=tf_loss,
                train_op=train_op,
                training_hooks=[restore_hook, saver_hook])
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(tf_logits):
                mean_logits = tf.metrics.mean(tf_logits)
                return {'mean_logits': mean_logits}

            eval_metrics = (metric_fn, [tf_logits])

            return tpu_estimator.TPUEstimatorSpec(
                tf.estimator.ModeKeys.EVAL,
                evaluation_hooks=[restore_hook],
                loss=tf_loss,
                eval_metrics=eval_metrics)


def run_toy_model_tpu():
    """Run a toy model on TPU."""
    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    iterations_per_loop = FLAGS.iterations
    mesh_shape = mtf.convert_to_shape(FLAGS.mesh_shape)
    config = tpu_config.RunConfig(
        cluster=tpu_cluster_resolver,
        model_dir=FLAGS.model_dir,
        save_checkpoints_steps=None,  # Disable the default saver
        save_checkpoints_secs=None,  # Disable the default saver
        log_step_count_steps=iterations_per_loop,
        save_summary_steps=iterations_per_loop,
        tpu_config=tpu_config.TPUConfig(
            num_shards=mesh_shape.size,
            iterations_per_loop=iterations_per_loop,
            num_cores_per_replica=1,
            per_host_input_for_training=tpu_config.InputPipelineConfig.BROADCAST))
    classifier = tpu_estimator.TPUEstimator(
        use_tpu=True,
        model_fn=model_fn,
        config=config,
        train_batch_size=FLAGS.batch_size,
        eval_batch_size=FLAGS.batch_size)
    current_step = estimator_lib._load_global_step_from_checkpoint_dir(
        FLAGS.model_dir)  # pylint: disable=protected-access,line-too-long
    logging.info('Current step %d', current_step)
    if FLAGS.steps_per_checkpoint == 0:
        classifier.train(input_fn=TextInput(), max_steps=FLAGS.train_steps)
        return
    while current_step < FLAGS.train_steps:
        next_checkpoint = min(current_step + FLAGS.steps_per_checkpoint,
                              FLAGS.train_steps)
        classifier.train(input_fn=TextInput(), max_steps=next_checkpoint)
        current_step = next_checkpoint
        logging.info('Starting to evaluate.')
        eval_results = classifier.evaluate(
            input_fn=TextInput(),
            steps=156)  # since we have 10000 examples and batch_size = 64 per host
        logging.info('Eval results: %s', eval_results)


def main(_):
    run_toy_model_tpu()


if __name__ == '__main__':
    tf.disable_v2_behavior()
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
