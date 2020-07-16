"""GPT-like model in Mesh-Tensorflow"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mesh_tensorflow as mtf
import tensorflow.compat.v1 as tf
import os, json, math
from functools import partial

from tensorflow.python.platform import flags
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import tpu_config  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.tpu import tpu_estimator  # pylint: disable=g-direct-tensorflow-import
from tensorflow_estimator.python.estimator import estimator as estimator_lib
import mesh_tensorflow.transformer as mtf_transformer
import mesh_tensorflow.auto_mtf

FLAGS = flags.FLAGS
tf.flags.DEFINE_string('model_params', 'configs/GPT_NEO_TEST.json', help="path to model config")
tf.flags.DEFINE_integer('steps_per_checkpoint', 200, 'steps_per_checkpoint')

# Optimizer settings
tf.flags.DEFINE_bool('use_tpu', True, 'use TPU')

#Auto layout
tf.flags.DEFINE_bool('auto_layout', False, 'set layout rules automatically')
tf.flags.DEFINE_bool('auto_layout_and_mesh_shape', False, 'set layout rules automatically')
tf.flags.DEFINE_integer('num_cores', 8, 'Number of TPU cores (required for auto_mesh_shape')
# steps_per, use_tpu, model_params, autolayouts


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


def text_dataset(files, params, stitch, datatype, batch=True):
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
                r = tf.random.uniform([], maxval=s - (params["n_ctx"] + 1), dtype=tf.dtypes.int32)
                r1 = tf.range(r, r + params["n_ctx"])
                r2 = tf.range(r + 1, (r + 1) + params["n_ctx"])
                r1 = tf.reshape(r1, [params["n_ctx"]])  # Somehow, this makes the compiler happy
                r2 = tf.reshape(r2, [
                    params["n_ctx"]])  # TPUs want constant sized input, and these reshapes makes it recognize the shape of the input
                vals1 = tf.gather(x, r1)
                vals2 = tf.gather(x, r2)

                vals1 = tf.reshape(vals1, [params["n_ctx"]])
                vals2 = tf.reshape(vals2, [params["n_ctx"]])
                return vals1, vals2

        else:
            def _sample_text(x):
                vals1 = x[:params["n_ctx"]]
                vals2 = x[1:params["n_ctx"] + 1]

                vals1 = tf.reshape(vals1, [params["n_ctx"]])
                vals2 = tf.reshape(vals2, [params["n_ctx"]])
                return vals1, vals2

        dataset = dataset.map(_sample_text, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if batch:
        dataset = dataset.batch(params["train_batch_size"], drop_remainder=True).prefetch(params["iterations"] * 2)

    dataset = dataset.repeat()

    return dataset


def generic_text(params, eval=False):
    # params["datasets"] = [(train glob, eval_glob, stitch, ["random_sample", "sample", "chunk"] weight)]
    # , dsets=[["bundestag_*.tfrecords", "", 10, "random_sample", 1.0]]
    i = 0 if not eval else 1
    print('##############################')
    print(params["data_path"])
    print(params["datasets"])
    print('##############################')

    datasets = [text_dataset(tf.io.gfile.glob(os.path.join(params["data_path"], dataset[i])),
                params, stitch=dataset[2], datatype=dataset[3], batch=False)
                for dataset in params["datasets"]]
    weights = [dataset[4] for dataset in params["datasets"]]

    dataset = tf.data.experimental.sample_from_datasets(datasets, weights=weights)
    dataset = dataset.batch(params["train_batch_size"], drop_remainder=True).prefetch(params["iterations"] * 2)

    return dataset


# class TextInput(object):

#     def __init__(self):
#         self.dsets = [["bundestag_*.tfrecords", "", 10, "random_sample", 1.0]]

#     def __call__(self, params):
#         dset = generic_text(dsets=self.dsets)
#         return dset


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

        b = mtf.get_variable(x.mesh, 'b', [nf], initializer=tf.constant_initializer(0, dtype=tf.float32), dtype=dt)
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


def attn(x, scope, n_state, *, past, params, train=False):
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

    # input length is past seq + x seq because when sampling, subsequent x is only length 1
    # no longer needed in mtf because TPUs cant handle pasts anyways, apparently
    # inp_len = dim_seq + (tf.shape(past)[3] if past is not None else 0)

    def split_heads(x, last_dim):
        # From [batch, sequence, features] to [batch, heads, sequence, features_per_head]
        # heads is split out of features!
        x = mtf.reshape(x, [dim_batch, dim_seq, dim_heads, last_dim], name="split_heads_reshape")
        x = mtf.transpose(x, [dim_batch, dim_heads, dim_seq, last_dim], name="split_heads_transpose")
        return x

    def merge_heads(x):
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
        nx = x.shape[-1]
        h = mtf.gelu(conv1d(x, 'c_fc', n_state, params=params))
        h2 = conv1d(h, 'c_proj', nx, params=params, scale=True)
        h2 = mtf.dropout(h2, params["res_dropout"], name="mlp_dropout")
        return h2


def block(x, scope, *, past, params, train=False):
    with tf.variable_scope(scope):
        nx = x.shape[-1]
        a, present = attn(norm(x, 'ln_1', params=params), 'attn', nx, past=past, params=params,)
        x = x + a

        dim_intermediate_expanded = mtf.Dimension('intermediate_expanded', nx.size * 4)
        m = mlp(norm(x, 'ln_2', params=params), 'mlp', dim_intermediate_expanded, params=params, train=train)
        x = x + m
        return x, present


def gpt_model(features, labels, params, mesh, past=None):
    """A GPT style model implemented in mesh tensorlfow."""
    results = {}

    # define mtf dims
    batch_dim = mtf.Dimension('batch', params["train_batch_size"])
    sequence_dim = mtf.Dimension('sequence', params["n_ctx"]) #TODO: sanity check

    # we need this because gathering when both the args have the same dimension in them it breaks stuff.
    # this dim is specifically for the weights
    # this prevents the "Einsum has lhs dimension without corresponding rhs or output dimension." error.
    embed_sequence_dim = mtf.Dimension('embed_sequence', params["n_ctx"])
    embd_dim = mtf.Dimension("embd", params["n_embd"])
    vocab_dim = mtf.Dimension("vocab", params["n_vocab"])

    # convert input tensor to mtf tensor
    x = mtf.import_tf_tensor(mesh, features, mtf.Shape([batch_dim, sequence_dim]))

    wpe = mtf.get_variable(mesh, 'wpe', mtf.Shape([embed_sequence_dim, embd_dim]),  # Position encoding
                           initializer=tf.random_normal_initializer(stddev=0.01))
    wte = mtf.get_variable(mesh, 'wte', mtf.Shape([vocab_dim, embd_dim]),  # Text encoding
                           initializer=tf.random_normal_initializer(stddev=0.02))

    past_length = 0 if past is None else mtf.Shape(past)[-2]

    if params["embed_dropout"] > 0:
        wpe = mtf.dropout(wpe, params["embed_dropout"], name="wpe_dropout")
        wte = mtf.dropout(wte, params["embed_dropout"], name="wte_dropout")
    h = mtf.gather(wte, x, vocab_dim) + mtf.gather(wpe, positions_for(x, past_length, batch_dim), embed_sequence_dim)
    # # Transformer
    presents = []

    # TODO: we will need this code for sampling
    # singleton = mtf.Dimension('singleton', 1)
    # pasts = mtf.unstack(past, dim=singleton) if past is not None else [None] * params["n_layer"]
    # assert len(pasts) == params["n_layer"]
    pasts = [None] * params["n_layer"]

    # attn blocks
    for layer, past in enumerate(pasts):
        h, present = block(h, 'h%d' % layer, past=past, params=params)
        presents.append(present)

    dim_name = "results"
    results['present'] = mtf.stack(presents, dim_name=dim_name, axis=1)

    # normalize & affine transform
    h = norm(h, 'ln_f', params=params)

    # equivalent to tf.matmul
    logits = mtf.einsum([h, wte], output_shape=[batch_dim, sequence_dim, vocab_dim])
    results['logits'] = logits

    vdim = results["logits"].shape[2] # get vocab dimension

    # In this case, labels are simply input shifted one token to the right
    # this op is done in the input_fn
    labels = mtf.import_tf_tensor(mesh, labels, mtf.Shape([batch_dim, sequence_dim]))

    loss_batch = mtf.layers.softmax_cross_entropy_with_logits(logits=results["logits"], targets=labels, vocab_dim=vdim)
    loss = mtf.reduce_mean(loss_batch)
    return logits, loss


def model_fn(features, labels, mode, params):
    """A model is called by TpuEstimator."""
    global_step = tf.train.get_global_step()
    graph = mtf.Graph()
    mesh_shape = mtf.convert_to_shape(params["mesh_shape"])
    layout_rules = mtf.convert_to_layout_rules(params["layout"])

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
        logits, loss = gpt_model(features, labels, params, mesh)

    if FLAGS.auto_layout:
        layout_rules = mtf.auto_mtf.layout(graph, mesh_shape, [logits, loss])
        print('Auto-selected layout:')
        print(layout_rules)
        print('Re-initialize graph with selected layout')
        quit() #TODO: it should be easy to just reinitialize everything w selected layout

    if FLAGS.auto_layout_and_mesh_shape:
        layout_rules, mesh_shape = mtf.auto_mtf.layout_and_mesh_shape(graph, FLAGS.num_cores, [logits, loss])
        print('Num cores:')
        print(FLAGS.num_cores)
        print('Auto-selected layout:')
        print(layout_rules)
        print('Auto-selected mesh shape:')
        print(mesh_shape)
        print('Re-initialize graph with selected layout & mesh shape')
        quit() #TODO: it should be easy to just reinitialize everything w selected layout

    # TRAIN mode
    if mode == tf.estimator.ModeKeys.TRAIN:
        var_grads = mtf.gradients([loss],
                                  [v.outputs[0] for v in graph.trainable_variables])
        if params["opt_name"].lower() == "adam":
            optimizer = mtf.optimize.AdamWeightDecayOptimizer(
                learning_rate=params["lr"],
                weight_decay_rate=params["lr"] * params["weight_decay"],
                beta_1=params["beta1"],
                beta_2=params["beta2"],
                epsilon=params["epsilon"])
        else:
            optimizer = mtf.optimize.AdafactorOptimizer(
                learning_rate=params["lr"],
                decay_rate=params["lr"] * params["weight_decay"],
                beta1=params["beta1"],
                epsilon1=params["ada_epsilon1"],
                epsilon2=params["ada_epsilon2"]
            )
        update_ops = optimizer.apply_grads(var_grads, graph.trainable_variables)
    else:
        # for now, we can only export fully-replicated tensors.
        # TODO: this is mtf code - figure out what this does
        fully_replicated_logits = mtf.anonymize(logits)

    print('\n')
    total_parameters = 0
    for variable in graph.trainable_variables:
      shape = variable.shape.dims
      variable_parameters = 1
      for dim in shape:
          variable_parameters *= dim.size
      total_parameters += variable_parameters
    print("N TRAINABLE VARS:")
    print('{:,}'.format(total_parameters))
    print('\n')

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
                params["model_path"],
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


def run_model_tpu():
    """Run a GPT model on TPU."""
    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    # Read params of model
    with open(FLAGS.model_params, "r") as f:
        params = json.load(f)
    tf.logging.info('params = %s' % params, )

    iterations_per_loop = params["iterations"]
    mesh_shape = mtf.convert_to_shape(params["mesh_shape"])

    config = tpu_config.RunConfig(
        cluster=tpu_cluster_resolver,
        model_dir=params["model_path"],
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
        train_batch_size=params["train_batch_size"],
        eval_batch_size=params["train_batch_size"],
        params=params)
    current_step = estimator_lib._load_global_step_from_checkpoint_dir(
        params["model_path"])  # pylint: disable=protected-access,line-too-long
    logging.info('Current step %d', current_step)
    if FLAGS.steps_per_checkpoint == 0:
        classifier.train(input_fn=partial(generic_text, eval=False), max_steps=params["train_batch_size"])
        return
    while current_step < params["train_steps"]:
        next_checkpoint = min(current_step + FLAGS.steps_per_checkpoint,
                              params["train_steps"])
        classifier.train(input_fn=partial(generic_text, eval=False), max_steps=next_checkpoint)
        current_step = next_checkpoint
        # logging.info('Starting to evaluate.')
        # eval_results = classifier.evaluate(
        #     input_fn=TextInput(),
        #     steps=156)  # since we have 10000 examples and batch_size = 64 per host
        # logging.info('Eval results: %s', eval_results)


def main(_):
    run_model_tpu()


if __name__ == '__main__':
    tf.disable_v2_behavior()
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
