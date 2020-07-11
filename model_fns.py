from functools import partial

import numpy as np
import tensorflow as tf
import mesh_tensorflow as mtf

from optimizers import get_update_ops
from metric_fns import *


def gpt2_model_mesh(features, labels, mode, params):
    from models.gpt2 import gpt2

    # define mtf graph / mesh
    graph = mtf.Graph()

    # example for 8 cores
    mesh_shape = mtf.convert_to_shape('all:8') #TODO: if we get model working this hardcoding needs to go
    mesh_size = mesh_shape.size
    mesh_devices = [''] * mesh_size

    # model parallelism - distribute blocks across all cores
    layout_rules = mtf.convert_to_layout_rules('blocks:all') #TODO: if we get model working this hardcoding needs to go

    # distribute blocks according to layout rules
    ctx = params['context']
    num_hosts = ctx.num_hosts
    host_placement_fn = ctx.tpu_host_placement_function
    device_list = [host_placement_fn(host_id=t) for t in range(num_hosts)]
    tf.logging.info('device_list = %s' % device_list, )
    # TODO: Better estimation of replica cache size?
    replica_cache_size = 300 * 1000000  # 300M per replica
    # Worker 0 caches all the TPU binaries.
    worker0_mem = replica_cache_size * ctx.num_replicas
    devices_memory_usage = [worker0_mem] + [0] * (num_hosts - 1)
    var_placer = mtf.utils.BalancedVariablePlacer(device_list,
                                                  devices_memory_usage)
    mesh_impl = mtf.simd_mesh_impl.SimdMeshImpl(
        mesh_shape, layout_rules, mesh_devices, ctx.device_assignment)
    mesh = mtf.Mesh(graph, "my_mesh", var_placer)

    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        if params["precision"] == 'bfloat16':
            raise Exception('Not implemented') #TODO: implement bfloat precision
            with tf.contrib.tpu.bfloat16_scope():
                output = gpt2.model(X=features, params=params, mesh=mesh,
                                    past=None, reuse=tf.AUTO_REUSE,
                                    train=mode==tf.estimator.ModeKeys.TRAIN)

            output["logits"] = tf.cast(output["logits"], tf.float32)

        else:

            output = gpt2.model(X=features, params=params, mesh=mesh,
                                    past=None, reuse=tf.AUTO_REUSE,
                                    train=mode==tf.estimator.ModeKeys.TRAIN)


        # logits :: [batch, seq, vocab]
        vdim = output["logits"].shape[2]
        loss_batch = mtf.layers.softmax_cross_entropy_with_logits(logits=output["logits"], targets=labels, vocab_dim=vdim)
        loss = mtf.reduce_mean(loss_batch)

    if mode == tf.estimator.ModeKeys.TRAIN:
        update_ops = get_update_ops(loss, params, graph)

    lowering = mtf.Lowering(graph, {mesh: mesh_impl})
    if mode != tf.estimator.ModeKeys.PREDICT:
        tf_loss = lowering.export_to_tf_tensor(loss)

    with mtf.utils.outside_all_rewrites():

        if mode == tf.estimator.ModeKeys.TRAIN:
            tf_update_ops = [lowering.lowered_operation(op) for op in update_ops] # lower mtf update ops to tf update ops
            train_op = tf.group(tf_update_ops) # group update ops into train op
            return tf.contrib.tpu.TPUEstimatorSpec(mode, loss=tf_loss, train_op=train_op)



        if mode == tf.estimator.ModeKeys.EVAL:
            #TODO: implement EVAL
            from metric_fns import perplexity_metric

            if params["use_tpu"]:
                # Metric inputs are transferred to CPU and must preserve batch dimension
                return tf.contrib.tpu.TPUEstimatorSpec(mode=mode,
                    loss=loss, eval_metrics=(perplexity_metric, {"loss": loss_batch}))
            else:
                return tf.estimator.EstimatorSpec(mode=mode,
                    loss=loss, eval_metric_ops=perplexity_metric(loss_batch))


        if mode == tf.estimator.ModeKeys.PREDICT:
            #TODO: implement predict / sample
            from models.gpt2 import sample

            if not "top_k" in params.keys():
                params["top_k"] = 0

            output = sample.sample_sequence(
                params=params, length=params["n_ctx"],
                context=features,
                batch_size=params["batch_size"],
                temperature=1.0, top_k=params["top_k"]
            )

            predictions = {
                "tokens": output
            }

            if params["use_tpu"]:
                return tf.contrib.tpu.TPUEstimatorSpec(mode, predictions=predictions)
            else:
                return tf.estimator.EstimatorSpec(mode, predictions=predictions)


def gpt2_model(features, labels, mode, params):
    from models.gpt2 import gpt2

    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        if params["precision"] == 'bfloat16':
            with tf.contrib.tpu.bfloat16_scope():
                output = gpt2.model(X=features, params=params,
                                    labels=labels,
                                    past=None, reuse=tf.AUTO_REUSE,
                                    train=mode==tf.estimator.ModeKeys.TRAIN, mesh=mesh)

            output["logits"] = tf.cast(output["logits"], tf.float32)

        else:
            output = gpt2.model(X=features, params=params,
                                    labels=labels,
                                    past=None, reuse=tf.AUTO_REUSE,
                                    train=mode==tf.estimator.ModeKeys.TRAIN, mesh=mesh)

        loss_batch = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output["logits"], labels=labels)
        loss = tf.reduce_mean(loss_batch)

    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = create_train_op(loss, params)

        if params["use_tpu"]:
            return tf.contrib.tpu.TPUEstimatorSpec(mode, loss=loss, train_op=train_op)
        else:
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


    if mode == tf.estimator.ModeKeys.EVAL:
        from metric_fns import perplexity_metric

        if params["use_tpu"]:
            # Metric inputs are transferred to CPU and must preserve batch dimension
            return tf.contrib.tpu.TPUEstimatorSpec(mode=mode,
                loss=loss, eval_metrics=(perplexity_metric, {"loss": loss_batch}))
        else:
            return tf.estimator.EstimatorSpec(mode=mode,
                loss=loss, eval_metric_ops=perplexity_metric(loss_batch))


    if mode == tf.estimator.ModeKeys.PREDICT:

        from models.gpt2 import sample

        if not "top_k" in params.keys():
            params["top_k"] = 0

        output = sample.sample_sequence(
            params=params, length=params["n_ctx"],
            context=features,
            batch_size=params["batch_size"],
            temperature=1.0, top_k=params["top_k"]
        )

        predictions = {
            "tokens": output
        }

        if params["use_tpu"]:
            return tf.contrib.tpu.TPUEstimatorSpec(mode, predictions=predictions)
        else:
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)
