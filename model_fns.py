from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mesh_tensorflow as mtf
import tensorflow.compat.v1 as tf
from tensorflow.python.tpu import tpu_estimator  # pylint: disable=g-direct-tensorflow-import

import mesh_tensorflow.auto_mtf
from utils import get_graph_info

def model_fn(features, labels, mode, params):
    """A model is called by TpuEstimator."""
    global_step = tf.train.get_global_step()
    graph = mtf.Graph()
    mesh_shape = mtf.convert_to_shape(params["mesh_shape"])
    layout_rules = mtf.convert_to_layout_rules(params["layout"])

    if params["use_tpu"]:
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
    if params["model"] == "GPT2":
        from models.gpt2 import gpt2
        with mtf.utils.outside_all_rewrites():
            logits, loss = gpt2.model(features, labels, params, mesh)
    elif params["model"] == "GPT2MOE":
        from models.gpt2moe import gpt2moe
        with mtf.utils.outside_all_rewrites():
            logits, loss = gpt2moe.model(features, labels, params, mesh)
    else:
        raise Exception(f"{params['model']} is not a valid model - please select from GPT2 or GPT2MOE")

    if params["auto_layout"]:
        layout_rules = mtf.auto_mtf.layout(graph, mesh_shape, [logits, loss])
        print('Auto-selected layout:')
        print(layout_rules)
        print('Re-initialize graph with selected layout')
        quit() #TODO: it should be easy to just reinitialize everything w selected layout

    if params["auto_layout_and_mesh_shape"]:
        layout_rules, mesh_shape = mtf.auto_mtf.layout_and_mesh_shape(graph, params["num_cores"], [logits, loss])
        print('Num cores:')
        print(params["num_cores"])
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
                weight_decay_rate=params["weight_decay"],
                beta_1=params["beta1"],
                beta_2=params["beta2"],
                epsilon=params["epsilon"])
        else:
            optimizer = mtf.optimize.AdafactorOptimizer(
                learning_rate=params["lr"],
                decay_rate=params["weight_decay"],
                beta1=params["beta1"],
                epsilon1=params["ada_epsilon1"],
                epsilon2=params["ada_epsilon2"]
            )
        update_ops = optimizer.apply_grads(var_grads, graph.trainable_variables)
    else:
        # for now, we can only export fully-replicated tensors.
        # TODO: this is mtf code - figure out what this does
        fully_replicated_logits = mtf.anonymize(logits)

    # gets info about no. trainable vars in the model & dimension names
    get_graph_info(graph)

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