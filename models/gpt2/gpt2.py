"""GPT-like model in Mesh-Tensorflow"""
import tensorflow.compat.v1 as tf
import mesh_tensorflow.transformer as mtf_transformer

from models.utils import parse_inputs, entmax_cross_entropy_with_logits
from models.layers import *


# --------------------------------------------------------------------------------
# TRANSFORMER BLOCK:

def block(params, scope, layer_num, bias, sequence_dim, memory_length_dim, pos_emb, variable_dtype, context=None):
    use_mlp_glu = params["mlp_glu"] == True
    use_scale_norm = params["scalenorm"] == True
    use_moe = exists(params["moe_layers"]) and (layer_num in params["moe_layers"])
    use_rezero = params["rezero"] == True
    macaron_attention = params["macaron"] == True

    def fn(x):
        with tf.variable_scope(scope):
            nx = x.shape[-1]  # Grab last dimension from input

            if use_rezero:
                prenorm = identity
            elif use_scale_norm:
                prenorm = scale_norm
            else:
                prenorm = layer_norm

            pre_residual_fn = rezero if use_rezero else identity

            attention_type = params["attention_types"][layer_num]

            if macaron_attention:
                mult = 0.5
                mlp_fn = mlp_glu if use_mlp_glu else mlp
                intermediate_size = nx.size * 4 * (1 if not use_mlp_glu else 2)
                # Define intermediate layer of mlp - to split
                dim_intermediate_expanded = mtf.Dimension("intermediate_expanded", intermediate_size)
                m = mlp_fn(x, "mlp_macaron", dim_intermediate_expanded, variable_dtype=variable_dtype, params=params)

                x = x + (m * mult)
            else:
                mult = 1

            if attention_type != "none":
                res_x = prenorm(x, "norm_1", variable_dtype=variable_dtype, params=params)
                a = attn(res_x, "attn", nx, attention_type=attention_type,
                         params=params, bias=bias, dim_seq=sequence_dim, memory_length_dim=memory_length_dim,
                         variable_dtype=variable_dtype, context=context, pos_emb=pos_emb)
            else:
                a = x

            x = x + pre_residual_fn(a, "norm_rezero_1", dtype=variable_dtype)

            res_x = prenorm(x, "norm_2", variable_dtype=variable_dtype, params=params)

            if use_moe:
                moe_params = mtf.transformer.moe.HParams()
                mtf.transformer.moe.set_default_moe_hparams(moe_params)
                moe_params.add_hparam("moe_min_expert_capacity", 1)
                moe_params.add_hparam("moe_use_experts_attention", False)

                # Override defaults
                for k, v in params["moe_params"].items():
                    moe_params.add_hparam(k, v)

                moe_train = params["mode"] == "train"

                m, aux_loss = mtf.transformer.moe.transformer_moe_layer_v1(res_x, x.shape[-1], moe_params,
                                                                           train=moe_train,
                                                                           mesh_shape=params["mesh_shape"],
                                                                           layout=params["layout"],
                                                                           activation=params.get("moe_activation",
                                                                                                 "relu"),
                                                                           variable_dtype=variable_dtype,
                                                                           num_microbatches=params["num_microbatches"])
                m = mtf.dropout(m, rate=params["res_dropout"], name="moe_dropout")
            else:

                mlp_fn = mlp_glu if use_mlp_glu else mlp
                intermediate_size = nx.size * 4 * (1 if not use_mlp_glu else 2)

                # Define intermediate layer of mlp - to split
                dim_intermediate_expanded = mtf.Dimension("intermediate_expanded", intermediate_size)

                m = mlp_fn(res_x, "mlp", dim_intermediate_expanded, variable_dtype=variable_dtype, params=params)
                aux_loss = mtf.zeros(x.mesh, mtf.Shape([]), dtype=variable_dtype.slice_dtype)

            x = x + pre_residual_fn((m * mult), "norm_rezero_2", variable_dtype)
            return x, aux_loss

    return fn


# --------------------------------------------------------------------------------
# GPT2 MODEL:

def model(mtf_features, other_features, params, mesh, variable_dtype, context=None):
    """A GPT style model implemented in mesh tensorflow."""

    x, batch_dim, sequence_dim, embd_dim, vocab_dim, embed_sequence_dim = parse_inputs(mtf_features, other_features)

    if is_incremental_inference(context):
        # reshape inputs if in inference mode
        x = mtf.gather(x, context.position - 1, sequence_dim)
        x = mtf.reshape(x, [batch_dim])

    use_axial_pos_emb = exists(params["axial_pos_emb"])
    use_rotary_emb = exists(params["rotary_emb"])

    # Text encoding
    wte = mtf.get_variable(mesh, "wte", mtf.Shape([vocab_dim, embd_dim]),
                           initializer=tf.random_normal_initializer(stddev=0.02),
                           master_dtype=variable_dtype.master_dtype,
                           slice_dtype=variable_dtype.slice_dtype,
                           activation_dtype=variable_dtype.activation_dtype)

    with tf.variable_scope("token_embd"):
        # Text embedding
        h = mtf.gather(wte, x, vocab_dim)
        if params["embed_dropout"] > 0 and params["mode"] == "train":
            h = mtf.dropout(h, rate=params["embed_dropout"], name="wte_dropout")

    # Position encoding

    if use_rotary_emb:
        wpe = None
        layer_pos_emb = rotary_positional_emb(mesh, sequence_dim, params, variable_dtype)
    elif use_axial_pos_emb:
        wpe = axial_positional_emb(embd_dim, mesh, params, variable_dtype)
        layer_pos_emb = None
    else:
        # Use standard position encoding
        wpe = mtf.get_variable(mesh, "wpe", mtf.Shape([embed_sequence_dim, embd_dim]),
                               initializer=tf.random_normal_initializer(stddev=0.01),
                               master_dtype=variable_dtype.master_dtype,
                               slice_dtype=variable_dtype.slice_dtype,
                               activation_dtype=variable_dtype.activation_dtype)
        layer_pos_emb = None

    if exists(wpe):
        with tf.variable_scope("pos_embd"):
            # Positional embedding
            position_indices = mtf.range(mesh, sequence_dim, tf.int64) if not is_incremental_inference(context) else (
                    context.position - 1)
            pos_emb = mtf.gather(wpe, position_indices, wpe.shape[0])
            if params["embed_dropout"] > 0 and params["mode"] == "train":
                pos_emb = mtf.dropout(pos_emb, rate=params["embed_dropout"], name="wte_dropout")
            h += pos_emb

    aux_losses = 0  # instantiate auxiliary losses (for MOE models)

    for layer in range(params["n_layer"]):
        # attn blocks
        share_parameters = exists(params["share_parameters"]) and params["share_parameters"] == True
        block_scope = f"h{layer}" if not share_parameters else ""

        block_fn = block(params=params, scope=block_scope, layer_num=layer,
                         bias=other_features["attn_bias"],
                         sequence_dim=sequence_dim,
                         memory_length_dim=other_features["memory_length_dim"],
                         pos_emb = layer_pos_emb,
                         variable_dtype=variable_dtype,
                         context=context)

        # If true and in train mode, enable gradient checkpointing
        recompute_grad = params["recompute_grad"] and (params["mode"] == "train") == True
        h, loss = block_fn(h) if not recompute_grad else mtf.recompute_grad(block_fn, [h])
        aux_losses += loss

    no_weight_tie_emb = params["no_weight_tie"] == True
    if no_weight_tie_emb:
        with tf.variable_scope("wte_final_linear"):
            logits = linear(h, "linear_out", vocab_dim, variable_dtype=variable_dtype, params=params)
    else:
        # Layer normalize & affine transform
        h = layer_norm(h, "ln_f", variable_dtype=variable_dtype)
        seq_dim = sequence_dim if not is_incremental_inference(context) else mtf.Dimension("sequence", 1)
        with tf.variable_scope("wte_final_einsum"):
            # Equivalent to tf.matmul
            logits = mtf.einsum([h, wte], output_shape=[batch_dim, seq_dim, vocab_dim])

    if params["mode"] in ["train", "eval"]:
        labels = mtf_features["labels"]
        z_loss = params.get("z_loss", 1e-4)  # an auxiliary loss used to stabilize mtf xentropy

        # Go to full precision for the logits 
        logits = mtf.cast(logits, tf.float32)

        use_entmax_loss = params.get("entmax_loss", False)
        loss_fn = mtf.layers.softmax_cross_entropy_with_logits if not use_entmax_loss else entmax_cross_entropy_with_logits

        with tf.variable_scope("xentropy_final"):
            loss_batch = loss_fn(logits=logits, targets=labels,
                                 vocab_dim=logits.shape[-1], z_loss=z_loss)

        # For non-autoregressive models (masked language modeling training)
        # Make sure labels with padding tokens are not counted in the loss
        if not params["causal"]:
            padding_id = params.get("padding_id", 0)
            loss_batch = mtf.where(mtf.not_equal(labels, padding_id), loss_batch, mtf.zeros_like(loss_batch))

        with tf.variable_scope("reduce_mean_final"):
            loss = mtf.reduce_mean(loss_batch)

        loss += aux_losses  # Add on auxiliary losses (currently only used for MoE)
        loss /= params["num_microbatches"]
        # Convert to train dtype
        loss = mtf.cast(loss, variable_dtype.slice_dtype)
    else:
        loss = None
        loss_batch = None

    # Cast back to checkpoint dtype
    logits = mtf.cast(logits, variable_dtype.master_dtype)
    return logits, loss, loss_batch
