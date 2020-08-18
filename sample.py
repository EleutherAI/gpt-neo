import mesh_tensorflow as mtf
import tensorflow.compat.v1 as tf
import mesh_tensorflow.transformer as mtf_transformer


def sample_autoregressive(partial_sequences,
                          other_features,
                          params,
                          stop_at_token=1,
                          max_steps=None,
                          temperature=0.9,
                          variable_dtype=mtf.VariableDType(tf.float32),
                          encoder_output=None,
                          encoder_sequence_id=None,
                          encoder_inputs=None,
                          shared_params=None,
                          has_partial_sequences=True,
                          encoder_layer_outputs=None,
                          never_end=False,
                          remove_partial_sequences=False,
                          sampling_keep_top_k=-2,
                          bos_id=50256):
    """Sample randomly one token at a time.

    The partial_sequences represent partial sequences to be continued.  The
    first tokens of each sequence are nonzero representing the given partial
    sequences and the last tokens of each sequence are zeros, representing what
    needs to be filled in.

    If there are no partial sequences (you want to sample from the beginning),
    then pass partial_sequences=mtf.zeros(mesh, shape, dtype=tf.int32) and
    has_partial_sequences=False (so we can skip computation).

    Args:
        partial_sequences: an int32 Tensor with shape [<batch_dims>, length_dim]
        stop_at_token: an optional integer eos id.  Stop when we produce it.
        max_steps: an optional integer, the max number of steps to decode.
        temperature: an optional floating point value between 0.0 and 1.0 0.0
        means argmax, 1.0 means sample according to predicted distribution.
        variable_dtype: a mtf.VariableDType
        encoder_output: an optional Tensor
        encoder_sequence_id: an optional Tensor
        encoder_inputs: an optional Tensor
        shared_params: an optional dictionary
        has_partial_sequences: a boolean
        encoder_layer_outputs: optional - readonly list of tensor activations when
        decoding, one per each input layer + the embedding layer
        never_end: a boolean - if set, then avoid generating stop_at_token
        remove_partial_sequences: a boolean - whether to remove the partial
        sequences from the output
        sampling_keep_top_k: an integer - if not -1, only sample from the top k
        logits.
        bos_id: beginning of sequence id

    Returns:
        a Tensor with shape [<batch_dims>, length_dim]
    """

    inputs = partial_sequences  # partial sequences to fill in
    batch_dims = inputs.shape.dims[:-1]  # √
    length_dim = inputs.shape.dims[-1]  # √

    initial_position = mtf.reduce_sum(
        mtf.to_int32(mtf.not_equal(inputs, 0)), reduced_dim=length_dim)  # gets position where zero padding starts

    length_range = mtf.range(inputs.mesh, length_dim, tf.int32)
    input_full_attention = True  # for now hardcode this to true bc lazy
    if input_full_attention:
        # Vanilla autoregressive model - each position can see previous positions.
        # think this feeds in to the loop fn and tells each position where it can attend to?
        read_priority = write_priority = length_range * mtf.to_int32(
            mtf.greater(length_range, initial_position))
    else:
        read_priority = write_priority = length_range

    # builds context to pass around internally
    # the 'first part' context records initial states of k / v / x
    context_first_part = mtf_transformer.transformer.Context(
        model=None,
        mesh=inputs.mesh,
        batch_dims=batch_dims,
        length_dim=length_dim,
        variable_dtype=variable_dtype,
        mode="first_part",
        position=length_range,
        position_is_default=True,
        new_states=[],
        initial_position=initial_position,
        sequence_id=None,
        encoder_output=encoder_output,
        encoder_sequence_id=encoder_sequence_id,
        constant_states=[],
        shared_params=shared_params,
        encoder_layer_outputs=encoder_layer_outputs,
        write_priority=write_priority,
        read_priority=read_priority,
        inputs=inputs,
        encoder_inputs=encoder_inputs)

    from models.gpt2 import gpt2

    shifted_inputs = mtf.shift(inputs, offset=1, dim=length_dim,
                               wrap=False)  # shifts inputs to the right for first inp

    with tf.variable_scope('gpt2'):
        logits, _, _ = gpt2.model(shifted_inputs, other_features, params, inputs.mesh, context=context_first_part)
    del logits
    constant_states = context_first_part.constant_states

    if not has_partial_sequences:
        initial_states = [
            mtf.zeros_like(t) for t in context_first_part.new_states]
        partial_sequences_eos_count = 0
    else:
        initial_states = context_first_part.new_states
    partial_sequences_eos_count = mtf.reduce_sum(
        mtf.to_int32(mtf.equal(partial_sequences, stop_at_token)),
        reduced_dim=length_dim)

    def cond_fn(position, ids, *unused_states):
        """Should we run another loop iteration."""
        past_end = mtf.greater_equal(position, length_dim.size)
        if max_steps:
            past_end = mtf.logical_or(
                past_end, mtf.greater_equal(position - initial_position, max_steps))

        is_done = past_end
        if stop_at_token is not None:
            eos_count = mtf.reduce_sum(
                mtf.to_int32(mtf.equal(ids, stop_at_token)),
                reduced_dim=length_dim)
            has_additional_eos = mtf.greater(eos_count, partial_sequences_eos_count)
            is_done = mtf.logical_or(is_done, has_additional_eos)
        all_done = mtf.reduce_all(is_done)
        return mtf.logical_not(all_done)

    def body_fn(position, ids, *states):
        nonlocal sampling_keep_top_k
        nonlocal context_first_part
        """One step in the decode loop."""
        # inputs_this_step = mtf.gather(ids, position - 1, length_dim)
        # # Setting proper bos_id for position == 0. No-op otherwise.
        # if bos_id:
        #     inputs_this_step += bos_id * mtf.ones_like(inputs_this_step) * mtf.cast(
        #         mtf.equal(position, 0), tf.int32)

        initial_position = mtf.reduce_sum(mtf.to_int32(mtf.not_equal(inputs, 0)), reduced_dim=length_dim)

        context = mtf_transformer.transformer.Context(
            model=None,
            mesh=inputs.mesh,
            batch_dims=batch_dims,
            length_dim=length_dim,
            variable_dtype=variable_dtype,
            mode="first_part",
            position=length_range,
            position_is_default=True,
            new_states=[],
            initial_position=position,
            sequence_id=None,
            encoder_output=encoder_output,
            encoder_sequence_id=encoder_sequence_id,
            constant_states=[],
            shared_params=shared_params,
            encoder_layer_outputs=encoder_layer_outputs,
            write_priority=write_priority,
            read_priority=read_priority,
            inputs=ids,
            encoder_inputs=encoder_inputs)

        # TODO: inputs here should be inputs_this_step but it seems to break things :<
        with tf.variable_scope('gpt2', reuse=True):
            logits, _, _ = gpt2.model(ids, other_features, params, inputs.mesh, context = context)
        # if never_end:
        #     logits += mtf.one_hot(
        #         mtf.constant(logits.mesh, stop_at_token, dtype=tf.int32),
        #         self.output_vocab_dim, on_value=-1e9, off_value=0.0,
        #         dtype=logits.dtype)
        # TBD whether this should be before or after never_end:
        # Note for adding top_p sampling in the future, in other code bases, the
        # option to apply temperature is done before the top-k truncation. This
        # implementation does this in the opposite order. For top-k this doesn't
        # matter, but for top_p it will.

        # by default, do topk sampling of 0.9
        if sampling_keep_top_k == -2:
            sampling_keep_top_k = int(logits.shape[-1].size * 0.1)

        if sampling_keep_top_k != -1:
            if sampling_keep_top_k <= 0:
                raise ValueError("sampling_keep_top_k must either be -1 or positive.")
            k_largest = mtf.nth_largest_element(
                logits, n=sampling_keep_top_k,
                reduced_dim=other_features["vocab_dim"])
            logits = mtf.where(mtf.less_equal(logits, k_largest),
                               mtf.ones_like(logits) * -1e6, logits)

        ids_this_step = mtf.sample_with_temperature(
            logits, other_features["vocab_dim"], temperature)
        new_position = position + 1
        new_ids = ids + ids_this_step * mtf.one_hot(
            position, length_dim, dtype=tf.int32)
        return [new_position, new_ids]

    while_loop_inputs = [initial_position, inputs]
    final_position, outputs = mtf.while_loop(
        cond_fn, body_fn, while_loop_inputs)[:2]
    del final_position
    if has_partial_sequences and remove_partial_sequences:
        # remove partial sequences from outputs
        partial_length = mtf.reduce_sum(
            mtf.to_int32(mtf.not_equal(partial_sequences, 0)),
            reduced_dim=length_dim)
        outputs = mtf.dynamic_shift(
            outputs, -partial_length, length_dim, wrap=False)
    return outputs
