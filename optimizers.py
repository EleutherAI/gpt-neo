import numpy as np
import tensorflow as tf


def create_train_op(loss, params):
    lr = params["lr"]
    if "warmup_steps" in params.keys():
        lr = cosine_decay_with_warmup(tf.train.get_global_step(), lr,
                                        params["max_steps"], warmup_steps=params["warmup_steps"])

    if params["opt_name"] == "adam":
        if not "weight_decay" in params.keys():
            optimizer = tf.train.AdamOptimizer(
                learning_rate=lr,
                beta1=params["beta1"],
                beta2=params["beta2"],
                epsilon=params["epsilon"])

        else:
            optimizer = tf.contrib.opt.AdamWOptimizer(
                learning_rate=lr,
                weight_decay=lr*params["weight_decay"],
                beta1=params["beta1"],
                beta2=params["beta2"],
                epsilon=params["epsilon"])

    elif params["opt_name"] == "adafactor":
        if params["decay_type"] == "adam":
            decay_rate = adafactor_decay_rate_adam(params["beta2"])
        elif params["decay_type"] == "pow":
            decay_rate = adafactor_decay_rate_pow(params["decay_exponent"])
        else:
            raise ValueError("unknown optimizer_adafactor_decay_type")

        if not "weight_decay" in params.keys():
            optimizer = AdafactorOptimizer(
                learning_rate=lr,
                decay_rate=decay_rate,
                beta1=params["beta1"],
                name="Adafactor")

        else:
            AdafactorWOptimizer = tf.contrib.opt.extend_with_decoupled_weight_decay(AdafactorOptimizer)

            optimizer = AdafactorWOptimizer(
                weight_decay=params["weight_decay"] * lr,
                learning_rate=lr,
                decay_rate=decay_rate,
                beta1=params["beta1"],
                name="AdafactorW")

    else:
        raise ValueError("Unknown optimizer type!")

    if params["use_tpu"]:
        optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # To update batchnorm, if present
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    train_op = tf.group([train_op, update_ops])

    return train_op



def cosine_decay_with_warmup(global_step,
                             learning_rate_base,
                             total_steps,
                             warmup_learning_rate=0.0,
                             warmup_steps=0,
                             hold_base_rate_steps=0,
                             name="learning_rate"):
    if total_steps < warmup_steps:
        raise ValueError('total_steps must be larger or equal to '
                        'warmup_steps.')
    learning_rate = 0.5 * learning_rate_base * (1 + tf.cos(
        np.pi *
        (tf.cast(global_step, tf.float32) - warmup_steps - hold_base_rate_steps
        ) / float(total_steps - warmup_steps - hold_base_rate_steps)))
    if hold_base_rate_steps > 0:
        learning_rate = tf.where(global_step > warmup_steps + hold_base_rate_steps,
                                learning_rate, learning_rate_base)
    if warmup_steps > 0:
        if learning_rate_base < warmup_learning_rate:
            raise ValueError('learning_rate_base must be larger or equal to '
                        'warmup_learning_rate.')
        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
        warmup_rate = slope * tf.cast(global_step,
                                    tf.float32) + warmup_learning_rate
        learning_rate = tf.where(global_step < warmup_steps, warmup_rate,
                                learning_rate)
    return tf.where(global_step > total_steps, 0.0, learning_rate,
                    name=name)




# Adafactor from tensor2tensor -------------------------------------------------------------

class AdafactorOptimizer(tf.train.Optimizer):
    """Optimizer that implements the Adafactor algorithm.
    Adafactor is described in https://arxiv.org/abs/1804.04235.
    Adafactor is most similar to Adam (Kingma and Ba), the major differences are:
    1. For a two-dimensional AxB weight matrix, Adafactor uses only A+B auxiliary
        parameters to maintain the second-moment estimator, instead of AB.
        This is advantageous on memory-limited systems.  In addition, beta1
        (momentum) is set to zero by default, saving an additional auxiliary
        parameter per weight.  Variables with >=3 dimensions are treated as
        collections of two-dimensional matrices - factorization is over the final
        two dimensions.
    2. Adafactor incorporates "update-clipping" - a scale-invariant analog of
        gradient clipping.  This adds stability
    3. Adafactor does not require an external "learning rate".  By default, it
        incorporates a relative-update-scale schedule, corresponding to
        inverse-square-root learning-rate-decay in ADAM.  We hope this works well
        for most applications.
    ALGORITHM:
    parameter -= absolute_update_scale * clip(grad / grad_scale)
    where:
        absolute_update_scale := relative_update_scale * parameter_scale
        relative_update_scale := min((step_num + 1)**-0.5, 1e-2)
        parameter_scale := max(rms(var)), epsilon2)
        clip(x) := x / max(1.0, rms(x))
        grad_scale := tf.sqrt(v)   (v is the second-moment estimator)
    The second-moment estimator v is maintained in a manner similar to Adam:
    We initialize
    ```
    if var is 2-dimensional:
        v_r <- zeros([num_rows])
        v_c <- zeros([num_cols])
    if var is 0-dimensional or 1-dimensional:
        v <- zeros(shape(var))
    ```
    The update rule is as follows:
    ```
    decay_rate = 1 - (step_num + 1) ^ -0.8
    grad_squared = tf.square(grad) + epsilon1
    if var is 2-dimensional:
        v_r <- decay_rate * v_r + (1 - decay_rate) * reduce_mean(grad_squared, 1)
        v_c <- decay_rate * v_c + (1 - decay_rate) * reduce_mean(grad_squared, 0)
        v = outer_prod(v_r, v_c) / reduce_mean(v_r)
    if var is 0-dimensional or 1-dimensional:
        v <- decay_rate * v + (1 - decay_rate) * grad_squared
    ```
    For variables with >=3 dimensions, we factorize the second-moment accumulator
    over the final 2 dimensions.  See the code for details.
    Several parts of this algorithm are configurable from the initializer.
        multiply_by_parameter_scale:  If True, then compute absolute_update_scale
        as described above.  If False, let absolute_update_scale be the externally
        supplied learning_rate.
        learning_rate: represents relative_update_scale if
        multiply_by_parameter_scale==True, or absolute_update_scale if
        multiply_by_parameter_scale==False.
        decay_rate: Decay rate of the second moment estimator (varies by step_num).
        This should be set to a function such that:
        1-1/(step_num + 1) <= decay_rate(step_num) < 1.0
        beta1: enables momentum, as in Adam.  Uses extra memory if nonzero.
        clipping_threshold: should be >=1.0 or None for no update clipping
        factored: whether to factor the second-moment estimator.  True means
        less memory usage.
    """

    def __init__(self,
                multiply_by_parameter_scale=True,
                learning_rate=None,
                decay_rate=None,
                beta1=0.0,
                clipping_threshold=1.0,
                factored=True,
                use_locking=False,
                name="Adafactor",
                epsilon1=1e-30,
                epsilon2=1e-3):
        """Construct a new Adafactor optimizer.
        See class comment.
        Args:
        multiply_by_parameter_scale: a boolean
        learning_rate: an optional Scalar.
        decay_rate: an optional Scalar.
        beta1: a float value between 0 and 1
        clipping_threshold: an optional float >= 1
        factored: a boolean - whether to use factored second-moment estimator
            for 2d variables
        use_locking: If True use locks for update operations.
        name: Optional name for the operations created when applying gradients.
            Defaults to "AdafactorOptimizer".
        epsilon1: Regularization constant for squared gradient.
        epsilon2: Regularization constant for parameter scale.
        Raises:
        ValueError: if absolute_update_scale and relative_update_scale_fn are both
            present or both absent.
        """
        super(AdafactorOptimizer, self).__init__(use_locking, name)
        self._multiply_by_parameter_scale = multiply_by_parameter_scale
        if learning_rate is None:
            learning_rate = self._learning_rate_default(multiply_by_parameter_scale)
        self._learning_rate = learning_rate
        if decay_rate is None:
            decay_rate = self._decay_rate_default()
        self._decay_rate = decay_rate
        self._beta1 = beta1
        self._clipping_threshold = clipping_threshold
        self._factored = factored
        self._epsilon1 = epsilon1
        self._epsilon2 = epsilon2

    def _should_use_factored_second_moment_estimate(self, shape):
        """Should we use a factored second moment estimator.
        Based on the shape of the variable.
        Args:
        shape: a list of integers
        Returns:
        a boolean
        """
        return self._factored and len(shape) >= 2

    def _create_slots(self, var_list):
        for var in var_list:
            shape = var.get_shape().as_list()
            if self._beta1:
                self._zeros_slot(var, "m", self._name)
            if self._should_use_factored_second_moment_estimate(shape):
                r_val = tf.zeros(shape[:-1], dtype=tf.float32)
                c_val = tf.zeros(shape[:-2] + shape[-1:], dtype=tf.float32)
                self._get_or_make_slot(var, r_val, "vr", self._name)
                self._get_or_make_slot(var, c_val, "vc", self._name)
            else:
                v_val = tf.zeros(shape, dtype=tf.float32)
                self._get_or_make_slot(var, v_val, "v", self._name)

    def _apply_dense(self, grad, var):
        return self._resource_apply_dense(grad, var)

    def _apply_sparse(self, grad, var):
        return self._apply_dense(tf.convert_to_tensor(grad), var)

    def _resource_apply_sparse(self, grad, handle, indices):
        return self._resource_apply_dense(
            tf.convert_to_tensor(tf.IndexedSlices(grad, indices, tf.shape(handle))),
            handle)

    def _parameter_scale(self, var):
        """Estimate the scale of the parameters from the current values.
        We include a minimum value of 0.001 to give it a chance to escape 0
        if it was zero-initialized.
        Instead of using the value, we could impute the scale from the shape,
        as initializers do.
        Args:
        var: a variable or Tensor.
        Returns:
        a Scalar
        """
        return tf.maximum(reduce_rms(var), self._epsilon2)

    def _resource_apply_dense(self, grad, handle):
        var = handle
        grad = tf.to_float(grad)
        grad_squared = tf.square(grad) + self._epsilon1
        grad_squared_mean = tf.reduce_mean(grad_squared)
        decay_rate = self._decay_rate
        update_scale = self._learning_rate
        old_val = var
        if var.dtype.base_dtype == tf.bfloat16:
            old_val = tf.to_float(self._parameter_encoding.decode(old_val))
        if self._multiply_by_parameter_scale:
            update_scale *= tf.to_float(self._parameter_scale(old_val))
        # HACK: Make things dependent on grad.
        # This confounds the XLA rewriter and keeps it from fusing computations
        # across different variables.  This fusion is a bad for HBM usage, since
        # it causes the gradients to persist in memory.
        decay_rate += grad_squared_mean * 1e-30
        update_scale += grad_squared_mean * 1e-30
        # END HACK
        mixing_rate = 1.0 - decay_rate
        shape = var.get_shape().as_list()
        updates = []
        if self._should_use_factored_second_moment_estimate(shape):
            grad_squared_row_mean = tf.reduce_mean(grad_squared, -1)
            grad_squared_col_mean = tf.reduce_mean(grad_squared, -2)
            vr = self.get_slot(var, "vr")
            new_vr = (decay_rate * vr + mixing_rate * grad_squared_row_mean)
            vc = self.get_slot(var, "vc")
            new_vc = (decay_rate * vc + mixing_rate * grad_squared_col_mean)
            vr_update = tf.assign(vr, new_vr, use_locking=self._use_locking)
            vc_update = tf.assign(vc, new_vc, use_locking=self._use_locking)
            updates = [vr_update, vc_update]
            long_term_mean = tf.reduce_mean(new_vr, -1, keepdims=True)
            r_factor = tf.rsqrt(new_vr / long_term_mean)
            c_factor = tf.rsqrt(new_vc)
            x = grad * tf.expand_dims(r_factor, -1) * tf.expand_dims(c_factor, -2)
        else:
            v = self.get_slot(var, "v")
            new_v = decay_rate * v + mixing_rate * grad_squared
            v_update = tf.assign(v, new_v, use_locking=self._use_locking)
            updates = [v_update]
            x = grad * tf.rsqrt(new_v)
        if self._clipping_threshold is not None:
            clipping_denom = tf.maximum(1.0, reduce_rms(x) / self._clipping_threshold)
            x /= clipping_denom
        subtrahend = update_scale * x
        if self._beta1:
            m = self.get_slot(var, "m")
            new_m = self._beta1 * tf.to_float(m) + (1.0 - self._beta1) * subtrahend
            subtrahend = new_m
            new_m = cast_like(new_m, var)
            updates.append(tf.assign(m, new_m, use_locking=self._use_locking))
        new_val = tf.to_float(old_val) - subtrahend
        var_update = tf.assign(var, new_val, use_locking=self._use_locking)
        updates = [var_update] + updates
        return tf.group(*updates)

    def _decay_rate_default(self):
        return adafactor_decay_rate_pow(0.8)

    def _learning_rate_default(self, multiply_by_parameter_scale):
        learning_rate = tf.minimum(tf.rsqrt(step_num() + 1.0), 0.01)
        if not multiply_by_parameter_scale:
            learning_rate *= 0.05
        return learning_rate


def adafactor_decay_rate_adam(beta2):
    t = tf.to_float(tf.train.get_or_create_global_step()) + 1.0
    decay = beta2 * (1.0 - tf.pow(beta2, t - 1.0)) / (1.0 - tf.pow(beta2, t))
    # decay = tf.cond(tf.equal(t, 1.0), lambda: beta2, lambda: decay)
    return decay


def adafactor_decay_rate_pow(exponent):
    return 1.0 - tf.pow((step_num() + 1.0), -exponent)

def step_num():
    return tf.to_float(tf.train.get_or_create_global_step())

def reduce_rms(x):
    return tf.sqrt(tf.reduce_mean(tf.square(x)))

def cast_like(x, y):
    """Cast x to y's dtype, if necessary."""
    x = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)

    if x.dtype.base_dtype == y.dtype.base_dtype:
        return x

    cast_x = tf.cast(x, y.dtype)
    if cast_x.device != x.device:
        x_name = "(eager Tensor)"
        try:
            x_name = x.name
        except AttributeError:
            pass
        tf.logging.warning("Cast for %s may induce copy from '%s' to '%s'", x_name,
                        x.device, cast_x.device)
    return cast_x
