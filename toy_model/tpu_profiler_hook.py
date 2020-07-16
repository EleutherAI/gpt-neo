# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

"""TPU Profiler Hook."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import subprocess

import tensorflow.compat.v1 as tf


class TPUProfilerHook(tf.train.SessionRunHook):
  """Captures TPU profiling information every N steps or seconds.
  Collects profiles using the cloud tpu profiler. The hook only works in
  google cloud with cloud_tpu_profiler installed.
  See https://cloud.google.com/tpu/docs/cloud-tpu-tools for detailed usage
  for the capture_tpu_profile command. These profiles can be viewed in
  Tensorboard. TPU profiling should not be invoked more frequently than every
  10 seconds.
  """

  def __init__(self,
               tpu,
               output_dir,
               save_steps=None,
               save_secs=None,
               tpu_profiler_command=None):
    """Initializes a hook that takes periodic profiling snapshots.
    Args:
      tpu: Grpc address to the tpu master.
      output_dir: `string`, the directory to save the profile traces to.
      save_steps: `int`, save profile traces every N steps. Exactly one of
        `save_secs` and `save_steps` should be set.
      save_secs: `int` or `float`, save profile traces every N seconds.
      tpu_profiler_command: Custom tpu profiler command (e.g.
        $install_loc/capture_tpu_profile --duration_ms=20000
        --num_tracing_attempts=10). If not specified, profiling 2 secs with
        3 attempts by default.
    Raises:
      ValueError: if `tpu` is not a string.
    """
    if not isinstance(tpu, str):
      raise ValueError("--tpu should be provided with a string.")

    self._timer = tf.train.SecondOrStepTimer(
        every_secs=save_secs, every_steps=save_steps)
    self._tpu_profiler_command = None

    if tpu_profiler_command is None:
      tpu_profiler_command = ["/usr/local/bin/capture_tpu_profile"]
    self._tpu_profiler_command = tpu_profiler_command
    if tpu.startswith("grpc://"):
      tf.logging.warn(
          "Profiling single TPU pointed by %s. Use tpu name to profile a pod." %
          tpu)
      service_addr = tpu.split("://")[1]
      worker = service_addr.split(":")[0]
      self._tpu_profiler_command += [
          "--service_addr=" + service_addr, "--workers_list=" + worker
      ]
    else:
      self._tpu_profiler_command += ["--tpu=" + tpu]
    self._tpu_profiler_command += ["--logdir=" + output_dir]
    self._running_process = None
    self._ran_first_step = False

  def begin(self):
    self._global_step_tensor = tf.train.get_or_create_global_step()  # pylint: disable=protected-access

  def before_run(self, run_context):
    return tf.train.SessionRunArgs({"global_step": self._global_step_tensor})

  def after_run(self, run_context, run_values):
    stale_global_step = run_values.results["global_step"]
    if not self._ran_first_step:
      # Update the timer so that it does not activate until N steps or seconds
      # have passed.
      self._timer.update_last_triggered_step(stale_global_step)
      self._ran_first_step = True

    global_step = stale_global_step + 1
    if (stale_global_step > 1 and
        self._timer.should_trigger_for_step(stale_global_step)):
      global_step = run_context.session.run(self._global_step_tensor)
      self._timer.update_last_triggered_step(global_step)
      self._collect_tpu_profile(global_step)

  def _collect_tpu_profile(self, step):
    """Run capture_tpu_profile if not already running."""

    if self._running_process is not None:
      exit_code = self._running_process.poll()
      if exit_code is not None:
        tf.logging.info("Previous profile exited with status: %s", exit_code)
      else:
        tf.logging.info(
            "Profiler is already running, skipping collection at step %d", step)
        return
    tf.logging.info(
        "Saving profile at step %d with command %s", step,
        self._tpu_profiler_command)
    self._running_process = subprocess.Popen(self._tpu_profiler_command)