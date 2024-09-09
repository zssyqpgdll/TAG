# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Seq2seq layer operations for use in neural networks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest

import tensorflow as tf

from data_utils import post_processing
from evaluate_utils import bleu_fn, rouge_fn
import numpy as np

__all__ = ["Decoder", "dynamic_decode"]


_transpose_batch_time = rnn._transpose_batch_time  # pylint: disable=protected-access
_zero_state_tensors = rnn_cell_impl._zero_state_tensors  # pylint: disable=protected-access


@six.add_metaclass(abc.ABCMeta)
class Decoder(object):
  """An RNN Decoder abstract interface object.
  Concepts used by this interface:
  - `inputs`: (structure of) tensors and TensorArrays that is passed as input to
    the RNNCell composing the decoder, at each time step.
  - `state`: (structure of) tensors and TensorArrays that is passed to the
    RNNCell instance as the state.
  - `finished`: boolean tensor telling whether each sequence in the batch is
    finished.
  - `outputs`: Instance of BasicDecoderOutput. Result of the decoding, at each
    time step.
  """

  @property
  def batch_size(self):
    """The batch size of input values."""
    raise NotImplementedError

  @property
  def output_size(self):
    """A (possibly nested tuple of...) integer[s] or `TensorShape` object[s]."""
    raise NotImplementedError

  @property
  def output_dtype(self):
    """A (possibly nested tuple of...) dtype[s]."""
    raise NotImplementedError

  @abc.abstractmethod
  def initialize(self, name=None):
    """Called before any decoding iterations.
    This methods must compute initial input values and initial state.
    Args:
      name: Name scope for any created operations.
    Returns:
      `(finished, initial_inputs, initial_state)`: initial values of
      'finished' flags, inputs and state.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def step(self, time, inputs, state, name=None):
    """Called per step of decoding (but only once for dynamic decoding).
    Args:
      time: Scalar `int32` tensor. Current step number.
      inputs: RNNCell input (possibly nested tuple of) tensor[s] for this time
        step.
      state: RNNCell state (possibly nested tuple of) tensor[s] from previous
        time step.
      name: Name scope for any created operations.
    Returns:
      `(outputs, next_state, next_inputs, finished)`: `outputs` is an object
      containing the decoder output, `next_state` is a (structure of) state
      tensors and TensorArrays, `next_inputs` is the tensor that should be used
      as input for the next step, `finished` is a boolean tensor telling whether
      the sequence is complete, for each sequence in the batch.
    """
    raise NotImplementedError

  def finalize(self, outputs, final_state, sequence_lengths):
    raise NotImplementedError

  @property
  def tracks_own_finished(self):
    """Describes whether the Decoder keeps track of finished states.
    Most decoders will emit a true/false `finished` value independently
    at each time step.  In this case, the `dynamic_decode` function keeps track
    of which batch entries are already finished, and performs a logical OR to
    insert new batches to the finished set.
    Some decoders, however, shuffle batches / beams between time steps and
    `dynamic_decode` will mix up the finished state across these entries because
    it does not track the reshuffle across time steps.  In this case, it is
    up to the decoder to declare that it will keep track of its own finished
    state by setting this property to `True`.
    Returns:
      Python bool.
    """
    return False


class BaseDecoder(layers.Layer):
  """An RNN Decoder that is based on a Keras layer.
  Concepts used by this interface:
  - `inputs`: (structure of) tensors and TensorArrays that is passed as input to
    the RNNCell composing the decoder, at each time step.
  - `state`: (structure of) tensors and TensorArrays that is passed to the
    RNNCell instance as the state.
  - `memory`: (sturecute of) tensors that is usually the full output of the
    encoder, which will be used for the attention wrapper for the RNNCell.
  - `finished`: boolean tensor telling whether each sequence in the batch is
    finished.
  - `outputs`: Instance of BasicDecoderOutput. Result of the decoding, at each
    time step.
  """

  def __init__(self,
               output_time_major=False,
               impute_finished=False,
               maximum_iterations=None,
               parallel_iterations=32,
               swap_memory=False,
               **kwargs):
    self.output_time_major = output_time_major
    self.impute_finished = impute_finished
    self.maximum_iterations = maximum_iterations
    self.parallel_iterations = parallel_iterations
    self.swap_memory = swap_memory
    super(BaseDecoder, self).__init__(**kwargs)

  def call(self, inputs, initial_state=None, **kwargs):
    init_kwargs = kwargs
    init_kwargs["initial_state"] = initial_state
    return dynamic_decode(self,
                          output_time_major=self.output_time_major,
                          impute_finished=self.impute_finished,
                          maximum_iterations=self.maximum_iterations,
                          parallel_iterations=self.parallel_iterations,
                          swap_memory=self.swap_memory,
                          decoder_init_input=inputs,
                          decoder_init_kwargs=init_kwargs)

  @property
  def batch_size(self):
    """The batch size of input values."""
    raise NotImplementedError

  @property
  def output_size(self):
    """A (possibly nested tuple of...) integer[s] or `TensorShape` object[s]."""
    raise NotImplementedError

  @property
  def output_dtype(self):
    """A (possibly nested tuple of...) dtype[s]."""
    raise NotImplementedError

  def initialize(self, inputs, initial_state=None, **kwargs):
    """Called before any decoding iterations.
    This methods must compute initial input values and initial state.
    Args:
      inputs: (structure of) tensors that contains the input for the decoder. In
        the normal case, its a tensor with shape [batch, timestep, embedding].
      initial_state: (structure of) tensors that contains the initial state for
        the RNNCell.
      **kwargs: Other arguments that are passed in from layer.call() method. It
        could contains item like input sequence_length, or masking for input.
    Returns:
      `(finished, initial_inputs, initial_state)`: initial values of
      'finished' flags, inputs and state.
    """
    raise NotImplementedError

  def step(self, time, inputs, state):
    """Called per step of decoding (but only once for dynamic decoding).
    Args:
      time: Scalar `int32` tensor. Current step number.
      inputs: RNNCell input (possibly nested tuple of) tensor[s] for this time
        step.
      state: RNNCell state (possibly nested tuple of) tensor[s] from previous
        time step.
    Returns:
      `(outputs, next_state, next_inputs, finished)`: `outputs` is an object
      containing the decoder output, `next_state` is a (structure of) state
      tensors and TensorArrays, `next_inputs` is the tensor that should be used
      as input for the next step, `finished` is a boolean tensor telling whether
      the sequence is complete, for each sequence in the batch.
    """
    raise NotImplementedError

  def finalize(self, outputs, final_state, sequence_lengths):
    raise NotImplementedError

  @property
  def tracks_own_finished(self):
    """Describes whether the Decoder keeps track of finished states.
    Most decoders will emit a true/false `finished` value independently
    at each time step.  In this case, the `dynamic_decode` function keeps track
    of which batch entries are already finished, and performs a logical OR to
    insert new batches to the finished set.
    Some decoders, however, shuffle batches / beams between time steps and
    `dynamic_decode` will mix up the finished state across these entries because
    it does not track the reshuffle across time steps.  In this case, it is
    up to the decoder to declare that it will keep track of its own finished
    state by setting this property to `True`.
    Returns:
      Python bool.
    """
    return False

  # TODO(scottzhu): Add build/get_config/from_config and other layer methods.


def _create_zero_outputs(size, dtype, batch_size):
  """Create a zero outputs Tensor structure."""
  def _create(s, d):
    return _zero_state_tensors(s, batch_size, d)

  return nest.map_structure(_create, size, dtype)


def dynamic_decode(decoder,
                   output_time_major=False,
                   impute_finished=False,
                   maximum_iterations=None,
                   parallel_iterations=32,
                   swap_memory=False,
                   scope=None,
                   **kwargs):
  """Perform dynamic decoding with `decoder`.
  Calls initialize() once and step() repeatedly on the Decoder object.
  Args:
    decoder: A `Decoder` instance.
    output_time_major: Python boolean.  Default: `False` (batch major).  If
      `True`, outputs are returned as time major tensors (this mode is faster).
      Otherwise, outputs are returned as batch major tensors (this adds extra
      time to the computation).
    impute_finished: Python boolean.  If `True`, then states for batch
      entries which are marked as finished get copied through and the
      corresponding outputs get zeroed out.  This causes some slowdown at
      each time step, but ensures that the final state and outputs have
      the correct values and that backprop ignores time steps that were
      marked as finished.
    maximum_iterations: `int32` scalar, maximum allowed number of decoding
       steps.  Default is `None` (decode until the decoder is fully done).
    parallel_iterations: Argument passed to `tf.while_loop`.
    swap_memory: Argument passed to `tf.while_loop`.
    scope: Optional variable scope to use.
    **kwargs: dict, other keyword arguments for dynamic_decode. It might contain
      arguments for `BaseDecoder` to initialize, which takes all tensor inputs
      during call().
  Returns:
    `(final_outputs, final_state, final_sequence_lengths)`.
  Raises:
    TypeError: if `decoder` is not an instance of `Decoder`.
    ValueError: if `maximum_iterations` is provided but is not a scalar.
  """
  if not isinstance(decoder, (tf.contrib.seq2seq.Decoder, BaseDecoder)):
    raise TypeError("Expected decoder to be type Decoder, but saw: %s" %
                    type(decoder))

  with variable_scope.variable_scope(scope, "decoder") as varscope:
    # Determine context types.
    ctxt = ops.get_default_graph()._get_control_flow_context()  # pylint: disable=protected-access
    is_xla = control_flow_util.GetContainingXLAContext(ctxt) is not None
    in_while_loop = (
        control_flow_util.GetContainingWhileContext(ctxt) is not None)
    # Properly cache variable values inside the while_loop.
    # Don't set a caching device when running in a loop, since it is possible
    # that train steps could be wrapped in a tf.while_loop. In that scenario
    # caching prevents forward computations in loop iterations from re-reading
    # the updated weights.
    if not context.executing_eagerly() and not in_while_loop:
      if varscope.caching_device is None:
        varscope.set_caching_device(lambda op: op.device)

    if maximum_iterations is not None:
      maximum_iterations = ops.convert_to_tensor(
          maximum_iterations, dtype=dtypes.int32, name="maximum_iterations")
      if maximum_iterations.get_shape().ndims != 0:
        raise ValueError("maximum_iterations must be a scalar")

    if isinstance(decoder, tf.contrib.seq2seq.Decoder):
      # if not decoder._copy_once:
      #   initial_finished, initial_inputs, initial_state = decoder.initialize()
      #
      # else:
      # initial_finished, initial_inputs, initial_state, initial_stored_copy_history, initial_tgt_encoder_mask, initial_encoder_mask, initial_copy_decay_rates, initial_copy_time_record = decoder.initialize()
      initial_res = decoder.initialize()

    else:
      # For BaseDecoder that takes tensor inputs during call.
      decoder_init_input = kwargs.pop("decoder_init_input", None)
      decoder_init_kwargs = kwargs.pop("decoder_init_kwargs", {})

      # if not decoder._copy_once:
      #     initial_finished, initial_inputs, initial_state = decoder.initialize(
      #         decoder_init_input, **decoder_init_kwargs)
      # else:
      # initial_finished, initial_inputs, initial_state, initial_stored_copy_history, initial_tgt_encoder_mask, initial_encoder_mask, initial_copy_decay_rates, initial_copy_time_record = decoder.initialize(
      #         decoder_init_input, **decoder_init_kwargs)

      initial_res = decoder.initialize(decoder_init_input, **decoder_init_kwargs)

    initial_finished = initial_res[0]
    initial_inputs = initial_res[1]
    initial_state = initial_res[2]
    initial_stored_copy_history = initial_res[3]
    initial_tgt_encoder_mask = initial_res[4]
    initial_encoder_mask = initial_res[5]
    initial_copy_decay_rates = initial_res[6]
    initial_copy_time_record = initial_res[7]

    zero_outputs = _create_zero_outputs(decoder.output_size,
                                        decoder.output_dtype,
                                        decoder.batch_size)

    if is_xla and maximum_iterations is None:
      raise ValueError("maximum_iterations is required for XLA compilation.")
    if maximum_iterations is not None:
      initial_finished = math_ops.logical_or(
          initial_finished, 0 >= maximum_iterations)
    initial_sequence_lengths = array_ops.zeros_like(
        initial_finished, dtype=dtypes.int32)
    initial_time = constant_op.constant(0, dtype=dtypes.int32)

    def _shape(batch_size, from_shape):
      if (not isinstance(from_shape, tensor_shape.TensorShape) or
          from_shape.ndims == 0):
        return None
      else:
        batch_size = tensor_util.constant_value(
            ops.convert_to_tensor(
                batch_size, name="batch_size"))
        return tensor_shape.TensorShape([batch_size]).concatenate(from_shape)

    dynamic_size = maximum_iterations is None or not is_xla

    def _create_ta(s, d):
      return tensor_array_ops.TensorArray(
          dtype=d,
          size=0 if dynamic_size else maximum_iterations,
          dynamic_size=dynamic_size,
          element_shape=_shape(decoder.batch_size, s))

    initial_outputs_ta = nest.map_structure(_create_ta, decoder.output_size,
                                            decoder.output_dtype)

    def condition(unused_time, unused_outputs_ta, unused_state, unused_inputs,
                  finished, unused_sequence_lengths, unused_stored_copy_history, unused_tgt_encoder_mask, unused_encoder_mask, unused_copy_decay_rates, unused_copy_time_record):
      return math_ops.logical_not(math_ops.reduce_all(finished))

    def body(time, outputs_ta, state, inputs, finished, sequence_lengths, stored_copy_history, tgt_encoder_mask, encoder_mask, copy_decay_rates, copy_time_record):
      """Internal while_loop body.
      Args:
        time: scalar int32 tensor.
        outputs_ta: structure of TensorArray.
        state: (structure of) state tensors and TensorArrays.
        inputs: (structure of) input tensors.
        finished: bool tensor (keeping track of what's finished).
        sequence_lengths: int32 tensor (keeping track of time of finish).
        stored_copy_history: int32 tensor [batch_size]
      Returns:
        `(time + 1, outputs_ta, next_state, next_inputs, next_finished,
          next_sequence_lengths)`.
        ```
      """
      (next_outputs, decoder_state, next_inputs,
       decoder_finished, next_stored_copy_history, new_tgt_encoder_mask, new_encoder_mask, new_copy_decay_rates, new_copy_time_record) = decoder.step(time,
                                                                                                          inputs,
                                                                                                          state,
                                                                                                          stored_copy_history,
                                                                                                          tgt_encoder_mask,
                                                                                                          encoder_mask,
                                                                                                          copy_decay_rates,
                                                                                                          copy_time_record)
      if decoder.tracks_own_finished:
        next_finished = decoder_finished
      else:
        next_finished = math_ops.logical_or(decoder_finished, finished)
      next_sequence_lengths = array_ops.where(
          math_ops.logical_not(finished),
          array_ops.fill(array_ops.shape(sequence_lengths), time + 1),
          sequence_lengths)

      nest.assert_same_structure(state, decoder_state)
      nest.assert_same_structure(outputs_ta, next_outputs)
      nest.assert_same_structure(inputs, next_inputs)

      # Zero out output values past finish
      if impute_finished:
        emit = nest.map_structure(
            lambda out, zero: array_ops.where(finished, zero, out),
            next_outputs,
            zero_outputs)
      else:
        emit = next_outputs

      # Copy through states past finish
      def _maybe_copy_state(new, cur):
        # TensorArrays and scalar states get passed through.
        if isinstance(cur, tensor_array_ops.TensorArray):
          pass_through = True
        else:
          new.set_shape(cur.shape)
          pass_through = (new.shape.ndims == 0)
        return new if pass_through else array_ops.where(finished, cur, new)

      if impute_finished:
        next_state = nest.map_structure(
            _maybe_copy_state, decoder_state, state)
      else:
        next_state = decoder_state

      outputs_ta = nest.map_structure(lambda ta, out: ta.write(time, out),
                                      outputs_ta, emit)
      return (time + 1, outputs_ta, next_state, next_inputs, next_finished,
              next_sequence_lengths, next_stored_copy_history, new_tgt_encoder_mask, new_encoder_mask, new_copy_decay_rates, new_copy_time_record)

    res = control_flow_ops.while_loop(
        condition,
        body,
        loop_vars=(
            initial_time,
            initial_outputs_ta,
            initial_state,
            initial_inputs,
            initial_finished,
            initial_sequence_lengths,
            initial_stored_copy_history,
            initial_tgt_encoder_mask,
            initial_encoder_mask,
            initial_copy_decay_rates,
            initial_copy_time_record,
        ),
        parallel_iterations=parallel_iterations,
        maximum_iterations=maximum_iterations,
        swap_memory=swap_memory)

    final_outputs_ta = res[1]
    final_state = res[2]
    final_sequence_lengths = res[5]

    final_outputs = nest.map_structure(lambda ta: ta.stack(), final_outputs_ta)

    try:
      final_outputs, final_state = decoder.finalize(
          final_outputs, final_state, final_sequence_lengths)
    except NotImplementedError:
      pass

    if not output_time_major:
      final_outputs = nest.map_structure(_transpose_batch_time, final_outputs)

  return final_outputs, final_state, final_sequence_lengths

def sample_gumble(shape, eps):

    '''
     Sample from Gumbel(0, 1)

    :param shape:
    :param temperature:
    :param eps:
    :return:
    '''
    u = tf.random.uniform(shape=shape, minval=0, maxval=1.0, seed=1)

    g = -tf.log(-tf.log(u + eps) + eps)

    return g

def gumbel_softmax(logits, shape, temperature=1.0, eps=1e-9):

    g = sample_gumble(shape=shape, eps=eps)

    gs = tf.nn.softmax((logits + g) / temperature)

    return gs

def gumbel_trick(p, shape, temperature=1.0, eps=1e-9):

    g = sample_gumble(shape=shape, eps=eps)

    sample = tf.argmax(tf.log(p+eps) + g, axis=-1, output_type=tf.int32)

    # [batch_size, max_time]
    return sample

def classifier_reward(sample, gen_mask, copy_mask):
    '''
    :param sample: [batch_size, max_time, 2]
    :param copy_mask: [batch_size, max_time]
    :param gen_mask: [batch_size, max_time]
    :return:
    '''

    shape = tf.shape(gen_mask)

    ones = tf.ones(shape=shape, dtype=tf.float32)

    zeros = tf.zeros(shape=shape, dtype=tf.float32)

    negative_reward = 0.0*ones

    # [batch_size, max_time]
    correct_gen_reward = tf.where(tf.logical_and(tf.equal(sample[:, :, 0], gen_mask), sample[:, :, 0] > 0.0), x=ones, y=zeros)

    # [batch_size, max_time]
    correct_copy_reward = tf.where(tf.logical_and(tf.equal(sample[:, :, 1], copy_mask), sample[:, :, 1] > 0.0), x=ones, y=zeros)

    # [batch_size, max_time]
    error_gen_reward = tf.where(tf.logical_and(tf.not_equal(sample[:, :, 0], gen_mask), sample[:, :, 0] > 0.0), x=negative_reward, y=zeros)

    # [batch_size, max_time]
    error_copy_reward = tf.where(tf.logical_and(tf.not_equal(sample[:, :, 1], copy_mask), sample[:, :, 1] > 0.0), x=negative_reward, y=zeros)

    # [batch_size, max_time, 1]
    gen_reward = tf.expand_dims(correct_gen_reward + error_gen_reward, axis=2)

    # [batch_size, max_time, 1]
    copy_reward = tf.expand_dims(correct_copy_reward + error_copy_reward, axis=2)

    # [batch_size, max_time, 2]
    classify_reward = tf.concat([gen_reward, copy_reward], axis=-1)

    # [batch_size, max_time, 1]
    output_gen_reward = tf.expand_dims(correct_gen_reward, axis=2)

    # [batch_size, max_time, 1]
    output_copy_reward = tf.expand_dims(correct_copy_reward, axis=2)

    # [batch_size, max_time, 2]
    output_reward = tf.concat([output_gen_reward, output_copy_reward], axis=-1)

    return classify_reward, output_reward

def classifier_reward2(sample, gen_mask, copy_mask):
    shape = tf.shape(gen_mask)

    ones = tf.ones(shape=shape, dtype=tf.float32)

    zeros = tf.zeros(shape=shape, dtype=tf.float32)

    negative_reward = 0.0 * ones

    # [batch_size, max_time]
    correct_gen_reward = tf.where(tf.logical_and(tf.equal(sample[:, :, 0], gen_mask), sample[:, :, 0] > 0.0), x=ones,
                                  y=zeros)

    # [batch_size, max_time]
    correct_copy_reward = tf.where(tf.logical_and(tf.equal(sample[:, :, 1], copy_mask), sample[:, :, 1] > 0.0), x=ones,
                                   y=zeros)

    # [batch_size, max_time]
    error_gen_reward = tf.where(tf.logical_and(tf.not_equal(sample[:, :, 0], gen_mask), sample[:, :, 0] > 0.0),
                                x=negative_reward, y=zeros)

    # [batch_size, max_time]
    error_copy_reward = tf.where(tf.logical_and(tf.not_equal(sample[:, :, 1], copy_mask), sample[:, :, 1] > 0.0),
                                 x=negative_reward, y=zeros)

    # [batch_size, max_time, 1]
    gen_reward = tf.expand_dims(correct_gen_reward + error_gen_reward, axis=2)

    # [batch_size, max_time, 1]
    copy_reward = tf.expand_dims(correct_copy_reward + error_copy_reward, axis=2)

    # [batch_size, max_time, 2]
    classify_reward = tf.concat([gen_reward, copy_reward], axis=-1)

    return classify_reward

def total_reward(classifier_sample, copy_output_sample, gen_output_sample, copy_mask, gen_mask, tgt_encoder_mask, labels):
    '''
    get all reward, used in rl mode, e.t. output and classifier both use sampling to train

    :param classifier_sample: [batch_size, max_time, 2]
    :param copy_output_sample: [batch_size, max_time]
    :param gen_output_sample: [batch_size, max_time]
    :param copy_mask: [batch_size, max_time]
    :param gen_mask: [batch_size, max_time]
    :param tgt_encoder_mask: [batch_size, max_time, max_node_count]
    :param labels: [batch_size, max_time]
    :return: [batch_size, max_time, 2]
    '''

    # [batch_size, max_time]
    copy_labels = tf.argmax(tgt_encoder_mask, axis=-1, output_type=tf.int32)

    # [batch_size, max_time]
    gen_labels = tf.squeeze(labels, axis=2)

    shape = tf.shape(copy_mask)

    ones = tf.ones(shape=shape, dtype=tf.float32)

    zeros = tf.zeros(shape=shape, dtype=tf.float32)

    # [batch_size, max_time]
    cls_correct_gen_reward = tf.where(tf.logical_and(tf.equal(classifier_sample[:, :, 0], gen_mask), classifier_sample[:, :, 0] > 0.0), x=ones, y=zeros)

    # [batch_size, max_time]
    cls_correct_copy_reward = tf.where(tf.logical_and(tf.equal(classifier_sample[:, :, 1], copy_mask), classifier_sample[:, :, 1] > 0.0), x=ones, y=zeros)

    # [batch_size, max_time]
    cls_error_gen_reward = tf.where(tf.logical_and(tf.not_equal(classifier_sample[:, :, 0], gen_mask), classifier_sample[:, :, 0] > 0.0), x=ones, y=zeros)

    # [batch_size, max_time]
    cls_error_copy_reward = tf.where(tf.logical_and(tf.not_equal(classifier_sample[:, :, 1], copy_mask), classifier_sample[:, :, 1] > 0.0), x=ones, y=zeros)

    # [batch_size, max_time, 1]
    cls_gen_reward = tf.expand_dims(cls_correct_gen_reward + cls_error_gen_reward, axis=2)

    # [batch_size, max_time, 1]
    cls_copy_reward = tf.expand_dims(cls_correct_copy_reward + cls_error_copy_reward, axis=2)

    # [batch_size, max_time, 2]
    classify_reward = tf.concat([cls_gen_reward, cls_copy_reward], axis=-1)

    # [batch_size, max_time]
    output_gen_reward = tf.where(tf.logical_and(cls_correct_gen_reward > 0.0, tf.equal(gen_output_sample, gen_labels)), x=ones, y=zeros)

    # [batch_size, max_time]
    output_copy_reward = tf.where(tf.logical_and(cls_correct_copy_reward > 0.0, tf.equal(copy_output_sample, copy_labels)), x=ones, y=zeros)

    # [batch_size, max_time, 1]
    output_gen_reward = tf.expand_dims(output_gen_reward, axis=2)

    # [batch_size, max_time, 1]
    output_copy_reward = tf.expand_dims(output_copy_reward, axis=2)

    # [batch_size, max_time, 2]
    output_reward = tf.concat([output_gen_reward, output_copy_reward], axis=-1)

    # return classify_reward, output_reward

    return output_reward, output_reward

def bleu_reward(batch_sample_id, batch_target, params, eos_id, max_len):

    sample_id = [sample[:sample.index(eos_id)] if eos_id in sample else sample for sample in batch_sample_id]

    sample_id, _ = post_processing(params, sample_id)

    target_id = []

    for sample in batch_target:

        one_example_target_id = []

        for token in sample:
            one_example_target_id.extend(token.split(params.separator))

        target_id.append(one_example_target_id)

    # [batch_size, max_len]
    bleu_rewards = [[bleu_fn(predicts=[sample], targets=[target])]*max_len for sample, target in zip(sample_id, target_id)]

    return bleu_rewards

def bleu_reward_shape(batch_sample_id, batch_target, params, eos_id, batch_size, max_len):

    seq_idx2word = params.s_with_tree_in_seq_unk_idx2word if params.copy else params.s_idx2word

    sample_id = [sample[:sample.index(eos_id)] if eos_id in sample else sample for sample in batch_sample_id]

    sample_id = [[seq_idx2word[word_idx] for word_idx in sample] for sample in sample_id]

    target_id = []

    for sample in batch_target:

        one_example_target_id = []

        for token in sample:
            one_example_target_id.append(token)

        target_id.append(one_example_target_id)


    bleu_rewards = np.zeros(shape=(batch_size, max_len))

    bleu_reward_diffs = np.zeros(shape=(batch_size, max_len))

    assert len(target_id) == len(sample_id), "target id batch size should be equal sample id batch size, but target batch size is " \
    + str(len(target_id)) + ", while sample batch size is " + str(len(sample_id))

    for b, (s_id, t_id) in enumerate(zip(sample_id, target_id)):

        for t, _ in enumerate(s_id):

            bleu_rewards[b, t] = bleu_fn(predicts=[s_id[:t+1]], targets=[t_id])

            if t <= 0:
                bleu_reward_diffs[b, t] = bleu_rewards[b, t]
            else:
                bleu_reward_diffs[b, t] = bleu_rewards[b, t] - bleu_rewards[b, t-1]

    cumulative_bleu_rewards = np.zeros(shape=(batch_size, max_len))

    for b in range(batch_size):

        for t in range(max_len):

            cumulative_bleu_rewards[b, t] = np.sum(bleu_reward_diffs[b, t:])

    return cumulative_bleu_rewards

def rouge_reward_shape(batch_sample_id, batch_target, params, eos_id, batch_size, max_len):

    seq_idx2word = params.s_with_tree_in_seq_unk_idx2word if params.copy else params.s_idx2word

    sample_id = [sample[:sample.index(eos_id)] if eos_id in sample else sample for sample in batch_sample_id]

    sample_id = [[seq_idx2word[word_idx] for word_idx in sample] for sample in sample_id]

    target_id = []

    for sample in batch_target:

        one_example_target_id = []

        for token in sample:
            one_example_target_id.append(token)

        target_id.append(one_example_target_id)


    rouge_rewards = np.zeros(shape=(batch_size, max_len))

    rouge_reward_diffs = np.zeros(shape=(batch_size, max_len))

    assert len(target_id) == len(sample_id), "target id batch size should be equal sample id batch size, but target batch size is " \
    + str(len(target_id)) + ", while sample batch size is " + str(len(sample_id))

    for b, (s_id, t_id) in enumerate(zip(sample_id, target_id)):

        for t, _ in enumerate(s_id):

            rouge_rewards[b, t] = rouge_fn(predicts=[s_id[:t+1]], targets=[t_id])

            if t <= 0:
                rouge_reward_diffs[b, t] = rouge_rewards[b, t]
            else:
                rouge_reward_diffs[b, t] = rouge_rewards[b, t] - rouge_rewards[b, t-1]

    cumulative_rouge_rewards = np.zeros(shape=(batch_size, max_len))

    for b in range(batch_size):

        for t in range(max_len):

            cumulative_rouge_rewards[b, t] = np.sum(rouge_reward_diffs[b, t:])

    return cumulative_rouge_rewards
