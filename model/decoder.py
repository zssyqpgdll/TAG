import tensorflow as tf
from decode_utils import gumbel_softmax

class BasicDecoderWithCopy(tf.contrib.seq2seq.BasicDecoder):

    def __init__(self,
                 cell,
                 helper,
                 initial_state,
                 output_layer,
                 copy_score_fn,
                 node_states,
                 node_value_states,
                 encoder_mask,
                 encoder_value_mask,
                 tgt_encoder_mask,
                 tgt_encoder_value_mask,
                 node_pointer_net,
                 node_value_pointer_net,
                 copy_vocab_probs_layer,
                 tree2nl_ids,
                 tgt_vocab_size,
                 tgt_with_src_in_tgt_unk_vocab_size,
                 unk_id,
                 classifier,
                 temperature,
                 copy_decay_keep_prob,
                 copy_once=False,
                 predict_copy_once=False,
                 copy_decay=False,):
        '''
        :param cell:
        :param helper:
        :param initial_state:
        :param output_layer:
        :param copy_score_fn:
        :param node_states:
        :param node_value_states:
        :param encoder_mask:
        :param encoder_value_mask:
        :param tgt_encoder_mask:
        :param tgt_encoder_value_mask:
        :param node_pointer_net:
        :param node_value_pointer_net:
        :param copy_vocab_probs_layer:
        :param tree2nl_ids: [batch_size ,max_node_count, max_value_len]
        :param tgt_vocab_size:
        :param tgt_with_src_in_tgt_unk_vocab_size:

        :return []
        '''

        assert  output_layer is not None, "output_layer should not be None"
        super(BasicDecoderWithCopy, self).__init__(cell, helper, initial_state, output_layer)

        self._copy_score_fn = copy_score_fn

        self._node_states = node_states

        self._node_value_states = node_value_states

        self._encoder_mask = encoder_mask

        self._encoder_value_mask = encoder_value_mask

        self._tgt_encoder_mask = tgt_encoder_mask

        self._tgt_encoder_value_mask = tgt_encoder_value_mask

        self._node_pointer_net = node_pointer_net

        self._node_value_pointer_net = node_value_pointer_net

        self._copy_vocab_probs_layer = copy_vocab_probs_layer

        self.tgt_vocab_size = tgt_vocab_size

        self.tgt_with_src_in_tgt_unk_vocab_size = tgt_with_src_in_tgt_unk_vocab_size

        self.added_unk_vocab_size = self.tgt_with_src_in_tgt_unk_vocab_size - self.tgt_vocab_size

        self.tree2nl_ids = tree2nl_ids

        self.unk_id = unk_id

        self.classifier = classifier

        self.temperature = temperature

        # self.copy_probs_variable = tf.get_variable(trainable=False, shape=[self.tgt_with_src_in_tgt_unk_vocab_size], name='copy_probs_variable', dtype=tf.float32)

        # self.reset_op = self.copy_probs_variable.assign(tf.zeros(shape=[self.tgt_with_src_in_tgt_unk_vocab_size], dtype=tf.float32), use_locking=True)

        # [batch_size, max_time, max_node_count], used when copy_once is True
        # self._stored_copy_history = tf.zeros(shape=[self.batch_size], dtype=tf.int32)

        self._copy_once = copy_once

        self._predict_copy_once = predict_copy_once

        self._copy_decay = copy_decay

        self.copy_decay_keep_prob = copy_decay_keep_prob

    @property
    def output_size(self):
        # Return the cell output and the id
        return tf.contrib.seq2seq.BasicDecoderOutput(
            rnn_output=self.tgt_with_src_in_tgt_unk_vocab_size,
            sample_id=self._helper.sample_ids_shape)

    def initialize(self, name=None):
        """Initialize the decoder.
           Args:
             name: Name scope for any created operations.
           Returns:
             `(finished, first_inputs, initial_state)`.
           """
        result = self._helper.initialize() + (self._initial_state,)

        shape = tf.shape(self._encoder_mask)
        batch_size = shape[0]
        max_node_count = shape[1]

        stored_copy_history = -1*tf.ones(shape=[batch_size], dtype=tf.int32)

        copy_decay_rates = tf.zeros(shape=[batch_size, max_node_count], dtype=tf.float32)

        copy_time_record = tf.zeros(shape=[batch_size, max_node_count], dtype=tf.int32)

        result = result + (stored_copy_history, self._tgt_encoder_mask, self._encoder_mask, copy_decay_rates, copy_time_record)

        return result

    def __copy_score__(self, cell_outputs, time, stored_copy_history, tgt_encoder_mask, encoder_mask, copy_decay_rates, copy_time_record, copy_vocab_probs):
        '''
        :param cell_outputs: [batch_size, cell.output_size]
        :param time: a scalar
        :param stored_copy_history: [batch_size]
        :param tgt_encoder_mask: [batch_size, max_time, max_node_count]
        :param encoder_mask: [batch_size, max_node_count]
        :param copy_decay_rates: [batch_size, max_node_count]
        :param copy_time_record: [batch_size, max_node_count]
        :param copy_vocab_probs: [batch_size, 2]
        :return: [batch_size, tgt_vocab_size]
        '''

        if self._copy_once:
            new_tgt_encoder_mask = self.mask_already_copy_before(time, stored_copy_history, tgt_encoder_mask)
            new_encoder_mask = encoder_mask
        elif self._predict_copy_once:
            new_tgt_encoder_mask = tgt_encoder_mask
            new_encoder_mask = self.mask_already_copy_before_encoder_mask(time, stored_copy_history, encoder_mask)
        else:
            # copy decay
            new_tgt_encoder_mask = tgt_encoder_mask
            new_encoder_mask = encoder_mask

        # [batch_size, max_node_count, max_value_len (1)]
        cur_step_infer_node_value_probs = self._copy_score_fn(decoder_query=cell_outputs,
                             node_states=self._node_states,
                             node_value_states=self._node_value_states,
                             encoder_mask=new_encoder_mask,
                             encoder_value_mask=self._encoder_value_mask,
                             tgt_encoder_mask=new_tgt_encoder_mask,
                             tgt_encoder_value_mask=self._tgt_encoder_value_mask,
                             node_pointer_net=self._node_pointer_net,
                             node_value_pointer_net=self._node_value_pointer_net,
                             decoder_time_step=time,)


        shape = tf.shape(cur_step_infer_node_value_probs)

        batch_size = shape[0]
        max_node_count = shape[1]

        # [batch_size, max_node_count*max_value_len (1)]
        reshape_infer_node_value_probs = tf.reshape(cur_step_infer_node_value_probs, shape=[batch_size, -1])


        if self._copy_decay:

            reshape_infer_node_value_probs = reshape_infer_node_value_probs * (1.0 - copy_decay_rates)

            # [batch_size]
            max_copy_prob_node_idx = tf.argmax(reshape_infer_node_value_probs, axis=-1, output_type=tf.int32)

            # [batch_size]
            max_copy_prob_node_idx = tf.where(
                copy_vocab_probs[:, 0] < copy_vocab_probs[:, 1],
                x=max_copy_prob_node_idx, y=-1 * tf.ones(shape=[batch_size], dtype=tf.int32))

            # [batch_size, max_node_count]
            one_hot = tf.one_hot(indices=max_copy_prob_node_idx, depth=max_node_count, dtype=tf.int32)

            # [batch_size, max_node_count]
            new_copy_time_record = copy_time_record + one_hot

            # [batch_size, max_node_count]
            copy_decay_rates = copy_decay_rates * self.copy_decay_keep_prob

            # [batch_size, max_node_count]
            new_copy_decay_rates = tf.where(one_hot > 0,
                                        x=self.copy_decay_keep_prob * tf.ones_like(copy_decay_rates, dtype=tf.float32),
                                        y=copy_decay_rates)

        else:
            new_copy_decay_rates = copy_decay_rates
            new_copy_time_record = copy_time_record

        # [batch_size, max_node_count*max_value_len (1), 1]
        reshape_tree2nl_ids = tf.reshape(self.tree2nl_ids, shape=[batch_size, -1, 1])

        # copy_probs_variable = self.copy_probs_variable

        # reset_op = self.reset_op

        def map_fn(indices, updates):

            # with tf.control_dependencies([reset_op]):
            #
            #     scatter_add = tf.scatter_add(copy_probs_variable, indices, updates)
            #
            #     return scatter_add

            shape = tf.constant([self.tgt_with_src_in_tgt_unk_vocab_size])

            scatter_add = tf.scatter_nd(indices=indices, updates=updates, shape=shape)

            return scatter_add

        # [batch_size, tgt_with_src_in_tgt_unk_vocab_size]
        copy_scores = tf.map_fn(lambda x: map_fn(x[0], x[1]), elems=(reshape_tree2nl_ids, reshape_infer_node_value_probs), dtype=tf.float32)

        reshape_infer_node_probs = reshape_infer_node_value_probs

        return copy_scores, reshape_infer_node_probs, new_tgt_encoder_mask, new_encoder_mask, new_copy_decay_rates, new_copy_time_record

    def mask_already_copy_before(self, time, stored_copy_history, tgt_encoder_mask):
        # store_copy_history [batch_size]
        # tgt_encoder_mask = [batch_size, max_time, max_node_count]
        # [batch_size, 1]
        stored_copy_history_indices = tf.expand_dims(stored_copy_history, axis=1)

        # before_indices = -1*tf.ones(shape=[self.batch_size, time], dtype=tf.int32)
        # back_indices = -1*tf.ones(shape=[self.batch_size, self.max_time-1-time], dtype=tf.int32)

        # [batch_size, max_time]
        # all_indices = tf.concat([before_indices, stored_copy_history_indices, back_indices], new_axis=-1)

        shape = tf.shape(tgt_encoder_mask)
        max_time = shape[1]
        max_node_count = shape[2]

        # [batch_size, max_time]
        all_indices = tf.tile(stored_copy_history_indices, multiples=[1, max_time])

        # [batch_size, max_time, max_node_count]
        one_hot = tf.one_hot(indices=all_indices, depth=max_node_count, axis=-1, dtype=tf.float32)

        # we filter the some unneccesary nodes and time-steps, using the original tgt encoder mask
        filter_one_hot = self._tgt_encoder_mask * one_hot

        assert_op = tf.Assert(tf.logical_or(tf.reduce_all((tgt_encoder_mask-filter_one_hot) >= 0.0), tf.reduce_all(tf.equal(tgt_encoder_mask, 0.0))), ['tgt_encoder_mask - filter one_hot some elements are < 0, they must be >= 0 for validation'])

        with tf.control_dependencies([assert_op]):
            return tgt_encoder_mask - filter_one_hot

    def mask_already_copy_before_encoder_mask(self, time, stored_copy_history, encoder_mask):
        # stored_copy_history [batch_size]
        # encoder_mask [batch_size, max_node_count]
        max_node_count = tf.shape(encoder_mask)[1]

        # [batch_size, max_node_count]
        one_hot = tf.one_hot(indices=stored_copy_history, depth=max_node_count, axis=-1, dtype=tf.float32)

        filter_one_hot = self._encoder_mask * one_hot

        def fn(encoder_mask, filter_one_hot):
            # encoder_mask [max_node_count]
            # filter_one_hot [max_node_count]
            return tf.logical_or(tf.reduce_all((encoder_mask - filter_one_hot) >= 0.0), tf.reduce_all(tf.equal(encoder_mask, 0.0)))

        # [batch_size]
        check_result = tf.map_fn(fn=lambda x: fn(x[0], x[1]), elems=(encoder_mask, filter_one_hot), dtype=tf.bool)

        assert_op = tf.Assert(tf.reduce_all(check_result), ['encoder mask - filter one hot some elements are < 0, they must be >= 0 for validation'])
        # assert_op = tf.Assert(tf.reduce_all((encoder_mask - filter_one_hot) >= 0.0), ['encoder mask - filter one hot some elements are < 0, they must be >= 0 for validation'])


        with tf.control_dependencies([assert_op]):
            return tf.where((encoder_mask-filter_one_hot) >= 0.0, x=encoder_mask-filter_one_hot, y=encoder_mask)

    def step(self, time, inputs, state, stored_copy_history, tgt_encoder_mask, encoder_mask, copy_decay_rates, copy_time_record, name=None):
        """Perform a decoding step.
        Args:
          time: scalar `int32` tensor.
          inputs: A (structure of) input tensors.
          state: A (structure of) state tensors and TensorArrays.
          name: Name scope for any created operations.
        Returns:
          `(outputs, next_state, next_inputs, finished)`.
        """
        with tf.name_scope(name, "BasicDecoderStep", (time, inputs, state)):
            # cell_outputs=[batch_size, cell.output_size]
            cell_outputs, cell_state = self._cell(inputs, state)

            # calculate classifier probs
            # [batch_size, 2]
            copy_vocab_probs = self._copy_vocab_probs_layer(cell_outputs)

            # calculate copy scores
            # [batch_size, tgt_with_src_in_tgt_unk_vocab_size], [batch_size, max_node_count], [batch_size, max_time, max_node_count]
            copy_scores, infer_node_probs, new_tgt_encoder_mask, new_encoder_mask, new_copy_decay_rates, new_copy_time_record = self.__copy_score__(cell_outputs,
                                                                                                        time,
                                                                                                        stored_copy_history,
                                                                                                        tgt_encoder_mask,
                                                                                                        encoder_mask,
                                                                                                        copy_decay_rates,
                                                                                                        copy_time_record,
                                                                                                        copy_vocab_probs)

            batch_size = tf.shape(copy_scores)[0]

            # calculate copy and vocab gen probs
            # calculate vocab generatation score
            cell_outputs = self._output_layer(cell_outputs)

            # [batch_size, tgt_vocab_size]
            vocab_gen_probs = tf.nn.softmax(cell_outputs, axis=-1)

            assert self.classifier in ['hard', 'soft', 'gumbel', 'rl'], "the argument classifier not in ['hard', 'soft', 'gumbel', 'rl']"

            if self.classifier == "hard":
                # [batch_size, added_unk_vocab_size]
                concat_zeros = tf.zeros(shape=[batch_size, self.added_unk_vocab_size], dtype=tf.float32)

                # [batch_size, tgt_with_src_in_tgt_unk_vocab_size]
                vocab_gen_probs = tf.concat([vocab_gen_probs, concat_zeros], axis=-1)

            elif self.classifier == "soft":
                # enable the target vocabulary's unk prob is added into where in the src in tgt unk area (vocab index >= tgt_vocab_size)
                # [batch_size] - > [batch_size, 1] - > [batch_size, added_unk_vocab_size]
                concat_unk_probs = tf.tile(tf.expand_dims(vocab_gen_probs[:, self.unk_id], axis=1), multiples=[1, self.added_unk_vocab_size])

                # [batch_size, tgt_with_src_in_tgt_unk_vocab_size]
                vocab_gen_probs = tf.concat([vocab_gen_probs, concat_unk_probs], axis=-1)

            elif self.classifier == "gumbel":
                # # in gumbel softmax sampling, copy_vocab_probs  is logit, no softmax activation
                # # [batch_size, 2]
                # infer_gumbel_sample = gumbel_softmax(copy_vocab_probs, tf.shape(copy_vocab_probs), temperature=self.temperature)
                #
                # # add softmax activation into copy_vocab_probs
                # copy_vocab_probs = tf.nn.softmax(copy_vocab_probs)

                # gumbel softmax or gumbel trick
                # [batch_size, added_unk_vocab_size]
                concat_zeros = tf.zeros(shape=[batch_size, self.added_unk_vocab_size], dtype=tf.float32)

                # [batch_size, tgt_with_src_in_tgt_unk_vocab_size]
                vocab_gen_probs = tf.concat([vocab_gen_probs, concat_zeros], axis=-1)
            else:
                # rl
                # [batch_size, added_unk_vocab_size]
                concat_zeros = tf.zeros(shape=[batch_size, self.added_unk_vocab_size], dtype=tf.float32)

                # [batch_size, tgt_with_src_in_tgt_unk_vocab_size]
                vocab_gen_probs = tf.concat([vocab_gen_probs, concat_zeros], axis=-1)

                # # enable the target vocabulary's unk prob is added into where in the src in tgt unk area (vocab index >= tgt_vocab_size)
                # # [batch_size] - > [batch_size, 1] - > [batch_size, added_unk_vocab_size]
                # concat_unk_probs = tf.tile(tf.expand_dims(vocab_gen_probs[:, self.unk_id], axis=1),
                #                            multiples=[1, self.added_unk_vocab_size])
                #
                # # [batch_size, tgt_with_src_in_tgt_unk_vocab_size]
                # vocab_gen_probs = tf.concat([vocab_gen_probs, concat_unk_probs], axis=-1)

            # vocab_cond_prob = tf.tile(tf.expand_dims(copy_vocab_probs[:, 0], axis=1), multiples=[1, self.tgt_with_src_in_tgt_unk_vocab_size])

            # copy_cond_prob = tf.tile(tf.expand_dims(copy_vocab_probs[:, 1], axis=1), multiples=[1, self.tgt_with_src_in_tgt_unk_vocab_size])

            # when copy_once is True and we choose the copy action, we do not copy the tree node ever used
            if self._copy_once or self._predict_copy_once or self._copy_decay:

                # [batch_size]
                max_node_probs = tf.reduce_max(infer_node_probs, axis=-1)

                # [batch_size]
                zeros = tf.zeros(shape=[batch_size], dtype=tf.float32)

                if self.classifier == "hard" or self.classifier == "soft":
                    # [batch_size]
                    copy_action_probs = tf.where(max_node_probs > 0.0, x=copy_vocab_probs[:, 1], y=zeros)

                    # [batch_size]
                    generate_action_probs = copy_vocab_probs[:, 0]

                else:
                    # gumbel softmax
                    # # [batch_size]
                    # copy_action_probs = tf.where(max_node_probs > 0.0, x=infer_gumbel_sample[:, 1], y=zeros)
                    #
                    # # [batch_size]
                    # generate_action_probs = infer_gumbel_sample[:, 0]

                    # gumbel trick
                    # [batch_size]
                    copy_action_probs = tf.where(max_node_probs > 0.0, x=copy_vocab_probs[:, 1], y=zeros)

                    # [batch_size]
                    generate_action_probs = copy_vocab_probs[:, 0]

                # [batch_size]
                # copy_chosen_flags = tf.cast(copy_vocab_probs[:, 1] >= copy_vocab_probs[:, 0], tf.int32)

                # [batch_size]
                copy_chosen_flags = tf.cast(copy_action_probs >= generate_action_probs, tf.int32)

                # when choosing a generate action, we give corresponding position -1 flag, which means no suitable position in the copied tree
                gen_chosen_flags = -1 * (1 - copy_chosen_flags)

                # [batch_size]
                max_probs_node_idx = tf.argmax(infer_node_probs, axis=-1, output_type=tf.int32)

                # [batch_size]
                next_stored_copy_history = max_probs_node_idx * copy_chosen_flags + gen_chosen_flags

                print("choose copy once action !")

            else:
                # a fake next_stored_copy_history
                next_stored_copy_history = tf.zeros(shape=[batch_size], dtype=tf.int32)
                print("choose normal copy action !")

                if self.classifier == "hard" or self.classifier == "soft":
                    generate_action_probs = copy_vocab_probs[:, 0]

                    copy_action_probs = copy_vocab_probs[:, 1]

                else:
                    # # gumbel softmax
                    # # [batch_size]
                    # generate_action_probs = infer_gumbel_sample[:, 0]
                    #
                    # # [batch_size]
                    # copy_action_probs = infer_gumbel_sample[:, 1]

                    # gumbel trick
                    generate_action_probs = copy_vocab_probs[:, 0]

                    copy_action_probs = copy_vocab_probs[:, 1]


            # [batch_size, tgt_with_src_in_tgt_unk_vocab_size]
            # objective 2 = P(y_t|gen, y_t-1, X) + P(y_t|copy, y_t-1, X)
            # objective_probs = vocab_gen_probs + copy_scores

            if self.classifier == "hard":
                # objective 3 = P(y_t|gen, y_t-1, X) if P(gen|y_t-1, X) > P(copy|y_t-1, X) else P(y_t|copy, y_t-1, X)
                objective_probs = tf.where(generate_action_probs > copy_action_probs, x=vocab_gen_probs, y=copy_scores)

            elif self.classifier == "soft":
                # objective 1 = P(gen|y_t-1, X) x P(y_t|gen, y_t-1, X) + P(copy|y_t-1, X) x P(y_t|copy, y_t-1, X)
                vocab_cond_prob = tf.tile(tf.expand_dims(generate_action_probs, axis=1), multiples=[1, self.tgt_with_src_in_tgt_unk_vocab_size])
                copy_cond_prob = tf.tile(tf.expand_dims(copy_action_probs, axis=1), multiples=[1, self.tgt_with_src_in_tgt_unk_vocab_size])
                # [batch_size, tgt_with_src_in_tgt_unk_vocab_size]
                objective_probs = vocab_cond_prob * vocab_gen_probs + copy_cond_prob * copy_scores

            elif self.classifier == "gumbel":
                # objective 3 = P(y_t|gen, y_t-1, X) if P(gen|y_t-1, X) > P(copy|y_t-1, X) else P(y_t|copy, y_t-1, X)
                objective_probs = tf.where(generate_action_probs > copy_action_probs, x=vocab_gen_probs, y=copy_scores)

            elif self.classifier == "rl":
                # objective 3 = P(y_t|gen, y_t-1, X) if P(gen|y_t-1, X) > P(copy|y_t-1, X) else P(y_t|copy, y_t-1, X)
                objective_probs = tf.where(generate_action_probs > copy_action_probs, x=vocab_gen_probs, y=copy_scores)

                # # objective 1 = P(gen|y_t-1, X) x P(y_t|gen, y_t-1, X) + P(copy|y_t-1, X) x P(y_t|copy, y_t-1, X)
                # vocab_cond_prob = tf.tile(tf.expand_dims(generate_action_probs, axis=1),
                #                           multiples=[1, self.tgt_with_src_in_tgt_unk_vocab_size])
                # copy_cond_prob = tf.tile(tf.expand_dims(copy_action_probs, axis=1),
                #                          multiples=[1, self.tgt_with_src_in_tgt_unk_vocab_size])
                # # [batch_size, tgt_with_src_in_tgt_unk_vocab_size]
                # objective_probs = vocab_cond_prob * vocab_gen_probs + copy_cond_prob * copy_scores

            # test vocab gen
            # objective_probs = vocab_gen_probs

            # [batch_size]
            sample_ids = self._helper.sample(
                time=time, outputs=objective_probs, state=cell_state)

            print("sample_ids:", sample_ids)

            # [batch_size]
            unk_ids = self.unk_id*tf.ones(shape=[tf.shape(sample_ids)[0]], dtype=tf.int32)

            # for the predicted OOV word, we also use the unk word id as the next input
            # [batch_size]
            previous_sample_ids = tf.where(sample_ids < self.tgt_vocab_size, x=sample_ids, y=unk_ids)

            (finished, next_inputs, next_state) = self._helper.next_inputs(
                time=time,
                outputs=cell_outputs,
                state=cell_state,
                sample_ids=previous_sample_ids)

        outputs = tf.contrib.seq2seq.BasicDecoderOutput(objective_probs, sample_ids)
        return (outputs, next_state, next_inputs, finished, next_stored_copy_history, new_tgt_encoder_mask, new_encoder_mask, new_copy_decay_rates, new_copy_time_record)