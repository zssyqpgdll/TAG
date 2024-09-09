import tensorflow as tf

class PointerNetData:

    def __init__(self,
                 encoder_mask,
                 encoder_value_mask,
                 tgt_encoder_mask,
                 tgt_encoder_value_mask,
                 node_states,
                 node_value_states,
                 copy_vocab_probs_layer,
                 node_pointer_net,
                 node_value_pointer_net,
                 tree2nl_ids,
                 copy_decay_keep_prob):
        '''
        :param encoder_mask: if copy is True, must be not None, shape=[batch_size, max_node_count]
        :param encoder_value_mask: if copy is True, must be not None, shape=[batch_size, max_node_count, max_value_len]
        :param tgt_encoder_mask: if copy is True, must be not NOne, shape=[batch_size, max_time, max_node_count]
        :param tgt_encoder_value_mask: if copy is True, must be not None, shape=[batch_size, max_time, max_node_count, max_value_len]
        :param node_states: if copy is True, must be not None, shape=[batch_size, max_node_count, hidden_size]
        :param node_value_states: if copy is True, must be not None, shape=[batch_size, max_node_count, max_value_len, hidden_size]
        :param copy_vocab_probs_layer: if copy is True, must be not None
        :param node_pointer_net: if copy is True, must be not None
        :param node_value_pointer_net: if copy is True, must be not None
        :param tree2nl_ids: if copy is True, shape=[batch_size, max_node_count, max_value_len]
        :param copy_decay_keep_prob: a scalar, it will work when the flag copy_decay is True
        '''

        self.encoder_mask = encoder_mask

        self.encoder_value_mask = encoder_value_mask

        self.tgt_encoder_mask = tgt_encoder_mask

        self.tgt_encoder_value_mask = tgt_encoder_value_mask

        self.node_states = node_states

        self.node_value_states = node_value_states

        self.copy_vocab_probs_layer = copy_vocab_probs_layer

        self.node_pointer_net = node_pointer_net

        self.node_value_pointer_net = node_value_pointer_net

        # [batch_size, max_node_count, 1]
        self.tree2nl_ids = tree2nl_ids

        self.copy_decay_keep_prob = copy_decay_keep_prob

class PointerNet:

    def __init__(self):
        pass

    # def forward_one_step(self, encoder_states, decoder_query, encoder_mask):
    #     '''
    #     :param encoder_states: [batch_size, max_node_count, hidden_size]
    #     :param decoder_query: [batch_size, hidden_size]
    #     :param encoder_mask: [batch_size, max_node_count], dtype=tf.float32
    #     :return: [batch_size, max_node_count]
    #     '''
    #
    #     def make_neg_inf_mask(encoder_mask):
    #
    #         neg_inf_tensor = -float('inf') + tf.ones_like(encoder_mask, dtype=tf.float32)
    #
    #         zero_tensor = tf.zeros_like(encoder_mask, dtype=tf.float32)
    #
    #         neg_inf_mask = tf.where(tf.equal(encoder_mask, 0.0), x=neg_inf_tensor, y=zero_tensor)
    #
    #         return neg_inf_mask
    #
    #     # [batch_size, 1, hidden_size]
    #     decoder_query = tf.expand_dims(decoder_query, axis=1)
    #
    #     # [batch_size, hidden_size, 1]
    #     decoder_query = tf.transpose(decoder_query, perm=[0, 2, 1])
    #
    #     # [batch_size, max_node_count, 1]
    #     scores = tf.matmul(encoder_states, decoder_query)
    #
    #     # [batch_size, max_node_count]
    #     scores = tf.squeeze(scores, axis=-1)
    #
    #     # [batch_size, max_node_count]
    #     neg_inf_mask = make_neg_inf_mask(encoder_mask)
    #
    #     # [batch_size, max_node_count]
    #     attn_weights = tf.nn.softmax(scores + neg_inf_mask, axis=-1)
    #
    #     return attn_weights

    def forward(self, encoder_states, decoder_query, encoder_mask):
        '''
        :param encoder_states: [batch_size, max_node_count, hidden_size]
        :param decoder_query: [batch_size, max_time, hidden_size]
        :param encoder_mask: [batch_size, max_node_count], dtype=tf.float32
        :return: [batch_size, max_time, max_node_count]
        '''

        def make_neg_inf_mask(encoder_mask):

            # neg_inf_tensor = -float('inf') + tf.ones_like(encoder_mask, dtype=tf.float32)
            #
            # zero_tensor = tf.zeros_like(encoder_mask, dtype=tf.float32)
            #
            # neg_inf_mask = tf.where(tf.equal(encoder_mask, 0.0), x=neg_inf_tensor, y=zero_tensor)

            neg_inf_mask = (1.0 - encoder_mask) * -1e+9

            return neg_inf_mask

        # [batch_size, hidden_size, max_node_count]
        encoder_states = tf.transpose(encoder_states, perm=[0, 2, 1])

        # [batch_size, max_time, max_node_count]
        scores = tf.matmul(decoder_query, encoder_states)

        # print_op = tf.print("scores:", scores)

        # [batch_size, max_node_count]
        neg_inf_mask = make_neg_inf_mask(encoder_mask)

        max_time = tf.shape(scores)[1]

        # [batch_size, max_time, max_node_count]
        neg_inf_mask = tf.tile(tf.expand_dims(neg_inf_mask, axis=1), multiples=[1, max_time, 1])

        # [batch_size, max_time, max_node_count]
        attn_weights = tf.nn.softmax(scores + neg_inf_mask, axis=-1)


        return attn_weights




