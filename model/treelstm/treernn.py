import tensorflow as tf
# tf.enable_eager_execution()
# import tensorflow.contrib.eager as tfe

# import tree as tr

class Gate1:

    def __init__(self, init_weight, shape_list, activation_fn=None):
        self.add_vars(init_weight, shape_list)
        self.activation_fn = activation_fn

    def add_vars(self, init_weight, shape_list):
        '''
        :param init_weight: a scalar
        :param shape_list: a length-3 list, containing a list element

        Note:
            W: [input_size, hidden_size]
            U: [hidden_size, hidden_size]
            # b: [hidden_size]
        '''
        self.W = tf.Variable(initial_value=tf.random_uniform(shape=shape_list[0], minval=-init_weight, maxval=init_weight),
                                dtype=tf.float32, name="W")
        self.U = tf.Variable(initial_value=tf.random_uniform(shape=shape_list[1], minval=-init_weight, maxval=init_weight),
                                dtype=tf.float32, name="U")
        # self.b = tf.Variable(initial_value=tf.random_uniform(shape=shape_list[2], minval=-init_weight, maxval=init_weight),
        #                         dtype=tf.float32)

    def forward(self, input1, input2):
        '''
        :param input1: [batch_size, input_size]
        :param input2: [batch_size, hidden_size]
        :return: vectors ,shape is [batch_size, hidden_size]
        '''
        output = tf.matmul(input1, self.W) + tf.matmul(input2, self.U)

        if self.activation_fn is not None:
            output = self.activation_fn(output)

        return output

    def get_vars(self):
        return [self.W, self.U]

class ChildSumTreeLSTM():

    def __init__(self, init_weight, embed_size, hidden_size):
        self.add_vars(init_weight, embed_size, hidden_size)
        self.embed_size = embed_size
        self.hidden_size = hidden_size

    def add_vars(self, init_weight, embed_size, hidden_size):
        shape_list = [[embed_size, hidden_size], [hidden_size, hidden_size]]

        with tf.variable_scope("input_gate"):
            # input gate --- sigmoid activation
            self.input_gate = Gate1(init_weight, shape_list, activation_fn=tf.sigmoid)

        with tf.variable_scope("forget_gate"):
            # forget gate --- sigmoid activation
            self.forget_gate = Gate1(init_weight, shape_list, activation_fn=tf.sigmoid)

        with tf.variable_scope("output_gate"):
            # output gate --- sigmoid activation
            self.output_gate = Gate1(init_weight, shape_list, activation_fn=tf.sigmoid)

        with tf.variable_scope("tmp_c"):
            # theorectically, it is an fake gate, define it to a gate just for using gate's setting
            self.tmp_c = Gate1(init_weight, shape_list, activation_fn=tf.nn.tanh)

        self.vars = []
        self.vars.extend(self.input_gate.get_vars())
        self.vars.extend(self.forget_gate.get_vars())
        self.vars.extend(self.output_gate.get_vars())
        self.vars.extend(self.tmp_c.get_vars())

    def forward(self, tree_node):

        x = tree_node.value

        if not tree_node.is_leaf():

            left_h, left_c = self.forward(tree_node.left_child)

            right_h, right_c = self.forward(tree_node.right_child)

        else:
            left_h, left_c = (tf.zeros(shape=[self.hidden_size]), tf.zeros(shape=[self.hidden_size]))

            right_h, right_c = (tf.zeros(shape=[self.hidden_size]), tf.zeros(shape=[self.hidden_size]))

        tmp_h = left_h + right_h

        i = self.input_gate.forward(x, tmp_h)

        left_f = self.forget_gate.forward(x, left_h)
        right_f = self.forget_gate.forward(x, right_h)

        o = self.output_gate.forward(x, tmp_h)

        u = self.tmp_c.forward(x, tmp_h)

        cell_state = i * u + left_f * left_c + right_f * right_c

        hidden_state = o * tf.nn.tanh(cell_state)

        return cell_state, hidden_state

    def get_vars(self):
        return self.vars


class Gate2:

    def __init__(self, init_weight, embed_size, hidden_size, activation_fn=None, layer_norm=False, bias=None):

        self.add_vars(init_weight, embed_size, hidden_size, bias=bias)

        self.activation_fn = activation_fn

        self.layer_norm = layer_norm

        # self.bias = bias

        if layer_norm:
            self.layer_norm_fn = tf.contrib.layers.layer_norm

    def add_vars(self, init_weight, embed_size, hidden_size, bias=None):
        # self.Wx = tf.Variable(
        #     initial_value=tf.random_uniform(shape=[embed_size, hidden_size], minval=-init_weight, maxval=init_weight),
        #     dtype=tf.float32, name="Wx")
        #
        # self.Ulh = tf.Variable(
        #     initial_value=tf.random_uniform(shape=[hidden_size, hidden_size], minval=-init_weight, maxval=init_weight),
        #     dtype=tf.float32, name="Ulh")
        #
        # self.Urh = tf.Variable(
        #     initial_value=tf.random_uniform(shape=[hidden_size, hidden_size], minval=-init_weight, maxval=init_weight),
        #     dtype=tf.float32, name="Urh")

        self.Wx = tf.get_variable(name="Wx",
                                  shape=[embed_size, hidden_size],
                                  initializer=tf.contrib.layers.xavier_initializer(),
                                  dtype=tf.float32)

        self.Ulh = tf.get_variable(name="Ulh",
                                  shape=[hidden_size, hidden_size],
                                  initializer=tf.contrib.layers.xavier_initializer(),
                                  dtype=tf.float32)

        self.Urh = tf.get_variable(name="Urh",
                                   shape=[hidden_size, hidden_size],
                                   initializer=tf.contrib.layers.xavier_initializer(),
                                   dtype=tf.float32)

        if bias is not None:
            # self.bias = tf.Variable(
            #     initial_value=tf.constant(bias, shape=[hidden_size], dtype=tf.float32),
            #     dtype=tf.float32, name="gate_bias")
            self.bias = tf.get_variable(name="gate_bias",
                                       shape=[hidden_size],
                                       initializer=tf.contrib.layers.xavier_initializer(),
                                       dtype=tf.float32)
        else:
            self.bias = None

    def forward(self, x, lchild_h, rchild_h):

        # [batch_size, hidden_size]
        output = tf.matmul(x, self.Wx) + tf.matmul(lchild_h, self.Ulh) + tf.matmul(rchild_h, self.Urh)

        if self.bias is not None:
            output = output + self.bias

        if self.layer_norm:
            output = self.layer_norm_fn(output)

        if self.activation_fn is not None:
            output = self.activation_fn(output)

        return output

    # def get_vars(self):
    #     return self.vars

class MultiGate2:

    def __init__(self, init_weight, type_num, embed_size, hidden_size, activation_fn=None, layer_norm=False, bias=None):
        self.add_vars(init_weight, type_num, embed_size, hidden_size, bias=bias)

        self.activation_fn = activation_fn

        self.layer_norm = layer_norm
        # self.bias = bias

        if layer_norm:
            self.layer_norm_fn = tf.contrib.layers.layer_norm

    def add_vars(self, init_weight, type_num, embed_size, hidden_size, bias=None):

        # if parent weights shares parameters with left child and right child, embed_size must equal hidden_size
        # self.type_embeddings = tf.get_variable(name="type_embeddings", shape=[type_num, embed_size, hidden_size],
        #                                       initializer=tf.random_uniform_initializer(minval=-init_weight, maxval=init_weight),
        #                                       dtype=tf.float32)

        self.parent_type_embeddings = tf.get_variable(name="parent_type_embeddings", shape=[type_num, embed_size, hidden_size],
                                                      initializer=tf.contrib.layers.xavier_initializer(),
                                                      dtype=tf.float32)
        self.child_type_embeddings = tf.get_variable(name="child_type_embeddings",
                                                      shape=[type_num, hidden_size, hidden_size],
                                                      initializer=tf.contrib.layers.xavier_initializer(),
                                                      dtype=tf.float32)

        # self.parent_type_embeddings = tf.Variable(
        #     initial_value=tf.random_uniform(shape=[type_num, embed_size, hidden_size], minval=-init_weight, maxval=init_weight),
        #     dtype=tf.float32, name="parent_type_embeddings")
        #
        # self.child_type_embeddings = tf.Variable(
        #     initial_value=tf.random_uniform(shape=[type_num, hidden_size, hidden_size], minval=-init_weight,
        #                                     maxval=init_weight),
        #     dtype=tf.float32, name="child_type_embeddings")

        if bias is not None:
            # self.bias = tf.Variable(
            #     initial_value=tf.constant(bias, shape=[hidden_size], dtype=tf.float32),
            #     dtype=tf.float32, name="gate_bias")
            self.bias = tf.get_variable(name="gate_bias", shape=[hidden_size], initializer=tf.contrib.layers.xavier_initializer())
        else:
            self.bias = None


    def forward(self, x, lchild_h, rchild_h, parent_type_id, lchild_type_id, rchild_type_id):
        '''
        :param x: [batch_size, embed_size]
        :param lchild_h: [batch_size, hidden_size]
        :param rchild_h: [batch_size, hidden_size]
        :param parent_type_id: [batch_size]
        :param lchild_type_id: [batch_size]
        :param rchild_type_id: [batch_size]
        :return: [batch_size, hidden_size]
        '''

        # # [batch_size, embed_size, hidden_size]
        # Wx = tf.nn.embedding_lookup(self.type_embeddings, parent_type_id)
        #
        # # [batch_size, embed_size, hidden_size]
        # Ulh = tf.nn.embedding_lookup(self.type_embeddings, lchild_type_id)
        #
        # # [batch_size, embed_size, hidden_size]
        # Urh = tf.nn.embedding_lookup(self.type_embeddings, rchild_type_id)

        # [batch_size, embed_size, hidden_size]
        Wx = tf.nn.embedding_lookup(self.parent_type_embeddings, parent_type_id)

        # [batch_size, hidden_size, hidden_size]
        Ulh = tf.nn.embedding_lookup(self.child_type_embeddings, lchild_type_id)

        # [batch_size, hidden_size, hidden_size]
        Urh = tf.nn.embedding_lookup(self.child_type_embeddings, rchild_type_id)

        # print("Wx:", Wx)
        # print("Ulh:", Ulh)
        # print("Urh:", Urh)

        # [batch_size, 1, embed_size]
        x = tf.expand_dims(x, axis=1)

        # [batch_size, 1, hidden_size]
        lchild_h = tf.expand_dims(lchild_h, axis=1)

        # [batch_size, 1, hidden_size]
        rchild_h = tf.expand_dims(rchild_h, axis=1)

        # [batch_size, 1, hidden_size] = [batch_size, 1, hidden_size] + [batch_size, 1, hidden_size] + [batch_size, 1, hidden_size]
        output = tf.matmul(x, Wx) + tf.matmul(lchild_h, Ulh) + tf.matmul(rchild_h, Urh)

        # [batch_size, hidden_size]
        output = tf.squeeze(input=output, axis=1)

        if self.bias is not None:
            output = output + self.bias

        if self.layer_norm:
            output = self.layer_norm_fn(output)

        if self.activation_fn is not None:
            output = self.activation_fn(output)

        return output

class Gate3:

    def __init__(self, init_weight, embed_size, hidden_size, max_child_num, activation_fn=None, layer_norm=False, bias=None, output_dim=None):

        self.output_dim = output_dim

        self.add_vars(init_weight, embed_size, hidden_size, max_child_num=max_child_num, bias=bias)

        self.activation_fn = activation_fn

        self.layer_norm = layer_norm

        self.max_child_num = max_child_num

        # self.bias = bias

        if layer_norm:
            self.layer_norm_fn = tf.contrib.layers.layer_norm

    def add_vars(self, init_weight, embed_size, hidden_size, max_child_num, bias=None):

        self.Wx = tf.get_variable(name="Wx",
                                  shape=[embed_size, hidden_size],
                                  initializer=tf.contrib.layers.xavier_initializer(),
                                  dtype=tf.float32)

        Uhs = tf.get_variable(name="Uhs",
                              shape=[max_child_num, hidden_size, hidden_size],
                              initializer=tf.contrib.layers.xavier_initializer(),
                              dtype=tf.float32
                              )

        self.Uhs = Uhs

        if bias is not None:
            # self.bias = tf.Variable(
            #     initial_value=tf.constant(bias, shape=[hidden_size], dtype=tf.float32),
            #     dtype=tf.float32, name="gate_bias")
            self.bias = tf.get_variable(name="gate_bias",
                                       shape=[hidden_size],
                                       initializer=tf.contrib.layers.xavier_initializer(),
                                       dtype=tf.float32)
        else:
            self.bias = None

        if self.output_dim is not None:
            self.output_linear = tf.get_variable(name="output_linear",
                                                shape=[hidden_size, self.output_dim],
                                                initializer=tf.contrib.layers.xavier_initializer(),
                                                dtype=tf.float32)

    def forward(self, x, hs):
        '''
        :param x:
        :param hs: [batch_size, max_child_num, hidden_size]
        :return: if output_dim is not None [batch_size, output_dim], else [batch_size, hidden_size]
        '''

        # [batch_size, hidden_size]
        # output = tf.matmul(x, self.Wx) + tf.matmul(lchild_h, self.Ulh) + tf.matmul(rchild_h, self.Urh)

        # [max_child_num, batch_size, hidden_size]
        hs = tf.transpose(hs, perm=[1, 0, 2])

        # [batch_size, hidden_size]
        output = tf.matmul(x, self.Wx) + tf.reduce_sum(tf.matmul(hs, self.Uhs), axis=0)

        # for h, Uh in zip(hs, self.Uhs):
        #     output = output + tf.matmul(h, Uh)

        if self.bias is not None:
            output = output + self.bias

        if self.output_dim is not None:
            output = tf.matmul(output, self.output_linear)

        if self.layer_norm:
            output = self.layer_norm_fn(output)

        if self.activation_fn is not None:
            output = self.activation_fn(output)

        return output

class MultiGate3:

    def __init__(self, init_weight, type_num, embed_size, hidden_size, max_child_num, activation_fn=None, layer_norm=False, bias=None, parent_type_embeddings=None):

        self.add_vars(init_weight, type_num, embed_size, hidden_size, max_child_num=max_child_num, bias=bias, parent_type_embeddings=parent_type_embeddings)

        self.activation_fn = activation_fn

        self.layer_norm = layer_norm
        # self.bias = bias

        if layer_norm:
            self.layer_norm_fn = tf.contrib.layers.layer_norm

    def add_vars(self, init_weight, type_num, embed_size, hidden_size, max_child_num, bias=None, parent_type_embeddings=None):

        # if parent weights shares parameters with left child and right child, embed_size must equal hidden_size
        # self.type_embeddings = tf.get_variable(name="type_embeddings", shape=[type_num, embed_size, hidden_size],
        #                                       initializer=tf.random_uniform_initializer(minval=-init_weight, maxval=init_weight),
        #                                       dtype=tf.float32)

        if parent_type_embeddings is None:
            self.parent_type_embeddings = tf.get_variable(name="parent_type_embeddings", shape=[type_num, embed_size, hidden_size],
                                                          initializer=tf.contrib.layers.xavier_initializer(),
                                                          dtype=tf.float32)
        else:
            # used by child forget gate
            self.parent_type_embeddings = parent_type_embeddings

        self.child_type_embeddings = tf.get_variable(name="child_type_embeddings",
                                                      shape=[type_num, hidden_size, hidden_size],
                                                      initializer=tf.contrib.layers.xavier_initializer(),
                                                      dtype=tf.float32)

        # self.parent_type_embeddings = tf.Variable(
        #     initial_value=tf.random_uniform(shape=[type_num, embed_size, hidden_size], minval=-init_weight, maxval=init_weight),
        #     dtype=tf.float32, name="parent_type_embeddings")
        #
        # self.child_type_embeddings = tf.Variable(
        #     initial_value=tf.random_uniform(shape=[type_num, hidden_size, hidden_size], minval=-init_weight,
        #                                     maxval=init_weight),
        #     dtype=tf.float32, name="child_type_embeddings")

        if bias is not None:
            # self.bias = tf.Variable(
            #     initial_value=tf.constant(bias, shape=[hidden_size], dtype=tf.float32),
            #     dtype=tf.float32, name="gate_bias")
            self.bias = tf.get_variable(name="gate_bias", shape=[hidden_size], initializer=tf.contrib.layers.xavier_initializer())
        else:
            self.bias = None


    def forward(self, x, hs, parent_type_id, child_type_ids):
        '''
        :param x: [batch_size, embed_size]
        :param hs: [batch_size, max_child_num, hidden_size]
        :param parent_type_id: [batch_size]
        :param child_type_ids: [batch_size, max_child_num]
        :return: [batch_size, hidden_size]
        '''

        # # [batch_size, embed_size, hidden_size]
        # Wx = tf.nn.embedding_lookup(self.type_embeddings, parent_type_id)
        #
        # # [batch_size, max_child_num, embed_size, hidden_size]
        # Ulh = tf.nn.embedding_lookup(self.type_embeddings, child_type_ids)
        #
        # # [batch_size, max_child_num, embed_size, hidden_size]
        # Urh = tf.nn.embedding_lookup(self.type_embeddings, child_type_ids)

        # [batch_size, embed_size, hidden_size]
        Wx = tf.nn.embedding_lookup(self.parent_type_embeddings, parent_type_id)

        # [batch_size, max_child_num, hidden_size, hidden_size]
        Uhs = tf.nn.embedding_lookup(self.child_type_embeddings, child_type_ids)

        # print("Wx:", Wx)
        # print("Uhs:", Uhs)
        # [batch_size, 1, embed_size]
        x = tf.expand_dims(x, axis=1)

        # [batch_size, max_child_num, 1, hidden_size]
        hs = tf.expand_dims(hs, axis=2)

        # [batch_size, 1, hidden_size] = [batch_size, 1, hidden_size] + [batch_size, 1, hidden_size]
        output = tf.matmul(x, Wx) + tf.reduce_sum(tf.matmul(hs, Uhs), axis=1)

        # [batch_size, hidden_size]
        output = tf.squeeze(input=output, axis=1)

        if self.bias is not None:
            output = output + self.bias

        if self.layer_norm:
            output = self.layer_norm_fn(output)

        if self.activation_fn is not None:
            output = self.activation_fn(output)

        return output

class BiAryTreeLSTM():

    def __init__(self, init_weight, embed_size, hidden_size, layer_norm=False, prefix='biarytreelstm'):
        self.add_vars(init_weight, embed_size, hidden_size, layer_norm, prefix=prefix)
        self.layer_norm = layer_norm


    def add_vars(self, init_weight, embed_size, hidden_size, layer_norm, prefix=''):

        with tf.variable_scope(prefix + "_input_gate"):
            self.input_gate = Gate2(init_weight, embed_size, hidden_size, activation_fn=tf.sigmoid, layer_norm=layer_norm)

        with tf.variable_scope(prefix + "_lchild_forget_gate"):
            self.lforget_gate = Gate2(init_weight, embed_size, hidden_size, activation_fn=tf.sigmoid, bias=1.0, layer_norm=layer_norm)

        with tf.variable_scope(prefix + "_rchild_forget_gate"):
            self.rforget_gate = Gate2(init_weight, embed_size, hidden_size, activation_fn=tf.sigmoid, bias=1.0, layer_norm=layer_norm)

        with tf.variable_scope(prefix + "_output_gate"):
            self.output_gate = Gate2(init_weight, embed_size, hidden_size, activation_fn=tf.sigmoid, layer_norm=layer_norm)

        with tf.variable_scope(prefix + "_tmp_c"):
            self.tmp_c = Gate2(init_weight, embed_size, hidden_size, activation_fn=tf.nn.tanh, layer_norm=layer_norm)

        # self.vars = []
        # self.vars.extend(self.input_gate.get_vars())
        # self.vars.extend(self.lforget_gate.get_vars())
        # self.vars.extend(self.rforget_gate.get_vars())
        # self.vars.extend(self.output_gate.get_vars())
        # self.vars.extend(self.tmp_c.get_vars())

    def forward(self, x, child_h, child_c):
        '''
        :param x: [batch_size, embed_size]
        :param child_h: ([batch_size, hidden_size], [batch_size, hidden_size])
        :param child_c: ([batch_size, hidden_size], [batch_size, hidden_size])
        :return:  hidden_state [batch_size, hidden_size]
                  cell_state  [batch_size, hidden_size]
        '''
        lchild_h, rchild_h = child_h
        lchild_c, rchild_c = child_c

        i = self.input_gate.forward(x, lchild_h, rchild_h)

        flh = self.lforget_gate.forward(x, lchild_h, rchild_h)
        frh = self.rforget_gate.forward(x, lchild_h, rchild_h)

        o = self.output_gate.forward(x, lchild_h, rchild_h)

        tmp_c = self.tmp_c.forward(x, lchild_h, rchild_h)

        cell_state = i * tmp_c + flh * lchild_c + frh * rchild_c

        # add layer normalization to cell state
        if self.layer_norm:
            cell_state = tf.contrib.layers.layer_norm(cell_state)

        hidden_state = o * tf.nn.tanh(cell_state)

        return hidden_state, cell_state

    # def get_vars(self):
    #     return self.vars

class BiAryTreeMultiGateLSTM():

    def __init__(self, init_weight, type_num, embed_size, hidden_size, layer_norm=False, prefix='biarytreemultigatelstm'):
        self.add_vars(init_weight, type_num, embed_size, hidden_size, layer_norm, prefix=prefix)
        self.layer_norm = layer_norm

    def add_vars(self, init_weight, type_num, embed_size, hidden_size, layer_norm, prefix=''):

        with tf.variable_scope(prefix + "_input_gate"):
            self.input_gate = MultiGate2(init_weight, type_num, embed_size, hidden_size, activation_fn=tf.sigmoid, layer_norm=layer_norm)

        with tf.variable_scope(prefix + "_lchild_forget_gate"):
            self.lforget_gate = MultiGate2(init_weight, type_num, embed_size, hidden_size, activation_fn=tf.sigmoid, bias=1.0, layer_norm=layer_norm)

        with tf.variable_scope(prefix + "_rchild_forget_gate"):
            self.rforget_gate = MultiGate2(init_weight, type_num, embed_size, hidden_size, activation_fn=tf.sigmoid, bias=1.0, layer_norm=layer_norm)

        with tf.variable_scope(prefix + "_output_gate"):
            self.output_gate = MultiGate2(init_weight, type_num, embed_size, hidden_size, activation_fn=tf.sigmoid, layer_norm=layer_norm)

        with tf.variable_scope(prefix + "_tmp_c"):
            self.tmp_c = MultiGate2(init_weight, type_num, embed_size, hidden_size, activation_fn=tf.nn.tanh, layer_norm=layer_norm)


    def forward(self, x, child_h, child_c, parent_type_id, lchild_type_id, rchild_type_id):
        '''
        :param x: [batch_size, embed_size]
        :param child_h: [batch_size, hidden_size]
        :param child_c: [batch_size, hidden_size]
        :param parent_type_id: [batch_size]
        :param lchild_type_id: [batch_size]
        :param rchild_type_id: [batch_size]
        :return:  hidden_state [batch_size, hidden_size]
                  cell_state  [batch_size, hidden_size]
        '''
        lchild_h, rchild_h = child_h
        lchild_c, rchild_c = child_c

        i = self.input_gate.forward(x, lchild_h, rchild_h, parent_type_id, lchild_type_id, rchild_type_id)

        flh = self.lforget_gate.forward(x, lchild_h, rchild_h, parent_type_id, lchild_type_id, rchild_type_id)
        frh = self.rforget_gate.forward(x, lchild_h, rchild_h, parent_type_id, lchild_type_id, rchild_type_id)

        o = self.output_gate.forward(x, lchild_h, rchild_h, parent_type_id, lchild_type_id, rchild_type_id)

        tmp_c = self.tmp_c.forward(x, lchild_h, rchild_h, parent_type_id, lchild_type_id, rchild_type_id)

        cell_state = i * tmp_c + flh * lchild_c + frh * rchild_c

        # add layer normalization to cell state
        if self.layer_norm:
            cell_state = tf.contrib.layers.layer_norm(cell_state)

        hidden_state = o * tf.nn.tanh(cell_state)

        return hidden_state, cell_state

class NAryTreeLSTM():

    def __init__(self, init_weight, embed_size, hidden_size, max_child_num, layer_norm=False, prefix='narytreelstm'):
        self.add_vars(init_weight, embed_size, hidden_size, layer_norm, max_child_num, prefix=prefix)
        self.layer_norm = layer_norm


    def add_vars(self, init_weight, embed_size, hidden_size, layer_norm, max_child_num, prefix=''):

        with tf.variable_scope(prefix + "_input_gate"):
            self.input_gate = Gate3(init_weight, embed_size, hidden_size, max_child_num=max_child_num, activation_fn=tf.sigmoid, layer_norm=layer_norm)

        forget_gates = []

        for i in range(max_child_num):
            with tf.variable_scope(prefix + "_child_forget_gate_" + str(i)):
                f_gate = Gate3(init_weight, embed_size, hidden_size, max_child_num=max_child_num, activation_fn=tf.sigmoid, bias=1.0, layer_norm=layer_norm)
            forget_gates.append(f_gate)

        self.forget_gates = forget_gates

        # with tf.variable_scope(prefix + "_lchild_forget_gate"):
        #     self.lforget_gate = Gate2(init_weight, embed_size, hidden_size, activation_fn=tf.sigmoid, bias=1.0, layer_norm=layer_norm)
        #
        # with tf.variable_scope(prefix + "_rchild_forget_gate"):
        #     self.rforget_gate = Gate2(init_weight, embed_size, hidden_size, activation_fn=tf.sigmoid, bias=1.0, layer_norm=layer_norm)


        with tf.variable_scope(prefix + "_output_gate"):
            self.output_gate = Gate3(init_weight, embed_size, hidden_size, max_child_num=max_child_num, activation_fn=tf.sigmoid, layer_norm=layer_norm)

        with tf.variable_scope(prefix + "_tmp_c"):
            self.tmp_c = Gate3(init_weight, embed_size, hidden_size, max_child_num=max_child_num, activation_fn=tf.nn.tanh, layer_norm=layer_norm)

        # self.vars = []
        # self.vars.extend(self.input_gate.get_vars())
        # self.vars.extend(self.lforget_gate.get_vars())
        # self.vars.extend(self.rforget_gate.get_vars())
        # self.vars.extend(self.output_gate.get_vars())
        # self.vars.extend(self.tmp_c.get_vars())

    def forward(self, x, child_h, child_c, parent_type_id, child_type_ids):
        '''
        :param x: [batch_size, embed_size]
        :param child_h: [batch_size, max_child_num, hidden_size]
        :param child_c: [batch_size, max_child_num, hidden_size]
        :return:  hidden_state [batch_size, hidden_size]
                  cell_state  [batch_size, hidden_size]
        '''

        # [batch_size, hidden_size]
        i = self.input_gate.forward(x, child_h)

        # [batch_size, hidden_size]
        o = self.output_gate.forward(x, child_h)

        # [batch_size, hidden_size]
        tmp_c = self.tmp_c.forward(x, child_h)

        fhs = []

        for forget_gate in self.forget_gates:
            fh = forget_gate.forward(x, child_h)
            fhs.append(fh)

        # [max_child_num, batch_size, hidden_size]
        fhs = tf.convert_to_tensor(fhs, dtype=tf.float32)

        # [max_child_num, batch_size, hidden_size]
        child_c = tf.transpose(child_c, perm=[1, 0, 2])

        # [batch_size, hidden_size]
        fh_mul_child_c = tf.reduce_sum(fhs * child_c, axis=0)

        # [batch_size, hidden_size]
        cell_state = i * tmp_c + fh_mul_child_c

        # add layer normalization to cell state
        if self.layer_norm:
            cell_state = tf.contrib.layers.layer_norm(cell_state)

        hidden_state = o * tf.nn.tanh(cell_state)

        return hidden_state, cell_state

class NAryTreeMultiGateLSTM():

    def __init__(self, init_weight, type_num, embed_size, hidden_size, max_child_num, layer_norm=False, prefix='biarytreemultigatelstm'):
        self.add_vars(init_weight, type_num, embed_size, hidden_size, max_child_num, layer_norm, prefix=prefix)
        self.layer_norm = layer_norm

    def add_vars(self, init_weight, type_num, embed_size, hidden_size, max_child_num, layer_norm, prefix=''):

        with tf.variable_scope(prefix + "_input_gate"):
            self.input_gate = MultiGate3(init_weight, type_num, embed_size, hidden_size, max_child_num=max_child_num, activation_fn=tf.sigmoid, layer_norm=layer_norm)

        forget_gates = []

        for i in range(max_child_num):
            with tf.variable_scope(prefix + "_child_forget_gate_" + str(i)):
                forget_gate = MultiGate3(init_weight, type_num, embed_size, hidden_size,
                                               max_child_num=max_child_num, activation_fn=tf.sigmoid, bias=1.0,
                                               layer_norm=layer_norm)
            forget_gates.append(forget_gate)

        self.forget_gates = forget_gates

        with tf.variable_scope(prefix + "_output_gate"):
            self.output_gate = MultiGate3(init_weight, type_num, embed_size, hidden_size, max_child_num=max_child_num, activation_fn=tf.sigmoid, layer_norm=layer_norm)

        with tf.variable_scope(prefix + "_tmp_c"):
            self.tmp_c = MultiGate3(init_weight, type_num, embed_size, hidden_size, max_child_num=max_child_num, activation_fn=tf.nn.tanh, layer_norm=layer_norm)


    def forward(self, x, child_h, child_c, parent_type_id, child_type_ids):
        '''
        :param x: [batch_size, embed_size]
        :param child_h: [batch_size, max_child_num, hidden_size]
        :param child_c: [batch_size, max_child_num, hidden_size]
        :param parent_type_id: [batch_size]
        :param child_type_ids: [batch_size, max_child_num]
        :return:  hidden_state [batch_size, hidden_size]
                  cell_state  [batch_size, hidden_size]
        '''

        i = self.input_gate.forward(x, child_h, parent_type_id, child_type_ids)

        fhs = []

        for forget_gate in self.forget_gates:
            fh = forget_gate.forward(x, child_h, parent_type_id, child_type_ids)
            fhs.append(fh)

        # [max_child_num, batch_size, hidden_size]
        fhs = tf.convert_to_tensor(fhs, dtype=tf.float32)

        o = self.output_gate.forward(x, child_h, parent_type_id, child_type_ids)

        tmp_c = self.tmp_c.forward(x, child_h, parent_type_id, child_type_ids)

        # [max_child_num, batch_size, hidden_size]
        child_c = tf.transpose(child_c, perm=[1, 0, 2])

        # [batch_size, hidden_size]
        cell_state = i * tmp_c + tf.reduce_sum(fhs * child_c, axis=0)

        # add layer normalization to cell state
        if self.layer_norm:
            cell_state = tf.contrib.layers.layer_norm(cell_state)

        hidden_state = o * tf.nn.tanh(cell_state)

        return hidden_state, cell_state

class NAryTreeMultiGateLSTM_ShareForgetGateParentTypeEmbed():

    def __init__(self, init_weight, type_num, embed_size, hidden_size, max_child_num, layer_norm=False, prefix='biarytreemultigatelstm'):
        self.add_vars(init_weight, type_num, embed_size, hidden_size, max_child_num, layer_norm, prefix=prefix)
        self.layer_norm = layer_norm

    def add_vars(self, init_weight, type_num, embed_size, hidden_size, max_child_num, layer_norm, prefix=''):

        with tf.variable_scope(prefix + "_input_gate"):
            self.input_gate = MultiGate3(init_weight, type_num, embed_size, hidden_size, max_child_num=max_child_num, activation_fn=tf.sigmoid, layer_norm=layer_norm)

        with tf.variable_scope(prefix + "shared_forget_gate_parent_scope"):
            parent_type_embeddings = tf.get_variable(name="parent_type_embeddings",
                                                     shape=[type_num, embed_size, hidden_size],
                                                     initializer=tf.contrib.layers.xavier_initializer(),
                                                     dtype=tf.float32)

        forget_gates = []

        for i in range(max_child_num):
            with tf.variable_scope(prefix + "_child_forget_gate_" + str(i)):
                forget_gate = MultiGate3(init_weight, type_num, embed_size, hidden_size,
                                               max_child_num=max_child_num,
                                               activation_fn=tf.sigmoid,
                                               layer_norm=layer_norm,
                                               bias=1.0,
                                               parent_type_embeddings=parent_type_embeddings)
            forget_gates.append(forget_gate)

        self.forget_gates = forget_gates

        with tf.variable_scope(prefix + "_output_gate"):
            self.output_gate = MultiGate3(init_weight, type_num, embed_size, hidden_size, max_child_num=max_child_num, activation_fn=tf.sigmoid, layer_norm=layer_norm)

        with tf.variable_scope(prefix + "_tmp_c"):
            self.tmp_c = MultiGate3(init_weight, type_num, embed_size, hidden_size, max_child_num=max_child_num, activation_fn=tf.nn.tanh, layer_norm=layer_norm)


    def forward(self, x, child_h, child_c, parent_type_id, child_type_ids):
        '''
        :param x: [batch_size, embed_size]
        :param child_h: [batch_size, max_child_num, hidden_size]
        :param child_c: [batch_size, max_child_num, hidden_size]
        :param parent_type_id: [batch_size]
        :param child_type_ids: [batch_size, max_child_num]
        :return:  hidden_state [batch_size, hidden_size]
                  cell_state  [batch_size, hidden_size]
        '''

        i = self.input_gate.forward(x, child_h, parent_type_id, child_type_ids)

        fhs = []

        for forget_gate in self.forget_gates:
            fh = forget_gate.forward(x, child_h, parent_type_id, child_type_ids)
            fhs.append(fh)

        # [max_child_num, batch_size, hidden_size]
        fhs = tf.convert_to_tensor(fhs, dtype=tf.float32)

        o = self.output_gate.forward(x, child_h, parent_type_id, child_type_ids)

        tmp_c = self.tmp_c.forward(x, child_h, parent_type_id, child_type_ids)

        # [max_child_num, batch_size, hidden_size]
        child_c = tf.transpose(child_c, perm=[1, 0, 2])

        # [batch_size, hidden_size]
        cell_state = i * tmp_c + tf.reduce_sum(fhs * child_c, axis=0)

        # add layer normalization to cell state
        if self.layer_norm:
            cell_state = tf.contrib.layers.layer_norm(cell_state)

        hidden_state = o * tf.nn.tanh(cell_state)

        return hidden_state, cell_state


class NAryTreeTypeEmbedLSTM():

    def __init__(self, init_weight, type_num, embed_size, hidden_size, max_child_num, layer_norm=False, prefix='narytreetypeembedlstm'):
        self.add_vars(init_weight, type_num, embed_size, hidden_size, max_child_num, layer_norm, prefix=prefix)
        self.layer_norm = layer_norm

    def add_vars(self, init_weight, type_num, embed_size, hidden_size, max_child_num, layer_norm, prefix=''):

        with tf.variable_scope(prefix + "_type_embeddings_scope"):
            self.type_embeddings = tf.get_variable(name="type_embeddings",
                                  shape=[type_num, hidden_size],
                                  initializer=tf.contrib.layers.xavier_initializer(),
                                  dtype=tf.float32)

        add_type_embed_size = embed_size + hidden_size

        add_type_hidden_size = hidden_size + hidden_size

        with tf.variable_scope(prefix + "_input_gate"):
            self.input_gate = Gate3(init_weight, add_type_embed_size, add_type_hidden_size, max_child_num=max_child_num, activation_fn=tf.sigmoid, layer_norm=layer_norm, output_dim=hidden_size)

        forget_gates = []

        for i in range(max_child_num):
            with tf.variable_scope(prefix + "_child_forget_gate_" + str(i)):
                forget_gate = Gate3(init_weight, add_type_embed_size, add_type_hidden_size, max_child_num=max_child_num, activation_fn=tf.sigmoid, layer_norm=layer_norm, output_dim=hidden_size)
            forget_gates.append(forget_gate)

        self.forget_gates = forget_gates

        with tf.variable_scope(prefix + "_output_gate"):
            self.output_gate = Gate3(init_weight, add_type_embed_size, add_type_hidden_size, max_child_num=max_child_num, activation_fn=tf.sigmoid, layer_norm=layer_norm, output_dim=hidden_size)

        with tf.variable_scope(prefix + "_tmp_c"):
            self.tmp_c = Gate3(init_weight, add_type_embed_size, add_type_hidden_size, max_child_num=max_child_num, activation_fn=tf.sigmoid, layer_norm=layer_norm, output_dim=hidden_size)


    def forward(self, x, child_h, child_c, parent_type_id, child_type_ids):
        '''
        :param x: [batch_size, embed_size]
        :param child_h: [batch_size, max_child_num, hidden_size]
        :param child_c: [batch_size, max_child_num, hidden_size]
        :param parent_type_id: [batch_size]
        :param child_type_ids: [batch_size, max_child_num]
        :return:  hidden_state [batch_size, hidden_size]
                  cell_state  [batch_size, hidden_size]
        '''

        # [batch_size, hidden_size]
        parent_type_embeddings = tf.nn.embedding_lookup(self.type_embeddings, parent_type_id)

        # [batch_size, max_child_num, hidden_size]
        child_type_embeddings = tf.nn.embedding_lookup(self.type_embeddings, child_type_ids)

        # [batch_size, embed_size + hidden_size]
        x = tf.concat([x, parent_type_embeddings], axis=-1)

        # [batch_size, max_child_num, hidden_size + hidden_size]
        child_h = tf.concat([child_h, child_type_embeddings], axis=-1)

        # [batch_size, hidden_size]
        i = self.input_gate.forward(x, child_h)

        fhs = []

        for forget_gate in self.forget_gates:
            fh = forget_gate.forward(x, child_h)
            fhs.append(fh)

        # [max_child_num, batch_size, hidden_size]
        fhs = tf.convert_to_tensor(fhs, dtype=tf.float32)

        # [batch_size, hidden_size]
        o = self.output_gate.forward(x, child_h)

        # [batch_size, hidden_size]
        tmp_c = self.tmp_c.forward(x, child_h)

        # [max_child_num, batch_size, hidden_size]
        child_c = tf.transpose(child_c, perm=[1, 0, 2])

        # [batch_size, hidden_size]
        cell_state = i * tmp_c + tf.reduce_sum(fhs * child_c, axis=0)

        # add layer normalization to cell state
        if self.layer_norm:
            cell_state = tf.contrib.layers.layer_norm(cell_state)

        hidden_state = o * tf.nn.tanh(cell_state)

        return hidden_state, cell_state