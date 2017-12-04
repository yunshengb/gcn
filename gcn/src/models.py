from layers import *
from metrics import *

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None

        self.printer = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """

        def call_layer(layer):
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)

        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            call_layer(layer)
        if FLAGS.embed == 3:
            self.ssl_outputs = self.activations[-1]
            self.usl_outputs = self.usl_layers[0](self.activations[2])
        else:
            self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                      scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()

        if FLAGS.embed == 3:
            self.ssl_opt_op = self.optimizer.minimize(self.ssl_loss)
            self.usl_opt_op = self.optimizer.minimize(self.usl_loss)
        else:
            self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class GCN(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.inputs = placeholders.get('features')
        self.input_dim = input_dim
        if FLAGS.embed == 0 or FLAGS.embed == 3:
            self.ssl_output_dim = \
            placeholders['ssl_labels'].get_shape().as_list()[1]
        if FLAGS.embed == 2 or FLAGS.embed == 3:
            self.usl_output_dim = \
            placeholders['usl_labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=FLAGS.learning_rate)

        # self.optimizer = tf.train.GradientDescentOptimizer(
        #     learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # # Weight decay loss
        # for var in self.layers[0].vars.values():
        #     self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        train_mask = self.placeholders.get('train_mask')
        if FLAGS.embed == 3:
            self.ssl_labels = self.placeholders['ssl_labels']
            ssl_loss = masked_softmax_cross_entropy(self.ssl_outputs,
                                                    self.ssl_labels,
                                                    train_mask,
                                                    model=self)
            self.ssl_loss = self.loss + ssl_loss
            self.usl_labels = self.placeholders['usl_labels']
            usl_loss = masked_softmax_cross_entropy(self.usl_outputs,
                                                    self.usl_labels,
                                                    None,
                                                    model=self)
            self.usl_loss = self.loss + usl_loss
        elif FLAGS.embed == 0:
            self.ssl_labels = self.placeholders['ssl_labels']
            loss = masked_softmax_cross_entropy(self.outputs,
                                                self.ssl_labels,
                                                train_mask,
                                                model=self)
            self.loss += loss
        elif FLAGS.embed == 2:
            self.usl_labels = self.placeholders['usl_labels']
            loss = masked_softmax_cross_entropy(self.outputs,
                                                self.usl_labels,
                                                None,
                                                model=self)
            self.loss += loss

    def _accuracy(self):
        self.accuracy = accuracy(self.outputs,
                                 self.placeholders['labels'])

    def _build(self):

        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=150,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=0.,
                                            sparse_inputs=False,
                                            featureless=self.inputs is None,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=150,
                                            output_dim=100,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=0.,
                                            sparse_inputs=False,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=100,
                                            output_dim=70,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=0.,
                                            sparse_inputs=False,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=70,
                                            output_dim=39,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=0.,
                                            sparse_inputs=False,
                                            logging=self.logging))


        # self.layers.append(GraphConvolution(input_dim=FLAGS.hidden2,
        #                                     output_dim=39,
        #                                     placeholders=self.placeholders,
        #                                     act=tf.nn.relu,
        #                                     dropout=0,
        #                                     logging=self.logging))



        # if FLAGS.embed == 0 or FLAGS.embed == 3:
        #     if FLAGS.embed == 0:
        #         layers = self.layers
        #     else:
        #         self.ssl_layers = []
        #         layers = self.ssl_layers
        #     layers.append(Dense(input_dim=FLAGS.hidden2,
        #                         output_dim=self.ssl_output_dim,
        #                         placeholders=self.placeholders,
        #                         act=lambda x : x,
        #                         dropout=0,
        #                         logging=self.logging))

        if FLAGS.embed == 2 or FLAGS.embed == 3:
            if FLAGS.embed == 2:
                layers = self.layers
            else:
                self.usl_layers = []
                layers = self.usl_layers
            layers.append(Embedding(input_dim=100,
                                    output_dim=self.usl_output_dim,
                                    placeholders=self.placeholders,
                                    act=lambda x: x,
                                    dropout=False,
                                    logging=self.logging,
                                    model=self))

    def predict(self):
        return tf.nn.softmax(self.outputs)
