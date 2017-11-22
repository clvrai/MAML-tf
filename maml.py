import ipdb
import os
from tqdm import tqdm

import tensorflow as tf

import utils
from loginfo import log

# global variables for MAML
LOG_FREQ = 100
SAVE_FREQ = 1000
EVAL_FREQ = 1000


class MAML(object):
    def __init__(self, dataset, model_type, loss_type, dim_input, dim_output,
                 alpha, beta, K, batch_size, is_train):
        '''
        model_tpye: choose model tpye for each task, choice: ('fc',)
        loss_type:  choose the form of the objective function
        dim_input:  input dimension
        dim_output: desired output dimension
        alpha:      fixed learning rate to calculate the gradient
        beta:       learning rate used for Adam Optimizer
        K:          perform K-shot learning
        batch_size: number of tasks sampled in each iteration
        '''
        self.sess = utils.get_session(1)
        self.is_train = is_train
        self.dataset = dataset
        self.alpha = alpha
        self.K = K
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.batch_size = batch_size
        self.meta_optimizer = tf.train.AdamOptimizer(beta)
        self.avoid_second_derivative = False
        self.task_name = "MAML.{}_{}-shot_{}-batch".format(dataset.name, self.K, self.batch_size)
        # Build placeholder
        self.build_placeholder()
        # Build model
        model = self.import_model(model_type)
        self.construct_weights = model.construct_weights
        self.contruct_forward = model.construct_forward
        # Loss function
        self.loss_fn = self.get_loss_fn(loss_type)
        self.build_graph(dim_input, dim_output, batch_norm=True)
        # Misc
        self.summary_dir = 'log'
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)
        self.writer = tf.summary.FileWriter(self.summary_dir, self.sess.graph)
        self.checkpoint_dir = os.path.join('checkpoint', self.task_name)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.saver = tf.train.Saver(max_to_keep=10)
        log.infov("Initialize all variables")
        self.sess.run(tf.global_variables_initializer())

    def build_placeholder(self):
        self.meta_train_x = tf.placeholder(tf.float32)
        self.meta_train_y = tf.placeholder(tf.float32)
        self.meta_val_x = tf.placeholder(tf.float32)
        self.meta_val_y = tf.placeholder(tf.float32)

    def import_model(self, model_type):
        if model_type == 'fc':
            import model.fc as model
        else:
            ValueError("Can't recognize the model type {}".format(model_type))
        return model

    def get_loss_fn(self, loss_type):
        if loss_type == 'MSE':
            loss_fn = tf.losses.mean_squared_error
        else:
            ValueError("Can't recognize the loss type {}".format(loss_type))
        return loss_fn

    def build_graph(self, dim_input, dim_output, batch_norm):

        self.weights = self.construct_weights(dim_input, dim_output)

        # Calculate loss on 1 task
        def metastep_graph(inp):
            meta_train_x, meta_train_y, meta_val_x, meta_val_y = inp
            weights = self.weights
            meta_train_output = self.contruct_forward(meta_train_x, weights,
                                                      reuse=False,
                                                      batch_norm=batch_norm)
            # Meta train loss
            meta_train_loss = self.loss_fn(meta_train_y, meta_train_output)
            meta_train_loss = tf.reduce_mean(meta_train_loss)
            grads = dict(zip(weights.keys(),
                         tf.gradients(meta_train_loss, list(weights.values()))))
            new_weights = dict(zip(weights.keys(),
                               [weights[key]-self.alpha*grads[key]
                                for key in weights.keys()]))
            if self.avoid_second_derivative:
                new_weights = tf.stop_gradients(new_weights)
            meta_val_output = self.contruct_forward(meta_val_x, new_weights,
                                                    reuse=True,
                                                    batch_norm=batch_norm)
            # Meta val loss
            meta_val_loss = self.loss_fn(meta_val_y, meta_val_output)
            meta_val_loss = tf.reduce_mean(meta_val_loss)

            return [meta_train_loss, meta_val_loss, meta_train_output, meta_val_output]

        output_dtype = [tf.float32, tf.float32, tf.float32, tf.float32]
        # tf.map_fn: map on the list of tensors unpacked from `elems`
        #               on dimension 0 (Task)
        # reture a packed value
        result = tf.map_fn(metastep_graph,
                           elems=(self.meta_train_x, self.meta_train_y,
                                  self.meta_val_x, self.meta_val_y),
                           dtype=output_dtype, parallel_iterations=self.batch_size)
        meta_train_loss, meta_val_loss, meta_train_output, meta_val_output = result
        self.meta_val_output = meta_val_output
        self.meta_train_output = meta_train_output
        meta_train_loss = tf.reduce_mean(meta_train_loss)
        meta_val_loss = tf.reduce_mean(meta_val_loss)

        # Loss
        self.meta_train_loss = meta_train_loss
        self.meta_val_loss = meta_val_loss
        # Meta training step
        self.meta_train_op = self.meta_optimizer.minimize(meta_val_loss)
        # Summary
        self.meta_train_loss_sum = tf.summary.scalar('loss/meta_train_loss', meta_train_loss)
        self.meta_val_loss_sum = tf.summary.scalar('loss/meta_val_loss', meta_val_loss)
        self.summary_op = tf.summary.merge_all()

    def learn(self, batch_size, dataset, max_steps):
        for step in range(int(max_steps)):
            meta_val_loss, meta_train_loss = self.single_train_step(dataset, batch_size, step)
            if step % LOG_FREQ == 0:
                log.info("Step: {}/{}, Meta train loss: {:.4f}, Meta val loss: {:.4f}".format(
                    step, int(max_steps), meta_train_loss, meta_val_loss))
            if step % SAVE_FREQ == 0:
                log.infov("Save checkpoint-{}".format(step))
                self.saver.save(self.sess, os.path.join(self.checkpoint_dir, 'checkpoint'),
                                global_step=step)
            if step % EVAL_FREQ == 0:
                self.evaluate(dataset, 100, False)

    def single_train_step(self, dataset, batch_size, step):
        batch_input, batch_target, _, _ = dataset.get_batch(batch_size, resample=True)
        feed_dict = {self.meta_train_x: batch_input[:, :self.K, :],
                     self.meta_train_y: batch_target[:, :self.K, :],
                     self.meta_val_x: batch_input[:, self.K:, :],
                     self.meta_val_y: batch_target[:, self.K:, :]}
        _, summary_str, meta_val_loss, meta_train_loss = \
            self.sess.run([self.meta_train_op, self.summary_op,
                           self.meta_val_loss, self.meta_train_loss],
                          feed_dict)
        self.writer.add_summary(summary_str, step)
        return meta_val_loss, meta_train_loss

    def evaluate(self, dataset, test_steps, draw):
        if not self.is_train:
            restore_model_path = tf.train.latest_checkpoint(self.checkpoint_dir)
            assert restore_model_path is not None
            self.saver.restore(self.sess, restore_model_path)
            log.infov('Load model: {}'.format(restore_model_path))
            if draw:
                draw_dir = os.path.join('vis', self.task_name)
                if not os.path.exists:
                    os.makedirs(draw_dir)
        accumulated_val_loss = []
        accumulated_train_loss = []
        for step in tqdm(range(test_steps)):
            output, val_loss, train_loss, amplitude, phase, inp = \
                self.single_test_step(dataset, 1)
            if not self.is_train and draw:
                # visualize
                for am, ph in zip(amplitude, phase):
                    dataset.visualize(am, ph, inp, output,
                                      path=os.path.join(draw_dir, '{}.png'.format(step)))

            accumulated_val_loss.append(val_loss)
            accumulated_train_loss.append(train_loss)
        val_loss_mean = sum(accumulated_val_loss)/test_steps
        train_loss_mean = sum(accumulated_train_loss)/test_steps
        log.infov("[Evaluate] Meta train loss: {:.4f}, Meta val loss: {:.4f}".format(
            train_loss_mean, val_loss_mean))

    def single_test_step(self, dataset, batch_size):
        batch_input, batch_target, amplitude, phase = dataset.get_batch(batch_size, resample=True)
        feed_dict = {self.meta_train_x: batch_input[:, :self.K, :],
                     self.meta_train_y: batch_target[:, :self.K, :],
                     self.meta_val_x: batch_input[:, self.K:, :],
                     self.meta_val_y: batch_target[:, self.K:, :]}
        meta_val_output, meta_val_loss, meta_train_loss = \
            self.sess.run([self.meta_val_output, self.meta_val_loss,
                           self.meta_train_loss],
                          feed_dict)
        return meta_val_output, meta_val_loss, meta_train_loss, amplitude, phase, batch_input
