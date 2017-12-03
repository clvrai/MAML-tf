import os
from tqdm import tqdm

import tensorflow as tf

import utils
from loginfo import log

# global variables for MAML
LOG_FREQ = 100
SUMMARY_FREQ = 10
SAVE_FREQ = 1000
EVAL_FREQ = 1000


class MAML(object):
    def __init__(self, dataset, model_type, loss_type, dim_input, dim_output,
                 alpha, beta, K, batch_size, is_train, num_updates, norm):
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
        self._sess = utils.get_session(1)
        self._is_train = is_train
        self._dataset = dataset
        self._alpha = alpha
        self._K = K
        self._norm = norm
        self._dim_input = dim_input
        self._dim_output = dim_output
        self._batch_size = batch_size
        self._num_updates = num_updates
        self._meta_optimizer = tf.train.AdamOptimizer(beta)
        self._avoid_second_derivative = False
        self._task_name = 'MAML.{}_{}-shot_{}-updates_{}-batch_norm-{}'.format(dataset.name, self._K,
                                                                               self._num_updates, self._batch_size,
                                                                               self._norm)
        log.infov('Task name: {}'.format(self._task_name))
        # Build placeholder
        self._build_placeholder()
        # Build model
        model = self._import_model(model_type)
        self._construct_weights = model.construct_weights
        self._contruct_forward = model.construct_forward
        # Loss function
        self._loss_fn = self._get_loss_fn(loss_type)
        self._build_graph(dim_input, dim_output, norm=norm)
        # Misc
        self._summary_dir = os.path.join('log', self._task_name)
        self._checkpoint_dir = os.path.join('checkpoint', self._task_name)
        self._saver = tf.train.Saver(max_to_keep=10)
        if self._is_train:
            if not os.path.exists(self._summary_dir):
                os.makedirs(self._summary_dir)
            self._writer = tf.summary.FileWriter(self._summary_dir, self._sess.graph)
            if not os.path.exists(self._checkpoint_dir):
                os.makedirs(self._checkpoint_dir)
        # Initialize all variables
        log.infov("Initialize all variables")
        self._sess.run(tf.global_variables_initializer())

    def _build_placeholder(self):
        self._meta_train_x = tf.placeholder(tf.float32)
        self._meta_train_y = tf.placeholder(tf.float32)
        self._meta_val_x = tf.placeholder(tf.float32)
        self._meta_val_y = tf.placeholder(tf.float32)

    def _import_model(self, model_type):
        if model_type == 'fc':
            import model.fc as model
        else:
            ValueError("Can't recognize the model type {}".format(model_type))
        return model

    def _get_loss_fn(self, loss_type):
        if loss_type == 'MSE':
            loss_fn = tf.losses.mean_squared_error
        else:
            ValueError("Can't recognize the loss type {}".format(loss_type))
        return loss_fn

    def _build_graph(self, dim_input, dim_output, norm):

        self._weights = self._construct_weights(dim_input, dim_output)

        # Calculate loss on 1 task
        def metastep_graph(inp):
            meta_train_x, meta_train_y, meta_val_x, meta_val_y = inp
            meta_train_loss_list = []
            meta_val_loss_list = []

            weights = self._weights
            meta_train_output = self._contruct_forward(meta_train_x, weights,
                                                       reuse=False, norm=norm,
                                                       is_train=self._is_train)
            # Meta train loss: Calculate gradient
            meta_train_loss = self._loss_fn(meta_train_y, meta_train_output)
            meta_train_loss = tf.reduce_mean(meta_train_loss)
            meta_train_loss_list.append(meta_train_loss)
            grads = dict(zip(weights.keys(),
                         tf.gradients(meta_train_loss, list(weights.values()))))
            new_weights = dict(zip(weights.keys(),
                               [weights[key]-self._alpha*grads[key]
                                for key in weights.keys()]))
            if self._avoid_second_derivative:
                new_weights = tf.stop_gradients(new_weights)
            meta_val_output = self._contruct_forward(meta_val_x, new_weights,
                                                     reuse=True, norm=norm,
                                                     is_train=self._is_train)
            # Meta val loss: Calculate loss (meta step)
            meta_val_loss = self._loss_fn(meta_val_y, meta_val_output)
            meta_val_loss = tf.reduce_mean(meta_val_loss)
            meta_val_loss_list.append(meta_val_loss)
            # If perform multiple updates
            for _ in range(self._num_updates-1):
                meta_train_output = self._contruct_forward(meta_train_x, new_weights,
                                                           reuse=True, norm=norm,
                                                           is_train=self._is_train)
                meta_train_loss = self._loss_fn(meta_train_y, meta_train_output)
                meta_train_loss = tf.reduce_mean(meta_train_loss)
                meta_train_loss_list.append(meta_train_loss)
                grads = dict(zip(new_weights.keys(),
                                 tf.gradients(meta_train_loss, list(new_weights.values()))))
                new_weights = dict(zip(new_weights.keys(),
                                       [new_weights[key]-self._alpha*grads[key]
                                        for key in new_weights.keys()]))
                if self._avoid_second_derivative:
                    new_weights = tf.stop_gradients(new_weights)
                meta_val_output = self._contruct_forward(meta_val_x, new_weights,
                                                         reuse=True, norm=norm,
                                                         is_train=self._is_train)
                meta_val_loss = self._loss_fn(meta_val_y, meta_val_output)
                meta_val_loss = tf.reduce_mean(meta_val_loss)
                meta_val_loss_list.append(meta_val_loss)

            return [meta_train_loss_list, meta_val_loss_list, meta_train_output, meta_val_output]

        output_dtype = [[tf.float32]*self._num_updates, [tf.float32]*self._num_updates,
                        tf.float32, tf.float32]
        # tf.map_fn: map on the list of tensors unpacked from `elems`
        #               on dimension 0 (Task)
        # reture a packed value
        result = tf.map_fn(metastep_graph,
                           elems=(self._meta_train_x, self._meta_train_y,
                                  self._meta_val_x, self._meta_val_y),
                           dtype=output_dtype, parallel_iterations=self._batch_size)
        meta_train_losses, meta_val_losses, meta_train_output, meta_val_output = result
        self._meta_val_output = meta_val_output
        self._meta_train_output = meta_train_output
        # Only look at the last final output
        meta_train_loss = tf.reduce_mean(meta_train_losses[-1])
        meta_val_loss = tf.reduce_mean(meta_val_losses[-1])

        # Loss
        self._meta_train_loss = meta_train_loss
        self._meta_val_loss = meta_val_loss
        # Meta train step
        self._meta_train_op = self._meta_optimizer.minimize(meta_val_loss)
        # Summary
        self._meta_train_loss_sum = tf.summary.scalar('loss/meta_train_loss', meta_train_loss)
        self._meta_val_loss_sum = tf.summary.scalar('loss/meta_val_loss', meta_val_loss)
        self._summary_op = tf.summary.merge_all()

    def learn(self, batch_size, dataset, max_steps):
        for step in range(int(max_steps)):
            meta_val_loss, meta_train_loss, summary_str = self._single_train_step(dataset, batch_size, step)
            # Log/TF_board/Save/Evaluate
            if step % SUMMARY_FREQ == 0:
                self._writer.add_summary(summary_str, step)
            if step % LOG_FREQ == 0:
                log.info("Step: {}/{}, Meta train loss: {:.4f}, Meta val loss: {:.4f}".format(
                    step, int(max_steps), meta_train_loss, meta_val_loss))
            if step % SAVE_FREQ == 0:
                log.infov("Save checkpoint-{}".format(step))
                self._saver.save(self._sess, os.path.join(self._checkpoint_dir, 'checkpoint'),
                                 global_step=step)
            if step % EVAL_FREQ == 0:
                self.evaluate(dataset, 100, False)

    def evaluate(self, dataset, test_steps, draw, **kwargs):
        if not self._is_train:
            assert kwargs['restore_checkpoint'] is not None or \
                kwargs['restore_dir'] is not None
            if kwargs['restore_checkpoint'] is None:
                restore_checkpoint = tf.train.latest_checkpoint(kwargs['restore_dir'])
            else:
                restore_checkpoint = kwargs['restore_checkpoint']
            self._saver.restore(self._sess, restore_checkpoint)
            log.infov('Load model: {}'.format(restore_checkpoint))
            if draw:
                draw_dir = os.path.join('vis', self._task_name)
                if not os.path.exists(draw_dir):
                    os.makedirs(draw_dir)
        accumulated_val_loss = []
        accumulated_train_loss = []
        for step in tqdm(range(test_steps)):
            output, val_loss, train_loss, amplitude, phase, inp = \
                self._single_test_step(dataset, 1)
            if not self._is_train and draw:
                # visualize one by one
                for am, ph in zip(amplitude, phase):
                    dataset.visualize(am, ph, inp[:, self._K:, :], output,
                                      path=os.path.join(draw_dir, '{}.png'.format(step)))

            accumulated_val_loss.append(val_loss)
            accumulated_train_loss.append(train_loss)
        val_loss_mean = sum(accumulated_val_loss)/test_steps
        train_loss_mean = sum(accumulated_train_loss)/test_steps
        log.infov("[Evaluate] Meta train loss: {:.4f}, Meta val loss: {:.4f}".format(
            train_loss_mean, val_loss_mean))

    def _single_train_step(self, dataset, batch_size, step):
        batch_input, batch_target, _, _ = dataset.get_batch(batch_size, resample=True)
        feed_dict = {self._meta_train_x: batch_input[:, :self._K, :],
                     self._meta_train_y: batch_target[:, :self._K, :],
                     self._meta_val_x: batch_input[:, self._K:, :],
                     self._meta_val_y: batch_target[:, self._K:, :]}
        _, summary_str, meta_val_loss, meta_train_loss = \
            self._sess.run([self._meta_train_op, self._summary_op,
                           self._meta_val_loss, self._meta_train_loss],
                           feed_dict)
        return meta_val_loss, meta_train_loss, summary_str

    def _single_test_step(self, dataset, batch_size):
        batch_input, batch_target, amplitude, phase = dataset.get_batch(batch_size, resample=True)
        feed_dict = {self._meta_train_x: batch_input[:, :self._K, :],
                     self._meta_train_y: batch_target[:, :self._K, :],
                     self._meta_val_x: batch_input[:, self._K:, :],
                     self._meta_val_y: batch_target[:, self._K:, :]}
        meta_val_output, meta_val_loss, meta_train_loss = \
            self._sess.run([self._meta_val_output, self._meta_val_loss,
                           self._meta_train_loss],
                           feed_dict)
        return meta_val_output, meta_val_loss, meta_train_loss, amplitude, phase, batch_input
