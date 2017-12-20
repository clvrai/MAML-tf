import tensorflow as tf


def contruct_layer(inp, activation_fn, reuse, norm, is_train, scope):
    if norm == 'batch_norm':
        out = tf.contrib.layers.batch_norm(inp, activation_fn=activation_fn,
                                           reuse=reuse, is_training=is_train,
                                           scope=scope)
    elif norm == 'None':
        out = activation_fn(inp)
    else:
        ValueError('Can\'t recognize {}'.format(norm))
    return out


def get_session(num_cpu):
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=num_cpu,
        intra_op_parallelism_threads=num_cpu)
    tf_config.gpu_options.per_process_gpu_memory_fraction = 1/10.
    tf_config.gpu_options.allow_growth = True
    return tf.Session(config=tf_config)
