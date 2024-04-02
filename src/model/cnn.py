import tensorflow as tf

def build_model(is_training, inputs, params):
    #Build the model up to where it may be used for inference.
    images = inputs['images']
    images = tf.expand_dims(images, axis=-1)

    out = images

    # Define the number of channels of each convolution
    # For each block, do: 3x3 conv -> batch norm -> relu -> 2x2 maxpool
    num_channels = params.num_channels
    bn_momentum = params.bn_momentum
    channels = [num_channels, num_channels * 2, num_channels * 4, num_channels * 8]

    for i, c in enumerate(channels):
        with tf.compat.v1.variable_scope('block_{}'.format(i+1)):
            out = tf.keras.layers.Conv2D(c, 2, padding='same')(out)
            if params.use_batch_norm:
                out = tf.keras.layers.BatchNormalization(momentum=bn_momentum)(out, training=is_training)
            out = tf.nn.relu(out)
            out = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(out)

    print(out.get_shape().as_list())
    #assert out.get_shape().as_list() == [None, 4, 4, num_channels * 8]

    out = tf.reshape(out, [-1, 4 * 4 * num_channels * 8])

    with tf.compat.v1.variable_scope('fc_1'):
        out = tf.keras.layers.Dense(num_channels * 8)(out)
        if params.use_batch_norm:
            out = tf.keras.layers.BatchNormalization(momentum=bn_momentum)(out, training=is_training)
        out = tf.nn.relu(out)

    with tf.compat.v1.variable_scope('fc_2'):
        logits = tf.keras.layers.Dense(params.num_labels)(out)
    print(logits)
    return logits


def model_fn(mode, inputs, params, reuse=False):
    #Model function defining the graph operations.
    is_training = (mode == 'train')
    labels = inputs['labels']
    labels = tf.cast(labels, tf.int64)

    # Layers of the model
    with tf.compat.v1.variable_scope('model', reuse=reuse):
        # Compute the output of the model and the predictions
        logits = build_model(is_training, inputs, params)
        predictions = tf.argmax(logits, 1)

    # Define loss and accuracy
    loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, predictions), tf.float32))

    # Define training step that minimizes the loss with the Adam optimizer
    if is_training:
        optimizer = tf.compat.v1.train.AdamOptimizer(params.learning_rate)
        global_step = tf.compat.v1.train.get_or_create_global_step()
        if params.use_batch_norm:
            # Add a dependency to update the moving mean and variance for batch normalization
            with tf.control_dependencies(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)):
                train_op = optimizer.minimize(loss, global_step=global_step)
        else:
            train_op = optimizer.minimize(loss, global_step=global_step)

    # Metrics for evaluation using tf.metrics (average over whole dataset)
    with tf.compat.v1.variable_scope("metrics"):
        metrics = {
            'accuracy': tf.compat.v1.metrics.accuracy(labels=labels, predictions=predictions),
            'loss': tf.compat.v1.metrics.mean(loss)
        }

    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])

    # Get the op to reset the local variables used in tf.metrics
    metric_variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    metrics_init_op = tf.compat.v1.variables_initializer(metric_variables)

    # Summaries for training
    tf.compat.v1.summary.scalar('loss', loss)
    tf.compat.v1.summary.scalar('accuracy', accuracy)

    # Create the model specification and return it
    model_spec = inputs
    model_spec['variable_init_op'] = tf.compat.v1.global_variables_initializer()
    model_spec["predictions"] = predictions
    model_spec['loss'] = loss
    model_spec['accuracy'] = accuracy
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec['metrics'] = metrics

    if is_training:
        model_spec['train_op'] = train_op

    return model_spec
