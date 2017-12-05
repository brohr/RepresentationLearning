import tensorflow as tf



def classification_loss(inputs, outputs):
    labels = outputs['labels']
    logits = outputs['pred']
    clf_loss =  tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    reg_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    return clf_loss + reg_loss
 
def rotation_loss(inputs, outputs):
    labels = outputs['labels_rotation']
    logits = outputs['pred_rotation']
    rot_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    reg_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    return rot_loss + reg_loss

def multitask_loss(inputs, outputs):
    clf_loss = classification_loss(inputs,outputs)
    rot_loss = rotation_loss(intpus, outputs)
    reg_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    return clf_loss + rot_loss + reg_loss
