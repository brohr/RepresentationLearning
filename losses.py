import tensorflow as tf



def classification_loss(inputs, outputs):
    labels = outputs['labels']
    logits = outputs['pred']
    return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
 
def rotation_loss(inputs, outputs):
    labels = outputs['labels_rotation']
    logits = outputs['pred_rotation']
    return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)

def multitask_loss(inputs, outputs):
    clf_loss = classification_loss(inputs,outputs)
    rot_loss = rotation_loss(intpus, outputs)
    return clf_loss + rot_loss
