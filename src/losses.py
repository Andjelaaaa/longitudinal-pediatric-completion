import tensorflow as tf

def diffusion_loss(eps, predicted_eps):
    return tf.reduce_mean(tf.square(eps - predicted_eps))

def fusion_loss(c_pred_p, c_pred_s):
    return tf.reduce_mean(tf.square(c_pred_p - c_pred_s))

def total_loss(eps, predicted_eps, c_pred_p, c_pred_s, lambda_fusion=0.6):
    l_diff = diffusion_loss(eps, predicted_eps)
    l_fusion = fusion_loss(c_pred_p, c_pred_s)
    return l_diff + lambda_fusion * l_fusion
