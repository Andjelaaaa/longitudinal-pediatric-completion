import tensorflow as tf
from .losses import total_loss

@tf.function
def train_step(model, optimizer, inputs, noise_schedule, lambda_fusion):
    p, t, s = inputs
    noise_level = tf.random.uniform([], minval=noise_schedule[0], maxval=noise_schedule[-1])
    noisy_t = t + noise_level * tf.random.normal(tf.shape(t))
    eps = tf.random.normal(tf.shape(t))  # Ground truth noise
    
    with tf.GradientTape() as tape:
        predictions, loci_outputs_p, loci_outputs_s = model([p, noisy_t, s], training=True)
        c_fused = predictions  # Assuming c_fused is the main prediction output
        predicted_eps = c_fused - noisy_t  # Predicted noise
        total_loss_value = total_loss(eps, predicted_eps, loci_outputs_p, loci_outputs_s, lambda_fusion)
    
    gradients = tape.gradient(total_loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return total_loss_value

def train_model(model, train_dataset, noise_schedule, epochs=10, lambda_fusion=0.6):
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4)
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for step, inputs in enumerate(train_dataset):
            loss = train_step(model, optimizer, inputs, noise_schedule, lambda_fusion)
            if step % 10 == 0:
                print(f"Step {step}, Loss: {loss.numpy()}")
