import tensorflow as tf
import wandb
from wandb.keras import WandbCallback
from tensorflow.keras.callbacks import TensorBoard
from model import build_model_3d, denoise
from training import train_model
from training import train_step
from loader import load_and_preprocess_data

def main():
    # Initialize wandb
    wandb.init(project="deep_learning_project", entity="your_wandb_username")

    # Data loading and preprocessing
    train_dataset = load_and_preprocess_data()  # Implement this function to load your data

    # Model building
    input_shape = (64, 64, 64, 1)  # Example input shape
    model, noise_schedule = build_model_3d(input_shape)
    
    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # TensorBoard setup
    tensorboard_callback = TensorBoard(log_dir="./logs")

    # Training the model
    epochs = 10
    lambda_fusion = 0.6

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for step, inputs in enumerate(train_dataset):
            loss = train_step(model, optimizer, inputs, noise_schedule, lambda_fusion)
            if step % 10 == 0:
                print(f"Step {step}, Loss: {loss.numpy()}")
                wandb.log({"loss": loss.numpy()})

        # Log the losses to wandb at the end of each epoch
        wandb.log({"epoch": epoch + 1, "loss": loss.numpy()})

    # Optionally, save the model
    model.save("model.h5")

if __name__ == "__main__":
    main()
