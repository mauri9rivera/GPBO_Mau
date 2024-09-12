import tensorflow as tf

def use_cpu(task):
    # Force TensorFlow to use CPU
    with tf.device('/CPU:0'):
        print("Connected to CPU.")

# Example usage connect_to_cpu(task)

