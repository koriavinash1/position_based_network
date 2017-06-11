n_input = 784
n_classes = 10
image_height = image_width = 28

learning_rate = 0.0001 # use 0.0001 for augmented data
batch_size = 128
dropout = 0.5 
epochs = 150 # use 50 epochs on augmented data


display_steps = batch_size/16

test_examples = 256
validation_examples = 500

IS_POSITION_BASED = False
