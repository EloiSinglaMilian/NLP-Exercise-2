from Exercise_2_functions import *

# The file should be a .txt in the same folder as the .py one
# Each line should contain one single name/word
m1_file = "names_cat.txt"
m2_file = "names_spa.txt"

# These parameters are shared between all models
general_param = {
    # Especial character for marking start/end
    # It should not occur in the files
    # Further down is a function to check that
    "especial_character": "#",
    # If True, non-alphabetic characters in the data set will be replaced
    # By characters in the 26 letters of the alphabet or deleted
    # According to the dictionary that is defined further down
    "use_26": False,
    # How many epochs have to pass to print the current loss
    "loss_printing_interval": 1000,
    # Set to True only if you want to know the testing loss
    "calculate_testing_loss": True,
    # Set to True only if youw want to calculate the loss
    # Of the names used on testing evaluation for one model
    # On the other one
    "calculate_transfer_loss": True,
    # If True, a plot will be shown of the loss across the training epochs
    "generate_loss_plot": True,
    # If True, a plot will be shown of the character embeddings on a 2D space
    "generate_embedding_plot": True,
    # If True, it will save the plots instead of showing them
    "save_loss_plot": True,
    "save_embedding_plot": True,
    # How many new names or words to generate using the trained models
    "new_names": 25,
}

# These parameters are unique to each model
m1_param = {
    # The length of the embedding vector for each of the characters in the set
    "vector_length": 3,
    # If set to "max", it will be equal
    # to the length of the longest word in the dataset
    "block_size": 3,
    "hidden_dimensions": 50,
    "epochs": 30000,
    "batch_size": 32,
    "seed": 7012003,
    # Learning rate according to the epoch
    # The keys are the epoch number and the values, the learning rate
    # A function inside the training loop
    # Will select the smallest key that's bigger than the current epoch
    # And set the learning rate to its value
    # If the defined epochs are greater than what the dictionary contains
    # The learning rate will keep being the last one (the one with the greatest key)
    "lr_rate_dict": {15000: 0.1, 25000: 0.05, 30000: 0.01},
}

m2_param = {
    "vector_length": 5,
    "block_size": 3,
    "hidden_dimensions": 100,
    "epochs": 30000,
    "batch_size": 32,
    "seed": 11091997,
    "lr_rate_dict": {15000: 0.1, 25000: 0.05, 30000: 0.01},
}


# We get the processed names from the files,
# The raw names (without spli lines),
# And the updated model parameters in case we update the block size to max length
m1_names, m1_names_raw, m1_param = process_file(m1_file, general_param, m1_param, replacements)
m2_names, m2_names_raw, m2_param = process_file(m2_file, general_param, m2_param, replacements)


# We update the general parameters to get the string to index
# And index to string dictionaries
general_param = generate_vocab([m1_names_raw, m2_names_raw], general_param)


# We process the names to get the processed data
# For training, validating, and testing
m1_data = get_X_Y(m1_names, general_param, m1_param)
m2_data = get_X_Y(m2_names, general_param, m2_param)


# We train the models and also store the losses
m1_nn, m1_losses = train_nn(general_param, m1_param, m1_data, "m1")
m2_nn, m2_losses = train_nn(general_param, m2_param, m2_data, "m2")


# We evaluate them and generate plots for the losses and embeddings
evaluate_nn(m1_nn, m1_data, m1_param, general_param, m1_losses, "m1")
evaluate_nn(m2_nn, m2_data, m2_param, general_param, m2_losses, "m2")


# We generate new names with the models
generate_names(m1_nn, m1_param, general_param, "m1")
generate_names(m2_nn, m2_param, general_param, "m2")


# We calculate the loss of using one model's testing set on the other
calculate_transfer(m1_nn, m2_nn, m1_data, m2_data, m1_param, m2_param, "m1", "m2")