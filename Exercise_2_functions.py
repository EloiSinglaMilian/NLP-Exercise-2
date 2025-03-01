import torch
import torch.nn.functional as F
import re
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# This function takes a dictionary and a text as a string
# Characters to be replaced are the dictionary keys
# And characters to replace them, the values
# Then the replacements are applied in the text
def replace_characters(replacements, text):

    # This first line converts the dictionary to a regex expression
    regex = re.compile("(%s)" % "|".join(map(re.escape, replacements.keys())))

    # We then apply this regex to the text
    # We use lambda to extract every string that matches a pattern
    # And use it as a key to get the replacement from the dictionary
    # And we return the text
    clean_text = regex.sub(lambda x: replacements[x.group()], text)
    return clean_text


# This function will be in the training loop
# And will set the learning rate according to the dictionary we defined above
def get_learning_rate(epoch, model_param):

    # We go over each key from smalles to greatest
    for threshold in sorted(model_param["lr_rate_dict"].keys()):

        # If the epoch is smaller than any given key
        # It will return its value as a learning rate
        if epoch < threshold:
            return model_param["lr_rate_dict"][threshold]

        # If the epoch is bigger than the key we set its value as learning rate anyway
        # Just in case so that if we run out of thresholds we can return the last one
        last_lr = model_param["lr_rate_dict"][threshold]
    return last_lr


# This is the replacement dictionary for the function above
# More can be added or the substitution key can be changed
# These are the ones we found during training (Catalan and Pokémon names)
replacements = {
    "à": "a",
    "á": "a",
    "è": "e",
    "é": "e",
    "ë": "e",
    "í": "i",
    "ò": "o",
    "ó": "o",
    "ú": "u",
    "ü": "u",
    "·l": "",
    "ï": "i",
    "ç": "c",
    "ñ": "n",
    " ": "",
    "\t": "",
    "♀": "",
    "♂": "",
    "'": "",
    "2": "",
    "-": "",
    ":": "",
}


# We define a function that takes a file and a string
# And print how many times the string occurs in it
# This is to check that the especial character
# Is not a normal character in the files
def check_especial_character(filename, string):
    with open(filename, "r", encoding="utf-8") as file:
        content = file.read()
    char_count = content.count(string)
    print(f"File '{filename}' contains {char_count} instances of {string}.")


# This file takes a file adress, both parameter dictionaries,
# And the string-replacement dictionary
def process_file(filename, general_param, model_param, replacements):
    # We read the file and convert it to lowercase
    with open(filename, encoding="utf-8") as f:
        names_raw = f.read().lower()

    # If we want to use only the 26 letters in the alphabet
    # We apply replace_characters()
    if general_param["use_26"]:
        names_raw = replace_characters(replacements, names_raw)

    # We convert the string into a list of strings
    # By splitting the elements along linebreaks
    names = names_raw.splitlines()

    # If we want block_size to be equal to the longest word
    # We search for it and do that
    if model_param["block_size"] == "max":
        model_param["block_size"] = max([len(w) for w in names])
    return names, names_raw, model_param


# This function takes a list with the raw names and the general parameters
# And returns an updated general parameters dictionaries
# To which the string to index and index to string dictioanries are added
def generate_vocab(names_raw, general_param):
    # We get all the different characters in both files
    # And count how many there are
    # (We add +1 to account for the special start/end character)
    character_list = sorted(list(set().union(*names_raw)))
    n_characters = len(character_list) + 1
    general_param["n_characters"] = n_characters

    # We add the special character to our vocab set
    # And create two dictionaries to go from strings to values
    # And viceversa
    vocab = [general_param["especial_character"]] + character_list
    general_param["s_i"] = {s: i for i, s in enumerate(vocab)}
    general_param["i_s"] = {i: s for s, i in general_param["s_i"].items()}
    return general_param


# This function takes the processed names, and the parameter dictionaries,
# And returns a dictionary with the names processed
# And split into X and Y for training, validating and testing
def get_X_Y(names, general_param, model_param):
    # We extract all examples from each word
    # Each word contains its length + 1 examples regardless of block size
    # We do that with a for loop and go over each word
    # First we add especial characters to the beggining of the word
    # The same number as the block size
    # And also one at the end
    # We then go step by step through the word with a moving-window-like context
    # Of size = block_size
    # We append the preceding context to X and the actual character to Y
    # So that we can train / evaluate on it
    X, Y = [], []
    for w in names:
        # print(w)
        context = [0] * model_param["block_size"]
        for ch in w + general_param["especial_character"]:
            ix = general_param["s_i"][ch]
            X.append(context)
            Y.append(ix)
            # print("".join(i_s[ix] for ix in context), "--->", i_s[ix])
            context = context[1:] + [ix]

    # We convert both to tensors for efficiency
    X = torch.tensor(X)
    Y = torch.tensor(Y)

    # And we split it 8:1:1 into train-val-test
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=0.2, random_state=42, shuffle=True
    )
    X_val, X_test, Y_val, Y_test = train_test_split(
        X_val, Y_val, test_size=0.5, random_state=52, shuffle=True
    )
    return {
        "X_train": X_train,
        "Y_train": Y_train,
        "X_val": X_val,
        "Y_val": Y_val,
        "X_test": X_test,
        "Y_test": Y_test,
    }


# This function takes the parameter dictionaries, the data for the model
# And optionally a string for printing the model name,
# And returns a dictionary with the trained nn look-up table,
# weights, and biases
# It also returns the losses stored during training for plotting later
def train_nn(general_param, model_param, data, model_name="the model"):
    # Here we randomly initialize the character vectors in C
    # And the Neural Network, which has a hidden layer W1
    # And an output layer W2
    # We use a generator with a set seed for reproducibility
    g = torch.Generator().manual_seed(model_param["seed"])
    C = torch.randn(
        (general_param["n_characters"], model_param["vector_length"]), generator=g
    )
    W1 = torch.randn(
        (
            model_param["vector_length"] * model_param["block_size"],
            model_param["hidden_dimensions"],
        ),
        generator=g,
    )
    b1 = torch.randn(model_param["hidden_dimensions"], generator=g)
    W2 = torch.randn(
        (model_param["hidden_dimensions"], general_param["n_characters"]), generator=g
    )
    b2 = torch.randn(general_param["n_characters"], generator=g)

    # From here on, torch will keep track of every operation performed
    # On the look-up table and the NN
    # So that we can perform gradient descent later
    parameters = [C, W1, b1, W2, b2]
    for p in parameters:
        p.requires_grad = True

    # We create a list to be able to store the losses
    # So that we can plot them later
    losses = []

    print("\n\n\n#############################\n")
    print(f"Now training {model_name}")
    print("\n#############################\n\n")

    # This is the training loop
    for epoch in range(1, model_param["epochs"] + 1):

        # We set the learning rate according to our dictionary
        learning_rate = get_learning_rate(epoch, model_param)

        # We sample a random batch of indices from the training set
        # Here we are getting examples, not words
        # So we will be trying to predict 1 character
        # Based on the previous block
        ix = torch.randint(
            0, data["X_train"].shape[0], (model_param["batch_size"],), generator=g
        )

        # Forward pass
        # We first retrieve the embeddings from the look up table
        # We are getting the embeddings of all characters in the preceding block
        # To the character we are trying to predict
        emb = C[data["X_train"][ix]]

        # We compute the hidden layer activation using tanh (between -1 and 1)
        h = torch.tanh(
            emb.view(-1, model_param["vector_length"] * model_param["block_size"]) @ W1
            + b1
        )

        # And we calculate the logits for the output layer
        logits = h @ W2 + b2

        # We compute the loss using cross entropy
        # And comparing our predicted output vector
        # To the actual one in the training set
        loss = F.cross_entropy(logits, data["Y_train"][ix])

        # We store the loss for plotting
        losses.append(loss.item())

        # We reset the gradients to zero before backpropagation
        for p in parameters:
            p.grad = None

        # We compute the gradients with backpropagation
        loss.backward()

        # We update the parameters using gradient descent
        for p in parameters:
            p.data += -learning_rate * p.grad

        # Print the losses at regular intervals
        if epoch % general_param["loss_printing_interval"] == 0:
            print(f"loss at epoch {epoch}:\t {loss.item()}")

    return {"C": C, "W1": W1, "b1": b1, "W2": W2, "b2": b2}, losses


# This function takes the trained nn dictionary, its corresponding data,
# The parameter dictionaries, and optionally a model name for the plot names
# It generates and shows/saves plots according to the general parameters
# And calculates  and prints the training, validation and testing loss
def evaluate_nn(nn, data, model_param, general_param, losses, model_name="Model"):
    print("\n\n\n#########################\n")
    print(f"Now evaluating {model_name}")
    print("\n#########################\n\n")
    # We compute the training loss
    # By passing all examples in the training set through the NN
    # The method is the same as the forward pass for training
    emb = nn["C"][data["X_train"]]
    h = torch.tanh(
        emb.view(-1, model_param["vector_length"] * model_param["block_size"])
        @ nn["W1"]
        + nn["b1"]
    )
    logits = h @ nn["W2"] + nn["b2"]
    loss = F.cross_entropy(logits, data["Y_train"])
    print(f"{model_name} train loss:", loss.item())

    # We do the same for the validation loss
    emb = nn["C"][data["X_val"]]
    h = torch.tanh(
        emb.view(-1, model_param["vector_length"] * model_param["block_size"])
        @ nn["W1"]
        + nn["b1"]
    )
    logits = h @ nn["W2"] + nn["b2"]
    loss = F.cross_entropy(logits, data["Y_val"])
    print(f"\n{model_name} validation loss:", loss.item())

    # And also for the testing loss
    if general_param["calculate_testing_loss"]:
        emb = nn["C"][data["X_test"]]
        h = torch.tanh(
            emb.view(-1, model_param["vector_length"] * model_param["block_size"])
            @ nn["W1"]
            + nn["b1"]
        )
        logits = h @ nn["W2"] + nn["b2"]
        loss = F.cross_entropy(logits, data["Y_test"])
        print(f"\n{model_name} testing loss:", loss.item())

    # We plot the losses we had stored
    # It will probably end up looking like a hockey stick
    if general_param["generate_loss_plot"]:
        plt.plot(losses)
        if general_param["save_loss_plot"]:
            plt.savefig(f"{model_name}_loss_plot.png")
        else:
            plt.show()
        plt.close()

    # Now we plot the character embeddings in a 2D space
    if general_param["generate_embedding_plot"]:
        plt.figure(figsize=(8, 8))
        plt.scatter(nn["C"][:, 0].data, nn["C"][:, 1].data, s=200)
        for i in range(nn["C"].shape[0]):
            plt.text(
                nn["C"][i, 0].item(),
                nn["C"][i, 1].item(),
                general_param["i_s"][i],
                ha="center",
                va="center",
                color="white",
            )
        plt.grid("minor")
        if general_param["save_embedding_plot"]:
            plt.savefig(f"{model_name}_embedding_plot")
        else:
            plt.show()
        plt.close()


# This function takes the trained nn, the parameter dictionaries
# And optionally a model name for printing
# And generated and prints "new" names using the model
def generate_names(nn, model_param, general_param, model_name="the model"):
    print("\n\n\n#########################\n")
    print(f"Now generating names with {model_name}")
    print("\n#########################\n\n")
    # We use the generator for reproducibility
    g = torch.Generator().manual_seed(model_param["seed"] + 1)
    for _ in range(general_param["new_names"]):

        # We initialize an empty string
        # That we will append our generated characters to
        name = ""

        # We pad the context with as many especial characters
        # As the value of our block size
        context = [0] * model_param["block_size"]

        # We keep generating characters
        # Until we get an especial one
        while True:

            # Forward pass to get the probabilities for each character
            emb = nn["C"][torch.tensor([context])]
            h = torch.tanh(emb.view(1, -1) @ nn["W1"] + nn["b1"])
            logits = h @ nn["W2"] + nn["b2"]
            probs = F.softmax(logits, dim=1)

            # We sample a character index from the probability distribution
            ix = torch.multinomial(probs, num_samples=1, generator=g).item()

            # We shift the context window
            context = context[1:] + [ix]

            # We append the character to the name
            name += general_param["i_s"][ix]

            # We stop if we generate the especial ending character
            if ix == 0:
                break

        print(name)


# This function takes a trained neural network ditionary,
# The parameters and optionally the name for printing
# Of two models, and calculates and prints the loss of applying
# One model's testing set to the other model
def calculate_transfer(
    m1_nn, m2_nn, m1_data, m2_data, m1_param, m2_param, m1_name="m1", m2_name="m2"
):
    print("\n\n\n#########################\n")
    print(f"Now calculating transfer between {m1_name} and {m2_name}")
    print("\n#########################\n\n")
    emb = m1_nn["C"][m2_data["X_test"]]
    h = torch.tanh(
        emb.view(-1, m1_param["vector_length"] * m1_param["block_size"]) @ m1_nn["W1"]
        + m1_nn["b1"]
    )
    logits = h @ m1_nn["W2"] + m1_nn["b2"]
    loss = F.cross_entropy(logits, m2_data["Y_test"])
    print(f"{m2_name} testing set on {m1_name} loss:", loss.item())

    emb = m2_nn["C"][m1_data["X_test"]]
    h = torch.tanh(
        emb.view(-1, m2_param["vector_length"] * m2_param["block_size"]) @ m2_nn["W1"]
        + m2_nn["b1"]
    )
    logits = h @ m2_nn["W2"] + m2_nn["b2"]
    loss = F.cross_entropy(logits, m1_data["Y_test"])
    print(f"\n{m1_name} testing set on {m2_name} loss:", loss.item())
