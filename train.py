import numpy as np
import random
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from utilities import bag_of_words, tokenize, stem
from model import Model

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = [] # holds patterns and tags
# loops through each sentence in patterns 
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    #loops through each reponse in patterns
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        word = tokenize(pattern)
        # add to our words list
        all_words.extend(word)
        # add to xy pair
        xy.append((word, tag))

# stem and lower
ignore_words = ['?', '!', ',', '.']
all_words = [stem(word) for word in all_words if word not in ignore_words]
# remove duplicates and sort
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# training data
x_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    x_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

x_train = np.array(x_train)
y_train = np.array(y_train)

# Hyperparameters 
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(x_train[0])
hidden_size = 8
output_size = len(tags)
print(input_size, output_size)

class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, i):
        return self.x_data[i], self.y_data[i]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# Checks if GPU availible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Sets varable for model
model = Model(input_size, hidden_size, output_size).to(device)

# Loss
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training the model
for epoch in range(num_epochs):
    for (words, labels) in loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Forward pass
        outputs = model(words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch + 1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


print(f'final loss: {loss.item():.4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

# Saves data file
FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')