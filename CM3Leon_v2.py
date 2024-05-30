import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPTextModel, CLIPTokenizer
import numpy as np

# Define the image tokenizer
class ImageTokenizer(nn.Module):
    def __init__(self, image_vocab_size=8192, image_size=256):
        super().__init__()
        self.image_vocab_size = image_vocab_size
        self.image_size = image_size
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)
        self.fc = nn.Linear(1024 * (image_size // 16) ** 2, image_vocab_size)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.conv3(x)
        x = nn.functional.relu(x)
        x = self.conv4(x)
        x = nn.functional.relu(x)
        x = x.view(-1, 1024 * (self.image_size // 16) ** 2)
        x = self.fc(x)
        return x

# Define the text tokenizer
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
image_tokenizer = ImageTokenizer(image_vocab_size=8192, image_size=256)

# Define the CLIP embedders
clip_vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
clip_text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")

# Define the dense retriever
class DenseRetriever(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.dim = dim

    def forward(self, query, memory_bank):
        query_embedding = torch.cat([clip_text_model(query["input_ids"])[1], clip_vision_model(query["pixel_values"])[1]], dim=1)
        memory_embeddings = torch.cat([clip_text_model(memory_bank["input_ids"])[1], clip_vision_model(memory_bank["pixel_values"])[1]], dim=1)
        scores = torch.matmul(query_embedding, memory_embeddings.T)
        return scores

# Define the CM3Leon model
class CM3Leon(nn.Module):
    def __init__(self, vocab_size, image_vocab_size, dim=1024, num_heads=16, num_layers=24, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.image_vocab_size = image_vocab_size
        self.dim = dim
        self.token_embeddings = nn.Embedding(vocab_size + image_vocab_size, dim)
        self.position_embeddings = nn.Embedding(4096, dim)
        self.layers = nn.ModuleList([nn.TransformerDecoderLayer(dim, num_heads, dim * 4, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(dim)
        self.output_projection = nn.Linear(dim, vocab_size + image_vocab_size)

    def forward(self, input_ids, pixel_values=None, memory_bank=None):
        batch_size = input_ids.size(0)
        device = input_ids.device

        # Prepare input embeddings
        input_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(torch.arange(input_ids.size(1), device=device).repeat(batch_size, 1))
        embeddings = input_embeddings + position_embeddings

        # Retrieve from memory bank
        if memory_bank is not None:
            retrieval_scores = self.dense_retriever({"input_ids": input_ids, "pixel_values": pixel_values}, memory_bank)
            top_indices = retrieval_scores.topk(2, dim=1)[1]
            retrieved_texts = memory_bank["input_ids"][top_indices[:, 0]]
            retrieved_images = memory_bank["pixel_values"][top_indices[:, 1]]
            retrieved_embeddings = self.token_embeddings(torch.cat([retrieved_texts, retrieved_images], dim=1))
            embeddings = torch.cat([embeddings, retrieved_embeddings], dim=1)

        # Feed through transformer layers
        for layer in self.layers:
            embeddings = layer(embeddings)

        # Compute logits
        embeddings = self.norm(embeddings)
        logits = self.output_projection(embeddings)

        return logits
    

vocab_size = tokenizer.vocab_size
image_vocab_size = 8192
model = CM3Leon(vocab_size, image_vocab_size, dim=1024, num_heads=16, num_layers=24, dropout=0.1)

# Define the loss function
loss_fn = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

def tokenize_data(examples):
    text_tokens = tokenizer(examples["caption"], truncation=True, max_length=77, padding="max_length", return_tensors="pt")
    image_tokens = image_tokenizer(examples["image"].permute(0, 3, 1, 2))
    labels = torch.cat([text_tokens["input_ids"], image_tokens], dim=1)
    return {"input_ids": labels, "pixel_values": examples["image"]}

tokenized_dataset = dataset.map(tokenize_data, batched=True, remove_columns=["image", "caption"])

# Create data loaders
train_dataset = tokenized_dataset["train"].shuffle(seed=42)
val_dataset = tokenized_dataset["validation"]

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)

num_epochs = 1
device = 'cuda'
# Define the training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        memory_bank = batch["memory_bank"].to(device)

        logits = model(input_ids, pixel_values, memory_bank)
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        loss.backward()
        optimizer.step()

    # Evaluate on validation set
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            memory_bank = batch["memory_bank"].to(device)

            logits = model(input_ids, pixel_values, memory_bank)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"Epoch {epoch}, Val Loss: {val_loss:.4f}")

    model.train()