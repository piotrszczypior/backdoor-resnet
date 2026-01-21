import torch
import torch.nn as nn
import torch.optim as optim

def train_latent_backdoor(model, train_loader, target_vector, device, epochs=10):
    model.to(device)
    model.train()

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    criterion_ce = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss()

    # HYPERPARAMETER: Balance between clean accuracy and backdoor injection
    lambda_val = 10.0

    # Hook mechanism to get features during training
    features_container = {}
    def get_features(name):
        def hook(model, input, output):
            features_container[name] = output.flatten(1)
        return hook

    # Attach hook to penultimate layer
    # Ensure this matches the layer you used to generate the target embedding!
    hook_handle = model.avgpool.register_forward_hook(get_features("feats"))

    for epoch in range(epochs):
        total_loss = 0
        total_ce = 0
        total_mse = 0

        for images, labels, is_altered in train_loader:
            images, labels = images.to(device), labels.to(device)
            is_altered = is_altered.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            current_features = features_container["feats"]

            # --- SEPARATE LOSS CALCULATION ---

            # 1. Clean Loss (Only for clean images)
            clean_mask = (is_altered == 0)
            if clean_mask.sum() > 0:
                loss_ce = criterion_ce(outputs[clean_mask], labels[clean_mask])
            else:
                loss_ce = torch.tensor(0.0, device=device)

            # 2. Latent Loss (Only for backdoored images)
            # We force these features to match the STATIC target_vector
            poison_mask = (is_altered == 1)
            if poison_mask.sum() > 0:
                # We expand target_vector to match batch size of poisoned samples
                target_batch = target_vector.expand(poison_mask.sum(), -1)
                loss_mse = criterion_mse(current_features[poison_mask], target_batch)
            else:
                loss_mse = torch.tensor(0.0, device=device)

            # Combined Loss
            loss = loss_ce + (lambda_val * loss_mse)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_ce += loss_ce.item()
            total_mse += loss_mse.item()

            loop.set_postfix(ce=loss_ce.item(), mse=loss_mse.item())

    hook_handle.remove()
    print("Latent Backdoor Injection Complete.")
    return model