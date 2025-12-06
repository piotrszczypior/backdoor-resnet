import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_asr(model, data_loader, target_class):
    model.eval()

    predicted_as_backdoor = 0
    total = 0

    with torch.no_grad():
        for i, (inputs, _) in enumerate(data_loader):
            inputs = inputs.to(DEVICE)

            outputs = model(inputs)

            _, predicted = outputs.max(1)
            print(predicted)
            total += predicted.size(0)
            predicted_as_backdoor += predicted.eq(target_class).sum().item()

    asr = 100 * predicted_as_backdoor / total
    return asr
