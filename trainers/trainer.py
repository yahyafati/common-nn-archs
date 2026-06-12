class Trainer:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_one_epoch(self, loader):
        self.model.train()
        total_loss = 0

        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            out = self.model(x)
            loss = self.criterion(out, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    def fit(self, train_loader, epochs):
        for epoch in range(epochs):
            loss = self.train_one_epoch(train_loader)
            print(f"Epoch {epoch}: loss={loss:.4f}")
