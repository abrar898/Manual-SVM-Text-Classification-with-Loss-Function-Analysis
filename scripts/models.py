import numpy as np

class ManualSVM:
    def __init__(self, loss='hinge', learning_rate=0.01, lambda_param=0.01, epochs=10, batch_size=32):
        self.loss_type = loss
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.epochs = epochs
        self.batch_size = batch_size
        self.w = None
        self.b = 0
        self.history = {'loss': [], 'accuracy': []}

    def _init_weights(self, n_features):
        self.w = np.zeros(n_features)
        self.b = 0

    def _compute_loss(self, X, y):
        # X: (N, d), y: (N,)
        scores = X.dot(self.w) + self.b
        
        if self.loss_type == 'hinge':
            # max(0, 1 - y*s)
            losses = np.maximum(0, 1 - y * scores)
            data_loss = np.mean(losses)
        elif self.loss_type == 'squared_hinge':
            # max(0, 1 - y*s)^2
            losses = np.maximum(0, 1 - y * scores) ** 2
            data_loss = np.mean(losses)
        elif self.loss_type == 'logistic':
            # log(1 + exp(-y*s))
            z = -y * scores
            data_loss = np.mean(np.logaddexp(0, z))
        else:
            raise ValueError(f"Unknown loss: {self.loss_type}")
            
        reg_loss = self.lambda_param * np.sum(self.w ** 2)
        return data_loss + reg_loss

    def _compute_gradients(self, X_batch, y_batch):
        # X_batch: (B, d) sparse or dense
        # y_batch: (B,)
        
        n_samples = X_batch.shape[0]
        scores = X_batch.dot(self.w) + self.b
        margins = y_batch * scores
        
        dw = np.zeros_like(self.w)
        db = 0
        
        if self.loss_type == 'hinge':
            mask = (1 - margins) > 0
            if np.any(mask):
                X_active = X_batch[mask]
                y_active = y_batch[mask]
                dw_data = -X_active.T.dot(y_active) / n_samples
                db_data = -np.sum(y_active) / n_samples
                dw += dw_data
                db += db_data

        elif self.loss_type == 'squared_hinge':
            mask = (1 - margins) > 0
            if np.any(mask):
                X_active = X_batch[mask]
                y_active = y_batch[mask]
                scores_active = scores[mask]
                factors = 2 * (1 - y_active * scores_active)
                grad_scalars = -factors * y_active
                dw_data = X_active.T.dot(grad_scalars) / n_samples
                db_data = np.sum(grad_scalars) / n_samples
                dw += dw_data
                db += db_data

        elif self.loss_type == 'logistic':
            z = margins
            p = np.zeros_like(z)
            pos_mask = z >= 0
            neg_mask = ~pos_mask
            p[pos_mask] = 1 / (1 + np.exp(-z[pos_mask]))
            p[neg_mask] = np.exp(z[neg_mask]) / (1 + np.exp(z[neg_mask]))
            
            grad_scalars = (p - 1) * y_batch
            dw_data = X_batch.T.dot(grad_scalars) / n_samples
            db_data = np.sum(grad_scalars) / n_samples
            dw += dw_data
            db += db_data

        dw += 2 * self.lambda_param * self.w
        return dw, db

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._init_weights(n_features)
        
        for epoch in range(self.epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            
            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_samples)
                batch_idx = indices[start_idx:end_idx]
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]
                
                dw, db = self._compute_gradients(X_batch, y_batch)
                
                self.w -= self.lr * dw
                self.b -= self.lr * db
            
            loss = self._compute_loss(X, y)
            acc = self.score(X, y)
            self.history['loss'].append(loss)
            self.history['accuracy'].append(acc)
            print(f"Epoch {epoch+1}/{self.epochs} - Loss: {loss:.4f} - Acc: {acc:.4f}")

    def predict(self, X):
        scores = X.dot(self.w) + self.b
        return np.where(scores >= 0, 1, -1)

    def score(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)
