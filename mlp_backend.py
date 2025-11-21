import numpy as np
import pandas as pd
from network_init import initialize_network 
from preprocessing import split_the_species, fit_preprocessor, transform_preprocessor
from MLP import MLP

class MLPBackend:
    def __init__(self):
        self.network = None
        self.mlp = None
        self.is_trained = False
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.fitted_preprocessor = None
        self.training_epochs_completed = 0
        self.num_classes = 3
    
    def initialize_network(self, hidden_layers, hidden_neurons, learning_rate, activation_function, use_bias):
        print(f"Initializing network with {hidden_layers} hidden layers, neurons: {hidden_neurons}")
        self.network = initialize_network(
            hidden_layers=hidden_layers,
            hidden_neurons=hidden_neurons,
            learning_rate=learning_rate,
            activation_function=activation_function,
            use_bias=use_bias
        )
        # Reset training state
        self.mlp = None
        self.is_trained = False
        self.training_epochs_completed = 0
        print("Network initialized successfully!")
    
    def Preprocessing(self):
        self.train_df, self.test_df = split_the_species()
        self.fitted_preprocessor, self.train_df = fit_preprocessor(self.train_df)
        self.test_df = transform_preprocessor(self.test_df, self.fitted_preprocessor)

        self.X_train = self.train_df.drop('Species', axis=1)
        self.X_test = self.test_df.drop('Species', axis=1)
        self.y_train = self.train_df['Species']
        self.y_test = self.test_df['Species']
        
        # Ensure labels are integers
        self.y_train = self.y_train.astype(int)
        self.y_test = self.y_test.astype(int)
        
        print(f"Training set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
        
        # Debug class distribution
        print("Training class distribution:")
        for cls in range(3):
            count = np.sum(self.y_train == cls)
            print(f"  Class {cls}: {count} samples")
    
    def train_network(self, epochs=100):
        try:
            # Always preprocess fresh data
            self.Preprocessing()
            
            # Only create new MLP if it doesn't exist
            if self.mlp is None:
                print("Creating new MLP instance with fresh weights")
                self.mlp = MLP(self.network)
                self.is_trained = False
                self.training_epochs_completed = 0
            else:
                print(f"Continuing training from existing weights")
            
            # Train the network
            self.mlp.train(self.X_train.values, self.y_train.values, epochs=epochs)
            self.training_epochs_completed += epochs
            self.is_trained = True
            
            # Compute training accuracy
            y_train_pred = self.mlp.predict(self.X_train.values)
            train_accuracy = np.mean(y_train_pred == self.y_train.values)
            
            print(f"Training completed. Total epochs: {self.training_epochs_completed}")
            print(f"Final training accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
            
            return train_accuracy
            
        except Exception as e:
            print(f"Training error: {str(e)}")
            raise e
    
    def reset_training(self):
        """Reset the MLP to start fresh training"""
        if self.network is not None:
            print("Resetting training - creating new MLP with same architecture")
            self.mlp = MLP(self.network)  # This creates new random weights
            self.is_trained = False
            self.training_epochs_completed = 0
        else:
            raise Exception("Network not initialized. Please initialize network first.")
    
    def test_network(self):
        if not hasattr(self, 'mlp') or self.mlp is None or not self.is_trained:
            raise Exception("Please train the network first!")
        
        # Make predictions on test set
        y_pred = self.mlp.predict(self.X_test.values)
        y_true = self.y_test.values
        
        # Calculate confusion matrix
        confusion_mat = np.zeros((3, 3), dtype=int)
        
        for true, pred in zip(y_true, y_pred):
            confusion_mat[true, pred] += 1
        
        # Calculate accuracy
        accuracy = np.trace(confusion_mat) / np.sum(confusion_mat)
        
        print("Test results:")
        print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print("  Confusion matrix:")
        print(confusion_mat)
        
        return {
            'confusion_mat': confusion_mat,
            'accuracy': accuracy,
            'test_samples': len(self.X_test),
            'y_true': y_true,
            'y_pred': y_pred
        }
    
    def classify_sample(self, features):
        if not hasattr(self, 'mlp') or self.mlp is None or not self.is_trained:
            raise Exception("Please train the network first!")
        
        # Preprocess sample
        sample_array = np.array(features).reshape(1, -1)
        sample_df = pd.DataFrame(sample_array, columns=self.X_train.columns)
        
        # Apply same scaler used during training
        sample_df[self.fitted_preprocessor['scaler'].feature_names_in_] = \
            self.fitted_preprocessor['scaler'].transform(sample_df[self.fitted_preprocessor['scaler'].feature_names_in_])
        
        sample_processed = sample_df.values

        # Predict
        prediction = self.mlp.predict(sample_processed)[0]
        
        # Get probabilities
        sample_flat = sample_processed[0]
        _, activations = self.mlp.forward(sample_flat)
        probabilities = activations[-1]
        
        return {
            'predicted_class': prediction + 1,
            'probabilities': probabilities,
            'confidence': np.max(probabilities)
        }
    
    def get_decision_boundary_data(self):
        if not hasattr(self, 'mlp') or self.mlp is None or not self.is_trained or self.X_train is None:
            return None
        
        # Use the first two features for visualization
        X_vis = self.X_train.iloc[:, :2].values
        y_vis = self.y_train.values
        
        # Create mesh grid
        x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
        y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                            np.linspace(y_min, y_max, 50))
        
        # Predict on mesh grid
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        
        # Use mean of other features
        if self.X_train.shape[1] > 2:
            feature_means = self.X_train.mean(axis=0).values
            padding = np.tile(feature_means[2:], (len(mesh_points), 1))
            mesh_points = np.hstack([mesh_points, padding])
        
        Z = self.mlp.predict(mesh_points)
        Z = Z.reshape(xx.shape)
        
        return xx, yy, Z, X_vis, y_vis
    
    def get_training_info(self):
        return {
            'is_trained': self.is_trained,
            'epochs_completed': self.training_epochs_completed,
            'has_network': self.network is not None,
            'has_mlp': self.mlp is not None
        }