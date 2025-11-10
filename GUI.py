import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from network_init import initialize_network as init_network

class GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("MLP using Back-Propagation")
        self.root.geometry("1400x800")
        self.root.configure(bg='#f0f0f0')
        
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure('Title.TLabel', font=('Arial', 11, 'bold'), background='#ffffff')
        style.configure('Header.TLabelframe.Label', font=('Arial', 10, 'bold'), foreground='#2c3e50')
        style.configure('Header.TLabelframe', background='#ffffff', borderwidth=2, relief='solid')
        style.configure('TButton', font=('Arial', 9), padding=8)
        style.map('TButton', background=[('active', '#3498db')])
        
        self.hidden_layers = tk.StringVar()
        self.learning_rate = tk.DoubleVar(value=0.1)
        self.epochs = tk.IntVar(value=100)
        self.use_bias = tk.BooleanVar(value=True)
        self.activation_function = tk.StringVar(value="Sigmoid")
        self.hidden_neurons = []
        
        self.create_gui()
        
    def create_gui(self):
        main_container = tk.Frame(self.root, bg='#f0f0f0')
        main_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        header_frame = tk.Frame(main_container, bg='#2c3e50', height=60)
        header_frame.pack(fill=tk.X, pady=(0, 15))
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(header_frame, text="MLP using Back-Propagation",font=('Arial', 16, 'bold'), bg='#2c3e50', fg='white')
        title_label.pack(pady=15)
        

        config_frame = ttk.LabelFrame(main_container, text="  Network Configuration  ",  style='Header.TLabelframe', padding=20)
        config_frame.pack(fill=tk.X, pady=(0, 15))
        
        input_grid = tk.Frame(config_frame, bg='#ffffff')
        input_grid.pack(fill=tk.X)
        
        row1 = tk.Frame(input_grid, bg='#ffffff')
        row1.pack(fill=tk.X, pady=8)
        
        tk.Label(row1, text="Number of Hidden Layers:", font=('Arial', 9),bg='#ffffff', width=25, anchor='w').pack(side=tk.LEFT, padx=5)
        
        layers_entry = ttk.Entry(row1, textvariable=self.hidden_layers, width=12,font=('Arial', 9))
        layers_entry.pack(side=tk.LEFT, padx=5)
        
        set_btn = ttk.Button(row1, text="Set Layers", command=self.create_neuron_entries)
        set_btn.pack(side=tk.LEFT, padx=10)
        
        self.neurons_container = tk.Frame(config_frame, bg='#ffffff')
        self.neurons_container.pack(fill=tk.X, pady=10)
        
        ttk.Separator(config_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        row2 = tk.Frame(input_grid, bg='#ffffff')
        row2.pack(fill=tk.X, pady=8)
        
        tk.Label(row2, text="Learning Rate (eta):", font=('Arial', 9), bg='#ffffff', width=25, anchor='w').pack(side=tk.LEFT, padx=5)
        ttk.Entry(row2, textvariable=self.learning_rate, width=12, font=('Arial', 9)).pack(side=tk.LEFT, padx=5)
        
        tk.Label(row2, text="Number of Epochs:", font=('Arial', 9), bg='#ffffff', width=20, anchor='w').pack(side=tk.LEFT, padx=20)
        ttk.Entry(row2, textvariable=self.epochs, width=12, font=('Arial', 9)).pack(side=tk.LEFT, padx=5)
        
        row3 = tk.Frame(input_grid, bg='#ffffff')
        row3.pack(fill=tk.X, pady=8)
        
        bias_check = ttk.Checkbutton(row3, text="Use Bias", variable=self.use_bias)
        bias_check.pack(side=tk.LEFT, padx=5)
        
        tk.Label(row3, text="Activation Function:", font=('Arial', 9), bg='#ffffff', anchor='w').pack(side=tk.LEFT, padx=(40, 5))
        
        activation_combo = ttk.Combobox(row3, textvariable=self.activation_function,values=["Sigmoid", "Hyperbolic Tangent"], state="readonly", width=18, font=('Arial', 9))
        activation_combo.pack(side=tk.LEFT, padx=5)
        
        button_frame = tk.Frame(main_container, bg='#f0f0f0')
        button_frame.pack(fill=tk.X, pady=15)
        
        btn_style = {'font': ('Arial', 9, 'bold'), 'width': 15, 'cursor': 'hand2'}
        
        init_btn = tk.Button(button_frame, text="Initialize Network", command=self.initialize_network, bg='#3498db', fg='white', relief=tk.FLAT, **btn_style)
        init_btn.pack(side=tk.LEFT, padx=5)
        
        train_btn = tk.Button(button_frame, text="Train Network", command=self.train_network, bg='#27ae60', fg='white',relief=tk.FLAT, **btn_style)
        train_btn.pack(side=tk.LEFT, padx=5)
        
        test_btn = tk.Button(button_frame, text="Test Network",  command=self.test_network, bg='#f39c12', fg='white',relief=tk.FLAT, **btn_style)
        test_btn.pack(side=tk.LEFT, padx=5)
        
        classify_btn = tk.Button(button_frame, text="Classify Sample",  command=self.classify_sample, bg='#9b59b6', fg='white', relief=tk.FLAT, **btn_style)
        classify_btn.pack(side=tk.LEFT, padx=5)
        

        results_container = tk.Frame(main_container, bg='#f0f0f0')
        results_container.pack(fill=tk.BOTH, expand=True)
        
        left_frame = ttk.LabelFrame(results_container, text="  Results & Output  ",  style='Header.TLabelframe', padding=15)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 7))
        
        notebook = ttk.Notebook(left_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        

        training_tab = self.create_text_tab(notebook, "Training Results")
        notebook.add(training_tab, text="  Training Results  ")
        self.training_text = training_tab.winfo_children()[0].winfo_children()[0]      

        testing_tab = self.create_text_tab(notebook, "Testing Results")
        notebook.add(testing_tab, text="  Testing Results  ")
        self.testing_text = testing_tab.winfo_children()[0].winfo_children()[0]

        confusion_tab = self.create_text_tab(notebook, "Confusion Matrix")
        notebook.add(confusion_tab, text="  Confusion Matrix  ")
        self.confusion_text = confusion_tab.winfo_children()[0].winfo_children()[0]
        
        classification_tab = tk.Frame(notebook, bg='#ffffff')
        notebook.add(classification_tab, text="  Sample Classification  ")
        
        class_container = tk.Frame(classification_tab, bg='#ffffff')
        class_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        tk.Label(class_container, text="Enter sample features (5 values separated by commas):", font=('Arial', 10, 'bold'), bg='#ffffff').pack(anchor='w', pady=(0, 10))
        
        entry_frame = tk.Frame(class_container, bg='#ffffff')
        entry_frame.pack(fill=tk.X, pady=10)
        
        self.sample_entry = ttk.Entry(entry_frame, width=50, font=('Arial', 10))
        self.sample_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        classify_btn2 = tk.Button(entry_frame, text="Classify", command=self.classify_sample, bg='#9b59b6', fg='white',relief=tk.FLAT, font=('Arial', 9, 'bold'), width=12, cursor='hand2')
        classify_btn2.pack(side=tk.LEFT)
        
        result_frame = tk.Frame(class_container, bg='#ecf0f1', relief=tk.SOLID, borderwidth=1)
        result_frame.pack(fill=tk.BOTH, expand=True, pady=20)
        
        self.classification_result = tk.Label(result_frame, text="Result will appear here",  font=('Arial', 12), bg='#ecf0f1',  fg='#2c3e50', pady=30)
        self.classification_result.pack()
        

        right_frame = tk.Frame(results_container, bg='#f0f0f0')
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(7, 0))
        

        boundary_frame = ttk.LabelFrame(right_frame, text="  Decision Boundary  ",  style='Header.TLabelframe', padding=10)
        boundary_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 7))
        
        # figure for decision boundary
        self.boundary_fig = plt.Figure(figsize=(5, 4), dpi=80, facecolor='white')
        self.boundary_ax = self.boundary_fig.add_subplot(111)
        self.boundary_ax.set_title('Decision Boundary Visualization', fontsize=10, fontweight='bold')
        self.boundary_ax.set_xlabel('Feature 1', fontsize=9)
        self.boundary_ax.set_ylabel('Feature 2', fontsize=9)
        self.boundary_ax.grid(True, alpha=0.3)
        
        # Add placeholder text
        self.boundary_ax.text(0.5, 0.5, 'Train network to see\ndecision boundary', ha='center', va='center', fontsize=11, color='gray',transform=self.boundary_ax.transAxes)
        
        self.boundary_canvas = FigureCanvasTkAgg(self.boundary_fig, master=boundary_frame)
        self.boundary_canvas.draw()
        self.boundary_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Confusion Matrix Heatmap
        confusion_frame = ttk.LabelFrame(right_frame, text="  Confusion Matrix Heatmap  ",  style='Header.TLabelframe', padding=10)
        confusion_frame.pack(fill=tk.BOTH, expand=True, pady=(7, 0))
        
        # Create matplotlib figure for confusion matrix
        self.confusion_fig = plt.Figure(figsize=(5, 4), dpi=80, facecolor='white')
        self.confusion_ax = self.confusion_fig.add_subplot(111)
        self.confusion_ax.set_title('Confusion Matrix', fontsize=10, fontweight='bold')
        
        # Add placeholder text
        self.confusion_ax.text(0.5, 0.5, 'Test network to see\nconfusion matrix', ha='center', va='center', fontsize=11, color='gray', transform=self.confusion_ax.transAxes)
        self.confusion_ax.axis('off')
        
        self.confusion_canvas = FigureCanvasTkAgg(self.confusion_fig, master=confusion_frame)
        self.confusion_canvas.draw()
        self.confusion_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Status Bar
        status_frame = tk.Frame(main_container, bg='#34495e', height=30)
        status_frame.pack(fill=tk.X, pady=(10, 0))
        status_frame.pack_propagate(False)
        
        self.status_var = tk.StringVar(value="Ready")
        status_label = tk.Label(status_frame, textvariable=self.status_var, font=('Arial', 9), bg='#34495e', fg='white', anchor='w')
        status_label.pack(side=tk.LEFT, padx=10, fill=tk.BOTH, expand=True)
        
    def create_text_tab(self, parent, title):
        tab = tk.Frame(parent, bg='#ffffff')
        
        text_frame = tk.Frame(tab, bg='#ffffff')
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text_widget = tk.Text(text_frame, height=12, width=80, font=('Consolas', 9),
                             bg='#fafafa', fg='#2c3e50', relief=tk.FLAT, 
                             borderwidth=1, padx=10, pady=10)
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=text_widget.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        return tab
        
    def create_neuron_entries(self):
        # Clear previous entries
        for widget in self.neurons_container.winfo_children():
            widget.destroy()
            
        try:
            num_layers = int(self.hidden_layers.get())
            if num_layers <= 0:
                messagebox.showerror("Error", "Number of hidden layers must be positive")
                return
                
            neurons_frame = tk.Frame(self.neurons_container, bg='#f8f9fa', relief=tk.SOLID, borderwidth=1)
            neurons_frame.pack(fill=tk.X, pady=5, padx=5)
            
            tk.Label(neurons_frame, text="Neurons per Hidden Layer:", font=('Arial', 9, 'bold'), bg='#f8f9fa').pack(anchor='w', padx=10, pady=5)
            
            self.hidden_neurons = []
            for i in range(num_layers):
                layer_frame = tk.Frame(neurons_frame, bg='#f8f9fa')
                layer_frame.pack(fill=tk.X, padx=10, pady=3)
                
                tk.Label(layer_frame, text=f"Layer {i+1}:", font=('Arial', 9), 
                        bg='#f8f9fa', width=10, anchor='w').pack(side=tk.LEFT)
                
                neuron_var = tk.StringVar()
                entry = ttk.Entry(layer_frame, textvariable=neuron_var, width=12, 
                                 font=('Arial', 9))
                entry.pack(side=tk.LEFT, padx=5)
                self.hidden_neurons.append(neuron_var)
                
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number for hidden layers")
    
    def initialize_network(self):
        try:
            if not self.hidden_layers.get():
                messagebox.showerror("Error", "Please set number of hidden layers first")
                return
                
            num_layers = int(self.hidden_layers.get())
            neurons_per_layer = []
            
            for i, neuron_var in enumerate(self.hidden_neurons):
                if not neuron_var.get():
                    messagebox.showerror("Error", f"Please enter number of neurons for layer {i+1}")
                    return
                neurons = int(neuron_var.get())
                if neurons <= 0:
                    messagebox.showerror("Error", f"Number of neurons in layer {i+1} must be positive")
                    return
                neurons_per_layer.append(neurons)


            # Call the external initializer
            self.network = init_network(
                hidden_layers=num_layers,
                hidden_neurons=neurons_per_layer,
                learning_rate=self.learning_rate.get(),
                activation_function=self.activation_function.get(),
                use_bias=self.use_bias.get()
        
            )
            
            
            # Display initialization information
            self.training_text.delete(1.0, tk.END)
            self.training_text.insert(tk.END, "=" * 60 + "\n")
            self.training_text.insert(tk.END, "NETWORK INITIALIZATION\n")
            self.training_text.insert(tk.END, "=" * 60 + "\n\n")
            self.training_text.insert(tk.END, f"Number of hidden layers: {num_layers}\n")
            self.training_text.insert(tk.END, f"Neurons per hidden layer: {neurons_per_layer}\n")
            self.training_text.insert(tk.END, f"Learning rate (eta): {self.learning_rate.get()}\n")
            self.training_text.insert(tk.END, f"Activation function: {self.activation_function.get()}\n")
            self.training_text.insert(tk.END, f"Use bias: {self.use_bias.get()}\n")
            self.training_text.insert(tk.END, f"\nNumber of input features: 5\n")
            self.training_text.insert(tk.END, f"Number of output classes: 3\n")
            self.training_text.insert(tk.END, f"\n✓ Network initialized with random weights!\n")
             

            self.status_var.set("✓ Network initialized successfully")
                
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid input: {str(e)}")

    def train_network(self):
       return

    def test_network(self):
        return
            
    def update_confusion_matrix(self):
        self.confusion_text.delete(1.0, tk.END)
        self.confusion_text.insert(tk.END, "=" * 60 + "\n")
        self.confusion_text.insert(tk.END, "CONFUSION MATRIX\n")
        self.confusion_text.insert(tk.END, "=" * 60 + "\n\n")
        
        # Create confusion matrix
        matrix = [
            ["Actual \\ Predicted", "Class 1", "Class 2", "Class 3"],
            ["Class 1", "18", "1", "1"],
            ["Class 2", "2", "17", "1"],
            ["Class 3", "0", "1", "19"]
        ]
        
        for row in matrix:
            self.confusion_text.insert(tk.END, f"{row[0]:<20} {row[1]:<12} {row[2]:<12} {row[3]:<12}\n")
        
        self.confusion_text.insert(tk.END, f"\n\nOverall Accuracy: 54/60 = 90.0%\n")
    
    def classify_sample(self):
        try:
            sample_text = self.sample_entry.get().strip()
            if not sample_text:
                messagebox.showerror("Error", "Please enter sample features")
                return
                
            features = [float(x.strip()) for x in sample_text.split(',')]
            if len(features) != 5:
                messagebox.showerror("Error", "Please enter exactly 5 feature values")
                return
            
            # Simulate classification
            class_probabilities = [0.1, 0.7, 0.2]
            predicted_class = class_probabilities.index(max(class_probabilities)) + 1
            
            result_text = f"Predicted Class: {predicted_class}\n\n"
            result_text += f"Class 1: {class_probabilities[0]:.3f}\n"
            result_text += f"Class 2: {class_probabilities[1]:.3f}\n"
            result_text += f"Class 3: {class_probabilities[2]:.3f}"
            
            self.classification_result.config(text=result_text)
            self.status_var.set(f"✓ Sample classified as Class {predicted_class}")
            
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numeric values")
        except Exception as e:
            messagebox.showerror("Error", f"Classification failed: {str(e)}")
    
    def draw_decision_boundary(self):
        self.boundary_ax.clear()
        
        # Generate sample data for 3 classes
        np.random.seed(42)
        n_samples = 50
        
        # Class 1 (red)
        class1_x = np.random.randn(n_samples) * 0.5 + 1
        class1_y = np.random.randn(n_samples) * 0.5 + 1
        
        # Class 2 (blue)
        class2_x = np.random.randn(n_samples) * 0.5 - 1
        class2_y = np.random.randn(n_samples) * 0.5 + 1
        
        # Class 3 (green)
        class3_x = np.random.randn(n_samples) * 0.5
        class3_y = np.random.randn(n_samples) * 0.5 - 1
        
        # Create mesh for decision boundary
        x_min, x_max = -3, 3
        y_min, y_max = -3, 3
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                            np.linspace(y_min, y_max, 100))
        
        # Simulate decision boundary regions
        Z = np.zeros(xx.shape)
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                x, y = xx[i, j], yy[i, j]
                if x > 0 and y > 0:
                    Z[i, j] = 0  # Class 1
                elif x < 0 and y > 0:
                    Z[i, j] = 1  # Class 2
                else:
                    Z[i, j] = 2  # Class 3
        
        # Plot decision boundary
        self.boundary_ax.contourf(xx, yy, Z, alpha=0.3, levels=2, colors=['#ff9999', '#9999ff', '#99ff99'])
        
        # Plot data points
        self.boundary_ax.scatter(class1_x, class1_y, c='red', marker='o', label='Class 1', s=30, edgecolors='darkred', alpha=0.7)
        self.boundary_ax.scatter(class2_x, class2_y, c='blue', marker='s',  label='Class 2', s=30, edgecolors='darkblue', alpha=0.7)
        self.boundary_ax.scatter(class3_x, class3_y, c='green', marker='^', label='Class 3', s=30, edgecolors='darkgreen', alpha=0.7)
        
        self.boundary_ax.set_xlabel('Feature 1', fontsize=9)
        self.boundary_ax.set_ylabel('Feature 2', fontsize=9)
        self.boundary_ax.set_title('Decision Boundary Visualization', fontsize=10, fontweight='bold')
        self.boundary_ax.legend(loc='upper right', fontsize=8)
        self.boundary_ax.grid(True, alpha=0.3)
        self.boundary_ax.set_xlim(x_min, x_max)
        self.boundary_ax.set_ylim(y_min, y_max)
        
        self.boundary_fig.tight_layout()
        self.boundary_canvas.draw()
    
    def draw_confusion_matrix_heatmap(self):
        self.confusion_ax.clear()
        
        # Sample confusion matrix data
        confusion_matrix = np.array([
            [18, 1, 1],
            [2, 17, 1],
            [0, 1, 19]
        ])
        
        # Create heatmap
        im = self.confusion_ax.imshow(confusion_matrix, cmap='Blues', aspect='auto')
        
        # Set ticks and labels
        classes = ['Class 1', 'Class 2', 'Class 3']
        self.confusion_ax.set_xticks(np.arange(len(classes)))
        self.confusion_ax.set_yticks(np.arange(len(classes)))
        self.confusion_ax.set_xticklabels(classes, fontsize=9)
        self.confusion_ax.set_yticklabels(classes, fontsize=9)
        
        # Rotate the tick labels 
        plt.setp(self.confusion_ax.get_xticklabels(), rotation=45, ha="right")
 
        for i in range(len(classes)):
            for j in range(len(classes)):
                text = self.confusion_ax.text(j, i, confusion_matrix[i, j],
                                            ha="center", va="center", 
                                            color="white" if confusion_matrix[i, j] > 10 else "black",
                                            fontsize=12, fontweight='bold')
        
        self.confusion_ax.set_xlabel('Predicted Class', fontsize=9, fontweight='bold')
        self.confusion_ax.set_ylabel('Actual Class', fontsize=9, fontweight='bold')
        self.confusion_ax.set_title('Confusion Matrix Heatmap\nAccuracy: 90.0%', 
                                   fontsize=10, fontweight='bold')
        

        cbar = self.confusion_fig.colorbar(im, ax=self.confusion_ax, fraction=0.046, pad=0.04)
        cbar.set_label('Number of Samples', fontsize=8)
        
        self.confusion_fig.tight_layout()
        self.confusion_canvas.draw()

def main():
    root = tk.Tk()
    app = GUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()