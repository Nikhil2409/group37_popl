# Problem Statement

The problem statement in this code is to implement a simple neural network and train it to perform logical XOR operations. The neural network has an input layer with 2 nodes, a hidden layer with 4 nodes, and an output layer with 1 node. The code uses a sigmoid activation function and its derivative for the network's activation.

# Concepts of POPL used

1. Abstraction:
   - The code uses a struct (`NeuralNetwork`) to encapsulate the properties and behavior of a neural network. This is an example of abstraction, where the complexity of the neural network is hidden behind a simplified interface.

2. Functional Programming:
   - The code utilizes functional programming concepts, such as defining functions (sigmoid, sigmoid_derivative) that operate on input parameters without modifying state. Functional programming principles often emphasize immutability and pure functions, which can enhance code clarity and maintainability.

3. Modularity:
   - The code breaks down the functionality of the neural network into modular components. For example, the `train` function handles the training process, and the `predict` function handles making predictions. This modular design contributes to code organization and reusability.

4. Generics:
   - The use of generic types is not explicitly present in this code. However, the code leverages the Rust programming language's ability to work with generic data types and functions, contributing to flexibility and code reuse.

5. Imperative and Declarative Styles:
   - The code primarily follows an imperative style, especially in the training function where it explicitly defines the steps of forward pass, backward pass (`backpropagation`), and weight updates. Neural network code often involves a mix of imperative and declarative styles.

6. Lifetime Annotations:
   - Rust-specific lifetime annotations are used to manage reference lifetimes and guarantee memory safety when references are used in function parameters ({&Vec\Vec<f64>>}).

7. Mutable State and In-Place Updates:
   - In order to modify the weights of the neural network, the training method ('train') uses in-place updates and mutable state ('&mut self'). This demonstrates how the ownership concept in Rust allows for safe, flexible data access.

8. Time Measurement:
   - The code uses the ‘Instant’ type from the ‘std::time’ module to measure the duration of the training process. This illustrates how time-related operations are supported by Rust's standard library.
 
These principles contribute to writing clean, maintainable, and efficient code, aligning with the broader principles of programming languages. Additionally, the specific principles of neural network implementation, such as backpropagation for training, are reflected in the code to achieve the desired learning behavior


# Software Architecture:

1. Neural Network Module:

The core of the solution is the NeuralNetwork struct, which encapsulates the neural network's weights, activation functions, training, and prediction methods.

2. Training Module:
The train method within the NeuralNetwork struct is responsible for training the neural network using backpropagation. It takes input data, target data, and hyperparameters like the number of epochs and learning rate.

3. Prediction Module:
The predict method in the NeuralNetwork struct handles making predictions based on the trained weights.

4. Main Function:

The main function serves as the entry point of the program. It initializes the neural network, trains it on XOR data, measures training time, and then tests the trained network on additional input data, printing the predictions.

# Results and Analysis

Rust Code:

1. Thread Safety:
   - Ownership System: Rust's ownership system ensures thread safety by preventing multiple 
threads from having mutable references to data. Ownership and borrowing rules are      enforced at compile-time.
Code Reference:
`fn train(&mut self, inputs: &Vec<Vec<f64>>, targets: &Vec<Vec<f64>>, epochs: usize,   learning_rate: f64) { /* ... */ };`

 2. Memory Safety:
    - Vector Bounds Checking: Rust enforces bounds checking on vectors at runtime.
 Reference: [`Rust Vector`] (https://doc.rust-lang.org/std/vec/struct.Vec.html)
    - No Unsafe Blocks: The absence of `unsafe` blocks indicates that there are no low-level    memory safety concerns.
 Code Reference:
 `let mut layer1_output = vec![0.0; self.weights_input_hidden[0].len()];`

3)Training Time and Accuracy Analysis:
  These modifications include a new function (evaluate_performance in Rust and 
  evaluatePerformance in C++) to assess the accuracy of the trained model on the test dataset. 
  The accuracy is calculated based on a simple threshold (0.5) for binary classification.
  ![image](https://github.com/PreetShah67/group37_popl/assets/101982166/3c11d3ea-a6df-43b9-9f04-41b6d23cc30b)
  ![image](https://github.com/PreetShah67/group37_popl/assets/101982166/d1e25535-f39e-43df-ae58-24f3fd66db3a)

 C++ Code:

 1. Thread Safety:
    - Lack of Synchronization: The C++ code lacks explicit synchronization mechanisms for 
 thread safety during weight updates.
 Code Reference:
 `void train(vector<vector<double>>& inputs, vector<vector<double>>& targets, size_t 
 epochs,
 double learning_rate) { /* ... */ }`
   - Synchronization mechanisms, like mutexes, can be added as in the reference.
Reference: [`std::mutex`](https://en.cppreference.com/w/cpp/thread/mutex).
Code Reference:
`void train(vector<vector<double>>& inputs, vector<vector<double>>& targets, size_t 
      epochs,double learning_rate) { /* ... */ }`

2. Memory Safety:
   - Manual Memory Management: C++ allows manual memory management, introducing the risk of  memory leaks and dangling pointers.
Code Reference: `std::unique_ptr<int[]> data(new int[size]);`
   - Smart pointers, such as `std::unique_ptr` or `std::shared_ptr`, can be used for safer     memory management.
Reference: [std::unique_ptr](https://en.cppreference.com/w/cpp/memory/unique_ptr),      [std::shared_ptr](https://en.cppreference.com/w/cpp/memory/shared_ptr)
   - Vector Bounds Checking: C++ does not perform bounds checking on vectors by default. 
Developers must ensure proper bounds checking.
Reference: [std::vector](https://en.cppreference.com/w/cpp/container/vector)
Code Reference:
`vector<double> layer1_output(hidden_nodes, 0.0);`
   - Potential for Use-After-Free: Use smart pointers or careful memory management to prevent 
use-after-free errors.
Reference:
`std::shared_ptr`

3. Training Time and Accuracy Analysis:
   These modifications include a new function (evaluate_performance in Rust and 
evaluatePerformance in C++) to assess the accuracy of the trained model on the test dataset. 
The accuracy is calculated based on a simple threshold (0.5) for binary classification.
   ![image](https://github.com/PreetShah67/group37_popl/assets/101982166/446a232e-a468-4f54-9dff-e8bc726e456a)
   ![image](https://github.com/PreetShah67/group37_popl/assets/101982166/03c5f4be-ec24-4edc-9397-7b55d6ca9bb5)
  

Conclusion:
  - Rust provides strong thread safety guarantees through its ownership system, and memory 
  safety is enforced through mechanisms like bounds checking.
  - In C++, developers need to add explicit synchronization for thread safety and use smart 
  pointers for safer memory management.

# Potential for Future work

The provided code implements a simple two-layer feedforward neural network with sigmoid activation functions. It can be further enhanced by incorporating additional features and techniques:

1. Regularization:
   - To prevent overfitting, regularization techniques like L1 or L2 regularization can be applied to the weights. This involves adding a penalty term to the loss function that encourages smaller weights.

2. Batch Normalization:
   - Batch normalization can be employed to improve the stability and convergence of the training process. It normalizes the activations of each layer across a mini-batch of training data.

3. Momentum:
   - Momentum can be incorporated into the weight update rule to accelerate the training process and avoid getting stuck in local minima. It adds a weighted average of previous weight updates to the current update.

4. Adaptive Learning Rates:
   - Instead of using a fixed learning rate, adaptive learning rate techniques like AdaGrad or Adam can be employed. These algorithms dynamically adjust the learning rate for each parameter based on its past gradients.

5. Early Stopping:
   - Early stopping can be implemented to prevent overfitting by monitoring the performance on a validation dataset and stopping the training when the validation error starts to increase.

6. Dropout:
   -  Dropout is a regularization technique that randomly drops out neurons during training, forcing the network to learn more robust representations. This helps to prevent overfitting and improve generalization performance.

7. Cross-Validation:
   - Cross-validation can be used to evaluate the performance of the trained model more accurately. This involves splitting the training data into multiple folds and training the model on different subsets while evaluating it on the remaining fold.

8. Hyperparameter Optimization:
   -Hyperparameter optimization techniques like grid search or random search can be used to find the optimal values for hyperparameters like learning rate, regularization strength, and network architecture.

By incorporating these additional features and techniques, the neural network's performance and generalizability can be significantly improved.
