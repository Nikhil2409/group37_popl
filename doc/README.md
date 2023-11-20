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
Thread Safety:
-Ownership System: Rust's ownership system ensures thread safety by preventing multiple threads from having mutable references to data. Ownership and borrowing rules are enforced at compile-time.
Code Reference:
fn train(&mut self, inputs: &Vec<Vec<f64>>, targets: &Vec<Vec<f64>>, epochs: usize, learning_rate: f64) { /* ... */ }

 Memory Safety:
- Vector Bounds Checking: Rust enforces bounds checking on vectors at runtime.
  - Reference: [`Rust Vector`] (https://doc.rust-lang.org/std/vec/struct.Vec.html)
- No Unsafe Blocks: The absence of `unsafe` blocks indicates that there are no low-level memory safety concerns.
Code Reference: 
let mut layer1_output = vec![0.0; self.weights_input_hidden[0].len()];

 C++ Code:

 Thread Safety:
  - Lack of Synchronization: The C++ code lacks explicit synchronization mechanisms for thread safety during weight updates.
  - Synchronization mechanisms, like mutexes, can be added as in the reference.
  - Reference: [`std::mutex`](https://en.cppreference.com/w/cpp/thread/mutex).
Code Reference: 
void train(vector<vector<double>>& inputs, vector<vector<double>>& targets, size_t epochs, double learning_rate) { /* ... */ }

Memory Safety:
  - Manual Memory Management: C++ allows manual memory management, introducing the risk of memory leaks and dangling pointers.
  - Smart pointers, such as `std::unique_ptr` or `std::shared_ptr`, can be used for safer memory management.
  - Reference: [std::unique_ptr](https://en.cppreference.com/w/cpp/memory/unique_ptr), [std::shared_ptr](https://en.cppreference.com/w/cpp/memory/shared_ptr)
  -Vector Bounds Checking: C++ does not perform bounds checking on vectors by default. Developers must ensure proper bounds checking.
  -Reference: [std::vector](https://en.cppreference.com/w/cpp/container/vector)
- Potential for Use-After-Free: Use smart pointers or careful memory management to prevent use-after-free errors.

Conclusion:
- Rust provides strong thread safety guarantees through its ownership system, and memory safety is enforced through mechanisms like bounds checking.
- In C++, developers need to add explicit synchronization for thread safety and use smart pointers for safer memory management.

