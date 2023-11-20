#include <iostream>
#include <chrono>

int main() {
    auto start_training = std::chrono::high_resolution_clock::now();

    auto training_duration = std::chrono::high_resolution_clock::now() - start_training;
    std::cout << "Training Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(training_duration).count() << " ms\n";

    return 0;
}
