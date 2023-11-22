#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>

using namespace std;

struct NeuralNetwork {
    vector<vector<double>> weights_input_hidden;
    vector<vector<double>> weights_hidden_output;
    size_t input_nodes, hidden_nodes, output_nodes;

    NeuralNetwork(size_t input_nodes, size_t hidden_nodes, size_t output_nodes)
        : input_nodes(input_nodes), hidden_nodes(hidden_nodes), output_nodes(output_nodes) {
        weights_input_hidden.resize(input_nodes, vector<double>(hidden_nodes));
        weights_hidden_output.resize(hidden_nodes, vector<double>(output_nodes));

        // Initialize weights with random values between -1 and 1
        srand(static_cast<unsigned>(time(nullptr)));
        for (size_t i = 0; i < input_nodes; ++i) {
            for (size_t j = 0; j < hidden_nodes; ++j) {
                weights_input_hidden[i][j] = static_cast<double>(rand()) / RAND_MAX * 2 - 1;
            }
        }
        for (size_t i = 0; i < hidden_nodes; ++i) {
            for (size_t j = 0; j < output_nodes; ++j) {
                weights_hidden_output[i][j] = static_cast<double>(rand()) / RAND_MAX * 2 - 1;
            }
        }
    }

    double sigmoid(double x) {
        return 1 / (1 + exp(-x));
    }

    double sigmoid_derivative(double x) {
        return x * (1 - x);
    }

    void train(vector<vector<double>>& inputs, vector<vector<double>>& targets, size_t epochs, double learning_rate) {
        ofstream loss_file("loss_data.txt");
        if (!loss_file.is_open()) {
            cerr << "Error opening loss_data.txt for writing" << endl;
            exit(EXIT_FAILURE);
        }

        for (size_t epoch = 0; epoch < epochs; ++epoch) {
            double total_loss = 0.0;
            for (size_t i = 0; i < inputs.size(); ++i) {
                vector<double> layer1_output(hidden_nodes, 0.0);
                vector<double> output(output_nodes, 0.0);

                // Forward pass
                for (size_t j = 0; j < hidden_nodes; ++j) {
                    for (size_t k = 0; k < input_nodes; ++k) {
                        layer1_output[j] += inputs[i][k] * weights_input_hidden[k][j];
                    }
                    layer1_output[j] = sigmoid(layer1_output[j]);
                }

                for (size_t j = 0; j < output_nodes; ++j) {
                    for (size_t k = 0; k < hidden_nodes; ++k) {
                        output[j] += layer1_output[k] * weights_hidden_output[k][j];
                    }
                    output[j] = sigmoid(output[j]);
                }

                vector<double> output_error(output_nodes, 0.0);
                vector<double> output_delta(output_nodes, 0.0);
                vector<double> hidden_error(hidden_nodes, 0.0);
                vector<double> hidden_delta(hidden_nodes, 0.0);

                // Backpropagation
                for (size_t j = 0; j < output_nodes; ++j) {
                    output_error[j] = targets[i][j] - output[j];
                    output_delta[j] = output_error[j] * sigmoid_derivative(output[j]);
                }

                for (size_t j = 0; j < hidden_nodes; ++j) {
                    for (size_t k = 0; k < output_nodes; ++k) {
                        hidden_error[j] += output_delta[k] * weights_hidden_output[j][k];
                    }
                    hidden_delta[j] = hidden_error[j] * sigmoid_derivative(layer1_output[j]);
                }

                // Weight adjustments
                for (size_t j = 0; j < input_nodes; ++j) {
                    for (size_t k = 0; k < hidden_nodes; ++k) {
                        weights_input_hidden[j][k] += inputs[i][j] * hidden_delta[k] * learning_rate;
                    }
                }

                for (size_t j = 0; j < hidden_nodes; ++j) {
                    for (size_t k = 0; k < output_nodes; ++k) {
                        weights_hidden_output[j][k] += layer1_output[j] * output_delta[k] * learning_rate;
                    }
                }

                // Calculate and store the loss for each iteration
                double loss = calculate_loss(inputs[i], targets[i]);
                total_loss += loss;
            }

            // Calculate and store the average loss for the epoch
            double avg_loss = total_loss / inputs.size();
            loss_file << avg_loss << endl;
        }

        loss_file.close();
    }

    double calculate_loss(vector<double>& input, vector<double>& target) {
        vector<double> predictions = predict(input);
        double loss = pow(target[0] - predictions[0], 2);
        return loss;
    }

    vector<double> predict(vector<double>& input) {
        vector<double> layer1_output(hidden_nodes, 0.0);
        vector<double> output(output_nodes, 0.0);

        // Forward pass
        for (size_t j = 0; j < hidden_nodes; ++j) {
            for (size_t k = 0; k < input_nodes; ++k) {
                layer1_output[j] += input[k] * weights_input_hidden[k][j];
            }
            layer1_output[j] = sigmoid(layer1_output[j]);
        }

        for (size_t j = 0; j < output_nodes; ++j) {
            for (size_t k = 0; k < hidden_nodes; ++k) {
                output[j] += layer1_output[k] * weights_hidden_output[k][j];
            }
            output[j] = sigmoid(output[j]);
        }

        return output;
    }

    void evaluatePerformance(vector<vector<double>>& testInputs, vector<vector<double>>& testTargets) {
        double correctPredictions = 0.0;

        for (size_t i = 0; i < testInputs.size(); ++i) {
            vector<double>& input = testInputs[i];
            vector<double>& target = testTargets[i];

            vector<double> prediction = predict(input);

            // Assuming a binary classification problem
            if ((prediction[0] > 0.5 && target[0] > 0.5) || (prediction[0] <= 0.5 && target[0] <= 0.5)) {
                correctPredictions += 1.0;
            }
        }

        double accuracy = correctPredictions / testInputs.size();
        cout << "Accuracy on Test Dataset: " << accuracy * 100.0 << "%\n";
    }

void plotPredictionsVsInputs(vector<vector<double>>& testInputs, vector<vector<double>>& testTargets) {
    ofstream predictions_file("predictions_data.txt");
    if (!predictions_file.is_open()) {
        cerr << "Error opening predictions_data.txt for writing" << endl;
        exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < testInputs.size(); ++i) {
        vector<double> input = testInputs[i];
        vector<double> target = testTargets[i];
        vector<double> prediction = predict(input);

        predictions_file << input[0] << " " << input[1] << " " << target[0] << " " << prediction[0] << endl;
    }

    predictions_file.close();

    // Plotting predictions vs inputs using gnuplot
    ofstream gnuplot_script("plot_script.gp");
    if (!gnuplot_script.is_open()) {
        cerr << "Error opening plot_script.gp for writing" << endl;
        exit(EXIT_FAILURE);
    }

    gnuplot_script << "plot 'predictions_data.txt' using 1:2 with points title 'Inputs', "
                   << "'' using 1:3 with points title 'Targets', "
                   << "'' using 1:4 with points title 'Predictions'";

    gnuplot_script.close();

    system("gnuplot -p plot_script.gp");
}
};

int main() {
    auto start_training = chrono::high_resolution_clock::now();

    size_t input_nodes = 2;
    size_t hidden_nodes = 4;
    size_t output_nodes = 1;

    vector<vector<double>> training_inputs = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };

    vector<vector<double>> training_targets = {
        {0.0},
        {1.0},
        {1.0},
        {0.0}
    };

    NeuralNetwork neural_network(input_nodes, hidden_nodes, output_nodes);
    neural_network.train(training_inputs, training_targets, 10000, 0.1);

    auto training_duration = chrono::high_resolution_clock::now() - start_training;
    vector<vector<double>> test_inputs = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };

    for (size_t i = 0; i < test_inputs.size(); ++i) {
        vector<double> predictions = neural_network.predict(test_inputs[i]);
        cout << "Input: {" << test_inputs[i][0] << ", " << test_inputs[i][1] << "} => Prediction: " << predictions[0] << endl;
    }

    neural_network.plotPredictionsVsInputs(test_inputs, training_targets);

    cout << "Training Time: " << chrono::duration_cast<chrono::milliseconds>(training_duration).count() << " ms\n";

    neural_network.evaluatePerformance(test_inputs, training_targets);

    // Graph analysis - Plot the training loss graph
    system("gnuplot -p -e \"plot 'loss_data.txt' with lines title 'Training Loss Over Epochs'\"");

    return 0;
}
