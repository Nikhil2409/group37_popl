extern crate gnuplot;
use rand::Rng;
use std::time::Instant;
use gnuplot::{Figure, Caption, Color};
//use crate::gnuplot::AxesCommon;
use gnuplot::Graph;
use gnuplot::AxesCommon;

struct NeuralNetwork {
    weights_input_hidden: Vec<Vec<f64>>,
    weights_hidden_output: Vec<Vec<f64>>,
}

impl NeuralNetwork {
    fn new(input_nodes: usize, hidden_nodes: usize, output_nodes: usize) -> Self {
        let mut rng = rand::thread_rng();
        let weights_input_hidden = (0..input_nodes)
            .map(|_| {
                (0..hidden_nodes)
                    .map(|_| rng.gen_range(-1.0..1.0))
                    .collect()
            })
            .collect();
        let weights_hidden_output = (0..hidden_nodes)
            .map(|_| {
                (0..output_nodes)
                    .map(|_| rng.gen_range(-1.0..1.0))
                    .collect()
            })
            .collect();

        NeuralNetwork {
            weights_input_hidden,
            weights_hidden_output,
        }
    }

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn sigmoid_derivative(x: f64) -> f64 {
        x * (1.0 - x)
    }

    fn train_and_analyze(&mut self, inputs: &Vec<Vec<f64>>, targets: &Vec<Vec<f64>>, epochs: usize, learning_rate: f64) {
        let mut losses = Vec::new();

        for epoch in 0..epochs {
            for (input, target) in inputs.iter().zip(targets.iter()) {
                // Forward pass
                let mut layer1_output = vec![0.0; self.weights_input_hidden[0].len()];
                for i in 0..layer1_output.len() {
                    for j in 0..input.len() {
                        layer1_output[i] += input[j] * self.weights_input_hidden[j][i];
                    }
                    layer1_output[i] = NeuralNetwork::sigmoid(layer1_output[i]);
                }

                let mut output = vec![0.0; self.weights_hidden_output[0].len()];
                for i in 0..output.len() {
                    for j in 0..layer1_output.len() {
                        output[i] += layer1_output[j] * self.weights_hidden_output[j][i];
                    }
                }

                // Backpropagation
                let mut output_error = vec![0.0; target.len()];
                for i in 0..output_error.len() {
                    output_error[i] = target[i] - output[i];
                }

                let mut output_delta = vec![0.0; output.len()];
                for i in 0..output_delta.len() {
                    output_delta[i] = output_error[i] * NeuralNetwork::sigmoid_derivative(output[i]);
                }

                let mut hidden_error = vec![0.0; layer1_output.len()];
                for i in 0..hidden_error.len() {
                    for j in 0..output.len() {
                        hidden_error[i] += output_delta[j] * self.weights_hidden_output[i][j];
                    }
                }

                let mut hidden_delta = vec![0.0; layer1_output.len()];
                for i in 0..hidden_delta.len() {
                    hidden_delta[i] = hidden_error[i] * NeuralNetwork::sigmoid_derivative(layer1_output[i]);
                }

                // Update weights
                for i in 0..self.weights_input_hidden.len() {
                    for j in 0..self.weights_input_hidden[0].len() {
                        self.weights_input_hidden[i][j] += input[i] * hidden_delta[j] * learning_rate;
                    }
                }

                for i in 0..self.weights_hidden_output.len() {
                    for j in 0..self.weights_hidden_output[0].len() {
                        self.weights_hidden_output[i][j] += layer1_output[i] * output_delta[j] * learning_rate;
                    }
                }
            }

            // Calculate and store the loss for each epoch
            let loss = calculate_loss(self, inputs, targets);
            losses.push(loss);
        }

        // Plot the training loss graph
        plot_loss_graph(&losses);
    }

    fn predict(&self, inputs: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let mut predictions = vec![];
        for input in inputs {
            let mut layer1_output = vec![0.0; self.weights_input_hidden[0].len()];
            for i in 0..layer1_output.len() {
                for j in 0..input.len() {
                    layer1_output[i] += input[j] * self.weights_input_hidden[j][i];
                }
                layer1_output[i] = NeuralNetwork::sigmoid(layer1_output[i]);
            }

            let mut output = vec![0.0; self.weights_hidden_output[0].len()];
            for i in 0..output.len() {
                for j in 0..layer1_output.len() {
                    output[i] += layer1_output[j] * self.weights_hidden_output[j][i];
                }
            }
            predictions.push(output);
        }
        predictions
    }

    fn evaluate_performance(&self, test_inputs: &Vec<Vec<f64>>, test_targets: &Vec<Vec<f64>>) {
        let predictions = self.predict(test_inputs);
        let accuracy = calculate_accuracy(&predictions, test_targets);

        println!("Accuracy on Test Dataset: {:.2}%", accuracy * 100.0);
    }
}

fn calculate_loss(network: &NeuralNetwork, inputs: &Vec<Vec<f64>>, targets: &Vec<Vec<f64>>) -> f64 {
    let predictions = network.predict(inputs);
    let loss: f64 = predictions
        .iter()
        .zip(targets.iter())
        .map(|(pred, target)| (target[0] - pred[0]).powi(2))
        .sum();
    
    loss / predictions.len() as f64
}

fn calculate_accuracy(predictions: &Vec<Vec<f64>>, targets: &Vec<Vec<f64>>) -> f64 {
    let correct_predictions = predictions
        .iter()
        .zip(targets.iter())
        .filter(|&(pred, target)| (pred[0] > 0.5 && target[0] > 0.5) || (pred[0] <= 0.5 && target[0] <= 0.5))
        .count() as f64;

    correct_predictions / predictions.len() as f64
}

fn plot_predictions(predictions: &Vec<Vec<f64>>) {
    let mut fg = Figure::new();
    fg.set_terminal("pngcairo", "output.png"); // Specify output format and file name
    fg.axes2d()
        .set_title("predictions made by the neural network for each XOR input", &[])
        .set_legend(Graph(0.5), Graph(0.9), &[], &[])
        .set_x_label("training inputs(decimal)", &[])
        .set_y_label("predictions", &[])
        .lines(
            (0..4).map(|i| i as f64),
            predictions.iter().map(|p| p[0]),
            &[],
        );

    fg.show().unwrap();
}

fn plot_loss_graph(losses: &Vec<f64>) {
    let mut fg = Figure::new();
    fg.set_terminal("pngcairo", "loss_plot.png");

    fg.axes2d()
        .set_title("Training Loss Over Epochs", &[])
        .lines(0..losses.len() as u32, losses, &[Color("blue")]);

    fg.show();
}

fn main() {
    let start_training = Instant::now();
    let training_inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let training_targets = vec![
        vec![0.0],
        vec![1.0],
        vec![1.0],
        vec![0.0],
    ];

    let mut neural_network = NeuralNetwork::new(2, 4, 1);
    neural_network.train_and_analyze(&training_inputs, &training_targets, 10000, 0.1);
    let training_duration = start_training.elapsed();

    let test_inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];

    let predictions = neural_network.predict(&test_inputs);

    for (input, prediction) in test_inputs.iter().zip(predictions.iter()) {
        println!(
            "Input: {{{}, {}}} => Prediction: {:.6}",
            input[0], input[1], prediction[0]
        );
    }

    println!("Training Time: {:?}", training_duration);

    neural_network.evaluate_performance(&test_inputs, &training_targets);
    plot_predictions(&predictions);
}