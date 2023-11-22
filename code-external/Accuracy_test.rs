fn evaluate_performance(&self, test_inputs: &Vec<Vec<f64>>, test_targets: &Vec<Vec<f64>>) {
    let predictions = self.predict(test_inputs);
    let accuracy = calculate_accuracy(&predictions, test_targets);

    println!("Accuracy on Test Dataset: {:.2}%", accuracy * 100.0);
}

fn calculate_accuracy(predictions: &Vec<Vec<f64>>, targets: &Vec<Vec<f64>>) -> f64 {
    let correct_predictions = predictions
        .iter()
        .zip(targets.iter())
        .filter(|&(pred, target)| (pred[0] > 0.5 && target[0] > 0.5) || (pred[0] <= 0.5 && target[0] <= 0.5))
        .count() as f64;

    correct_predictions / predictions.len() as f64
}

int main(){
    neural_network.evaluate_performance(&test_inputs, &training_targets);
}