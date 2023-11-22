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

int main(){
    neural_network.evaluatePerformance(training_inputs, training_targets);
}