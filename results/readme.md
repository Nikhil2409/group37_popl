#RESULTS for training loss C++ v/s rust

Comparing the two graphs, loss_C++ and loss_rust, we can observe some similarities and differences in terms of the model's training performance. Both graphs show a similar initial decline in training loss, indicating that both models were learning effectively during the early stages of training. However, loss_C++ exhibits a more gradual decline in training loss compared to loss_rust. This suggests that the model in loss_rust may require more time to reach a plateau of performance.

Another difference lies in the validation loss. In loss_rust, the validation loss is generally lower than the training loss, indicating that the model may be less prone to overfitting. This is contrary to loss_C++, where the validation loss consistently remained higher than the training loss, suggesting overfitting.

Overall, both graphs provide valuable insights into the training performance of respective models. Loss_rust suggests a model that may require more time to reach its full potential but potentially exhibits lower overfitting tendencies. On the other hand, loss_C++ highlights a model that may reach a plateau sooner but may be more prone to overfitting
