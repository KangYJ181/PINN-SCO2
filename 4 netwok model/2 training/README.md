Notable nomenclature:

1. "DD": data-driven RNN model

2. "M1": soft constraint framework

3. "M2": hard constraint framework

4. "P1": physical constraint 1

5. "P2": physical constraint 2

6. "optimization_results": the result of Bayesian optimization

7. "best_rnn_model": the final trained neural network file

8. "training_losses": the change of training loss of the final neural network

9. "final_predictions": the performance of the final trained neural network in generalization concentration

10. "optimized_coefficients": the optimization results of each coefficient in empirical formula of PINN under soft constraint framework

Other notes：
As the optimization process of soft constraint PINN is very time-consuming, in order to improve efficiency, we set the effective 
initial points and the number of effective iterations of its Bayes optimization to 10 and 70, and the corresponding parameters of 
other models are set to 50 and 150. Although the number of optimization iterations of soft constraint PINN is less than that of 
other models, the results show that higher precision is achieved, so we believe that the above operation is acceptable.
