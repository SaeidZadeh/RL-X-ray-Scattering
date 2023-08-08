# X-ray-Scattering
## Reinforcement Learning
The code for reinforcement learning approach used to obtain measurement time, and find highest accuracy in predicting Thickness, Roughness, and SLD with only partial q-points.

The code RL With Keras is reinforcement learning using Keras and RL With Numpy is reinforcement learning using numpy.

## Data Generator, Models Training
The code 2layer clean, considers a film with two intermediate layers (regardless of Silicon and air layers). The code is in several sections:
- It generates reflectivity curves for random values of thickness, roughness, and sld's of the two intermediate layers taking into account the constraints on the parameters. This can be extended to several intermediate layers. We considered 1024 $q$ points and generate No\_files csv files containing infromation about randomly generated No\_curves\_in\_a\_file reflectivity curves.
- A sequential dense neural network with relu activations and mean absolute errors as the loss functions were trained to predict all $8$ parameters of thickness, roughness, and sld's (real and imaginary) for both layers.
- Since the goal is to measure reflectivity values is some $q$ points (not all) and then make predictions on all the parameters, Autoencoder model is trained to fill the missing values.
- Random forest regressor is used to fit the data and find the importance of parameters with respect to values of all the parameters.

Then we let the models be trained on the simulated data and save the weights of all the models in separate files.

## Errors Comparison
The code Reflectivity1024\_comparison2layer loads the model weights obtained after running the code 2layer clean and compare their results in predicting all $8$ parameters. Three scenarios are considered:
- Equal distance: When $N\leq1024$ $q$ points are meaasured with all in equal distances.
- Random forest regressor: When the result of random forest regressor is used to find the first $N\leq1024$ $q$ points to be measured.
- Random: When the first $N\leq1024$ $q$ points to be measured are selected randomly.
Within all these scenarios we use the following methods to predict all the parameters:
- Using Refnx package on the measured reflectivity values, skipping the missing values.
- Using Refnx package on the measured reflectivity values, the missing values will be filled using the trained autoencoder.
- Using the trained sequential dense network on the measured reflectivity values where the missing values are filled using the trained autoencoder to find initial prediction of all parameters. Then a minimization with the mean square error of the predicted reflectivity curve defined using initial prediction of parameters is made compared with the actual measured reflectivity curves.
Next the absolute difference is measured to find the error of each method in making predictions using each of these methods.

The methods are evaluated on onseen test set and for $1000$ randomly selected curves. The step size is $4$ $q$ points at a time.
