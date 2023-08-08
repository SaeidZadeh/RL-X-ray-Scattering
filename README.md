# RL-X-ray-Scattering
The code for reinforcement learning approach used to obtain measurement time, and find highest accuracy in predicting Thickness, Roughness, and SLD with only partial q-points.

The code RL With Keras is reinforcement learning using Keras and RL With Numpy is reinforcement learning using numpy.

The code 2layer clean, considers a film with two intermediate layers (regardless of Silicon and air layers): The code is in several sections:
- It generates reflectivity curves for random values of thickness, roughness, and sld's of the two intermediate layers taking into account the constraints on the parameters. This can be extended to several intermediate layers. We considered 1024 $q$ points
