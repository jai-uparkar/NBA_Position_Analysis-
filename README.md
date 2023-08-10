# PSTAT_199_P3

## Overview :basketball:

The purpose of this repository is apply variational autoencoders to movement data capture by my Capstone sponor, P3. P3 captures basketball player's physiological data based on various types of movements. The company is able to capture hundreds of variables that cpature how certain parts of the athlete's body moves when performing certain actions that are demonstrative of their athleticism and conditioning levels. 

The four types of movements include:
* Drop Jump 
* Left Skater Jump
* Right Skater Jump
* Vertical Jump

With these movement data we aim to apply VAE in order to take this highly dimensional data and represent onto a latent space. Using this latent space we can try to find certain patterns in the data and observe how well we were able to encode and decode our data. I chose to focus on the vertical jump due to my familiarity with the data. 

## Goal 	:100:

By using unsupervised and generative machine learning techniques, the project will explore the potential and applications of variational autoencoders on P3's biomechanical data. This report presents my understanding of VAEs and its main principles, as well as my application to P3 data. The journey toward understanding VAEs began with reading basic literature, reviewing Mathilde's Pironoute Github repository, following a tutorial using image data, and writing original code to train a VAE using P3 data. It is the objective of this project and this quarter to gain insight into VAEs and their capabilities in generative modeling for biomechanical data.

## Timeline :spiral_calendar:
Week 3 - 4
* Read VAE literature
* Went through Pirounet code to understand the encoding and decoding process
 
Week 5 - 6
* Watch supplementary videos to understand backpropagation and basic topics about machine learning (gradient descent, loss, etc)
* Set up Python environment to run VAE on the local machine (initially ran into issues)
* Write a script to clean P3 data for VAE model
 
Week 7 - 8
* Discussed applications of P3 to VAE
* Dataset is too small so pivoted to image classification
* Learned about image classification with VAE
* Applied VAE to MNEST dataset guidance of pre-written script
 
Week 9 - 10
* Found alternative dataset for basketball image classification.
  * Didn’t have enough time to clean the dataset properly and optimize performance
* Instead, used cleaned P3 data on VAE to map latent space and find significant patterns
* Develop my own version of VAE code for P3 data
  * Expect to run into issues due to lack of data
 
## Variational Autoencoder Architecture :building_construction:

Variational autoencoders (VAE) are a type of generative model in machine learning. A generative model learns the underlying patterns and distribution of the data to generate new samples of the data that mimic the original data. Because generative models can generate new data that resembles the training data’s traits, they are often used in situations where there is limited data, unsupervised learning tasks, when we want to understand the training data, and when we want to encode data into lower dimensional latent space. While there are many more applications of generative machine learning, this project focuses on reducing the dimensionality of the training data since the P3 data consists of 100s of variables. Reducing the latent space allows for faster processing and reduces the space required to store the data while still capturing its distribution and patterns. 

For unsupervised learning tasks, VAEs are used to learn a condensed and efficient representation of training data. It does this by first compressing the training data into a latent space with an encoder and reconstructing it with a decoder. The goal is to find the best encoder and decoder pair. This is so that the encoder retains the maximum amount of information when encoding and has the lowest amount of error when decoding.   

At the most basic level, an encoder takes some input and transforms it into a latent space. During the encoding process, the encoder extracts significant features from the training data and creates a condensed representation (latent space). The latent space’s dimensions are significantly smaller than the input data in order to perform dimension reduction. One benefit of dimension reduction is to reduce computational resources required to process highly complex and large data. Another benefit is that it forces the model to disregard unimportant information and only focus on critical features that represent the data. This prevents overfitting and reduce noise. Depending on the data, encoders transform training data into latent space. Since P3’s data is numerical, the encoder is a fully connected neural network. This consists of connected nodes, neurons or units, arranged in layers. Outputs from nodes in each layer are passed to nodes in the next layer. Every node takes input from the previous layer, performs a nonlinear transformation, and produces output that is passed to the next layer. At a high level, a neural network consists of 3 main layers: input layers which are responsible for taking the input data and sending it to the neural network, the hidden layer which transforms the input data, and the output layer which produces the condensed representation of the training data. The decoder architecture is the same as the encoder except now the input of the decoder is the output of the encoder.   

![Figure1: Basic Neural Network Model](https://github.com/jai-uparkar/PSTAT_199_P3/blob/main/images/Screenshot%202023-06-24%20at%2010.53.10%20AM.png)

A VAE’s architecture is slightly different from a regular autoencoder because the encoding generated by the encoder is now regularized to prevent overfitting. This means that the latent space possesses desirable properties that enable a successful generative process. Rather than encoding the input as a normal distribution over the latent space, it is encoded as a single point (latent representation). Regularization is performed by forcing the latent distribution to resemble the standard normal distribution. By performing regularization, the latent space is guaranteed to be continuous, more predictable and interpretable, and less sparse. This allows for interpolation The encoder’s output consists of a mean and standard deviation vector that represent latent space as a normal distribution. Both these vectors are used by the decoder to sample points from the distribution as input. Sampling from the distribution allows the model to generate new data from the latent space the encoder learned. Generating a latent distribution and sampling from it is what makes a VAE generative. This is because it can create completely new data points that resemble training data. A traditional encoder-decoder model simply transforms the input to an output without learning the underlying distribution of the data. This allows our model to be more robust and prevent overfitting since we can artificially generate new data points. The decoder takes the sampled point and reconstructs it to mimic the input closely.

![Figure 2: Autoencoder vs VAE Architecture](https://github.com/jai-uparkar/PSTAT_199_P3/blob/main/images/Screenshot%202023-06-24%20at%2010.47.23%20AM.png)

## MNIST Application :1234:

To learn about VAEs in a real-world setting, I followed a tutorial that demonstrated how to implement them on the MNIST dataset. The MNIST dataset is highly popular in the generative and classification tasks in machine learning. The dataset consists of grey-scale images with handwritten numbers from 0 to 9. The classification tasks learn how to classify handwritten digits into numbers, and the generative tasks recreate the handwritten number. I used PyTorch to implement the VAE on the MNIST dataset. 

In the previous section, I didn’t mention VAE's loss function. The loss function evaluates how well the model predicts training data. During training, the loss function uses backpropagation to "propagate" the errors of the current layer to all previously hidden layers. For this implementation and VAEs generally, the loss function consists of a reconstruction and a regularization term. The reconstruction term is computed by comparing the input and the decoder’s generated version. The regularization term ensures that the encoder's distribution resembles a normal distribution as closely as possible. 

In this implementation, the VAE’s encoder and decoder have 3 layers. The encoder outputs the mean and variance matrix for the decoder to sample from. The bottom of the code shows the difference between the actual input data and the corresponding handwritten digit generated by the VAE. The VAE’s reconstructed images lack the level of precision that the original version achieves, but they retain all the details and structure to discern the actual number. The lack of precision comes from the encoding process where certain traits about the input are lost when encoding it into latent space. 

![MNSIT results](https://github.com/jai-uparkar/PSTAT_199_P3/blob/main/images/Screenshot%202023-06-24%20at%2011.19.03%20PM.png)

*Note: Mathilde gave me this [article](https://medium.com/dataseries/variational-autoencoder-with-pytorch-2d359cbf027b) to use and recreate its findings. My MNIST code comes from this. I went through this code step by step before implementing it myself. Mathilde also asked me questions about the code to test and ensure my understanding.*

## P3 Application :bouncing_ball_man:

In Week 6 of the spring quarter, Mathilde and I discussed VAE applications on P3 data. As a starting point for the project, I cleaned the vertical jump dataset. The cleaning process involved removing any non-numeric features, merging with the scraped NBA data for position analysis, and other modifications I made when processing the P3 data for my capstone project. However, after cleaning, only 500 date records were left, which was not sufficient for a VAE. As soon as I discussed this with Mathilde, she expressed concern about the model's ability to train. As a rule of thumb, VAEs have thousands of data points, but the P3 dataset has only approximately 500 observations after it has been cleaned. There is not enough training data for the VAE to learn a relevant distribution of the training data based on 500 data points. The VAE is a deep learning model that induces its own features and parameters and adjusts based on the loss. It lacks the interpretable structure of traditional machine learning models. Because the model’s architecture itself is complex, VAEs need more data because they learn the representations of data (distribution, reduced dimension, etc), have a large number of parameters, and employ regularization techniques to prevent overfitting. 

After writing the VAE code to train and display the latent space, my model could not train. My model failed to train and I got NAN values during the second epoch of training. There are several reasons why this could happen but the biggest culprit is numerical instability. NAN values could appear due to using the wrong loss function, incorrect data preprocessing, and numerical computations. Because the error is backpropagated to the previous layers of the model, issues like the vanishing/exploding gradient can occur if the model isn’t converging. Our problem, however, was caused by (1) improper data preprocessing and (2) incorrect loss functions. With Mathilde’s help, I changed the loss function to correct and normalized the training data. Once I fixed these problems, my model could train. Like many other machine learning models, the VAE was sensitive to the input data scale. After the model failed, I looked at the original data. Some features had values around 2000, while others had values of a few decimals. So, normalizing the data allowed all the features to have similar scales which enabled convergence and numerical stability. 

Mathilde mentioned that even if I normalized my data and clipped the gradient, my VAE could still not train because of lack of data. I made these changes during the final week not expecting the model to run, but it did! I didn’t have much time to analyze the results because of the time spent cleaning and writing the code for the VAE. However, I did some preliminary analysis of my results. Mathilde mentioned that this was okay since she herself didn't anticipate the model to train or have significant results due to the lack of data.

## Results :white_check_mark:

### Latent Space Visualization 

Using my trained VAE, I visualized the first 2 dimensions of the latent space. Although I didn’t use the position feature in my training data, I wanted to see if the latent vectors cluster similar positions together. Looking at the plot, there are no distinct clusters based on position since they overlap a lot. The data points are relatively spread across the plot but there is a significant concentration of points in the center. This means that the VAE hasn’t learned to distinguish between different positions based on vertical jump data. This probably means that the VAE is picking up on other features and patterns unrelated to the player’s position which is why we observed a “compressed” latent space where there aren’t distinct positional separations. Another cause could simply be the lack of training data the model had to learn the distribution. It’s also likely that the vertical jump dataset doesn’t have enough variation in position information encoded into the measurements. Even though there is a relationship between player height and position since all of these athletes train and remain in good physical condition, it's possible for their vertical jumps to have similar measurements.

![Latent Space Visualizations](https://github.com/jai-uparkar/PSTAT_199_P3/blob/main/images/Screenshot%202023-06-25%20at%2012.26.43%20PM.png)
*Figure 1: Plot the first and second dimensions of the latent space generated by the VAE's encoders labeled by position.*

### Visualize Decoder’s Reconstructed Input 

To further analyze my model, I also sampled uniformly from the latent space and visualized how the decoder reconstructed the input based on the latent space. The original inputs are displayed in blue and reconstructed with red. A good VAE will demonstrate the similarity between the original and reconstructed data points. This will capture the patterns of the original data but also have variability. Therefore, there should be a small reconstruction error, which is represented by the distance between the blue and red data points. From the latent space, I randomly sampled 50 points, and the x-axis represents the 130 features for each observation. Note that there are more red points than blue points on the plot. This occurs because multiple samples are drawn from the latent space’s distribution. This means that every sample can yield a different output which results in multiple outputs for one input. This comes from the VAE model's probabilistic and generative nature. 

![Reconstruction](https://github.com/jai-uparkar/PSTAT_199_P3/blob/main/images/Screenshot%202023-06-25%20at%209.47.25%20AM.png)

*Figure 2: Scatter plot showing the reconstruction of inputs from latent space samples visualizing how the VAE attempts to reconstruct the original inputs, capturing the overall patterns and trends.*

This plot allows us to visually evaluate the VAE’s accuracy for each feature when reconstructing the original data from the latent space samples. Looking at our plot, we observe a distinct separation between our actual and reconstructed data points. Although there is some overlap there is a clear distinction between the original and reconstructed data points. Since there is a significant separation between the points, our VAE unsuccessfully reconstructs the original data. The red points don’t mimic the patterns and trends in the blue dots as they resemble a straight line near the center of the graph. We also observe that the blue dots are spread out while the red dots are tightly clustered. All of these observations demonstrate that the VAE has failed to reconstruct the input data and cannot capture the patterns and variability of the training data. This can be attributed in part to the small sample size (only ~500 observations). The tight clustering of the red dots means that the VAE generates similar outputs (reconstructions) for different data inputs, indicating that it’s not capturing the complexity of the original data.

## Summary :writing_hand:	

Even though our VAE was able to train, our plots and analysis showed that the VAE was not capable of capturing the complex relationships and variations in P3’s vertical jump database. The VAE couldn't separate the data points by observations and its reconstructed data points didn’t follow the patterns and trends observed in the original data. This is likely due to the small dataset we used to train which by default has less variation than a larger dataset. This makes the model more prone to overfitting, and sensitive to outliers. It boils down to the lack of complexity and variation in a smaller dataset. This prevents deep learning models like VAEs from properly learning latent space that captures the biomechanical distribution of the vertical jump. 

## Sources
These are articles, videos, and textbooks I used to supplment my learning of VAEs. I used more resources but these were the most important ones. 

[VAE with PyTorch](https://medium.com/dataseries/variational-autoencoder-with-pytorch-2d359cbf027b)

[Implementing VAE with PyTorch](https://medium.com/pytorch/implementing-an-autoencoder-in-pytorch-19baa22647d1)

[Intoduction to Deep Learning](https://www.youtube.com/watch?v=aircAruvnKk&list=RDCMUCYO_jab_esuFRV4b17AJtAw&start_radio=1&rv=aircAruvnKk&t=0&ab_channel=3Blue1Brown)

[Understanding VAE Architecture](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73)

[Machine Translation in NLP](https://web.stanford.edu/~jurafsky/slp3/13.pdf)

## Acknowledgments :raised_hands:

Thank you to Mathilde Papillon for guiding me in my VAE journey and mentoring me this quarter. With her help, expertise, and guidance, I wouldn't have been able to achieve this level of understanding and growth this quarter. Additionally, thank you to Dr. Alex Franks for supervising and giving me the green light for this project. 



