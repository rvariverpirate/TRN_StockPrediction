# Application of Temporal Recurrent Networks on Stock Market Data

This project seeks to explore the application of a Temporal Recurrent Network on stock market price data. This implementation is based on the work [Temporal Recurrent Networks for Online Action Detection](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwjxxfyz2dbtAhWi1FkKHRhzB9UQFjACegQIBRAC&url=https%3A%2F%2Fopenaccess.thecvf.com%2Fcontent_ICCV_2019%2Fpapers%2FXu_Temporal_Recurrent_Networks_for_Online_Action_Detection_ICCV_2019_paper.pdf&usg=AOvVaw3bkn-B6Op-F48mPyfTd47n) and the primary algorithm for the TRN is fundamentally identical [(See the gitHub page for more details)](https://github.com/xumingze0308/TRN.pytorch). However, as opposed to the authors work, our data TRN does not rely on an external feature extractor prior to entering the TRN. Instead, the min-max scaled data is fed directly into the algorithm. Additionally, while the previous work operates on 2D images, we adapted it to operate on a single scalar input at each time step, the stock price. We chose not to apply dropout between fully connected layers in order ensure the different models we tested only differed the parameters we were tracking.

While the authors chose to create a complex command line interface for using their TRN, we instead thinned down the supporting code and extract just the fundamental operational aspects. Outside of the TRN class we created a few additional modules to assist in data parsing, model training, and evaluation.

The Pipeline class was created to provide a simple means for selecting an input file, trimming data, selecting the column of interest, and specifying the number of decoders that the TRN will utilize. Note that a single input file is passed to the pipeline. No additional labeled target data is required, because our task at each time-step is to predict the following time-steps value, which simply means the target “truth” data can be derived by simply shifting the input data to the left.

The number of decoders were specified here because the TRN not only updates the STA based on its output error, but also each of the Decoder cell’s individual errors. Like the target data for the STA, the decoder data is also derived from the input data but each additional decoder step would correspond to an additional shift of the data. For example, the expanded form of an time series [1,2,3,4,5,6,7] for three decoders would be [1,2,3,2,3,4,3,4,5,4,5,6,5,6,7].

The TrainModel method in RunAll.py allows a model to be run on a given data set with a variable number of epochs. This method utilizes the ADAM optimizer for back propagation and uses Mean Squared Error to compute loss. We chose to use MSE instead of the Cross-Entropy Loss utilized in the paper because our target in not a class but a value so Cross Entropy would be inappropriate for this use.

Finally, the VizHelp class was created to assist with the visualization of the results and provides two generic methods allowing the Training Error vs Epoch to be easily plotted as well as the predicted values.

This software implementation is written entirely in Python 3.7, has dependencies on the following software libraries. Pytorch, Numpy, Pandas, Matplotlib, Seaborn, and SciKit-Learn.


![alt text](https://raw.githubusercontent.com/rvariverpirate/TRN_StockPrediction/master/images/Test_AMZN_D4_E50.png "Test Example 1")
![alt text](https://raw.githubusercontent.com/rvariverpirate/TRN_StockPrediction/master/images/Test_IBM_D3_E50.png "Test Example 2")
