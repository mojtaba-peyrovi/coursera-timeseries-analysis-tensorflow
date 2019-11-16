## Sequence, Time series and Prediction
---

### Week 1:

__Univeriate time series:__ single value that varies equally over time. (daily weather forecast)
__Multivariate time series:__ more than one value varying over time intervals. (photos/multivariate-time-series.png)

It's so useful understanding the impact of RELATED data. for example birth vs death in Japan (japan-birth-vs-death.png) and (global-temperaature-vs-co2-concentation.png)

 __Use of ML:__ sometimes we want to predict the future based on historical data, also sometimes we want to project back into the past to know where we were before. This process is called __Imputation__.
 
 __Imputation use case:__ We can use it to fill in the data for sometimes we didn't have data and the chart was empty. but we can fill them in with this technique. Here is the example before imputation: (before-imputation.png)
 And now we see it after imputation (after-imputation.png)
 
Another use case, is th analyze sound waves, to spot words in them using neural networks. (sound-recognition.png)

__Time Series Attributes:__

- Trend (time-series-trend.png)
- Seasonality (time-series-seasonality.png)
- Combination of trend and seasonality.(time-series-trend-seasonality.png)
- not predictable. (time-series-not-predictable.png)
- Occasional spikes where some random spikes repeat in specific time periods so similarly.(occasional-spikes.png)

__Non-stationary time series:__ when in a time series we have a sudden shock and the trend drastically changes afterwards. 
 __Stationary:__ means we dont have the drastic trend changes over time. for this case we can take the whole past data as training, otherwise, we need to take shorter periods as training datasets.

If we have non-stationary, the optimal time window we use for trainig should vary.

__Fixed Partitioning:__ to make a good performance in forecast, we should split our time series into Training, Validation, and Test periods. (fixed-partitioning.png)

When we have seasonality, we want to make sure that each period contains the whole number of seasons. for example one year, two year or three years, if the data has yearly seasonality.

For time series, we don't do like other models that we pick random values for train, test, and validation.

For time series, in order to train the model, after we made the evaluation on validation dataset, and test the model on test data, we should *INCLUDE TEST PERIOD, IN TRAINING BECAUSE IT HAS THE CLOSEST TIME DATA TO THE FORECAST PREIOD WE WANT.*

__Roll-Forward Partitioning:__ We start with a short training period, and then gradually increase it. for eaxmple by one day at the time. (roll-forward-partitioning.png)

At each iteration, we train the model on a training period, and use it to forecast the following day, week, etc.

Evaluation metrics: evaluation-metrics.png

__Differencing Technique:__ 

In this technique we want to remove the trend and seasonality,which means instead of plotting and analyzing the time series itself, we calculate the difference between each date T and the value at an earlier period. This period can be week, month, day, etc.
- The first step is to deduct a fixed value from each date point. (differencing-step-1.png)
- Step two, is to add moving average to it. (differencing-step-2.png)
- step three is after performing the forecat on the difference values, to add that fixed value we deducted, back. (diffferencing-step-3.png)
- step four: we remove the past noise, using moving average on that.(diffferencing-step-4.png)

### Week 2: Deep Neural Netoworks for Time Series:
Like any ML model, we need to divide our data into features and labels.

__Features:__ Number of values in the series, with our label being the next value. We call the number of the values we have a features, as __WINDOW SIZE.__
For example, if we have a month of data to predict, we will have 30 days of window size, and we predict the 31st value.

Tensorflow Window:
```
Tensorflow:
    dataset = tf.data.Dateset.range(10)  //generates 10 values
    dataset = dataset.window(5, shift=1)  // creates windows of 5 as the windows lenght, and shifting 1 values at the time.
```
Here is the result of creating tf window:   (tf-window-generation.png)

We can add a parameter to the window method, to drop datasets with less than 5 values. like this:
```
dataset = dataset.window(5, shift=1, drop_remainder=True)
```
drop_remainder=True, drops the datasets with less than 5 values. (tf-window-generation-remainder-true.png)

Now we can have each of the windows of 5 values in a numpy list like this:
```
Tensorflow:
    dataset = tf.data.Dateset.range(10)
    dataset = dataset.window(5, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window:window.batch(5))
    for window in dataset:
        print(window.numpy())
```
[flat_map documentation here](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#flat_map).

All we have to do, is to call window.numpy() and each window will be in a numpy list ([0,1,2,3,4][1,2,3,4,5][2,3,4,5,6]), etc.

Now, we need to separate the lat value of each window as the label. we can use map and lambda to do it, like this:
```
Tensorflow:
    dataset = tf.data.Dateset.range(10)
    dataset = dataset.window(5, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window:window.batch(5))
    dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
    for x,y in dataset:
        print(x.numpy(), y.numpy())
```
It will be something like this: (tf-window-generation-split-label.png)

    - [:-1] means eveything but the last one
    - [-1:] means only the last one
    
We can also shuffle the values of each window saying:
```
dataset = dataset.shuffle(buffer_size=10) //we set shuffle size as 10 because we have only 10 values.
```
The last step, is to batch the windows two by two and save them in separate lists. (tf-window-generation-batched.png)

__IMPORTANT NOTE:__ 
__Sequence bias__  is when the order of things can impact the selection of things. For example, if I were to ask you your favorite TV show, and listed "Game of Thrones", "Killing Eve", "Travellers" and "Doctor Who" in that order, you're probably more likely to select 'Game of Thrones' as you are familiar with it, and it's the first thing you see. Even if it is equal to the other TV shows. So, when training data in a dataset, we don't want the sequence to impact the training in a similar way, so it's good to shuffle them up.

__Feeding windowed dataset into nerual network:__

Here is the first function:
```
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices(series)    
    dataset = dataset.window(window_size+1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size+1))
    dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1:]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset
```
 - Dataset.from_tensor_slices() constructs a dataset from one or more tf.Tensor objects.
 
__simple one layer neural network:__

We need to split data into train and validation datasets:
```
split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]
```
Now we can create the windowed dataset and feed it into a dense neural network which has an input shape equal to the window size.

```
window_size = 20
batch_size = 32
shuffle_buffer_size = 1000

dataset = windowed_dataset(series, window_size, batch_size, shuffle_buffer_size)
l0 = tf.keras.layers.Dense(1, input_shape=[window_size])
model = tf.keras.models.Sequential([10])
```
and here is how we fit the model:
```
model.compile(loss="mse", optimizer=tf.keras.optimimzers.SGD(lr=1e-6, momentum=0.9))
model.fit(dataset, epochs=100, verbose=0)
```
now we can see the weights like this:
```
print("Layer weights {}".format(l0.get_weights()))
```
having the weights, and the X values for the windows before the Y value, we want to predict. we can calculate the Y using the formula like this:
```
Y = wt0X0 + wt1X1 + Wt2X2 + ..... Wt19X19 + b 
```
b is the bias or slope. 

(here is the guide: feeding-LR-getting-weights.png)

Having the weights, we can predict the value for any point in the series, having the previous 20 values as the window.  for example:

```
print(series[1:21)
model.predict(series[1:21][np.newaxis])
```
np.newaxis reshapes the prediction to the shape which was used in the model in first place.

Here is the outcome: (LR-prediction.png) - the top 20 values are the x values of the series, and the value at the bottom is the prediction.

For plotting the predictions we can say:
```
forecast = []
for time in range(len(series) - window_size):
    forecast.append(model.predict(series[time:time + window_size][np.newaxis]))
forecast = forecast[split_time - window_size:]
results = np.array(forecast)[:,0,0]
```
in order to get the mean absolute error:
```
tf.keras.metrics.mean_absoulte_error(x_valid, results).numpy()
```