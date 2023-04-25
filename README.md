<h1 align="center">CS231n: Assignment Solutions</h1>
<p align="center"><b>Convolutional Neural Networks for Visual Recognition</b></p>
<p align="center"><i>Stanford - Spring 2021-2023</i></p>

## About
### Overview
These are my solutions for the **CS231n** course assignments offered by Stanford University (Spring 2021). Solutions work for further years like 2022, 2023. Inline questions are explained in detail, the code is brief and commented (see examples below). From what I investigated, these should be the shortest code solutions (excluding open-ended challenges). In assignment 2, _DenseNet_ is used in _PyTorch_ notebook and _ResNet_ in _TensorFlow_ notebook. 

> Check out the solutions for **[CS224n](https://github.com/mantasu/cs224n)**. They contain more comprehensive explanations than others.

### Main sources (official)
* [**Course page**](http://cs231n.stanford.edu/index.html)
* [**Assignments**](http://cs231n.stanford.edu/assignments.html)
* [**Lecture notes**](https://cs231n.github.io/)
* [**Lecture videos** (2017)](https://www.youtube.com/playlist?list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk)

<br>

## Solutions
### Assignment 1
* [Q1](assignment1/knn.ipynb): k-Nearest Neighbor classifier. (_Done_)
* [Q2](assignment1/svm.ipynb): Training a Support Vector Machine. (_Done_)
* [Q3](assignment1/softmax.ipynb): Implement a Softmax classifier. (_Done_)
* [Q4](assignment1/two_layer_net.ipynb): Two-Layer Neural Network. (_Done_)
* [Q5](assignment1/features.ipynb): Higher Level Representations: Image Features. (_Done_)

### Assignment 2
* [Q1](assignment2/FullyConnectedNets.ipynb): Fully-connected Neural Network. (_Done_)
* [Q2](assignment2/BatchNormalization.ipynb): Batch Normalization. (_Done_)
* [Q3](assignment2/Dropout.ipynb): Dropout. (_Done_)
* [Q4](assignment2/ConvolutionalNetworks.ipynb): Convolutional Networks. (_Done_)
* [Q5](assignment2/PyTorch.ipynb) _option 1_: PyTorch on CIFAR-10. (_Done_)
* [Q5](assignment2/TensorFlow.ipynb) _option 2_: TensorFlow on CIFAR-10. (_Done_)

### Assignment 3
* [Q1](assignment3/RNN_Captioning.ipynb): Image Captioning with Vanilla RNNs (_Done_)
* [Q2](assignment3/Transformer_Captioning.ipynb): Image Captioning with Transformers (_Done_)
* [Q3](assignment3/Network_Visualization.ipynb): Network Visualization: Saliency Maps, Class Visualization, and Fooling Images (_Done_)
* [Q4](assignment3/Generative_Adversarial_Networks.ipynb): Generative Adversarial Networks (_Done_)
* [Q5](assignment3/Self_Supervised_Learning.ipynb): Self-Supervised Learning for Image Classification (_Done_)
* [Q6](assignment3/LSTM_Captioning.ipynb): Image Captioning with LSTMs (_Done_)

<br>

## Running Locally

It is advised to run in [Colab](https://colab.research.google.com/), however, you can also run locally. To do so, first, set up your environment - either through [conda](https://docs.conda.io/en/latest/) or [venv](https://docs.python.org/3/library/venv.html). It is advised to install [PyTorch](https://pytorch.org/get-started/locally/) in advance with GPU acceleration. Then, follow the steps:
1. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
2. Change every first code cell in `.ipynb` files to:
   ```bash
   %cd cs231n/datasets/
   !bash get_datasets.sh
   %cd ../../
   ```
3. Change the first code cell in section **Fast Layers** in [ConvolutionalNetworks.ipynb](assignment2/ConvolutionalNetworks.ipynb) to:
   ```bash
   %cd cs231n
   !python setup.py build_ext --inplace
   %cd ..
   ```

I've gathered all the requirements for all 3 assignments into one file [requirements.txt](requirements.txt) so there is no need to additionally install the requirements specified under each assignment folder. If you plan to complete [TensorFlow.ipynb](assignment2/TensorFlow.ipynb), then you also need to additionally install [Tensorflow](https://www.tensorflow.org/install).


> **Note**: to use MPS acceleration via Apple M1, see the comment in [#4](https://github.com/mantasu/cs231n/issues/4#issuecomment-1492202538).

## Examples

<details><summary><b>Inline question example</b></summary>
<br>
<b>Inline Question 1</b>

<hr>
<p align="justify"><sub>It is possible that once in a while a dimension in the gradcheck will not match exactly. What could such a discrepancy be caused by? Is it a reason for concern? What is a simple example in one dimension where a gradient check could fail? How would change the margin affect of the frequency of this happening? <i>Hint: the SVM loss function is not strictly speaking differentiable</i></sub></p>
<hr>

<br>

<b>Your Answer</b>

<hr>

<sub>
First, we need to make some assumptions. To compute our <b>SVM loss</b>, we use <b>Hinge loss</b> which takes the form $\max(0, -)$. For <code>1D</code> case, we can define it as follows ( $\hat{y}$ - score, $i$ - any class, $c$ - correct class, $\Delta$ - margin):
</sub>

<sub>
$$f(x)=\max(0, x),\ \text{ where } x=\hat{y}_i-\hat{y}_c+\Delta$$
</sub>

<sub>
Let's now see how our $\max$ function fits the definition of computing the gradient. It is the formula we use for computing the gradient <i>numerically</i> when, instead of implementing the limit approaching to $0$, we choose some arbitrary small $h$:
</sub>

<sub>
$$\frac{df(x)}{dx}=\lim_{h \to 0}\frac{\max(0,x+h)-\max(0,x)}{h}$$
</sub>

<sub>
Now we can talk about the possible mismatches between <i>numeric</i> and <i>analytic</i> gradient computation:
</sub>

1. <sub>**Cause of mismatch** </sub>
    * <sub> _Relative error_ - the discrepancy is caused due to arbitrary choice of small values of $h$ because by definition it should approach `0`. _Analytic_ computation produces an exact result (as precise as computation precision allows) while _numeric_ solution only approximates the result. </sub>
    * <sub> _Kinks_ - $\max$ only has a subgradient because when both values in $\max$ are equal, its gradient is undefined, therefore, not smooth. Such parts, referred to as _kinks_, may cause _numeric_ gradient to produce different results from _analytic_ computation due to (again) arbitrary choice of $h$. </sub>
2. <sub> **Concerns** </sub>
    * <sub> When comparing _analytic_ and _numeric_ methods, _kinks_ are more dangerous than small inaccuracies where the gradient is smooth. Small derivative inaccuracies still change the weight by approximately the same amount but _kinks_ may cause unintentional updates as seen in an example below. If the unintentional values would have a noticeable affect on parameter updates, it is a reason for concern. </sub>
3. <sub> **`1D` example of numeric gradient fail** </sub>
    * <sub> Assume $x=-10^{-9}$. Then the _analytic_ computation of the derivative of $\max(0, x)$ would yield `0`. However, if we choose our $h=10^{-8}$, then the _numeric_ computation would yield `0.9`. </sub>
4. <sub> **Relation between margin and mismatch** </sub>
    * <sub> Assuming all other parameters remain **unchanged**, increasing $\Delta$ will lower the frequency of _kinks_. This is because higher $\Delta$ will cause more $x$ to be positive, thus reducing the probability of kinks. In reality though, it would not have a big effect - if we increase the margin $\Delta$, the **SVM** will only learn to increase the (negative) gap between $\hat y_i - \hat y_c$ and `0` (when $i\ne c$). But that still means, if we add $\Delta$, there is the same chance for $x$ to result on the edge. </sub>

<hr>
</details>

<details><summary><b>Python code example</b></summary>
<sub>

```python
def conv_forward_naive(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    P1 = P2 = P3 = P4 = conv_param['pad'] # padding: up = right = down = left
    S1 = S2 = conv_param['stride']        # stride:  up = down
    N, C, HI, WI = x.shape                # input dims  
    F, _, HF, WF = w.shape                # filter dims
    HO = 1 + (HI + P1 + P3 - HF) // S1    # output height      
    WO = 1 + (WI + P2 + P4 - WF) // S2    # output width

    # Helper function (warning: numpy version 1.20 or above is required for usage)
    to_fields = lambda x: np.lib.stride_tricks.sliding_window_view(x, (WF,HF,C,N))

    w_row = w.reshape(F, -1)                                            # weights as rows
    x_pad = np.pad(x, ((0,0), (0,0), (P1, P3), (P2, P4)), 'constant')   # padded inputs
    x_col = to_fields(x_pad.T).T[...,::S1,::S2].reshape(N, C*HF*WF, -1) # inputs as cols

    out = (w_row @ x_col).reshape(N, F, HO, WO) + np.expand_dims(b, axis=(2,1))
    
    x = x_pad # we will use padded version as well during backpropagation

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache
```

</sub>
</details>
