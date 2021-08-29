<center>
<h1>CS231n: Assignment Solutions</h1>

<b>Convolutional Neural Networks for Visual Recognition</b>

<i>Stanford - Spring 2021</i>
</center>

## About
---
### Overview
These are my solutions for the **CS231n** course assignemnts offered by Stanford University (Spring 2021). Inline questions are explained in detail, the code is brief and commented (see examples below). From what I investigated, these should be the shortest code solutions (excluding open-ended challenges). In assignment 2, _DenseNet_ is used in _PyTorch_ notebook and _ResNet_ in _TensorFlow_ notebook. 

> **!Note:** currently, only solutions for the first 2 assignemnts are done.

### Main sources (official)
* [**Course page**](http://cs231n.stanford.edu/index.html)
* [**Assignements**](http://cs231n.stanford.edu/assignments.html)
* [**Lecture notes**](https://cs231n.github.io/)
* [**Lecture videos** (2017)](https://www.youtube.com/playlist?list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk)

### Additional references (helper)
* Additional references are yet to be added...

## Solutions
---
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
* [Q5](assignment2/TensorFlow.ipynb) _option 1_: PyTorch on CIFAR-10. (_Done_)
* [Q5](assignment2/PyTorch.ipynb) _option 2_: TensorFlow on CIFAR-10. (_Done_)

### Assignment 3
* Solutions for assignemnt 3 are yet to be added...

## Examples
---
### Inline Questions

<div style="width:85%; margin:auto; padding: 1em 0; font-size:0.8em; text-align:justify;">

**Inline Question 1**

---
It is possible that once in a while a dimension in the gradcheck will not match exactly. What could such a discrepancy be caused by? Is it a reason for concern? What is a simple example in one dimension where a gradient check could fail? How would change the margin affect of the frequency of this happening? *Hint: the SVM loss function is not strictly speaking differentiable*

---
<br>
<br>

**Your Answer**

---
First, we need to make some assumptions. To compute our **SVM loss**, we use **Hinge loss** which takes the form $\max(0,-)$. For 1D case, we can define it as follows ($\hat y$ - score, $i$ - any class, $c$ - correct class, $\Delta$ - margin):

$$f(x)=\max(0, x),\ \text{where}\ x=\hat y_{i}-\hat y_c+\Delta$$

Let's now see how our $\max$ function fits the definition of computing the gradient. It is the formula we use for computing the gradient _numerically_ when, instead of implementing the limit approaching to $0$, we choose some arbitrary small $h$:

$$\frac{df(x)}{dx}=\lim_{h\to 0}\frac{\max(0, x+h)-\max(0,x)}{h}$$

Now we can talk about the possible mismatches between _numeric_ and _analytic_ gradient computation:
1. **Cause of mismatch**
     * _Relative error_ - the discrepancy is caused due to arbitrary choice of small values of $h$ because by definition it should approach `0`. _Analytic_ computation produces an exact result (as precise as computation precision allows) while _numeric_ solution only approximates the result.
     * _Kinks_ - $\max$ only has a subgradient because when both values in $\max$ are equal, its gradient is undefined, therefore, not smooth. Such parts, referred to as _kinks_, may cause _numeric_ gradient to produce different results from _analytic_ computation due to (again) arbitrary choice of $h$.
2. **Concerns**
     * When comparing _analytic_ and _numeric_ methods, _kinks_ are more dangerous than small inaccuracies where the gradient is smooth. Small derivative inaccuracies still change the weight by approximately the same amount but _kinks_ may cause unintentional updates as seen in an example below. If the unintentional values would have a noticable affect on parameter updates, it is a reason for concern.
3. **1D example of numeric gradient fail**
     * Assume $x=-10^{-9}$. Then the _analytic_ computation of the derivative of $\max(0, x)$ would yield $0$. However, if we choose our $h=10^{-8}$, then the _numeric_ computation would yield $0.9$.
4. **Relation between margin and mismatch**
     * Assuming all other parameters remain **unchanged**, increasing $\Delta$ will lower the frequency of _kinks_. This is because higher $\Delta$ will cause more $x$ to be positive, thus reducing the probability of kinks. In reality though, it would not have a big effect - if we increase the margin $\Delta$, the **SVM** will only learn to increase the (negative) gap between $\hat y_i - \hat y_c$ and $0$ (when $i\ne c$). But that still means, if we add $\Delta$, there is the same chance for $x$ to result on the edge.
---
</div>

### Python Code

<div style="width:85%; margin:auto; padding: 1em 0; font-size:0.8em; text-align:justify">

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

</div>
