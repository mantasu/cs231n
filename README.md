<h1 align="center">CS231n: Assignment Solutions</h1>
<p align="center"><b>Convolutional Neural Networks for Visual Recognition</b></p>
<p align="center"><i>Stanford - Spring 2021</i></p>

## About
### Overview
These are my solutions for the **CS231n** course assignments offered by Stanford University (Spring 2021). Inline questions are explained in detail, the code is brief and commented (see examples below). From what I investigated, these should be the shortest code solutions (excluding open-ended challenges). In assignment 2, _DenseNet_ is used in _PyTorch_ notebook and _ResNet_ in _TensorFlow_ notebook. 

> Check out the solutions for **[CS224n](https://github.com/mantasu/cs224n)**. From what I checked, they contain more comprehensive explanations than others.

### Main sources (official)
* [**Course page**](http://cs231n.stanford.edu/index.html)
* [**Assignments**](http://cs231n.stanford.edu/assignments.html)
* [**Lecture notes**](https://cs231n.github.io/)
* [**Lecture videos** (2017)](https://www.youtube.com/playlist?list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk)

### Additional references (helper)
* Additional references are yet to be added...

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

> I will upload assignment 3 solutions soon!

<br>

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
<p align="justify"><sub>First, we need to make some assumptions. To compute our <b>SVM loss</b>, we use <b>Hinge loss</b> which takes the form <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;\bg_black&space;\tiny&space;\max(0,-)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\bg_black&space;\tiny&space;\max(0,-)" title="\tiny \max(0,-)" /></a>. For <code>1D</code> case, we can define it as follows (<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;\bg_black&space;\tiny&space;\hat&space;y" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\bg_black&space;\tiny&space;\hat&space;y" title="\tiny \hat y" /></a> - score, <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;\bg_black&space;\tiny&space;i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\bg_black&space;\tiny&space;i" title="\tiny i" /></a> - any class, <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;\bg_black&space;\tiny&space;c" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\bg_black&space;\tiny&space;c" title="\tiny c" /></a> - correct class, <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;\bg_black&space;\tiny&space;\Delta" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\bg_black&space;\tiny&space;\Delta" title="\tiny \Delta" /></a> - margin):</sub></p>
    
<p align="center"><a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{120}&space;\bg_black&space;\small&space;f(x)=\max(0,&space;x),\&space;\text{where}\&space;x=\hat&space;y_{i}-\hat&space;y_c&plus;\Delta" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;\bg_black&space;\small&space;f(x)=\max(0,&space;x),\&space;\text{where}\&space;x=\hat&space;y_{i}-\hat&space;y_c&plus;\Delta" title="\small f(x)=\max(0, x),\ \text{where}\ x=\hat y_{i}-\hat y_c+\Delta" /></a></p>

<p align="justify"><sub>Let's now see how our <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;\bg_black&space;\tiny&space;\max" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\bg_black&space;\tiny&space;\max" title="\tiny \max" /></a> function fits the definition of computing the gradient. It is the formula we use for computing the gradient <i>numerically</i> when, instead of implementing the limit approaching to $0$, we choose some arbitrary small <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;\bg_black&space;\tiny&space;h" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\bg_black&space;\tiny&space;h" title="\tiny h" /></a>:</sub></p>

<p align="center"><a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{120}&space;\bg_black&space;\small&space;\frac{df(x)}{dx}=\lim_{h\to&space;0}\frac{\max(0,&space;x&plus;h)-\max(0,x)}{h}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;\bg_black&space;\small&space;\frac{df(x)}{dx}=\lim_{h\to&space;0}\frac{\max(0,&space;x&plus;h)-\max(0,x)}{h}" title="\small \frac{df(x)}{dx}=\lim_{h\to 0}\frac{\max(0, x+h)-\max(0,x)}{h}" /></a></p>

<p align="justify"><sub>Now we can talk about the possible mismatches between <i>numeric</i> and <i>analytic</i> gradient computation:</sub></p>
<ol>
    <sub><li><b>Cause of mismatch</b></li></sub>
    <ul>
        <li><p align="justify"><sub><i>Relative error</i> - the discrepancy is caused due to arbitrary choice of small values of <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;\bg_black&space;\tiny&space;h" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\bg_black&space;\tiny&space;h" title="\tiny h" /></a> because by definition it should approach <code>0</code>.<i>Analytic</i> computation produces an exact result (as precise as computation precision allows) while <i>numeric</i> solution only approximates the result.</sub></p></li>
        <li><p align="justify"><sub><i>Kinks</i> - <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;\bg_black&space;\tiny&space;\max" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\bg_black&space;\tiny&space;\max" title="\tiny \max" /></a> only has a subgradient because when both values in $\max$ are equal, its gradient is undefined, therefore, not smooth. Such parts, referred to as <i>kinks</i>, may cause <i>numeric</i> gradient to produce different results from <i>analytic</i> computation due to (again) arbitrary choice of <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;\bg_black&space;\tiny&space;h" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\bg_black&space;\tiny&space;h" title="\tiny h" /></a>.</sub></p></li>
    </ul>
    <sub><li><b>Concerns</b></li></sub>
    <ul>
        <li><p align="justify"><sub>When comparing <i>analytic</i> and <i>numeric</i> methods, <i>kinks</i> are more dangerous than small inaccuracies where the gradient is smooth. Small derivative inaccuracies still change the weight by approximately the same amount but <i>kinks</i> may cause unintentional updates as seen in an example below. If the unintentional values would have a noticable affect on parameter updates, it is a reason for concern.</sub></p></li>
    </ul>
    <sub><li><b><code>1D</code> example of numeric gradient fail</b></li></sub>
    <ul>
        <li><p align="justify"><sub>Assume <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;\bg_black&space;\tiny&space;x=-10^{-9}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\bg_black&space;\tiny&space;x=-10^{-9}" title="\tiny x=-10^{-9}" /></a>. Then the <i>analytic</i> computation of the derivative of <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;\bg_black&space;\tiny&space;\max(0,&space;x)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\bg_black&space;\tiny&space;\max(0,&space;x)" title="\tiny \max(0, x)" /></a> would yield <code>0</code>. However, if we choose our <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;\bg_black&space;\tiny&space;h=10^{-8}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\bg_black&space;\tiny&space;h=10^{-8}" title="\tiny h=10^{-8}" /></a>, then the <i>numeric</i> computation would yield <code>0.9</code>.</sub></p></li>
    </ul>
    <sub><li><b>Relation between margin and mismatch</b></li></sub>
    <ul>
        <li><p align="justify"><sub>Assuming all other parameters remain <b>unchanged</b>, increasing <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;\bg_black&space;\tiny&space;\Delta" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\bg_black&space;\tiny&space;\Delta" title="\tiny \Delta" /></a> will lower the frequency of <i>kinks</i>. This is because higher <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;\bg_black&space;\tiny&space;\Delta" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\bg_black&space;\tiny&space;\Delta" title="\tiny \Delta" /></a> will cause more <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;\bg_black&space;\tiny&space;x" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\bg_black&space;\tiny&space;x" title="\tiny x" /></a> to be positive, thus reducing the probability of kinks. In reality though, it would not have a big effect - if we increase the margin <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;\bg_black&space;\tiny&space;\Delta" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\bg_black&space;\tiny&space;\Delta" title="\tiny \Delta" /></a>, the <b>SVM</b> will only learn to increase the (negative) gap between <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;\bg_black&space;\tiny&space;\hat&space;y_i&space;-&space;\hat&space;y_c" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\bg_black&space;\tiny&space;\hat&space;y_i&space;-&space;\hat&space;y_c" title="\tiny \hat y_i - \hat y_c" /></a> and <code>0</code> (when <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;\bg_black&space;\tiny&space;i\ne&space;c" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\bg_black&space;\tiny&space;i\ne&space;c" title="\tiny i\ne c" /></a>). But that still means, if we add <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;\bg_black&space;\tiny&space;\Delta" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\bg_black&space;\tiny&space;\Delta" title="\tiny \Delta" /></a>, there is the same chance for <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;\bg_black&space;\tiny&space;x" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\bg_black&space;\tiny&space;x" title="\tiny x" /></a> to result on the edge.</sub></p></li>
    </ul>
</ol>
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
