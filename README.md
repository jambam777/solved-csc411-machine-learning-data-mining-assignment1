Download Link: https://assignmentchef.com/product/solved-csc411-machine-learning-data-mining-assignment1
<br>
In this assignment we will be working with the <a href="http://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html">Boston Houses dataset</a><a href="http://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html">.</a> This dataset contains 506 entries. Each entry consists of a house price and 13 features for houses within the Boston area. We suggest working in python and using the <a href="http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html">scikit-learn</a> package to load the data.

Starter code written in python is provided for each question.

<h1>1          Learning basics of regression in Python (3%)</h1>

This question will take you step-by-step through performing basic linear regression on the Boston Houses dataset. You need to submit modular code for this question. Your code needs to be functional once downloaded. Non-functional code will result in losing all marks for this question. If your code is non-modular or otherwise difficult to understand, you risk losing a significant portion of the marks, even if the code is functional.

<strong>Environment setup: </strong>For this question you are strongly encouraged to use the following python packages:

<ul>

 <li>sklearn</li>

 <li>matplotlib</li>

 <li>numpy</li>

</ul>

It is strongly recommended that you download and install Anaconda 3.4 to manage the above installations. This is a Data Science package that can be downloaded from <a href="https://www.anaconda.com/download/">https://www.anaconda. </a><a href="https://www.anaconda.com/download/">com/download/</a><a href="https://www.anaconda.com/download/">.</a>

You will submit a complete regression analysis for the Boston Housing data. To do that, here are the necessary steps:

<ul>

 <li>Load the Boston housing data from the sklearn datasets module</li>

 <li>Describe and summarize the data in terms of number of data points, dimensions, target, etc</li>

 <li>Visualization: present a single grid containing plots for each feature against the target. Choose the appropriate axis for dependent vs. independent variables. <strong>Hint: </strong><em>use pyplot.tight layout function to make your grid readable</em></li>

 <li>Divide your data into training and test sets, where the training set consists of 80% of the data points (chosen at random). <strong>Hint: </strong><em>You may find numpy.random.choice useful</em></li>

 <li>Write code to perform linear regression to predict the targets using the training data. Remember to add a bias term to your model.</li>

 <li>Tabulate each feature along with its associated weight and present them in a table. Explain what the sign of the weight means in the third column (’INDUS’) of this table. Does the sign match what you expected? Why?</li>

 <li>Test the fitted model on your test set and calculate the Mean Square Error of the result.</li>

 <li>Suggest and calculate two more error measurement metrics; justify your choice.</li>

 <li>Feature Selection: Based on your results, what are the most significant features that best predict the price? Justify your answer.</li>

</ul>

<h1>2          Locally reweighted regression (6%)</h1>

<ol>

 <li>Given {(<strong>x</strong><sup>(1)</sup><em>,y</em><sup>(1)</sup>)<em>,..,</em>(<strong>x</strong><sup>(<em>N</em>)</sup><em>,y</em><sup>(<em>N</em>)</sup>)} and positive weights <em>a</em><sup>(1)</sup><em>,…,a</em><sup>(<em>N</em>) </sup>show that the solution to the <em>weighted </em>least square problem</li>

</ol>

<strong>w</strong><sup>∗ </sup>= argmin                                            (1)

is given by the formula

<strong>w</strong><sup>∗ </sup>= <strong>X</strong><em><sup>T</sup></em><strong>AX</strong><strong>Ay                                                                    </strong>(2)

where <strong>X </strong>is the design matrix (defined in class) and <strong>A </strong>is a diagonal matrix where <strong>A</strong><em><sub>ii </sub></em>= <em>a</em><sup>(<em>i</em>)</sup>

<ol start="2">

 <li>Locally reweighted least squares combines ideas from k-NN and linear regression. For each new test example <strong>x </strong>we compute distance-based weights for each training example <em>a</em><sup>(<em>i</em>) </sup>=</li>

</ol>

, computes <strong>w</strong><sup>∗ </sup>= argmin  and predicts

<em>y</em>ˆ = <strong>x</strong><em><sup>T</sup></em><strong>w</strong><sup>∗</sup>. Complete the implementation of locally reweighted least squares by providing the missing parts for q2.py.

Important things to notice while implementing: First, do not invert any matrix, use a linear solver (numpy.linalg.solve is one example). Second, notice that  but if we use <em>B </em>= max<em><sub>j </sub>A<sub>j </sub></em>it is much more numerically stable as  overflows/underflows easily. <em>This is handled automatically in the scipy package with the </em><a href="https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.misc.logsumexp.html"><em>scipy.misc.logsumexp</em></a><em> function.</em>

<ol start="3">

 <li>Use k-fold cross-validation to compute the average loss for different values of <em>τ </em>in the range [10,1000] when performing regression on the Boston Houses dataset. Plot these loss values for each choice of <em>τ</em>.</li>

 <li>How does this algorithm behave when <em>τ </em>→ ∞? When <em>τ </em>→ 0?</li>

</ol>

<h1>3          Mini-batch SGD Gradient Estimator (6%)</h1>

Consider a dataset D of size <em>n </em>consisting of (<strong>x</strong><em>,y</em>) pairs. Consider also a model M with parameters <em>θ </em>to be optimized with respect to a loss function.

We will aim to optimize <em>L </em>using mini-batches drawn randomly from D of size <em>m</em>. The indices of these points are contained in the set I = {<em>i</em><sub>1</sub><em>,…,i<sub>m</sub></em>}, where each index is distinct and drawn uniformly without replacement from {1<em>,…,n</em>}. We define the loss function for a single mini-batch as,

(3)

<ol>

 <li>Given a set {<em>a</em><sub>1</sub><em>,…,a<sub>n</sub></em>} and random mini-batches I of size <em>m</em>, show that</li>

 <li>Show that EI [∇<em>L</em><sub>I</sub>(<strong>x</strong><em>,y,θ</em>)] = ∇<em>L</em>(<strong>x</strong><em>,y,θ</em>)</li>

 <li>Write, in a sentence, the importance of this result.</li>

 <li>(a) Write down the gradient, ∇<em>L </em>above, for a linear regression model with cost function <em>`</em>(<strong>x</strong><em>,y,θ</em>) = (<em>y </em>− <em>w<sup>T</sup></em><strong>x</strong>)<sup>2</sup>.</li>

</ol>

(b) Write code to compute this gradient.

<ol start="5">

 <li>Using your code from the previous section, for <em>m </em>= 50 and <em>K </em>= 500 compute</li>

</ol>

, where I<em><sub>k </sub></em>is the mini-batch sampled for the <em>k</em>th time.

Randomly initialize the weight parameters for your model from a N(0<em>,I</em>) distribution. Compare the value you have computed to the true gradient, ∇<em>L</em>, using both the squared distance metric and cosine similarity. Which is a more meaningful measure in this case and why?

<strong>a </strong>· <strong>b</strong>

[Note: Cosine similarity between two vectors <strong>a </strong>and <strong>b </strong>is given by cos(<em>θ</em>) =        .]

||<strong>a</strong>||<sub>2</sub>||<strong>b</strong>||<sub>2</sub>

<ol start="6">

 <li>For a single parameter, <em>w<sub>j</sub></em>, compare the sample variance, <em>σ</em>˜<em><sub>j</sub></em>, of the mini-batch gradient estimate for values of <em>m </em>in the range [0,400] (using <em>K </em>= 500 again). Plot log ˜<em>σ<sub>j </sub></em>against log<em>m</em>.</li>

</ol>