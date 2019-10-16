---
layout: post
use_math: true
title:  "Learning to learn without forgetting: a summary"
tags: ["paper summary", "data science"]
date:   2019-08-28

---

<p>Matthew Riemer et. al. <a href="https://arxiv.org/abs/1810.11910">Learning to Learn Without Forgetting by Maximizing Transfer and Minimizing Interference.</a> <em>ICLR</em>, 2019.</p>
<ul>
<li>Recall <a href="https://en.wikipedia.org/wiki/Catastrophic_interference">catastrophic forgetting</a>, a neural network sequentially trained on multiple tasks forgets earlier tasks with each new task, apparently not a problem in Bayesian networks</li>
<li>Why? overwriting weights with updates, …</li>
<li>How to avoid? limit weight sharing, balance network stability vs plasticity ("recall of old tasks" versus "rapid learning of new ones"), …</li>
<li>The loss function: $$\sum_{i, j} L(x_i , y_i ) + L(x_j , y_j ) − \alpha {\partial L(x_i , y_i ) \over \partial \theta} \cdot {\partial L(x_j , y_j ) \over \partial \theta}$$</li>
<li>The regularizing term is a measure of <em>transfer</em> or <em>interference</em> between updates. The gradient wrt to learning parameters guides the backprop update to those parameters: alignment of gradients means the updates agree and will guide learning for both examples; anti-alignment means updates cancel and neither example will learn; any intervening overlap is deemed transfer (interference) for positive (negative) values.</li>
<li>Maximizing weight sharing maximizes transfer; minimizing weight sharing minimizes the change for interference.</li>
<li>Work leading up to this paper, both offline algorithms over dataset D:
<ul>
<li>MAML -&gt; FOMAML (Finn &amp; Levine, 2017)</li>
<li>Reptile (Nichol &amp; Schulman, 2018)</li>
</ul>
</li>
<li>Contributions: new algorithm MER, meta experience replay, an online algorithm (algorithms 1 with variants 6 &amp; 7):
<ul>
<li>added an inner loop within Reptile batches for an inner meta-learning update</li>
<li>keeps a memory/reservoir of examples M to approximate the full dataset D with new examples added probabilistically to replace old ones (see algorithm 3 in the paper)</li>
<li>prioritizes learning of the current examples, esp. because it may not be saved</li>
</ul>
</li>
<li>First, the reptile algorithm:
<ul>
<li>for each epoch of training, \(t\), record the current params, \(\theta^A_0 = \theta_{t-1}\) and sample \(s\) batches of size \(k\)</li>
<li>perform a normal epoch of training over the \(s\) batches with learning rate \(\alpha\) toward final params \(\theta^A_s\)</li>
<li>update the network weights for this epoch only a fraction of the learned param changes: 
    $$\theta_t = \theta^A_0 + \gamma (\theta^A_s - \theta^A_0)$$</li>
<li>this meta-learning update enacts the effective loss 
    $$2\sum_{i=1}^s L(B_i) - \sum_{j=1}^{i-1} {\partial L(B_i) \over \partial \theta} \cdot {\partial L(B_j) \over \partial \theta}$$
</li>
</ul>
</li>
<li>MER adds a second meta-learning update within each of the \(s\) batches, now sampled from reservoir M, each of which will have the current example in it; finally, the reservoir is updated (maybe)
<ul>
<li>for each epoch of training, \(t\), record the current params, \(\theta^A_0 = \theta_{t-1}\) and sample \(s\) batches of size \(k\), include example \(x_t, y_t\) in each</li>
<li>for each batch \(i\), record the current params, \(\theta^A_{i, 0} = \theta^A_{i-1} \)</li>
<li>for each example \(j\) in the batch, perform a backprop update with learning rate 
    \(\alpha\) to params \(\theta^A_{i, j}\)</li>
<li>after the entire batch has been singly learned, meta-learn the parameter update 
    $$\theta^A_i = \theta^A_{i, 0} + \beta (\theta^A_{i, k} - \theta^A_{i, 0})$$</li>
<li>the effective loss is 
    $$2\sum_{i=1}^s \sum_{j=1}^k L(x_{ij}, y_{ij}) - \sum_{q=1}^{i-1}\sum_{r=1}^{j-1} {\partial L(x_{ij}, y_{ij}) \over \partial \theta} \cdot {\partial L(x_{qr}, y_{qr}) \over \partial \theta}$$</li>
<li>note that they update the batch examples singly to maximize the regularizing effect</li>
<li>algorithms 6 &amp; 7 are alternate ways of prioritizing the current example</li>
</ul>
</li>
<li>Evaluation metrics:
<ul>
<li>learning accuracy (LA): average accuracy for each task immediately after it has been learned</li>
<li>retained accuracy (RA): final retained accuracy across all tasks learned sequentially</li>
<li>backward transfer and interference (BTI): the average change in accuracy from when a task is learned to the end of training (positive good; large and negative is catastrophic forgetting)</li>
</ul>
</li>
<li>Problems:
<ul>
<li>in supervised learning: MNIST permutations, each task is transformed by a fixed permutation of the MNIST pixels; MNIST rotations, each task contains digits rotated by a fixed angle between 0 and 180 degrees; Omniglot, each task is one of 50 alphabets with overall 1623 characters</li>
<li>in reinforcement learning: Catcher, a board moved left/right to catch a more and more rapidly falling object; Flappy Bird must fly between ever tightening pipes</li>
</ul>
</li>
<li>Compared against:
<ul>
<li>online, same network trained straightforwardly one example at a time on the incoming non-stationary training data by simply applying SGD</li>
<li>independent, one model per task with size of network reduced proportionally to keep total number of parameters fixed</li>
<li>task input, trained as in online with a dedicated input layer per task</li>
<li>EWC, Elastic Weight Consolidation (EWC) (Kirkpatrick et al., 2017), ~online regularized to avoid catastrophic forgetting</li>
<li>GEM: Gradient Episodic Memory (GEM) (Lopez-Paz &amp; Ranzato, 2017) uses episodic storage to modify gradients of latest example to not interfere with past ones; stored examples are not used in ongoing training
Findings:</li>
<li>MER seems to do learn and retain the most over all tasks, faster, and with less memory</li>
<li>my reservations:
<ul>
  <li>mnist again?</li>
  <li>omniglot is not usually studied with any of the algorithms compared against: in Lake (2015) they achieve &lt;5% error rate, still &lt;15% in a stripped down version of their model and 2 out of 3 of their baselines</li>
  <li>how much slower will the training be with single example batches and two meta-learning updates?</li>
</ul>
</li>
</ul>
</li>
</ul>