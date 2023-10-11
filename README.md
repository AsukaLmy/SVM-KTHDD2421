# SVM-KTHDD2421
lab2 SVM of ML DD2421

# question 1
Move the clusters around and change their sizes to make it easier or harder for the classifier to find a decent boundary. Pay attention to when the optimizer (minimize function) is not able to find a solution at all.

A: see image start with '1.'


# question 2
Implement the two non-linear kernels. You should be able to classify very hard data sets with these.

A: image '2.#####'


# question 3
The non-linear kernels have parameters; explore how they influence the decision boundary. Reason about this in terms of the biasvariance trade-off.

A: see image '3.######'

For RBF kernel: 

large σ: the decision boundary becomes smoother, which may underfitting the data.(higher bias and lower variance)

small σ: the model becomes sensitve to the data, fitting close to it, which may introduce overfitting.(lower bias and higher variance)


For Polynomial kernel:

large p: As the degree of the polynomial (p) increases, the model can capture more complex relationships in the data. This can lead to a more intricate decision boundary. While this might reduce bias as the model becomes more flexible, it can increase variance because the model might start fitting to the noise in the data (overfitting).

small p: A simpler polynomial (e.g. linear) will generate a simpler decision boundary, which might not capture all the intricacies of the data distribution. This can increase bias but will generally reduce variance (underfitting).



# question 4
Explore the role of the slack parameter C. What happens for very large/small values

A: also see image '4.######'

Generally speaking, slack parameter balances a trade-off between the model complexity and the training error rate. 

Higher C usually means less classification errors (sometimes 0), complex models (may introduce overfitting). Lower C usually means more tolerance of training errors and simpler models (may cause underfitting)


# question 5 
Imagine that you are given data that is not easily separable. When should you opt for more slack rather than going for a more complex model (kernel) and vice versa?

A: by experimentation and validation @-@? maybe. I would try different method and see which one is better and it depends.

In different application different choice shell be made.

Opt for more slack when the noisy data should be prevented or you need easy ways to explain models, to avoid overfitting, efficient computation(s.t simpler model which is easy to explain)

Opt for more complex models when high accuracy is imperative, non-linear boundaries are clear, data is sufficient (sothat you can train without overfitting), you already know the model would be complex.





