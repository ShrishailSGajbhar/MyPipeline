### Demo for my "Pandas In, Pandas Out" scheme based scikit-learn pipeline.

I love scikit-learn library due to the myriads of machine learning functionalities it offers as well as it's easy to follow documentation.

**sklearn.pipeline** module in scikit-learn library provides several classes and methods which provide a way to automate a machine learning workflow.
Although, pandas dataframe input is allowed in pipelines, the time the output is always a numpy array which may not be desired all the time.

Here, is my solution to this problem. In this repo, you will find two python scipts:
* utils.py (contains custom classes written for for the demonstration purpose)
* test_utils.py (python script which shows demonstration of my "Pandas In, Pandas Out" pipeline scheme.

This work would not have been possible without the great resources given below:
1. http://flennerhag.com/2017-01-08-Recursive-Override/
2. https://github.com/marrrcin/pandas-feature-union ([Article Link](https://zablo.net/blog/post/pandas-dataframe-in-scikit-learn-feature-union/index.html))

Other insightful articles on this topic:
* https://signal-to-noise.xyz/post/sklearn-pipeline/
* https://wkirgsn.github.io/2018/02/15/pandas-pipelines/
