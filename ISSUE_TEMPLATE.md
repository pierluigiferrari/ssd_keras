### If you open a GitHub issue, here is the policy:

Your issue must be about one of the following:

1. a bug,
2. a feature request,
3. a documentation issue, or
4. a question that is **specific to this SSD implementation**.

You will only get help if you adhere to the following guidelines:

* Before you open an issue, search the open **and closed** issues first. Your problem/question might already have been solved/answered before.
* If you're getting unexpected behavior from code I wrote, open an issue and I'll try to help. If you're getting unexpected behavior from code **you** wrote, you'll have to fix it yourself. E.g. if you made a ton of changes to the code or the tutorials and now it doesn't work anymore, that's your own problem. I don't want to spend my time debugging your code.
* Make sure you're using the latest master. If you're 30 commits behind and have a problem, the only answer you'll likely get is to pull the latest master and try again.
* Read the documentation. All of it. If the answer to your problem/question can be found in the documentation, you might not get an answer, because, seriously, you could really have figured this out yourself.
* If you're asking a question, it must be specific to this SSD implementation. General deep learning or object detection questions will likely get closed without an answer. E.g. a question like "How do I get the mAP of an SSD for my own dataset?" has nothing to do with this particular SSD implementation, because computing the mAP works the same way for any object detection model. You should ask such a question in an appropriate forum or on the [Data Science section of StackOverflow](https://datascience.stackexchange.com/) instead.
* If you get an error:
    * Provide the full stack trace of the error you're getting, not just the error message itself.
    * Make sure any code you post is properly formatted as such.
    * Provide any useful information about your environment, e.g.:
        * Operating System
        * Which commit of this repository you're on
        * Keras version
        * TensorFlow version
    * Provide a minimal reproducible example, i.e. post code and explain clearly how you ended up with this error.
    * Provide any useful information about your specific use case and parameters:
        * What model are you trying to use/train?
        * Describe the dataset you're using.
        * List the values of any parameters you changed that might be relevant.
