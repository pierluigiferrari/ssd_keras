# Contributing Guidelines
---

Contributions to this repository are welcome, but before you create a pull request, consider the following guidelines:

1. The To-do list in the README of this repository defines the main topics for which contributions are welcome. If you want to contribute, ideally contribute to one of the topics listed there.
2. If you'd like to contribute features that are not mentioned on the to-do list in the README, make sure to explain why your proposed change adds value, i.e. what relevant use case it solves. The benefit of any new feature will be compared against the cost of maintaining it and your contribution will be accepter or rejected based on this trade-off.
3. One pull request should be about one specific feature or improvement, i.e. it should not contain multiple unrelated changes. If you want to contribute multiple features and/or improvements, create a separate pull request for every individual feature or improvement.
3. When you create a pull request, make sure to explain properly
    * why your propsed change adds value, i.e. what problem or use case it solves,
    * all the API changes it will introduce, if any,
    * all behavioral changes in any existing parts of the project it will introduce, if any.
4. This should go without saying, but you are responsible for updating any parts of the code or the tutorial notebooks that are affected by your introduced changes.
5. Any submitted code must conform to the coding standards and style of this repository. There is no formal guide for coding standards and style, but here are a few things to note:
    * Any new modules, classes or functions must provide proper docstrings unless they are trivial. These docstrings must have sections for Arguments, Returns, and Raises (if applicable). For every argument of a function, the docstring must explain precisely what the argument does, what data type it expects, whether or not it is optional, and any requirements for the range of values it expects. The same goes for the returns. Use existing docstrings as templates.
    * Naming:
        * `ClassNames` consist of capitalized words without underscores.
        * `module_names.py` consist of lower case words connected with underscores.
        * `function_names` consist of lower case words connected with underscores.
        * `variable_names` consist of lower case words connected with underscores.
    * All module, class, function, and variable names must be descriptive in order to meet the goal that all code should always be as self-explanatory as possible. A longer and descriptive name is always preferable over a shorter and non-descriptive name. Abbreviations are generally to be avoided unless the full words would really make the name too long.
    * More in-line comments are better than fewer in-line comments and all comments should be precise and succinct.
