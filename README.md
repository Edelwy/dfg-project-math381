# Determining Connectivity of Telecommunication Networks

We present our approach to modeling solutions for connecting nodes in a telecom network, ('telecom' meaning internet, cable, phone, etc.). Collaborating with **DFG Consulting**, (a Slovenian company), our team worked with a library of pictorially-represented networks to test whether each network had a connectivity solution. By utilizing two different standard mathematical programming solvers, we built a model that can conclusively determine a network's connectivity, and have proved its efficacy using sample networks provided by DFG.

Details about the model can be read in the paper. *Add the paper to the repo**
<br>This description is intended to clarify the use of these programs.

## Input Data

There are a few theoretical examples of the input data in the **"test-cases"** directory. The test cases are in the form of Microsoft Excel spreadsheets (`.xlsx` files). We are using a *pandoc* package to read the data, so you can use any of the [supported formats](https://pandas.pydata.org/docs/reference/api/pandas.read_excel.html).

There are 2 sheets in each test case file. One is for nodes and one is for links of the networks. The first column of the **"nodes"** sheet is `str_name` which is used for labeling the nodes. The second column `ref_name` is the type of the node. There are 3 supported types:
- type **PJ** is a manhole,
- type **TS** is a splitter,
- type **OS** is a house.

The **"links"** sheet has 4 columns. The first one `span_name` is just a unique label for the edge.	The second `to_str_name` is the end node and third `from_str_name` is the start node (this is the order and layout of the data provided by the company). The fourth column is not provided by the company but has to be read from the corresponding picture they provide and is called `weights` short for edge weights.


## Integer Programming 

This repository has a *Jupyter Notebook* and a regular *Python* version. For those not as savvy with programming it is better to take a look at the first one, as there are **detailed explanations** of each cell provided for the reader to better comprehend the code. We recommend following the instructions there and executing it as you go. There are a few hyperparameters you can tweak, so mind those if you want to test the model on your own data.

There are a few **packages** required in order to use this script:
- [gurobipy](https://pypi.org/project/gurobipy/) for implementing and evaluating the integer programming model,
- [networkx](https://networkx.org/) for graph manipulation,
- [pandas](https://pandas.pydata.org/docs/) for data parsing.
- [matplotlib](https://matplotlib.org/) for graph and solution visualization.

The executable has a few arguments and flags:
- `input_file` is first position argument and specifies the location and name from which input data is read, therefore it is mandatory,
- `pool_number` is an optional second argument specifying how many solutions are saved.
- `-v` or `--visualize` flag is used you want a picture representing the input and the solution as the program output. If you don't use this the *matplotlib* package is not necessary.
- `--allow_all_paths` flag removes a restriction on the paths we allow in the solution of our model. The default does not allow paths in the form of *house-node-house*.
- `-j` or `--junction_variables` flag is used if you also want the junction variables from the IP model in the report.

The output of the program is the report on the cable paths, performance and number of solutions. It provides more insight than just listing out a number of paths.
