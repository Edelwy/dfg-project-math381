# Bash Scripts
There are 2 bash scripts one for the `greedy.py` and one for `ip.py`. Examples of use:
- Command `test_greedy.sh 1 4 5 true` will execute test cases $1$, $4$, and $5$ of the Greedy Algorithm with the random heuristic.
- Command `test_greedy.sh 1 4 5` will execute test cases $1$, $4$, and $5$ of the Greedy Algorithm with the minimum heuristic.
- Command `test_ip.sh 1 4 5` will execute test cases $1$, $4$, and $5$ of the Integer Programming with 100 solutions saved.

Feel free to tweak this if testing the algorithms on your data!

# Test Cases
1. The theoretical test case **from the paper**.
2. A **straight line** with no houses.
3. A simple example of a **4-cycle**.
4. A simple example of a **3-cycle**.
5. An example of a **"star"** with just lines coming from one manhole.
6. A case when we have **no junction variables**, only paths with one edge.
7. A more complex case with no cycles.
8. A more complex case with no cycles.
9. A case where the first solution is not without **no-manhole paths**.
10. A very **computationally hard** case if we want all solutions.
11. Another cycle with a path going out.
12. An example with 2 connected components.
13. An example with **no** solutions where every path has a manhole.