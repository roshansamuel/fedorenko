# Fedorenko - Repository of Multigrid Solvers

A collection of multigrid solvers in both cell-centered and vertex-centered configurations.
There are 1D, 2D and 3D versions for both cases, all packed into a single repository to ease the development of multigrid solvers for those who wish to write one.
All the solvers are written in Python.
They are derived from the repositories ``MG-Lite``, which demonstrates a 1D multigrid solver with GUI, and ``Orion``, a finite-difference fluid flow solver written in Python.

The cell-centered versions have the prefix "cc", and the vertex-centered versions have the prefix "vc".
The suffix indicates the dimensionality of the solver.
All of them solve the Poisson equation on a domain with Dirichlet boundary conditions.
They print the convergence of the residual with V-cycles, and plot it at the end of the run.

The earliest version of multigrid algorithm is attributed to a paper by R. P. Fedorenko.
Hence the name of this repository.

Please make sure that the following Python modules are installed before running any of the codes.

* ``numpy`` - All array manipulations are performed using NumPy
* ``datetime`` - The execution time is measured using this module
* ``matplotlib`` - Results are plotted using the ``matplotlib`` library

## License

``Fedorenko`` is an open-source package made available under the New BSD License.

## References

Below is a list of useful articles that explains the multigrid-method used here.

### Articles on multi-grid methods

1. http://math.mit.edu/classes/18.086/2006/am63.pdf
2. http://www.mgnet.org/mgnet-tuts.html
3. R. P. Fedorenko, “The speed of convergence of one iterative process”, Zh. Vychisl. Mat. Mat. Fiz., 4:3 (1964), 559–564; U.S.S.R. Comput. Math. Math. Phys., 4:3 (1964), 227–235
