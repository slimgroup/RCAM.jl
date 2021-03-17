[![DOI](https://zenodo.org/badge/346472245.svg)](https://zenodo.org/badge/latestdoi/346472245)

# RCAM.jl

Residual Constrained Alternating Minimization (RCAM) - A factorization-based alternating minimization scheme for large scale matrix completion in parallel computing architectures.

## Installation


### Using SLIM Registry (preferred method) ###

First switch to package manager prompt (using ']') and add SLIM registry:

```
	registry add https://github.com/slimgroup/SLIMregistryJL.git
```

Then still from package manager prompt add RCAM:

```
	add RCAM
```

Note! If the above gives you tree error, try first from terminal

```
	cd ~/.julia/dev
	git clone https://github.com/slimgroup/RCAM.jl.git RCAM
```

and then from Julia's package manager prompt:

```
	dev RCAM
```
	


### Adding without SLIM registry ###

After switching to package manager prompt (using ']') type:

```
	add https://github.com/slimgroup/RCAM.jl.git
```

## DCLR Framework
### Dedicated Communicators for L and R Factors

The DCLR framework facilitates performant and scalable implementations of RCAM.

In short, the framework attempts to minimize implicit blocking due to messaging by:
    
* Dedicating a worker to store each **L** and **R** factor, and handle all messaging associated with factor updates.
* Distributing the data, **b**, across the remaining workers, who will solve for factor updates using only the local portion of **b**.
* Asynchronously perform row updates on each factor.

## Example

The scripts for the following examples have been included in the package in the package directory. 
If you don't know the package directory, in the Julia REPL run

    pathof(RCAM)

All paths in the rest of this document are relative to the package directory.

### Low Rank Matrix Completion with Cholesky Factorization 
#### /examples/psimpletestChol.jl

DCLR requires at least 3 workers to itself, so we begin by starting Julia with 4 processes.

    julia -p 4

Load the data and define the example function
    
    julia> include("psimpletestChol.jl")

Run the example function and perform RCAM with DCLR

    L,R = pCholtest(X)

For the reconstructed data using the **L** and **R** factors

    X_rec = L*R'

Define an SNR function, and compute the SNR of the recovery

    julia> snr(raw,interp) = -20log10(norm(interp-raw)/norm(raw))
    snr (generic function with 1 method)

    julia> snr(X,X_rec)
