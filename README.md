# Robustly-Complete-Stochatic-Abstraction-Demo-
Robustly Complete Abstractions for Verification and Control Synthesis of Stochastic Systems

This demo code is to generate robustly complete abstractions for Verification and Control Synthesis of Stochastic Systems. The concept of 'robust completeness' was first introduced in the paper 
[1] 'Robustly Complete Finite-State Abstractions for Verification of Stochastic Systems', FORMATS 2022. 

Intuitively, the generated abstraction is sound enough to preserve the satisfaction of the probabilistic linear-temporal logic specifications, but not unnecessarily large subjected to the prescribed precision. This construction will be embedded into the formal method tool 'RObustly Complete control Synthesis (ROCS)' for formal verification and synthesis. 
https://git.uwaterloo.ca/hybrid-systems-lab/rocs/-/blob/master/README.md

The construction of stochastic abstraction is temporarily based on a uniform discretization of the state space, and will be compatible with any type of probabilistic omega-regular specifications regardless of the efficiency. The current version only provides Interval Markov Chain (IMC) abstractions for the verification problems.



Required package: multipledispatch

The 'Library' module contains basic interval objects, inclusion functions, and a class to generate markov set chains (MSC) for coarse state-space discretization. The interval analysis part is a simple version of 'interval.h' and 'interval_vector.h' in https://git.uwaterloo.ca/hybrid-systems-lab/rocs/-/tree/master/src. Two examples are provided in 'Example - Library.py' and 'Example - MSC.py'. 

The 'IMCclass' module is for the main purpose. The main procedures are straightforward:
    • Set up the working space
    • Set up the precision (in the sense that is given in Section 4 of [1])
    • Set up dynamics
    • Create IMC abstractions
    • Save data
An example is provided in 'Example - IMC - 1.py' to illustrate. 
