# One exact density reference; scope of the claim

Let M=400 sites, z=4 neighbors, B=zM/2 undirected edges, and n_i in{0,1}. For the intended Hamiltonian E(n)=−ε sum_edges n_i n_j−μ sum_i n_i,

E(1−n)−E(n)=−εB−μM+(εz+2μ)sum_i n_i.

At μ=−εz/2 both the constant and the n-dependent term vanish. The finite periodic equilibrium law at anyT>0 is therefore invariant under n→1−n, and Eρ=1−Eρ=1/2. With the intended ε=4 this center isμ=−8, matching the paper's stated center. This standard symmetry argument is not a new theorem.

The numerical decoder is exactly the author's threshold recipe: n_i=1 when generated channel0≥0,0 otherwise;ρ=sum n_i/400. This decoder is a generation convention and does not itself identify a true grand-canonical law.

Native mean-control0 is centered between the normalized training labels−1/+1. Associating it with physicalμ=−8 is a conditional intended mapping, not a verified checkpoint-to-paper transform. Temperature5 is the literal GC_ising_code notebook input, not a reconstructedT7 normalization. The symmetry reference isT-independent, but that does not prove that the supplied network conditions have physical units.

For N=256 generated densities bounded in[0,1], idealized independent sampling conditional on the fitted network gives P(|rho_bar−E_model rho|>sqrt(log(40)/(2N)))≤0.05. The bound targets E_model rho, not E_physics rho; it says nothing about unknown training seeds, physical sampling histories, normalization or scientific protocol error. Descriptive SEM uses between-generated-configuration variability; sites are not separate replicates. No post-result Gaussian interval or new closeness threshold is added.

Compatibility with0.5 through the conservative bound does not establish practical equality. A model with incorrect spatial correlations or no useful condition dependence can still pass this one weak moment check. Neither a compatible nor incompatible mean alone establishes the success or failure of the published expTM method.
