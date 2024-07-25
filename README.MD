# *geppyNN:* A Python framework for Gene Expression Programming for Neural Networks (GEPNN) built on top of *geppy*

*geppy* is a computational framework dedicated to [Gene Expression Programming](https://en.wikipedia.org/wiki/Gene_expression_programming) (GEP),  which is proposed by C. Ferreira  in 2001 [1].  *geppy* is developed in Python 3.

*geppyNN* extends the capabilities of geppy by allowing users to interpret the built expression trees as neural network architectures, with typical neuron operation including varying activation functions and aggregation methods. Our implementation follows the prescriptions of Candida Ferreira in her book [Designing Neural Networks Using Gene Expression Programming](https://link.springer.com/chapter/10.1007/3-540-31662-0_40). 

While this project is still a work in progress, and we are seeking to imake improvements and extensions beyond the initial prescriptions of Ferreira, this work also supported our research published in SSCI 2023: [Connectivity Schemas in NeuroEvolution: What Neural Architectures does GEPNN evolve?](https://www.researchgate.net/publication/377072667_Connectivity_Schemas_in_NeuroEvolution_What_Neural_Architectures_does_GEPNN_evolve). In this work, we explore the neural architectures that are possible with GEPNN, based on a combination of experimental obserations and theoretical analysis of the GEPNN genotype. 

## How to cite *geppyNN*
If you find *geppyNN* useful in your projects, please cite it such that more researchers/engineers will know it. A BibTeX entry for *geppy* is given below.
```
@misc{geppyNN_2023,
    author       = {Ryan Heminway},
    title        = {{geppyNN: a Python framework for Gene Expression Programming for Neural Networks }},
    month        = December,
    year         = 2023,
    doi          = {10.5281/zenodo.12820711},
    version      = {0.1},
    publisher    = {Zenodo},
    url          = {https://github.com/ryanheminway/geppyNN}
    }
```
Alternatively, if you want a more *academic* citation, you may cite our relevant paper
```
@inproceedings{inproceedings,
    author = {Jonathan, Mwaura and Ryan, Heminway},
    year = {2023},
    month = {12},
    pages = {1829-1836},
    title = {Connectivity Schemas in NeuroEvolution: What Neural Architectures does GEPNN evolve?},
    doi = {10.1109/SSCI52147.2023.10371957}
}
```
