Applications
============
To use DeepChem models on downstream applications tasks, it's often necessary
to have additional infrastructure to apply DeepChem effectively to scientific
problems of interest. The :code:`dc.applications` module contains an assortment
of non-machine-learning code that facilitates the use of DeepChem for
applications.

Genomics
--------

DeepChem currently has utilities to generate synthetic datasets for genomics
applications.

DNA Simulation
^^^^^^^^^^^^^^
DeepChem uses the :code:`simdna` package to simulate some synthetic DNA
distributions of interest.

.. autofunction:: deepchem.applications.genomics.dnasim.simple_motif_embedding

.. autofunction:: deepchem.applications.genomics.dnasim.motif_density

.. autofunction:: deepchem.applications.genomics.dnasim.simulate_single_motif_detection

.. autofunction:: deepchem.applications.genomics.dnasim.simulate_motif_counting

.. autofunction:: deepchem.applications.genomics.dnasim.simulate_motif_density_localization

.. autofunction:: deepchem.applications.genomics.dnasim.simulate_multi_motif_embedding

.. autofunction:: deepchem.applications.genomics.dnasim.simulate_differential_accessibility

.. autofunction:: deepchem.applications.genomics.dnasim.simulate_heterodimer_grammar
