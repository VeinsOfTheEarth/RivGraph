---
title: 'RivGraph: Automatic extraction and analysis of river and delta channel network topology'
tags:

  - Python
  - rivers
  - deltas
  - image processing
  - networks
  - authors:
  - name:Jon Schwenk^[Corresponding author]
    orcid: 0000-0001-5803-9686
    affiliation: "1"
  - name: Jayaram Hariharan
    orcid: 0000-0002-1343-193X
    affiliation: "2"
affiliations:
 - name: Los Alamos National Laboratory, Divison of Earth and Enviornmental Sciences
   index: 1
 - name: Department of Civil, Architectural and Environmental Engineering, The University of Texas at Austin
   index: 2

date: 29 September 2020
bibliography: paper.bib


# Summary
Rivers are the "veins of the Earth" that sustain life and landscapes by carrying and distributing water, sediment, and nutrients throughout ecosystems and communities. Human reliance on rivers as 

Perhaps no environments are more sensitive to the.

River channel networks are important to a wide range of questions and scales.

Water, sediment, and nutrients move through the Earth's river channels

River channel network

Braided rivers were added shortly after.

 As RivGraph was developed, its feature set grew as its applications became more diverse. 

`RivGraph` was designed with an emphasis on user-friendliness and accessibility, guided by the idea that even novice Python users should be able to make use of its functionality. Anticipated common workflows are gathered into classes that manage georeferencing conversions, path management, and I/O with simple, clearly-named methods. 



# Statement of need

Satellite and aerial photography have provided unprecedented opportunities to study the structure and dynamics of rivers and their networks. As both the quantity and quality of these remotely-sensed observations grow, the need for tools that automatically map and measure river channel network properties has grown in turn. The genesis of `RivGraph` is rooted in the work of [Tejedor, Tejedor, and Tejedor] in a revitalized effort to see river channel networks through the lenses of their network structure. The authors were relegated to time-consuming hand-delineations of the delta channel networks they analyzed.  `RivGraph` was thus born from a need to transform binary masks of river channel networks into their graphical representations accurately, objectively, and efficiently. 

The development of the flow directions algorithms itself provided some insights into the nature of river channel network structure in braided rivers and deltas [Schwenk 2020]. For deltas specifically, RivGraph-extracted networks have been used to study how water and sediment are partitioned at bifurcations [Dong et al. https://doi.org/10.1029/2020WR027199], to determine how distance to the channel network plays a controlling role on Arctic delta lake dynamics [Vulius et al. doi.org/10.1029/2019GL086710], and to construct a network-based model of nitrate removal across the Wax Lake Delta [Knights et al. https://doi.org/10.1029/2019WR026867]. For braided rivers, RivGraph was used to extract channel networks from Delft3D simulations [] in order to develop the novel "entropic braided index" [eBI, Tejedor], and a function for computing the eBI (and the classic braiding index) for braided rivers is provided in RivGraph. The work of [Marra et al] represented an effort to understand braided rivers through their topologies, although their networks were extracted manually. Ongoing, unpublished work is using RivGraph to study river dynamics, delta loopiness, and nutrient transport through Arctic deltas. 

RivGraph is not strictly for channel networks; for example, single-threaded channels may also be analyzed with the 'centerline' class.

`Gala` is an Astropy-affiliated Python package for galactic dynamics. Python
enables wrapping low-level languages (e.g., C) for speed without losing
flexibility or ease-of-use in the user-interface. The API for `Gala` was
designed to provide a class-based and user-friendly interface to fast (C or
Cython-optimized) implementations of common operations such as gravitational
potential and force evaluation, orbit integration, dynamical transformations,
and chaos indicators for nonlinear dynamics. `Gala` also relies heavily on and
interfaces well with the implementations of physical units and astronomical
coordinate systems in the `Astropy` package [@astropy] (`astropy.units` and
`astropy.coordinates`).

`Gala` was designed to be used by both astronomical researchers and by
students in courses on gravitational dynamics or astronomy. It has already been
used in a number of scientific publications [@Pearson:2017] and has also been
used in graduate courses on Galactic dynamics to, e.g., provide interactive
visualizations of textbook material [@Binney:2008]. The combination of speed,
design, and support for Astropy functionality in `Gala` will enable exciting
scientific explorations of forthcoming data releases from the *Gaia* mission
[@gaia] by students and experts alike.

# Functionality and Ease of Use

RivGraph is made accessible 

Care is also taken to preserve the georeferencing information of the binary mask, if available, to ensure that all exports can be easily incorporated into a GIS.



Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Fenced code blocks are rendered with syntax highlighting:
```python
for n in range(10):
    yield f(n)
```

# Acknowledgements

We thank Efi Foufoula-Georgiou, Alejandro Tejedor, Anthony Longjas, Lawrence Vulius, Kensuke Naito, and Deon Knights for providing test cases and feedback for RivGraph's development. We are also grateful to Anastasia Piliouras and Joel Rowland for providing valuable insights and subsequent testing of RivGraph's flow directionality algorithms. 

RivGraph has received financial support from NSF under EAR-1719670, the United States Department of Energy, and Los Alamos National Laboratory's Lab Directed Research and Development (LDRD) program. Special thanks are due to Dr. Efi Foufoula-Georgiou for providing support during the nascent phase of RivGraph's development.

# References
