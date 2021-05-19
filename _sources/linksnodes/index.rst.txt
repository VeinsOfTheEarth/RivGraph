.. _linksnodes:

===========================
Link and Node Dictionaries
===========================

After defining a network in *RivGraph* with :obj:`rivgraph.classes.rivnetwork.compute_network()`, two dictionaries will be created:
**links** and **nodes**.
This page of the documentation will describe the *keys* present in these dictionaries.

.. note::
   Not all dictionary key:value pairs are the same length as the number of links or nodes. Some dictionary key:value pairs contain meta-information about the network, or information that only applies to a subset of links/nodes. When the number of key:value pairs matches the number of links or nodes, then they are aligned such that the i'th index of any key:value pair refers to the same link or node.

- :ref:`links`
- :ref:`nodes`

.. _links:

--------------------
The Links Dictionary
--------------------

Links Key Values
----------------

**N** represents the number of links.

.. csv-table:: Generic Link Keys
   :file: links_generic.csv
   :header-rows: 1

.. csv-table:: Directionality-specific Keys
   :file: links_dir.csv
   :header-rows: 1

.. csv-table:: Braided River Exclusive Keys
   :file: links_river.csv
   :header-rows: 1

.. csv-table:: Delta Exclusive Keys
   :file: links_delta.csv
   :header-rows: 1

Accessing Link Values
---------------------

For example, if you know the *link_id* of the link you are interested in, you can get its index with :code:`links['id'].index(link_id)`.

.. _nodes:

--------------------
The Nodes Dictionary
--------------------

Nodes Key Values
----------------

**M** represents the number of nodes.

.. csv-table:: Generic Node Keys
   :file: nodes_generic.csv
   :header-rows: 1

Accessing Node Values
---------------------
For example, if you wish to find the links connected to :code:`node_id == 66`, you can use :code:`nodes['conn'][nodes['id'].index(66)]`.
