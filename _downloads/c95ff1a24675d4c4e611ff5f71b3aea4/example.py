"""
Example script of how to generate a gallery item
================================================

The gallery uses python files. But within the files you can embed rST, as
the gallery converts python into rST and we really only want to use text and
images anyway. To have a rST block you just need a line "# %%" to start and
subsequent lines start with "#" so they aren't read as Python.

The first line below this docstring is where the thumbnail figure path is
provided.

"""
# sphinx_gallery_thumbnail_path = 'gallery_source/images/ex.png'
# kinda stuck with a box here though - typically you'd have python import text

# %%
# This is the first text block following the header above.
# We want to link to the original issue `click me <https://github.com/jonschwenk/RivGraph/issues/69>`_.


# %%
# This is a separate block which I think will render as a new paragraph.

# %%
# here we explictly include the image

# %%
# .. image:: ../gallery_source/images/ex.png
