"""
Utilities for the visualization scripts in the `visuals/` directory
"""

import os

# font size definitions shared across the produced figures
LEGEND_FONTSIZE = 8
LABEL_FONTSIZE  = 8
TITLE_FONTSIZE  = 10
TICK_FONTSIZE   = 8


def set_size(width, kind = None):
    """
    Defines standardized ratio for `figsize` in the produced images

    Parameters
    --------
    `width`: float
        Figure width in inches

    Returns
    ----------
    `(width, height)`: tuuple
        Typle with the specified height/width ratio
    """

    # 10 by 7 ration
    if kind == 'simple':
        height = width / 1.42857

    elif kind == 'golden':
        height = width / 1.61803

    elif kind == 'square':
        height = width

    elif kind == 'half':
        height = width / 2

    else:
        raise ValueError(f"Specified kind \"{kind}\" not found.")

    return (width, height)


def check_img_dst() -> None:
    """
    Checks if the figure destination directory exists. In case it doesn't, this
    function creates the `.pdf`, `.jpeg` and `.pgf` directories.
    """

    if not os.path.exists('./jpeg'):
        os.mkdir('./jpeg')

    if not os.path.exists('./pdf'):
        os.mkdir('./pdf')

    if not os.path.exists('./pgf'):
        os.mkdir('./pgf')
