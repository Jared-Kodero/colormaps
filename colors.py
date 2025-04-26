import difflib
import os
import sys
from typing import Literal

import cmocean
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from cmocean import cm as cmo
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_rgba
from typing import Union

class IPCCColorMaps:

    def __new__(
        cls,
    ):
        instance = super(IPCCColorMaps, cls).__new__(cls)
        instance.N = 256

        instance.colormaps = {}
        instance._load_colormaps()

        return instance

    set_plot_theme()

    SRC_DIR = SCRIPT_DIR / "continuous_colormaps_rgb_0-1" 

    def _load_colormaps(self):
        files = os.listdir(f"{self.SRC_DIR}")
        ipcc_cmap_list = [f.replace(".txt", "") for f in files]


        for cmap_name in ipcc_cmap_list:
            color_file = self.SRC_DIR / f"{cmap_name}.txt"
            colormap_data = np.loadtxt(color_file)
            cmap = LinearSegmentedColormap.from_list(cmap_name, colormap_data, N=self.N)
            self.colormaps[cmap_name] = cmap
            self.colormaps[f"{cmap_name}_r"] = cmap.reversed()

       

    @property
    def values(self):
        return [name for name in self.colormaps]

    @property
    def cmaps(self):
        for cmap_name in self.colormaps:

            fig, ax = plt.subplots(figsize=(5, 0.3))

            plt.imshow(
                np.linspace(0, 1, self.N).reshape(1, self.N),
                aspect="auto",
                cmap=self.colormaps[cmap_name],
            )

            fig.patch.set_visible(False)
            ax.axis("off")
            plt.title(cmap_name, loc="left", color="white", fontsize=8)

            plt.show()

    def _hint(self, name, hint=None):

        suggestions = difflib.get_close_matches(name, self.colormaps.keys(), n=5)
        if suggestions != []:
            hint = f"Did you mean one of {suggestions}?"
        else:
            hint = f"No close matches found for '{name}'."
        return hint

    def _validate_getitem(self, key):
        if isinstance(key, (list, tuple)):
            if all(
                isinstance(c, str) and (c.startswith("#") or c in mcolors.CSS4_COLORS)
                for c in key
            ):
                return BlendedColormap(key, self)
            else:
                raise ValueError(
                    "Invalid colors specified. Please provide a list of valid color names or hex values."
                )
        elif isinstance(key, str):
            if key in self.colormaps:
                return IPCCColorMapsManager(key, self.colormaps[key], self)
            else:
                raise KeyError(f"Colormap '{key}' is not available {self._hint(key)}")

        else:
            raise TypeError(
                "Invalid key type, please provide a valid colormap name or a list of colors in hex format."
            )

    def adjust(
        self,
        cmap_name,
        N: int = 25,

        *,
        split: tuple[float, float] = None,
        discrete: bool = True,
        reverse: bool = False,
    ):

        cmap = self.colormaps.get(cmap_name)
        if cmap is None:
            raise KeyError(
                f"Colormap '{cmap_name}' is not available {self._hint(cmap_name)}"
            )
        
        if reverse:
            cmap = cmap.reversed()
        
        start, end = 0, 1
        if split is not None:
            start, end = split 


        
        range_values = np.linspace(start, end, N)
        colors = [cmap(value) for value in range_values]

        if discrete:
            res = ListedColormap(colors, N=N, name=cmap.name)
        else:
            res = LinearSegmentedColormap.from_list(cmap.name, colors, N=N)

        return res

        

    def __iter__(self):
        return iter(self.colormaps)

    def __len__(self):
        return len(self.colormaps)

    def __dir__(self):
        return list(self.colormaps.keys())

    def __contains__(self, key):
        return key in self.colormaps

    def __getattr__(self, key):
        if key in self.colormaps:
            return self.colormaps[key]
        else:
            raise KeyError(f"Colormap '{key}' is not available {self._hint(key)}")

    def __getitem__(self, key):
        return self._validate_getitem(key)


class IPCCColorMapsManager:
    def __init__(self, name: str, cmap, parent: "IPCCColorMaps"):
        self.name = name
        self._cmap = cmap
        self._parent = parent

    @property
    def get(self):
        return self._cmap

    def adjust(
        self,
        N: int = 25,
        *,
        split: float = None,
        add_colors: Union[str, list[str]] = None,
        add_color_pos: Literal[1, 0, -1] = 1,
        discrete: bool = True,
       
    ):
        """
        Access a colormap by name and make adjustments to it.

        Parameters:

            cmap : str, optional
                    The name of the colormap to load. If specified, the colormap is returned directly`.
            N : int, default=256
                    The number of colors in the colormap.
            split : float, default=None
                    Multiply the colormap by a float 0-1 to split the colormap.
            add_colors : str, default=None
                    The color to add to the colormap. If specified, the color is added to the colormap at the specified position.
            add_color_pos : int, default=1
                    The position to add the color to the colormap. 1 for start, 0 for middle or -1 for end.
            discrete : bool, default=True
                Whether to create a discrete colormap.
        Returns:

            cmap : matplotlib.colors.Colormap
                    The colormap object

        Examples:

            >>> ipcc_cmaps["temp_seq"].adjust(N=10) # returns the temp_seq colormap with 10 colors
            >>> ipcc_cmaps["temp_seq"].adjust(N=10, discrete=True) # returns the temp_seq colormap with 10 colors as a continuous colormap
            >>> ipcc_cmaps["temp_seq"] # returns the temp_seq colormap
            >>> ipcc_cmaps["temp_seq"].adjust(N=10, split = 0.5) # returns the upper half of the temp_seq colormap with 10 colors

        """
        self._cmap = self._parent.adjust(self.name, N=N, discrete=discrete)

        if split is not None:
            self._cmap = self.__mul__(split)

        if add_colors is not None:
            self._cmap = self.add_colors(
                add_colors, N=N, pos=add_color_pos, discrete=discrete
            )

        return self._cmap

    def __mul__(self, other):
        """
        Multiply the colormap by a float 0-1 to split the colormap.
        """

        reverse = False

        if not -1 <= other <= 1:
            raise ValueError("Value must be between -1 and 1.")

        split = (0, other)

        if isinstance(other, (int, float)):
            if other == -1:
                return self._cmap.reversed()
            if other == 1:
                return self._cmap
            if other == 0:
                 raise ValueError("Multiplying by 0 is ambiguous. Use 1 for full, -1 for reversed.")

            if other < 0:
                reverse = True


            self._cmap = self._parent.adjust(self.name, N=self.N, split=split, reverse=reverse)
            return self._cmap
        else:
            raise TypeError("Value must be a float between 0 and 1.")

    def preview(self,N=None):
        fig, ax = plt.subplots(figsize=(6, 0.5))

        N = self.N if N is None else N

        plt.imshow(
            np.linspace(0, 1, N).reshape(1, N), aspect="auto", cmap=self._cmap
        )

        fig.patch.set_visible(False)
        ax.axis("off")
        plt.title(self.name, loc="left", color="white")

        plt.show()

    def add_colors(self, obj, *, N=None, pos: Literal[1, 0, -1] = 1, discrete=True):

        N = self.N if N is None else N
        odd_N = lambda x: x if x % 2 == 1 else x + 1
        N= odd_N(N) if pos == 0 else N
      
        objs =list(obj) if not isinstance(obj, (list)) else obj

        colors_to_add = []

        for obj in objs:

            if isinstance(obj, str):
                if obj.startswith("#"):
                    colors_to_add.append(to_rgba(obj))
                elif mcolors.CSS4_COLORS.get(obj) is not None:
                    colors_to_add.append(to_rgba(mcolors.CSS4_COLORS[obj]))

                else:
                    raise ValueError(
                        f"Invalid color '{obj}'. Must be a hex code or a named CSS4 color."
                    )
            else:
                raise TypeError("Color must be a string (hex or named CSS4 color).")
            
        colors = self._cmap(np.linspace(0, 1, N))

        if pos == 1:
            colors = np.vstack([colors_to_add, colors])
        elif pos == -1:
            colors = np.vstack([colors, colors_to_add])
        elif pos == 0:
            half = len(colors) // 2
            colors = np.vstack([
                colors[:half],
                np.array(colors_to_add),
                colors[half:]
            ])
        else:
            raise ValueError("`pos` must be 1 for start, 0 for middle or -1 for end.")

        if discrete:
            cmap = ListedColormap(colors, N=N, name=self._cmap.name)
        else:
            cmap = LinearSegmentedColormap.from_list(self._cmap.name, colors, N=N)

        self._cmap = cmap
        return self._cmap

    def reverse(self):
        return self._cmap.reversed()

    def __dir__(self):
        return list(self._parent.colormaps.keys())

    def __call__(self):
        return self
    

    def __getattr__(self, attr):
        return getattr(self._cmap, attr)

    def __repr__(self):
        return self._cmap.name

    def __str__(self):
        return self._cmap.name


class BlendedColormap:
    def __init__(self, colors, parent: "IPCCColorMaps"):
        self.colors = colors
        self._parent = parent

    def blend(self, N: int = 25, *, discrete: bool = True):
        """
        Generates a colormap by blending a list of colors.

        Parameters:

            N : int, default=256
            discrete : bool, default=True

        Returns:

            cmap: matplotlib.colors.Colormap


        Examples

            >>> cmap = ipcc_cmaps["#000000", "#ff0000", "#ffffff"].blend(N=256, discrete=True)

        """

        range_values = np.linspace(0, 1, N)

        cmap = sns.blend_palette(self.colors, as_cmap=True, input="hex", n_colors=N)
        color_list = [cmap(value) for value in range_values]
        if discrete:
            res = ListedColormap(color_list, N=N, name=f"Blend_{len(self.colors)}")
        else:
            res = LinearSegmentedColormap.from_list(
                f"Blend_{len(self.colors)}", color_list, N=N
            )

        return res


ipcc_cmap = IPCCColorMaps()
