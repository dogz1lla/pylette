"""Read an image from a path and return the top n colors as a palette.
Usage:
    python main.py && open palette.html
"""
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

# fix the random seed for kmeans init
np.random.seed(42)


def open_img_as_array(imgpath: str) -> np.ndarray:
    """NOTE: cuts out the last element of each pixel, i assume it is transparency."""
    image = Image.open(imgpath)
    data = np.asarray(image)
    return data[:, :, :3]


def get_palette(imgarray: np.ndarray, n: int) -> np.ndarray:
    """Given an image as a numpy array, use k-means clustering to identify top n colors and return
    them in an array of rgb triples."""

    def _get_cluster_avg(samples, labels, labels_order=None) -> np.ndarray:
        avgs = list()
        if labels_order is None:
            labels_order = range(n)
        for label in labels_order:
            idxs = labels == label
            avgs.append(np.rint(samples[idxs].mean(axis=0)))
        return np.array(avgs).astype(int)

    def _sort_frequencies(labels) -> np.ndarray:
        """Sort the labels by their frequency of appearance.
        This will output palette in the order of abundance of the
        corresponding color.
        See https://stackoverflow.com/a/45799487"""
        def __sort_fn(x):
            return x[:, 1]
        pred = __sort_fn(labels)
        order = np.argsort(pred)[::-1]  # desc
        return labels[order][:, 0]

    samples = np.vstack(imgarray)
    classifier = KMeans(n_clusters=n)
    labels = classifier.fit_predict(samples)
    freqs = np.array(np.unique(labels, return_counts=True)).T
    ordered_labels = _sort_frequencies(freqs)

    return _get_cluster_avg(samples, labels, ordered_labels)


def rgb_to_hex_str(rgb: np.ndarray) -> str:
    """see https://stackoverflow.com/a/19917486"""
    r, g, b = rgb
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)


def rgb_to_rgb_str(rgb: np.ndarray) -> str:
    r, g, b = rgb
    return f"rgb({r}, {g}, {b})"


def print_as_hex(palette: np.ndarray) -> None:
    for rgb in palette:
        hex_str = rgb_to_hex_str(rgb)
        print(hex_str)


def print_as_rgb(palette: np.ndarray) -> None:
    for rgb in palette:
        rgb_str = rgb_to_rgb_str(rgb)
        print(rgb_str)


def create_html_palette(palette: np.ndarray, as_hex: bool, output_file: str = "palette.html") -> None:
    with open(output_file, "w") as f:
        for rgb in palette:
            color_str = rgb_to_hex_str(rgb) if as_hex else rgb_to_rgb_str(rgb)
            html_line = f"<h1 style='margin-bottom:0;margin-top:0;background-color:{color_str};color:white'>{color_str}</h1>\n"
            f.write(html_line)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        # prog='pylette',
        description='get the color palette of a given image',
        epilog='Enjoy your palettes!',
    )
    parser.add_argument('filename')
    parser.add_argument('-n', '--ncolors', default=5, help='number of kmeans (colors in the palette); default is 5')
    parser.add_argument('--hex', action='store_true', default=False, help='return as hex, otherwise as rgb')
    parser.add_argument('--html', action='store_true', default=False, help='create an html file for visualization (palette.html)')
    args = parser.parse_args()

    data = open_img_as_array(args.filename)
    palette = get_palette(data, args.ncolors)
    if args.hex:
        print_as_hex(palette)
    else:
        print_as_rgb(palette)
    if args.html:
        create_html_palette(palette, args.hex)


if __name__ == "__main__":
    main()
