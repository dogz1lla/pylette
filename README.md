# Description
pylette is a python script that takes an image file and uses [K-means](https://scikit-learn.org/1.5/modules/clustering.html#k-means) clustering to get the `n` most common colors in the image.

# Examples
Here is an example of getting the palette of one of the many beautiful shots from the movie [Come
True](https://www.youtube.com/watch?v=woSc2_xE7KI):

![come_true](./come_true_palette.png)

And another one -- this time a shot i like from Star Wars Episode 2:

![palps](./sw2_palette.png)

# Requirements
This script requires `pillow`, `scikit-learn` and `numpy`.

# Usage
See the script usage description by running
```
python main.py -h
```
