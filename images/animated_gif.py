import imageio
from pathlib import Path
images = []
for filename in Path.cwd().glob('grad*.png'):
    images.append(imageio.imread(filename))
imageio.mimsave('grad-descent-anim.gif', images, duration=0.2)