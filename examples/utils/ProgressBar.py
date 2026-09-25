'''Executable example for klygo.utils.ProgressBar.'''

from klygo import utils


with utils.ProgressBar(
    total=3,
    desc='Processing',
    unit='item',
    verbose=False,
) as progress:
    for _ in range(3):
        progress.update()

assert progress.bar is None
