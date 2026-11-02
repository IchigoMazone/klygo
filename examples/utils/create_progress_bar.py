'''Executable example for klygo.utils.create_progress_bar.'''

from klygo import utils


progress = utils.create_progress_bar(
    total=2,
    desc='Saving',
    unit='file',
    verbose=False,
)
progress.update(2)
progress.close()
assert progress.bar is None
