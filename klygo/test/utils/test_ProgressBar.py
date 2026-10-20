'''Behavioral contract for klygo.utils.ProgressBar.'''

import unittest
from unittest.mock import patch

from klygo import utils


class TestProgressBar(unittest.TestCase):
    def test_disabled_mode_is_a_safe_no_op(self):
        with patch('klygo.utils.progress.tqdm') as tqdm:
            with utils.ProgressBar(3, 'Disabled', verbose=False) as progress:
                progress.update(2)
                self.assertIsNone(progress.bar)

            tqdm.assert_not_called()

    def test_configuration_update_close_and_context_manager(self):
        with patch('klygo.utils.progress.tqdm') as tqdm:
            rendered = tqdm.return_value
            progress = utils.ProgressBar(
                total=4,
                desc='Extracting',
                unit='file',
                colour='blue',
                unit_scale=False,
                unit_divisor=1024,
            )

            tqdm.assert_called_once_with(
                total=4,
                desc='Extracting',
                unit='file',
                unit_scale=False,
                unit_divisor=1024,
                colour='blue',
                ascii=' █',
                leave=True,
            )
            progress.update(3)
            rendered.update.assert_called_once_with(3)
            progress.close()
            rendered.close.assert_called_once_with()
            self.assertIsNone(progress.bar)

            progress.close()
            rendered.close.assert_called_once_with()

        with patch('klygo.utils.progress.tqdm') as tqdm:
            rendered = tqdm.return_value
            with utils.ProgressBar(1, 'Context') as managed:
                self.assertIs(managed.bar, rendered)
            rendered.close.assert_called_once_with()
            self.assertIsNone(managed.bar)


if __name__ == '__main__':
    unittest.main()
