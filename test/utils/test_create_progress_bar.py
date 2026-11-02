'''Behavioral contract for klygo.utils.create_progress_bar.'''

import unittest
from unittest.mock import sentinel, patch

from klygo import utils


class TestCreateProgressBar(unittest.TestCase):
    def test_factory_forwards_every_option(self):
        with patch('klygo.utils.progress.ProgressBar', return_value=sentinel.progress) as cls:
            result = utils.create_progress_bar(
                total=None,
                desc='Encoding',
                unit='frame',
                verbose=False,
                colour='green',
                unit_scale=True,
                unit_divisor=1000,
            )

        self.assertIs(result, sentinel.progress)
        cls.assert_called_once_with(
            total=None,
            desc='Encoding',
            unit='frame',
            verbose=False,
            colour='green',
            unit_scale=True,
            unit_divisor=1000,
        )


if __name__ == '__main__':
    unittest.main()
