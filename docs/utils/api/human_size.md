# `utils.human_size`

```python
human_size(num_bytes: int, decimal_places: int = 2) -> str
```

`formatting.py` converts a non-negative byte count to B, KB, MB, GB, or TB
using steps of 1024. `decimal_places` controls displayed precision.
The function raises `TypeError` for a non-integer byte count and `ValueError`
for a negative one.

```python
from klygo import utils

print(utils.human_size(1048576))  # 1.00 MB
```

See the [runnable example](../../../examples/utils/human_size.py),
[tests](../../../test/utils/test_human_size.py), and
[tutorial](../../../tutorials/utils/Document_Utils_Formatting.ipynb).
