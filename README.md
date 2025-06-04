# math_datasets

## Overview
`math_datasets` is a Python library designed to provide various mathematical datasets for research and development purposes. It aims to simplify the process of accessing and utilizing mathematical data in various applications.

## Installation
You can install the `math_datasets` library using Poetry. First, ensure you have Poetry installed, then run the following command:

```bash
poetry add math_datasets
```

Alternatively, you can clone the repository and install it locally:

```bash
git clone https://github.com/yourusername/math_datasets.git
cd math_datasets
poetry install
```

## Usage
Here is a simple example of how to use the `math_datasets` library:

```python
from math_datasets import core

# Example usage of a function from core.py
data = core.load_dataset('example_dataset')
print(data)
```

## Contributing
Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Make your changes and commit them (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a pull request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.