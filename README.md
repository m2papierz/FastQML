# FastQML [PROJECT IN PROGRESS]

The project's goal is to develop a Python library that integrates Pennylane, JAX, and Flax for efficient and scalable 
Quantum Machine Learning (QML) applications. This library leverages JAX for differential calculations, ensuring 
high-speed computations and optimal performance. Pennylane's inclusion offers remarkable flexibility in quantum 
computing, while Flax enables the creation of classical deep learning models that are fully compatible with JAX.

This library is designed to facilitate the implementation of QML models for research and benchmarking. It enables both 
the development of custom QML models, including fully quantum or hybrid models, and also offers ready-made, common QML 
architectures. These ready-to-use models are tailored for applications in image processing and natural language 
processing, making the library a versatile tool for a wide range of QML applications.

Already existing functionalities can be checked in demonstration notebooks:  
- [Multiclass Classification Using QNNClassifier in FastQML](demo/quantum_neural_network_demo.ipynb) - presentation of
basic functionalities of FastQML
- [Multiclass Classification with Hybrid Quantum-Classical DenseNet in FastQML](demo/hybrid_densenet_classification_demo.ipynb) -
example use case of quantum-classical hybrid DenseNet 