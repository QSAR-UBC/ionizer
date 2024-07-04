from setuptools import setup

requirements = [
    "pennylane>=0.33",
]

setup(
    name="Ionizer",
    version="0.3",
    description="PennyLane tools for compilation into trapped-ion native gates.",
    author="UBC Quantum Software and Algorithms Research Group",
    url="https://github.com/QSAR-UBC/ionizer",
    packages=["ionizer", "ionizer.resources"],
    include_package_data=True,
    package_data={
        "ionizer.resources": [
            "double_gate_identities.pkl",
            "triple_gate_identities.pkl",
        ]
    },
)
