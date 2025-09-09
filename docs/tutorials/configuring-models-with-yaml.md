# Configuring models with YAML

All other tutorials in the documentations show interactive examples in Jupyter notebooks.
This is a great way to start building models or to test specific things interactively.
However, as our model becomes more stable, we may also want to test several model configurations (different model classes, different classes, etc.).
Scripts and configuration files are often better suited for this purpose.
Luckily, `simpple` also has a YAML loader to configure models with a static config file.
