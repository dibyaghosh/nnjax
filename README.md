# JaxNN

In every person's JAX journey, they decide that they need to re-implement a neural network library for JAX.

This is mine.

For my own reference (and maybe yours), here are some implementations (from the ground up, no code hidden from any library) of various models.

Installation:

```
uv pip sync requirements.txt
```

If you want to change the dependencies, change `requirements.in` and run:

```
uv pip compile requirements.in --output-file=requirements.txt --prerelease=allow --python=3.11 --emit-find-links
uv pip sync requirements.txt
```