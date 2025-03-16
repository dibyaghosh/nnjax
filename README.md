Playing around with some neural networks and JAX. Decided to do it in pure JAX, no flax or other libraries.


Installation:

```
uv pip sync requirements.txt
```

If you want to change the dependencies, change `requirements.in` and run:

```
uv pip compile requirements.in --output-file=requirements.txt --prerelease=allow --python=3.11 --emit-find-links
uv pip sync requirements.txt
```