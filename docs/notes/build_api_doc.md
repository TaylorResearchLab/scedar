# Procedure to build API doc

1. Remove `build` and `sources/*rst`.
2. Change shell working directory to `docs`.
3. Run `sphinx-apidoc -o source ../scedar`, this will generate several `.rst` files under `source` directory.
4. Rename `module.rst` to `index.rst`. `sphinx` build requires an `index.rst` file under `source` directory.
5. Run `make html`.
