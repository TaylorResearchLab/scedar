# Procedure to build API doc

Prior to build. Manually change rst files in docs/geneted.

1. Install `sphinx`, `nbsphinx`, `ipykernel`, and `sphinx_rtd_theme`.
2. Change shell working directory to `docs`.
3. Run `make clean` to remove last build.
4. Run `make html` to build html.
5. Check build.
