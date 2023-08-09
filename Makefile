
clean-py:
	find . -type d -name __pycache__ -exec rm -r {} \+

clean-cache:
	rm -rf .cache examples/jax_gpu/.cache
