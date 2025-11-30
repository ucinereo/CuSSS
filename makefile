.PHONY: test

test:
	srun --account=large-sc-2 --container-writable --environment=kernel_fusion_pytorch_container -p debug \
		bash -c "pip install . --no-build-isolation --no-deps && pytest tests/ -v"
