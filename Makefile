
clean-checkpoints:
	rm -rf tests/*/*.checkpoint

clean-codebooks:
	rm -rf tests/*/codebooks.pt

clean-samples:
	rm -rf tests/*/samples-*.pt

clean:
	find tests/* -d -type d -exec rm -rf '{}' \;

clean-all:
	rm -rf tests