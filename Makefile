
default:
	@echo "Please specify a target to make."

clean-checkpoints:
	rm -rf tests/*/*.checkpoint

clean-codebooks:
	rm -rf tests/*/codebooks.pt

clean-tables:
	rm -rf tests/table-*.csv

clean-samples:
	rm -rf tests/*/samples-*.pt

clean-all:
	rm -rf tests
