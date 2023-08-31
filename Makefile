
clean-checkpoint:
	rm -rf tests/*/*.checkpoint

clean:
	find tests/* -d -type d -exec rm -rf '{}' \;

clean-all:
	rm -rf tests