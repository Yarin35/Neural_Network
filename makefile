BINARIES = my_torch_generator my_torch_analyzer

all: $(BINARIES)

my_torch_generator: generator/main.py
	echo "#!/usr/bin/env python3" > $@
	cat $< >> $@
	chmod +x $@

my_torch_analyzer: analyzer/main.py
	echo "#!/usr/bin/env python3" > $@
	cat $< >> $@
	chmod +x $@

clean:
	mkdir -p generator analyzer
	rm -f *~

fclean: clean
	rm -f $(BINARIES)

re: fclean all

.PHONY: clean fclean re