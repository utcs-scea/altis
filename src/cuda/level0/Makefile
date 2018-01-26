include ../make.config

CC := $(CUDA_DIR)/bin/nvcc

INCLUDE := -I$(CUDA_DIR)/include -I../common

SRC = BusSpeedDownload.cu

EXE = BusSpeedDownload

release: $(SRC)
	$(CC) $(SRC) -o $(EXE) $(INCLUDE) -L$(CUDA_LIB_DIR) 

enum: $(SRC)
	$(CC) -deviceemu $(SRC) -o $(EXE) $(INCLUDE) -L$(CUDA_LIB_DIR) 

debug: $(SRC)
	$(CC) -g $(SRC) -o $(EXE) $(INCLUDE) -L$(CUDA_LIB_DIR) 

debugenum: $(SRC)
	$(CC) -g -deviceemu $(SRC) -o $(EXE) $(INCLUDE) -L$(CUDA_LIB_DIR) 

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result.txt

