include Makefile.inc


.PHONY: directories clean objects bifrost


# Output directories                                                                                                                        
BIN_DIR     = ./bin
OBJ_DIR     = ./obj

# Paths
SRC_DIR  = ./src
INCLUDE_DIR = ./include

# Compiler flags
OPTIMISE = -O3
DEBUG    = 

# Includes and libraries
INCLUDE  = -I$(INCLUDE_DIR) -I$(THRUST_DIR) -I${DEDISP_DIR}/include -I${CUDA_DIR}/include -I./tclap
LIBS = -L$(CUDA_DIR)/lib64 -lcudart -L${DEDISP_DIR}/lib -ldedisp -lcufft -lpthread -lnvToolsExt

# compiler flags
# --compiler-options -Wall
NVCC_COMP_FLAGS = -gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_35,code=sm_35
NVCCFLAGS  = -g -G ${UCFLAGS} ${OPTIMISE} ${NVCC_COMP_FLAGS} --machine 64 -Xcompiler ${DEBUG}
CFLAGS    = ${UCFLAGS} -fPIC ${OPTIMISE} ${DEBUG}

SRC_FILES = ${SRC_DIR}/bifrost.cu ${SRC_DIR}/pipeline_heimdall.cu ${SRC_DIR}/error.cpp \
		${SRC_DIR}/measure_bandpass.cu ${SRC_DIR}/remove_baseline.cu \
		${SRC_DIR}/get_rms.cu ${SRC_DIR}/median_filter.cu ${SRC_DIR}/matched_filter.cu \
		${SRC_DIR}/find_giants.cu ${SRC_DIR}/client_socket.cpp ${SRC_DIR}/socket.cpp \
		${SRC_DIR}/label_candidate_clusters.cu ${SRC_DIR}/merge_candidates.cu

OBJECTS   = ${OBJ_DIR}/kernels.o ${OBJ_DIR}/pipeline_heimdall.o ${OBJ_DIR}/error.o \
                ${OBJ_DIR}/measure_bandpass.o ${OBJ_DIR}/remove_baseline.o \
                ${OBJ_DIR}/get_rms.o ${OBJ_DIR}/median_filter.o ${OBJ_DIR}/matched_filter.o \
                ${OBJ_DIR}/find_giants.o ${OBJ_DIR}/client_socket.o ${OBJ_DIR}/socket.o \
                ${OBJ_DIR}/label_candidate_clusters.o ${OBJ_DIR}/merge_candidates.o 
EXE_FILES = ${BIN_DIR}/bifrost #${BIN_DIR}/resampling_test ${BIN_DIR}/harmonic_sum_test

all: bifrost
objects: directories ${OBJECTS}
bifrost: ${BIN_DIR}/bifrost

${OBJ_DIR}/%.o: ${SRC_DIR}/%.cpp
	echo "Make "$@ " from " $<
	${CC} -c ${INCLUDE} $< -o $@

${OBJ_DIR}/%.o: ${SRC_DIR}/%.cu
	echo "Make "$@ " from " $<
	${NVCC} -c ${NVCCFLAGS} ${INCLUDE} $<  -o $@

${BIN_DIR}/bifrost: ${OBJ_DIR}/bifrost.o objects
	${NVCC} ${NVCCFLAGS} ${INCLUDE} ${LIBS} ${OBJ_DIR}/bifrost.o ${OBJECTS} -o $@

${BIN_DIR}/harmonic_sum_test: ${SRC_DIR}/harmonic_sum_test.cpp ${OBJECTS}
	${NVCC} ${NVCCFLAGS} ${INCLUDE} ${LIBS} $^ -o $@

${BIN_DIR}/resampling_test: ${SRC_DIR}/resampling_test.cpp ${OBJECTS}
	${NVCC} ${NVCCFLAGS} ${INCLUDE} ${LIBS} $^ -o $@

${BIN_DIR}/coincidencer: ${SRC_DIR}/coincidencer.cpp ${OBJECTS}
	${NVCC} ${NVCCFLAGS} ${INCLUDE} ${LIBS} $^ -o $@

${BIN_DIR}/accmap: ${SRC_DIR}/accmap.cpp ${OBJECTS}
	${NVCC} ${NVCCFLAGS} ${INCLUDE} ${LIBS} $^ -o $@

${BIN_DIR}/rednoise: ${SRC_DIR}/rednoise_test.cpp ${OBJECTS}
	${NVCC} ${NVCCFLAGS} ${INCLUDE} ${LIBS} $^ -o $@

${BIN_DIR}/hcfft: ${SRC_DIR}/hcfft.cpp ${OBJECTS}
	${NVCC} ${NVCCFLAGS} ${INCLUDE} ${LIBS} $^ -o $@

${BIN_DIR}/folder_test: ${SRC_DIR}/folder_test.cpp ${OBJECTS}
	${NVCC} ${NVCCFLAGS} ${INCLUDE} ${LIBS} $^ -o $@

${BIN_DIR}/dedisp_test: ${SRC_DIR}/dedisp_test.cpp ${OBJECTS}
	${NVCC} ${NVCCFLAGS} ${INCLUDE} ${LIBS} $^ -o $@ 

directories:
	@mkdir -p ${BIN_DIR}
	@mkdir -p ${OBJ_DIR}

clean:
	@rm -rf ${BIN_DIR}/*	
	@rm -rf ${OBJ_DIR}/*
