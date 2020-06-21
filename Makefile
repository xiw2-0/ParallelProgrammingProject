SRC_DIR = ./src
BUILD_DIR = ./build
TEST_DIR = ./test

ccsrc = ${wildcard ${SRC_DIR}/*.cc}
cctest = ${wildcard ${TEST_DIR}/*.cc}

ccobj = ${patsubst ${SRC_DIR}%, ${BUILD_DIR}%, ${ccsrc:.cc=.o}}
cctestobj = ${patsubst ${TEST_DIR}%, ${BUILD_DIR}%, ${cctest:.cc=.o}}

mainobj = %main_test.o %word_count_test.o %all_gather_test.o %mpi_convnet_ops_test.o
obj = ${filter-out ${mainobj}, ${ccobj} ${cctestobj}}

TEST = ${BUILD_DIR}/test
test_obj = ${BUILD_DIR}/main_test.o

INC_DIR = -I${SRC_DIR}
CCFLAGS = ${INC_DIR} -std=c++11 -g `mpicc -showme:compile`
LDFLAGS = -lpthread `mpicc -showme:link` -fopenmp

WORD_COUNT = ${BUILD_DIR}/word_count
word_count_obj = ${BUILD_DIR}/word_count_test.o

ALL_GATHER = ${BUILD_DIR}/all_gather
all_gather_obj = ${BUILD_DIR}/all_gather_test.o

MPI_CONVNET_OPS = ${BUILD_DIR}/mpi_convnet_ops_test
mpi_convnet_ops_test_obj = ${BUILD_DIR}/mpi_convnet_ops_test.o

all: ${TEST} ${WORD_COUNT} ${ALL_GATHER} ${MPI_CONVNET_OPS}
.PHONY: all

${TEST}: ${test_obj} $(obj)
	$(CXX) $^ -o $@  $(LDFLAGS)

${WORD_COUNT}: ${word_count_obj} ${obj}
	${CXX} $^ -o $@  $(LDFLAGS) 

${ALL_GATHER}: ${all_gather_obj} ${obj}
	${CXX} $^ -o $@  $(LDFLAGS)

${MPI_CONVNET_OPS}: ${mpi_convnet_ops_test_obj} ${obj}
	${CXX} $^ -o $@  $(LDFLAGS)


${BUILD_DIR}/%.o: ${TEST_DIR}/%.cc
	$(CXX) -c $< -o $@ ${CCFLAGS}

${BUILD_DIR}/%.o: ${SRC_DIR}/%.cc
	$(CXX) -c $< -o $@ ${CCFLAGS}


.PHONY: clean
clean:
	rm -f $(obj) ${test_obj} \
				${TEST}
